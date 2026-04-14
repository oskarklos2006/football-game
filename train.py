"""
train.py
Train team_a (blue) with PPO against a random opponent.
Run:  python train.py
"""

import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from football_env import (
    SoccerEnv, FIELD_W, FIELD_H, GOAL_H,
    PLAYER_RADIUS, BALL_RADIUS,
)

KICK_RANGE = PLAYER_RADIUS + BALL_RADIUS + 0.5   # same as env internals

# Soft x-zone for each team_a player [min_x, max_x].
# Players are rewarded for staying in their zone when out of possession
# and lightly penalised for leaving it.
#   0 = GK        1,2 = defenders    3,4 = midfielders    5 = forward
ZONE_X = [
    (0,   10),   # GK – stay in own box
    (5,   25),   # left defender
    (5,   25),   # right defender
    (15,  40),   # left midfielder
    (15,  40),   # right midfielder
    (25,  50),   # forward – attacking half
]

# Opponent goal geometry (team_a attacks right)
GOAL_Y0   = (FIELD_H - GOAL_H) / 2
GOAL_Y1   = GOAL_Y0 + GOAL_H
OPP_GOAL  = np.array([FIELD_W, FIELD_H / 2], dtype=np.float32)


class TeamAEnv(gym.Env):
    """
    Wraps SoccerEnv for single-agent SB3 training.
    team_a is trained; team_b acts randomly (or via an optional model).
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, opponent_model=None):
        super().__init__()
        # Disable built-in shaping — all shaping is handled below.
        self._env = SoccerEnv(render_mode=render_mode, reward_cfg={
            "ball_to_goal":   0.0,
            "move_cost":      0.0,
            "spread":         0.0,
            "goal_scored":   100.0,
            "goal_conceded": -100.0,
        })
        self._opponent       = opponent_model
        self._last_obs_b     = None
        self._last_kicker    = None   # index 0-5 of last team_a player who kicked
        self._prev_ball_dist = 0.0    # distance from ball to opponent goal, last step

        self.observation_space = self._env.observation_space["team_a"]
        self.action_space      = self._env.action_space["team_a"]
        self.render_mode       = render_mode

    # ------------------------------------------------------------------ #
    def _opponent_action(self, obs_b):
        if self._opponent is None:
            return self._env.action_space["team_b"].sample()
        action, _ = self._opponent.predict(obs_b, deterministic=False)
        return action

    def reset(self, seed=None, options=None):
        obs, info = self._env.reset(seed=seed, options=options)
        self._last_obs_b     = obs["team_b"]
        self._last_kicker    = None
        self._prev_ball_dist = float(np.linalg.norm(self._env._ball_pos - OPP_GOAL))
        return obs["team_a"], info

    # ------------------------------------------------------------------ #
    def step(self, action_a):
        # Snapshot positions/ball BEFORE the physics step so kick detection
        # uses the state in which the kick was issued.
        ball_pre = self._env._ball_pos.copy()
        pos_pre  = self._env._pos[:6].copy()

        action_b = self._opponent_action(self._last_obs_b)
        obs, env_rewards, terminated, truncated, info = self._env.step({
            "team_a": action_a,
            "team_b": action_b,
        })
        self._last_obs_b = obs["team_b"]

        # Detect which team_a player actually kicked this step.
        for i in range(6):
            if action_a[i, 2] > 0:
                if np.linalg.norm(ball_pre - pos_pre[i]) < KICK_RANGE:
                    self._last_kicker = i
                    break

        reward = self._custom_reward(env_rewards["team_a"])
        return obs["team_a"], reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    def _custom_reward(self, goal_reward: float) -> float:
        """
        Pure goal-scoring reward. Every signal points directly at scoring.

          1. Goal scored / conceded     ±100  (dominant signal)
          2. Ball progression           +0.5 * delta toward opponent goal
          3. Shot on target             up to +0.5 when ball in goal-mouth zone
          4. Possession                 +0.1 when team has ball
          5. Ball chase                 +0.2 when closing down loose ball
          6. GK positioning             penalty for leaving own box
        """
        pos    = self._env._pos
        ball   = self._env._ball_pos
        reward = goal_reward

        team_a       = pos[:6]
        dists        = np.linalg.norm(team_a - ball, axis=1)
        closest_i    = int(np.argmin(dists))
        closest_dist = float(dists[closest_i])
        in_possession = closest_dist < KICK_RANGE

        # ── 1. Ball progression toward opponent goal ──────────────────────
        ball_dist = float(np.linalg.norm(ball - OPP_GOAL))
        if goal_reward == 0.0:
            delta = self._prev_ball_dist - ball_dist
            reward += 1.0 * float(np.clip(delta, -2.0, 2.0))
        self._prev_ball_dist = ball_dist

        # ── 2. Shot on target ─────────────────────────────────────────────
        # Strong reward — ball near goal mouth is exactly where it needs to be.
        bx, by = float(ball[0]), float(ball[1])
        if bx > FIELD_W * 0.6 and GOAL_Y0 <= by <= GOAL_Y1:
            closeness = (bx - FIELD_W * 0.6) / (FIELD_W * 0.4)
            reward += 2.0 * closeness

        # ── 3. Possession ─────────────────────────────────────────────────
        if in_possession:
            reward += 0.2

        # ── 4. Ball carrier advances toward goal ──────────────────────────
        # Rewards the player holding the ball for running forward.
        # This is the missing link: possession alone isn't enough,
        # the agent needs to learn to drive toward the goal with the ball.
        if in_possession:
            carrier_vx = float(self._env._vel[closest_i, 0])
            if carrier_vx > 0:
                reward += 0.5 * carrier_vx   # max ~0.5/step at full speed

        # ── 5. Chase loose ball ───────────────────────────────────────────
        if not in_possession:
            reward += 0.3 * max(0.0, 1.0 - closest_dist / 15.0)

        # ── 5b. Soft anti-clustering ──────────────────────────────────────
        # Non-closest players within 5u of the ball get a gentle deterrent.
        # Keeps them from all chasing in a pack without causing kick-and-run.
        # -0.02 vs original -0.2 — 10x weaker, nudge not a shove.
        for i in range(6):
            if i != closest_i and dists[i] < 5.0:
                reward -= 0.02

        # ── 6. GK must stay in goal ───────────────────────────────────────
        gk_x = float(team_a[0, 0])
        if gk_x > 12.0:
            reward -= 0.05 * (gk_x - 12.0) / FIELD_W

        # ── 7. Wall penalty — all 4 walls ────────────────────────────────
        # Gentle nudge only — must stay well below goal reward in magnitude.
        # Max per player per step: 0.003 * 4 = 0.012. All 6: 0.072/step = 72/episode.
        WALL_M = 4.0
        for i in range(6):
            px = float(team_a[i, 0])
            py = float(team_a[i, 1])
            if py < WALL_M:
                reward -= 0.003 * (WALL_M - py)
            elif py > FIELD_H - WALL_M:
                reward -= 0.003 * (py - (FIELD_H - WALL_M))
            if i != 0 and px < WALL_M:
                reward -= 0.003 * (WALL_M - px)
            elif px > FIELD_W - WALL_M:
                reward -= 0.003 * (px - (FIELD_W - WALL_M))

        # ── 8. Too far from ball ──────────────────────────────────────────
        # Max per player: 0.0005 * 30 = 0.015/step = 15/episode. Nudge only.
        for i in range(1, 6):
            if dists[i] > 20.0:
                reward -= 0.0005 * (dists[i] - 20.0)

        if goal_reward != 0.0:
            self._last_kicker = None

        return reward

    # ------------------------------------------------------------------ #
    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()


# ======================================================================= #
if __name__ == "__main__":
    TOTAL_STEPS    = 1_000_000
    SAVE_EVERY     = 50_000
    MODEL_NAME     = "team_a"
    LOG_DIR        = "./logs/"
    CHECKPOINT_DIR = "./checkpoints/"
    N_ENVS         = 4   # parallel games — raise if you have more CPU cores

    # SubprocVecEnv runs each env in its own process (true parallelism).
    # If it crashes on startup, replace SubprocVecEnv with DummyVecEnv.
    env = make_vec_env(TeamAEnv, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)

    if os.path.exists(f"{MODEL_NAME}.zip"):
        print(f"Found '{MODEL_NAME}.zip' — resuming training.")
        model = PPO.load(MODEL_NAME, env=env)
        model.learning_rate = 1e-4
        model.n_epochs      = 5
        model.clip_range    = lambda _: 0.1
    else:
        print("No existing model found — starting from scratch.")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=LOG_DIR,
            n_steps=2048,
            batch_size=256,
            n_epochs=5,
            learning_rate=1e-4,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,        # higher entropy → more exploration of new strategies
            clip_range=0.1,
            use_sde=True,
            policy_kwargs={"squash_output": True, "net_arch": [256, 256]},
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=SAVE_EVERY,
        save_path=CHECKPOINT_DIR,
        name_prefix=MODEL_NAME,
        verbose=1,
    )

    print(f"Training for {TOTAL_STEPS:,} steps...")
    model.learn(total_timesteps=TOTAL_STEPS, callback=checkpoint_cb)

    model.save(MODEL_NAME)
    print(f"\nDone. Model saved to '{MODEL_NAME}.zip'")
    env.close()
