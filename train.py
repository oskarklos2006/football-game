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

from football_env import (
    SoccerEnv, FIELD_W, FIELD_H,
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

# Opponent goal centre (team_a attacks right)
OPP_GOAL = np.array([FIELD_W, FIELD_H / 2], dtype=np.float32)


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
            "ball_to_goal": 0.0,
            "move_cost":    0.0,
            "spread":       0.0,
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
        Reward components:
          1. Ball progression toward opponent goal   (delta distance, per step)
          2. Possession                              (+0.1 when closest player has ball)
          3. Ball chase                              (closest player, only when no possession)
          4. Zone adherence                          (per player, scale depends on possession)
          5. Pass                                    (+1.5 once when ball changes hands via kick)
          6. GK positioning                          (strong penalty for leaving own box)
          7. Goal scored / conceded                  (±10 from env, passed in as goal_reward)
        """
        pos    = self._env._pos       # (12, 2)
        ball   = self._env._ball_pos  # (2,)
        reward = goal_reward          # ±10 on goals, 0 otherwise

        team_a       = pos[:6]
        dists        = np.linalg.norm(team_a - ball, axis=1)   # (6,)
        closest_i    = int(np.argmin(dists))
        closest_dist = float(dists[closest_i])
        in_possession = closest_dist < KICK_RANGE

        # ── 1. Ball progression ───────────────────────────────────────────
        # Skip on goal steps: after a goal the ball teleports to centre,
        # which would produce a large spurious negative delta.
        ball_dist = float(np.linalg.norm(ball - OPP_GOAL))
        if goal_reward == 0.0:
            delta = self._prev_ball_dist - ball_dist   # positive = ball closer to goal
            reward += 0.3 * float(np.clip(delta, -2.0, 2.0))
        self._prev_ball_dist = ball_dist

        # ── 2. Possession reward ──────────────────────────────────────────
        if in_possession:
            reward += 0.1

        # ── 3. Chase ball (only when team lacks possession) ───────────────
        # Encourages the nearest free player to close down the ball.
        if not in_possession:
            reward += 0.2 * max(0.0, 1.0 - closest_dist / 15.0)

        # ── 4. Zone adherence ─────────────────────────────────────────────
        # When defending (no possession) players should hold shape strongly.
        # When attacking they get more freedom to make runs.
        zone_scale = 0.03 if in_possession else 0.06
        for i in range(6):
            px        = float(team_a[i, 0])
            zx_min, zx_max = ZONE_X[i]
            if zx_min <= px <= zx_max:
                reward += zone_scale
            else:
                x_out   = max(zx_min - px, px - zx_max)
                reward -= zone_scale * min(x_out / 5.0, 2.0)

        # ── 5. Pass reward ────────────────────────────────────────────────
        # Fires ONCE when the ball arrives at a different player after a kick.
        # _last_kicker is set in step() only when a kick action was executed.
        if in_possession:
            if self._last_kicker is not None and self._last_kicker != closest_i:
                reward += 1.5
                self._last_kicker = None   # consume — don't fire again next step
            elif self._last_kicker is None:
                self._last_kicker = closest_i  # first touch after reset

        # ── 6. GK positioning ─────────────────────────────────────────────
        # Player 0 is the goalkeeper. Penalise leaving the defensive third.
        gk_x = float(team_a[0, 0])
        if gk_x > 12.0:
            reward -= 0.5 * (gk_x - 12.0) / FIELD_W

        # ── 7. Own-goal-area camping (non-GK) ────────────────────────────
        for i in range(1, 6):
            if float(team_a[i, 0]) < 4.0:
                reward -= 0.3

        # Reset shaping state after a goal so next kickoff starts clean
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

    env = TeamAEnv(render_mode=None)

    if os.path.exists(f"{MODEL_NAME}.zip"):
        print(f"Found '{MODEL_NAME}.zip' — resuming training.")
        model = PPO.load(MODEL_NAME, env=env)
    else:
        print("No existing model found — starting from scratch.")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=LOG_DIR,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,        # higher entropy → more exploration of new strategies
            clip_range=0.2,
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
