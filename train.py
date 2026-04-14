"""
train.py
Train team_a (blue) with PPO. team_b plays randomly.
Run:  python train.py
"""

import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from football_env import (
    SoccerEnv, FIELD_W, FIELD_H,
    PLAYER_RADIUS, BALL_RADIUS
)

KICK_RANGE = PLAYER_RADIUS + BALL_RADIUS + 0.5   # same as env internals


class TeamAEnv(gym.Env):
    """
    Wraps SoccerEnv so SB3 sees a single-agent environment.
    team_a is trained with custom rewards; team_b acts randomly.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, opponent_model=None):
        super().__init__()
        # Zero out built-in shaping rewards — custom shaping below handles everything.
        # Goal scored/conceded are kept at their defaults (+10 / -10).
        self._env = SoccerEnv(render_mode=render_mode, reward_cfg={
            "ball_to_goal": 0.0,
            "touch_ball":   0.0,
            "move_cost":    0.0,
            "spread":       0.0,
        })
        self._opponent  = opponent_model
        self._last_obs_b        = None
        self._last_toucher      = None   # index 0-5 of last team_a player who touched ball
        self._last_ball_pos     = None

        self.observation_space = self._env.observation_space["team_a"]
        self.action_space      = self._env.action_space["team_a"]
        self.render_mode       = render_mode

    def _opponent_action(self, obs_b):
        if self._opponent is None:
            return self._env.action_space["team_b"].sample()
        action, _ = self._opponent.predict(obs_b, deterministic=False)
        return action

    def reset(self, seed=None, options=None):
        obs, info = self._env.reset(seed=seed, options=options)
        self._last_obs_b    = obs["team_b"]
        self._last_toucher  = None
        self._last_ball_pos = self._env._ball_pos.copy()
        return obs["team_a"], info

    def _custom_reward(self, goal_reward):
        """
        Reward shaping for team_a.

          1. Closest player approaches ball             +0.3 (scaled by distance)
          2. Ball moving toward opponent goal (vx > 0) +0.3 per step
          3. Pass completed (different player picks up) +1.5
          4. Team spread reward (mean pairwise dist)    +0.0001 per unit
          5. Clustering penalty (<3u from ball)         -0.2 per non-closest player
          6. Too-far penalty (>30u from ball)           -0.1 per player
          7. Own-goal-area / corner penalty             -0.25 per player
          8. Goal scored / conceded                     +10 / -10
        """
        pos      = self._env._pos        # (12,2)
        ball     = self._env._ball_pos   # (2,)
        ball_vx  = self._env._ball_vel[0]
        reward   = goal_reward           # +10 / -10 on goals

        team_a = pos[:6]

        dists = np.linalg.norm(team_a - ball, axis=1)   # (6,)
        closest_i    = int(np.argmin(dists))
        closest_dist = dists[closest_i]

        # ── 1. Closest player chases ball ─────────────────────────────────────
        reward += 0.3 * max(0.0, 1.0 - closest_dist / 20.0)

        # ── 2. Ball moving toward opponent goal ───────────────────────────────
        if ball_vx > 0:
            reward += 0.3 * (ball_vx / 1.0)   # scaled by speed, max ~0.3

        # ── 3. Pass — fires when a DIFFERENT player picks up the ball ─────────
        # Check BEFORE updating last_toucher so the comparison is to the previous holder
        if closest_dist < KICK_RANGE and \
           self._last_toucher is not None and \
           self._last_toucher != closest_i:
            reward += 1.5
        if closest_dist < KICK_RANGE:
            self._last_toucher = closest_i

        # ── 4. Spread reward — incentivises wide play and positioning ──────────
        # Right winger earns more reward staying wide than crowding the ball.
        # Mean pairwise distance ~20-25u when spread, ~2-3u when crowded.
        n = 6
        spread_sum = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                spread_sum += float(np.linalg.norm(team_a[i] - team_a[j]))
        reward += 0.0001 * spread_sum / (n * (n - 1) / 2)

        # ── 5. Clustering penalty — push non-closest players away from ball ────
        for i in range(6):
            if i == closest_i:
                continue
            d = dists[i]
            if d < 6.0:
                reward -= 0.2    # too close — get out of the way
            elif d > 30.0:
                reward -= 0.1    # too far to contribute

        # ── 6. Boundary penalty — own goal area and corners only ──────────────
        # No y-side penalty: wingers are supposed to play wide near the sidelines.
        # Player 0 is the GK — exempt from own-goal penalty, it belongs there.
        for i in range(6):
            px, py = team_a[i]
            if i != 0 and px < 4.0:                         # non-GK camping own goal
                reward -= 0.3
            if (px < 4.0 or px > FIELD_W - 4.0) and \
               (py < 4.0 or py > FIELD_H - 4.0):           # corner camping
                reward -= 0.25

        return reward

    def step(self, action_a):
        action_b = self._opponent_action(self._last_obs_b)
        obs, env_rewards, terminated, truncated, info = self._env.step({
            "team_a": action_a,
            "team_b": action_b,
        })
        self._last_obs_b = obs["team_b"]

        # env_rewards["team_a"] here only contains goal_scored/goal_conceded
        # (ball_to_goal and touch_ball are zeroed out above)
        reward = self._custom_reward(env_rewards["team_a"])

        return obs["team_a"], reward, terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()


if __name__ == "__main__":
    TOTAL_STEPS    = 1_000_000
    SAVE_EVERY     = 50_000
    MODEL_NAME     = "team_a"
    LOG_DIR        = "./logs/"
    CHECKPOINT_DIR = "./checkpoints/"

    env = TeamAEnv(render_mode=None)

    if os.path.exists(f"{MODEL_NAME}.zip"):
        print(f"Found existing '{MODEL_NAME}.zip' — continuing training from checkpoint.")
        model = PPO.load(MODEL_NAME, env=env)
        model.ent_coef = 0.003      # override saved value — reduce randomness
        model.learning_rate = 1e-4  # stabilize after reward scale change
    else:
        print("No existing model found — starting from scratch.")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=LOG_DIR,
            n_steps=2048,
            batch_size=64,
            n_epochs=5,
            learning_rate=1e-4,
            ent_coef=0.003,
            clip_range=0.2,
            use_sde=True,
            policy_kwargs={"squash_output": True},
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
