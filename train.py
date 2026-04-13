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

          1. Closest player chases ball      +0.3   ← scaled by proximity
          2. Ball touch                      +0.3
          3. Pass completed                  +2.0   ← dominant signal
          4. Support positioning             +0.05  ← non-chasers spread out
          5. Too-far penalty                 -0.1   ← stops hiding
          6. Boundary penalty                -0.15  ← stops hugging walls
          7. Goal scored / conceded          +10 / -10
        """
        pos      = self._env._pos        # (12,2)
        ball     = self._env._ball_pos   # (2,)
        reward   = goal_reward           # +10 / -10 on goals

        team_a = pos[:6]

        dists = np.linalg.norm(team_a - ball, axis=1)   # (6,)
        closest_i    = int(np.argmin(dists))
        closest_dist = dists[closest_i]

        # ── 1. Closest player chases ball ────────────────────────────────────
        reward += 0.3 * max(0.0, 1.0 - closest_dist / 20.0)

        # ── 2 & 3. Touch + Pass ──────────────────────────────────────────────
        if closest_dist < KICK_RANGE:
            reward += 0.3   # touching the ball

            if self._last_toucher is not None and self._last_toucher != closest_i:
                reward += 2.0   # pass completed — strongest signal

            self._last_toucher = closest_i
        else:
            self._last_toucher = None

        # ── 4 & 5. Support positions for non-chasing players ─────────────────
        for i in range(6):
            if i == closest_i:
                continue
            d = dists[i]
            if 5.0 < d < 20.0:
                reward += 0.05   # good support position
            elif d > 30.0:
                reward -= 0.1    # too far away

        # ── 6. Boundary penalty ───────────────────────────────────────────────
        margin = 4.0
        for i in range(6):
            px, py = team_a[i]
            if px < margin or px > FIELD_W - margin or \
               py < margin or py > FIELD_H - margin:
                reward -= 0.15

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
    else:
        print("No existing model found — starting from scratch.")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=LOG_DIR,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            learning_rate=3e-4,
            ent_coef=0.01,
            clip_range=0.2,
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
