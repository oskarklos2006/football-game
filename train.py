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
    PLAYER_RADIUS, BALL_RADIUS, KICK_POWER, MAX_SPEED
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
        # Zero out the env's built-in shaping rewards — we replace them below.
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
        Custom reward shaping exclusively for team_a.

          1. Closest player chases ball      +0.2   ← scales with proximity
          2. Ball touch (closest only)       +0.15
          3. Pass completed                  +0.8   ← most important skill
          4. Support positioning             +0.02  ← non-chasers near ball (5-20u)
          5. Too-far penalty                 -0.05  ← stops hiding in corners
          6. Boundary penalty                -0.12  ← stops hugging walls
          7. Ball moving toward goal         +0.25
          8. Ball position progress          +0.04
          9. Goal rewards                    kept from env (+10 / -10)
        """
        pos      = self._env._pos        # (12,2)
        ball     = self._env._ball_pos   # (2,)
        ball_vel = self._env._ball_vel   # (2,)
        reward   = goal_reward

        team_a = pos[:6]

        dists = np.linalg.norm(team_a - ball, axis=1)   # (6,)
        closest_i    = int(np.argmin(dists))
        closest_dist = dists[closest_i]

        # ── 1. Closest player rewarded for being near ball ───────────────────
        # Smooth gradient: full reward at 0 distance, zero at 20 units away
        reward += 0.2 * max(0.0, 1.0 - closest_dist / 20.0)

        # ── 2 & 3. Touch + Pass ──────────────────────────────────────────────
        if closest_dist < KICK_RANGE:
            reward += 0.15

            if self._last_toucher is not None and self._last_toucher != closest_i:
                reward += 0.8   # pass completed

            self._last_toucher = closest_i
        else:
            self._last_toucher = None

        # ── 4 & 5. Support positions for non-chasing players ─────────────────
        # They should stay 5-20 units from ball (available for a pass, not clustering)
        for i in range(6):
            if i == closest_i:
                continue
            d = dists[i]
            if 5.0 < d < 20.0:
                reward += 0.02   # good support position
            elif d > 30.0:
                reward -= 0.05   # too far away, probably hiding

        # ── 6. Boundary penalty ───────────────────────────────────────────────
        # Explicitly punish hugging the edges/corners of the field
        MARGIN = 4.0
        for i in range(6):
            px, py = team_a[i]
            if px < MARGIN or px > FIELD_W - MARGIN or \
               py < MARGIN or py > FIELD_H - MARGIN:
                reward -= 0.12

        # ── 7. Ball moving toward opponent goal ───────────────────────────────
        if ball_vel[0] > 0.5:
            reward += 0.25 * (ball_vel[0] / KICK_POWER)

        # ── 8. Ball position progress ─────────────────────────────────────────
        reward += 0.04 * (ball[0] / FIELD_W)

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
    TOTAL_STEPS    = 500_000
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
