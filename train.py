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

        Components (rough priority order):
          1. Pass completed          +1.5   ← most important
          2. Ball touch (1 player)   +0.15  ← only the closest, not everyone
          3. Ball moving toward goal +0.3   ← when ball velocity points right
          4. Ball position progress  +0.06  ← per step, scaled by ball x
          5. Team spread bonus       +0.04  ← reward spatial coverage
          6. Clustering penalty      -0.15  ← punish two players < 3 units apart
          7. Goal rewards            kept from env (+10 / -10)
        """
        pos      = self._env._pos        # (12,2)
        ball     = self._env._ball_pos   # (2,)
        ball_vel = self._env._ball_vel   # (2,)
        reward   = goal_reward           # start with +10/-10 if a goal happened

        team_a = pos[:6]   # players 0-5

        # ── 1 & 2. Touch + Pass ──────────────────────────────────────────────
        dists = np.linalg.norm(team_a - ball, axis=1)   # (6,)
        closest_i   = int(np.argmin(dists))
        closest_dist = dists[closest_i]

        if closest_dist < KICK_RANGE:
            # Only the closest player gets a touch reward
            reward += 0.15

            # Pass: a different team_a player is now closest → pass completed
            if self._last_toucher is not None and self._last_toucher != closest_i:
                reward += 1.5

            self._last_toucher = closest_i
        else:
            # Ball not near anyone — reset pass chain
            self._last_toucher = None

        # ── 3. Ball moving toward opponent goal ──────────────────────────────
        # Reward when ball velocity has a strong rightward (positive x) component
        if ball_vel[0] > 0.5:
            reward += 0.3 * (ball_vel[0] / KICK_POWER)

        # ── 4. Ball position progress ────────────────────────────────────────
        reward += 0.06 * (ball[0] / FIELD_W)

        # ── 5. Team spread bonus ─────────────────────────────────────────────
        # Reward standard deviation of player positions — spread out team plays better
        spread = float(np.std(team_a, axis=0).mean())
        reward += 0.04 * spread

        # ── 6. Clustering penalty ────────────────────────────────────────────
        # Penalise pairs of team_a players that are too close to each other
        for i in range(6):
            for j in range(i + 1, 6):
                d = np.linalg.norm(team_a[i] - team_a[j])
                if d < 3.0:
                    reward -= 0.15

        self._last_ball_pos = ball.copy()
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
