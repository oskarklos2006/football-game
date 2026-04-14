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

          1. ALL players rewarded for proximity to ball  +0.1 each (max 6x per step)
          2. Closest player bonus                        +0.3
          3. Ball touch                                  +0.3
          4. Pass completed                              +1.5
          5. Support positioning (5-20u from ball)       +0.05 per player
          6. Too-far penalty (>30u)                      -0.1  per player
          7. Boundary penalty                            -0.15 per player
          8. Goal scored / conceded                      +10 / -10
        """
        pos      = self._env._pos        # (12,2)
        ball     = self._env._ball_pos   # (2,)
        reward   = goal_reward           # +10 / -10 on goals

        team_a = pos[:6]

        dists = np.linalg.norm(team_a - ball, axis=1)   # (6,)
        closest_i    = int(np.argmin(dists))
        closest_dist = dists[closest_i]

        # ── 1. Only closest player chases ball ───────────────────────────────
        # Others are rewarded for support spacing, NOT for clustering at the ball
        reward += 0.3 * max(0.0, 1.0 - closest_dist / 20.0)

        # ── 2. Touch ──────────────────────────────────────────────────────────
        if closest_dist < KICK_RANGE:
            reward += 0.3
            self._last_toucher = closest_i   # remember who had it last

        # ── 3. Pass — fires when a DIFFERENT player picks up a loose ball ─────
        # _last_toucher is NOT reset when the ball is loose, so a real kicked
        # pass (ball rolls, second player picks it up) is correctly detected
        if closest_dist < KICK_RANGE and \
           self._last_toucher is not None and \
           self._last_toucher != closest_i:
            reward += 1.5   # pass completed

        # ── 4. Support positions — spread out, be available for a pass ────────
        for i in range(6):
            if i == closest_i:
                continue
            d = dists[i]
            if 5.0 < d < 20.0:
                reward += 0.08   # good support position
            elif d < 3.0:
                reward -= 0.2    # clustering — get out of the way
            elif d > 30.0:
                reward -= 0.1    # too far away

        # ── 7. Boundary penalty ───────────────────────────────────────────────
        # Must be stronger than support reward (+0.08) so corners are never worth it
        margin = 6.0
        for i in range(6):
            px, py = team_a[i]
            near_x = px < margin or px > FIELD_W - margin
            near_y = py < margin or py > FIELD_H - margin
            if near_x or near_y:
                reward -= 0.25             # edge — clearly bad
            if near_x and near_y:
                reward -= 0.25             # corner — double penalty

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

    policy_kwargs = {"squash_output": True}

    if os.path.exists(f"{MODEL_NAME}.zip"):
        print(f"Found existing '{MODEL_NAME}.zip' — continuing training from checkpoint.")
        model = PPO.load(MODEL_NAME, env=env, policy_kwargs=policy_kwargs)
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
            n_epochs=10,
            learning_rate=1e-4,
            ent_coef=0.003,
            clip_range=0.2,
            policy_kwargs=policy_kwargs,
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
