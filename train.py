import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from football_env import SoccerEnv, FIELD_W, FIELD_H, GOAL_H, PLAYER_RADIUS, BALL_RADIUS

GOAL_Y0  = (FIELD_H - GOAL_H) / 2
GOAL_Y1  = GOAL_Y0 + GOAL_H
OPP_GOAL = np.array([FIELD_W, FIELD_H / 2], dtype=np.float32)
KICK_RANGE = PLAYER_RADIUS + BALL_RADIUS + 0.5


# wraps SoccerEnv for single-agent SB3 training; team_b acts randomly
class TeamAEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self._env = SoccerEnv(render_mode=render_mode, n_players=1)
        self._prev_ball_dist = 0.0
        self._last_obs_b     = None

        self.observation_space = self._env.observation_space["team_a"]
        self.action_space      = self._env.action_space["team_a"]
        self.render_mode       = render_mode

    def reset(self, seed=None, options=None):
        obs, info = self._env.reset(seed=seed, options=options)
        self._last_obs_b     = obs["team_b"]
        self._prev_ball_dist = float(np.linalg.norm(self._env._ball_pos - OPP_GOAL))
        return obs["team_a"], info

    def step(self, action_a):
        action_b = self._env.action_space["team_b"].sample()
        obs, env_rewards, terminated, truncated, info = self._env.step({
            "team_a": action_a,
            "team_b": action_b,
        })
        self._last_obs_b = obs["team_b"]
        return obs["team_a"], self._reward(env_rewards["team_a"]), terminated, truncated, info

    def _reward(self, goal_reward: float) -> float:
        # goal signal dominates; shaping guides the agent between goals
        ball   = self._env._ball_pos
        reward = goal_reward * 100.0

        ball_dist = float(np.linalg.norm(ball - OPP_GOAL))
        if goal_reward == 0.0:
            reward += float(np.clip(self._prev_ball_dist - ball_dist, -2.0, 2.0))
        self._prev_ball_dist = ball_dist

        bx, by = float(ball[0]), float(ball[1])
        if bx > FIELD_W * 0.6 and GOAL_Y0 <= by <= GOAL_Y1:
            reward += 2.0 * (bx - FIELD_W * 0.6) / (FIELD_W * 0.4)

        return reward

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()


# prints mean episode reward every print_freq steps
class RewardCallback(BaseCallback):
    def __init__(self, print_freq=10_000):
        super().__init__()
        self.print_freq = print_freq
        self._current    = None
        self._ep_rewards = []

    def _on_step(self) -> bool:
        if self._current is None:
            self._current = np.zeros(len(self.locals["rewards"]))
        self._current += self.locals["rewards"]
        for i, done in enumerate(self.locals["dones"]):
            if done:
                self._ep_rewards.append(float(self._current[i]))
                self._current[i] = 0.0
        if self.num_timesteps % self.print_freq == 0 and self._ep_rewards:
            print(f"  step {self.num_timesteps:>9,} | mean ep reward: {np.mean(self._ep_rewards[-100:]):.2f}")
        return True


if __name__ == "__main__":
    # --- training config ---
    TOTAL_STEPS    = 1_000_000
    SAVE_EVERY     = 50_000
    MODEL_NAME     = "team_a"
    LOG_DIR        = "./logs/"
    CHECKPOINT_DIR = "./checkpoints/"
    N_ENVS         = 4

    env = make_vec_env(TeamAEnv, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)

    if os.path.exists(f"{MODEL_NAME}.zip"):
        print(f"Found '{MODEL_NAME}.zip' — resuming.")
        model = PPO.load(MODEL_NAME, env=env)
        model.learning_rate = 1e-4
        model.clip_range    = lambda _: 0.1
    else:
        print("Starting from scratch.")
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
            ent_coef=0.01,
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
    model.learn(total_timesteps=TOTAL_STEPS, callback=CallbackList([checkpoint_cb, RewardCallback()]))
    model.save(MODEL_NAME)
    print(f"Saved to '{MODEL_NAME}.zip'")
    env.close()
