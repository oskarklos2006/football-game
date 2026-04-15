import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from football_env import SoccerEnv, FIELD_W

PA_W = 8.0   # penalty area width (matches renderer)


# wraps SoccerEnv for single-agent SB3 training
class TeamAEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, opponent_holder=None):
        super().__init__()
        self._env             = SoccerEnv(render_mode=render_mode, n_players=1)
        self._opponent_holder = opponent_holder
        self._last_obs_b      = None

        self.observation_space = self._env.observation_space["team_a"]
        self.action_space      = self._env.action_space["team_a"]
        self.render_mode       = render_mode

    def reset(self, seed=None, options=None):
        obs, info = self._env.reset(seed=seed, options=options)
        self._last_obs_b = obs["team_b"]
        return obs["team_a"], info

    def step(self, action_a):
        opponent = self._opponent_holder[0] if self._opponent_holder else None
        if opponent is not None:
            action_b, _ = opponent.predict(self._last_obs_b, deterministic=False)
        else:
            action_b = self._env.action_space["team_b"].sample()

        obs, env_rewards, terminated, truncated, info = self._env.step({
            "team_a": action_a,
            "team_b": action_b,
        })
        self._last_obs_b = obs["team_b"]
        return obs["team_a"], self._reward(env_rewards["team_a"]), terminated, truncated, info

    def _reward(self, goal_reward: float) -> float:
        reward = goal_reward * 100.0 - 0.01
        bx = float(self._env._ball_pos[0])

        # exponential zone rewards based purely on ball position
        if bx > FIELD_W - PA_W:
            depth = (bx - (FIELD_W - PA_W)) / PA_W
            reward += (np.exp(depth) - 1) / (np.e - 1) * 3.0
        elif bx < PA_W:
            depth = (PA_W - bx) / PA_W
            reward -= (np.exp(depth) - 1) / (np.e - 1) * 3.0

        return reward

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()



if __name__ == "__main__":
    # --- training config ---
    TOTAL_STEPS    = 500_000
    SAVE_EVERY     = 50_000
    MODEL_NAME     = "team_a"
    LOG_DIR        = "./logs/"
    CHECKPOINT_DIR = "./checkpoints/"
    N_ENVS         = 4

    # load previous model as frozen opponent if it exists, otherwise team_b plays randomly
    opponent_holder = [None]
    if os.path.exists(f"{MODEL_NAME}.zip"):
        opponent_holder[0] = PPO.load(MODEL_NAME)
        print(f"Loaded '{MODEL_NAME}.zip' as opponent.")

    env = DummyVecEnv([lambda: Monitor(TeamAEnv(opponent_holder=opponent_holder))] * N_ENVS)

    if os.path.exists(f"{MODEL_NAME}.zip"):
        print(f"Found '{MODEL_NAME}.zip' — resuming training.")
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

    callbacks = CheckpointCallback(save_freq=SAVE_EVERY, save_path=CHECKPOINT_DIR, name_prefix=MODEL_NAME, verbose=1)

    print(f"Training for {TOTAL_STEPS:,} steps...")
    model.learn(total_timesteps=TOTAL_STEPS, callback=callbacks)
    model.save(MODEL_NAME)
    print(f"Saved to '{MODEL_NAME}.zip'")
    env.close()
