import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from football_env import SoccerEnv, FIELD_W, FIELD_H
from train import get_relative_obs, TeamAEnv  # Reusing your logic

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
SOURCE_MODEL = "best_round_5"  # Start from your latest champion
TARGET_OPPONENT = "correct_weights"  # The specific "Boss" to beat
TOTAL_STEPS = 1_000_000
N_ENVS = 8


# ---------------------------------------------------------------------------
# Fine-Tuning Environment
# ---------------------------------------------------------------------------
class FinalBossEnv(TeamAEnv):
    def __init__(self, opponent_path):
        # We set frozen_fraction to 1.0 to ensure 100% smart opponent
        super().__init__(opponent_path=opponent_path, frozen_fraction=1.0)

    def reset(self, seed=None, options=None):
        # Disable curriculum; ball always starts in the center for the final test
        obs, info = self._env.reset(seed=seed)
        self._last_obs_b = self._get_my_obs(obs, 1)
        self._steps_since_reset = 0
        self._was_in_contact = False

        # Always use the smart opponent
        if self._cached_opponent is None:
            self._cached_opponent = PPO.load(self._opponent_path)
        self._use_frozen = True

        return self._get_my_obs(obs, 0), info


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"\n>>> FINE-TUNING: {SOURCE_MODEL} vs {TARGET_OPPONENT} (100% match)")

    # Create vectorized environment
    env = SubprocVecEnv([lambda: Monitor(FinalBossEnv(TARGET_OPPONENT))] * N_ENVS)

    # Load your champion
    if os.path.exists(f"{SOURCE_MODEL}.zip"):
        model = PPO.load(SOURCE_MODEL, env=env)
        # Lower the learning rate slightly for fine-tuning so we don't break the brain
        model.learning_rate = 5e-5
        model.ent_coef = 0.005  # Lower entropy because he already knows how to play
    else:
        print(f"Error: {SOURCE_MODEL}.zip not found!")
        exit()

    # Train
    model.learn(total_timesteps=TOTAL_STEPS, progress_bar=True)

    # Save the Final Champion
    model.save("final_champion")
    print("\n>>> Done! Final champion saved as final_champion.zip")
    env.close()