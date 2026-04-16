import os
import sys
import numpy as np
import pygame
from stable_baselines3 import PPO
from football_env import SoccerEnv, FIELD_W, FIELD_H

# Ensure get_relative_obs is defined or imported from train.py
from train import get_relative_obs

RENDER = "--no-render" not in sys.argv
N_EPISODES = 5
# Curriculum check: adjust path if necessary
MODEL_PATH = "best_phase0" if os.path.exists("best_phase0.zip") else "team_a"

# ---------------------------------------------------------------------------
# Load Model
# ---------------------------------------------------------------------------
model = None
if os.path.exists(f"{MODEL_PATH}.zip"):
    model = PPO.load(MODEL_PATH)
    print(f"Loaded '{MODEL_PATH}.zip'")
else:
    print("No model found — playing randomly")

# Initialize Env
env = SoccerEnv(render_mode="human" if RENDER else None, n_players=1)

# Fix for window visibility/speed
if RENDER:
    pygame.init()
    clock = pygame.time.Clock()

# ---------------------------------------------------------------------------
# Run Episodes
# ---------------------------------------------------------------------------
try:
    for ep in range(N_EPISODES):
        obs_raw, _ = env.reset(seed=ep)

        # Build initial GPS observation
        obs_a = get_relative_obs(env._pos, env._vel, env._ball_pos, env._ball_vel, 0, 1)

        done = False
        ep_touches = 0

        print(f"Starting Episode {ep + 1}...")

        while not done:
            # 1. Keep Window Responsive
            if RENDER:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("Exiting...")
                        env.close()
                        sys.exit()

            # 2. Predict Action
            if model is not None:
                action_a, _ = model.predict(obs_a, deterministic=True)
            else:
                action_a = env.action_space["team_a"].sample()

            # 3. Environment Step
            obs_raw, _, terminated, truncated, info = env.step({
                "team_a": action_a.reshape(1, 2),
                "team_b": env.action_space["team_b"].sample(),
            })

            # 4. Update GPS Observation
            obs_a = get_relative_obs(env._pos, env._vel, env._ball_pos, env._ball_vel, 0, 1)

            done = terminated or truncated

            # Track touches
            agent_pos = env._pos[0]
            from football_env import PLAYER_RADIUS, BALL_RADIUS

            if np.linalg.norm(env._ball_pos - agent_pos) < PLAYER_RADIUS + BALL_RADIUS + 0.6:
                ep_touches += 1

            # 5. Render at Human Speed
            if RENDER:
                env.render()
                # Slow down the simulation so human eyes can see the 98 goals!
                clock.tick(30)

        print(f"Episode {ep + 1} Finished: Goals: {info['score'][0]}, Touches: {ep_touches}")

except SystemExit:
    pass
except Exception as e:
    print(f"\nCaught Error: {e}")
finally:
    env.close()
    if RENDER:
        pygame.quit()
    print("Cleanup complete.")