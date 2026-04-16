import os
import sys
import numpy as np
import pygame
from stable_baselines3 import PPO
from football_env import SoccerEnv

# Import the GPS logic from your train script
from train import get_relative_obs

# ==========================================
# SET YOUR MODELS HERE
# ==========================================
TEAM_A_PATH = "best_round_2"  # Your new pro
TEAM_B_PATH = "best_round_5"  # The old boss
# ==========================================

N_EPISODES = 10
RENDER = True  # Set to False to get results instantly


def run_tournament():
    # 1. Load Models
    print(f"\n>>> LOADING MATCH: {TEAM_A_PATH} vs {TEAM_B_PATH}")
    model_a = PPO.load(TEAM_A_PATH)
    model_b = PPO.load(TEAM_B_PATH)

    # 2. Setup Env
    env = SoccerEnv(render_mode="human" if RENDER else None, n_players=1)
    if RENDER:
        pygame.init()
        clock = pygame.time.Clock()

    a_total_goals = 0
    b_total_goals = 0

    try:
        for ep in range(N_EPISODES):
            obs_raw, _ = env.reset(seed=ep)
            done = False

            while not done:
                if RENDER:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            env.close();
                            sys.exit()

                # Get specific GPS observations for both players
                # Team A (Index 0)
                obs_a = get_relative_obs(env._pos, env._vel, env._ball_pos, env._ball_vel, 0, 1)
                # Team B (Index 1)
                obs_b = get_relative_obs(env._pos, env._vel, env._ball_pos, env._ball_vel, 1, 1)

                # Predict actions
                act_a, _ = model_a.predict(obs_a, deterministic=True)
                act_b, _ = model_b.predict(obs_b, deterministic=True)

                # Step the environment
                obs_raw, _, terminated, truncated, info = env.step({
                    "team_a": act_a.reshape(1, 2),
                    "team_b": act_b.reshape(1, 2)
                })

                done = terminated or truncated

                if RENDER:
                    env.render()
                    clock.tick(60)  # Smooth 60 FPS gameplay

            # Tally Scores
            score = info['score']
            a_total_goals += score[0]
            b_total_goals += score[1]
            print(f"Episode {ep + 1}: [{TEAM_A_PATH}] {score[0]} - {score[1]} [{TEAM_B_PATH}]")

        print("\n" + "=" * 30)
        print("      FINAL TOURNAMENT SCORE")
        print("=" * 30)
        print(f"{TEAM_A_PATH}: {a_total_goals} goals")
        print(f"{TEAM_B_PATH}: {b_total_goals} goals")

        if a_total_goals > b_total_goals:
            print(f"\nWINNER: {TEAM_A_PATH}! 🎉")
        elif b_total_goals > a_total_goals:
            print(f"\nWINNER: {TEAM_B_PATH}! 🏆")
        else:
            print("\nIT'S A DRAW! ⚽")
        print("=" * 30)

    except KeyboardInterrupt:
        print("\nTournament cancelled.")
    finally:
        env.close()
        if RENDER: pygame.quit()


if __name__ == "__main__":
    run_tournament()