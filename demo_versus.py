import os
import sys
import numpy as np
import pygame
from stable_baselines3 import PPO
from football_env import SoccerEnv, FIELD_W

# Import the GPS logic from your train script
from train import get_relative_obs

# ==========================================
# SET YOUR MODELS HERE
# ==========================================
TEAM_A_PATH = "basic_correct_weights"  # Your new pro
TEAM_B_PATH = "best_model_a"  # The old boss
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
                            env.close()
                            sys.exit()

                # Team A — normal observation
                obs_a = get_relative_obs(env._pos, env._vel, env._ball_pos, env._ball_vel, 0, 1)

                # Team B — mirror the world so it looks like it's Team A attacking right
                pos_flipped = env._pos.copy()
                pos_flipped[:, 0] = FIELD_W - pos_flipped[:, 0]  # flip all x positions
                pos_flipped[[0, 1]] = pos_flipped[[1, 0]]         # swap so B's player is index 0

                vel_flipped = env._vel.copy()
                vel_flipped[:, 0] = -vel_flipped[:, 0]            # flip all x velocities
                vel_flipped[[0, 1]] = vel_flipped[[1, 0]]         # swap to match

                ball_pos_flipped = np.array([FIELD_W - env._ball_pos[0], env._ball_pos[1]])
                ball_vel_flipped = np.array([-env._ball_vel[0], env._ball_vel[1]])

                obs_b = get_relative_obs(pos_flipped, vel_flipped, ball_pos_flipped, ball_vel_flipped, 0, 1)

                # Predict actions
                act_a, _ = model_a.predict(obs_a, deterministic=True)
                act_b, _ = model_b.predict(obs_b, deterministic=True)

                # Flip B's x action back to real coordinates
                act_b_real = np.array([[-act_b[0], act_b[1]]])

                # Step the environment
                obs_raw, _, terminated, truncated, info = env.step({
                    "team_a": act_a.reshape(1, 2),
                    "team_b": act_b_real
                })

                done = terminated or truncated

                if RENDER:
                    env.render()
                    clock.tick(60)

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
        if RENDER:
            pygame.quit()


if __name__ == "__main__":
    run_tournament()