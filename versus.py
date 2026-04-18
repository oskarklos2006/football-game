import os
import sys
import numpy as np
import pygame
from stable_baselines3 import PPO
from football_env import SoccerEnv, FIELD_W, FIELD_H

# Import both builders
from train_attacker import get_relative_obs as get_attacker_obs
from train_defender import get_defender_obs

# ==========================================
# SET YOUR MODELS HERE
# ==========================================
TEAM_A_PATH = "best_attacker"  # Usually 12 features
TEAM_B_PATH = "best_defender"  # Usually 16 features
# ==========================================

N_EPISODES = 5
RENDER = True


def get_obs_for_model(model, env, team_idx):
    """
    Checks the model's expected input shape and returns the
    correct observation (12 or 16 features).
    """
    # Get the shape the model expects (e.g., 12 or 16)
    expected_shape = model.observation_space.shape[0]

    if expected_shape == 16:
        # It's a Defender model
        return get_defender_obs(env._pos, env._vel, env._ball_pos, env._ball_vel, team_idx, 1)
    else:
        # It's an Attacker model
        # Note: Attacker's get_relative_obs handles team_idx 0 or 1 internally
        return get_attacker_obs(env._pos, env._vel, env._ball_pos, env._ball_vel, team_idx, 1)


def run_tournament():
    print(f"\n>>> LOADING MATCH: {TEAM_A_PATH} vs {TEAM_B_PATH}")
    model_a = PPO.load(TEAM_A_PATH)
    model_b = PPO.load(TEAM_B_PATH)

    env = SoccerEnv(render_mode="human" if RENDER else None, n_players=1)
    if RENDER:
        pygame.init()
        clock = pygame.time.Clock()

    a_goals, b_goals = 0, 0

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

                # Get correct observations based on what each model was trained with
                obs_a = get_obs_for_model(model_a, env, 0)  # Team A (Left)
                obs_b = get_obs_for_model(model_b, env, 1)  # Team B (Right)

                # Predict
                act_a, _ = model_a.predict(obs_a, deterministic=True)
                act_b, _ = model_b.predict(obs_b, deterministic=True)

                # Step
                obs_raw, _, terminated, truncated, info = env.step({
                    "team_a": act_a.reshape(1, 2),
                    "team_b": act_b.reshape(1, 2)
                })

                done = terminated or truncated
                if RENDER:
                    env.render()
                    clock.tick(60)

            score = info['score']
            a_goals += score[0]
            b_goals += score[1]
            print(f"Ep {ep + 1}: [{TEAM_A_PATH}] {score[0]} - {score[1]} [{TEAM_B_PATH}]")

        print("\n" + "=" * 30 + "\n FINAL: " + f"{a_goals} - {b_goals}" + "\n" + "=" * 30)

    finally:
        env.close()
        if RENDER: pygame.quit()


if __name__ == "__main__":
    run_tournament()