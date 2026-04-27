import os
import sys
import numpy as np
import pygame
from stable_baselines3 import PPO
from football_env import SoccerEnv, FIELD_W, FIELD_H

from train_attacker import get_relative_obs as get_attacker_obs
from train_defender import get_defender_obs

# ---------------------------------------------------------------------------
# Model paths — swap these to pit any two saved models against each other.
# ---------------------------------------------------------------------------
TEAM_A_PATH = "best_attacker"
TEAM_B_PATH = "best_defender"

N_EPISODES = 5
RENDER     = True


# ---------------------------------------------------------------------------
# Observation dispatcher.
# Models trained as attackers expect 12 features; defenders expect 16.
# Rather than hard-coding which team uses which builder, we read the model's
# own observation_space so this works for any attacker/defender combination.
# ---------------------------------------------------------------------------
def get_obs_for_model(model, env, team_idx):
    expected_shape = model.observation_space.shape[0]
    if expected_shape == 16:
        return get_defender_obs(env._pos, env._vel,
                                env._ball_pos, env._ball_vel, team_idx, 1)
    else:
        return get_attacker_obs(env._pos, env._vel,
                                env._ball_pos, env._ball_vel, team_idx, 1)


# ---------------------------------------------------------------------------
# Main match loop — runs N_EPISODES and prints a running + final scoreline.
# Both models run deterministically (no sampling) so results are reproducible.
# ---------------------------------------------------------------------------
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
                            env.close()
                            sys.exit()

                obs_a = get_obs_for_model(model_a, env, 0)
                obs_b = get_obs_for_model(model_b, env, 1)

                act_a, _ = model_a.predict(obs_a, deterministic=True)
                act_b, _ = model_b.predict(obs_b, deterministic=True)

                obs_raw, _, terminated, truncated, info = env.step({
                    "team_a": act_a.reshape(1, 2),
                    "team_b": act_b.reshape(1, 2),
                })

                done = terminated or truncated
                if RENDER:
                    env.render()
                    clock.tick(60)

            score     = info["score"]
            a_goals  += score[0]
            b_goals  += score[1]
            print(f"Ep {ep + 1}: [{TEAM_A_PATH}] {score[0]} - {score[1]} [{TEAM_B_PATH}]")

        print("\n" + "=" * 30 + "\n FINAL: "
              + f"{a_goals} - {b_goals}" + "\n" + "=" * 30)

    finally:
        env.close()
        if RENDER:
            pygame.quit()


if __name__ == "__main__":
    run_tournament()