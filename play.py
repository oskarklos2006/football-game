import os
import pygame
from stable_baselines3 import PPO
from football_env import SoccerEnv

env = SoccerEnv(render_mode="human", n_players=1)
obs, _ = env.reset(seed=42)
env.render()

if os.path.exists("team_a.zip"):
    model = PPO.load("team_a")
    expected = env.observation_space["team_a"].shape
    if model.observation_space.shape != expected:
        print(f"Model obs shape {model.observation_space.shape} != env {expected} — ignoring, playing randomly.")
        model = None
    else:
        print("Loaded trained model.")
else:
    model = None
    print("No model found — both teams play randomly.")

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False

    if model is not None:
        action_a, _ = model.predict(obs["team_a"], deterministic=True)
    else:
        action_a = env.action_space["team_a"].sample()

    obs, _, terminated, truncated, info = env.step({
        "team_a": action_a,
        "team_b": env.action_space["team_b"].sample(),
    })
    env.render()

    if truncated or terminated:
        print(f"Final score — A: {info['score'][0]}  B: {info['score'][1]}")
        obs, _ = env.reset()

env.close()
