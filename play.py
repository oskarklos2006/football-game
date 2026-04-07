import os
import pygame
from stable_baselines3 import PPO
from football_env import SoccerEnv

env = SoccerEnv(render_mode="human")
obs, _ = env.reset(seed=42)
env.render()

# Load team_a model if it exists, otherwise play randomly
if os.path.exists("team_a.zip"):
    model = PPO.load("team_a")
    print("Loaded trained team_a model.")
else:
    model = None
    print("No trained model found — both teams play randomly.")

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

    actions = {
        "team_a": action_a,
        "team_b": env.action_space["team_b"].sample(),
    }

    obs, rewards, terminated, truncated, info = env.step(actions)
    env.render()

    if truncated or terminated:
        print(f"Final score — A (you): {info['score'][0]}  B (random): {info['score'][1]}")
        obs, _ = env.reset()

env.close()
