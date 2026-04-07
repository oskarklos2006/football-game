"""
watch.py
Watch your trained team_a play against a random team_b.
Run:  python watch.py
"""

import pygame
from stable_baselines3 import PPO
from train import TeamAEnv

MODEL_PATH = "team_a"   # loads team_a.zip

model = PPO.load(MODEL_PATH)
env   = TeamAEnv(render_mode="human", opponent_model=None)

obs, _ = env.reset(seed=0)
env.render()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False

    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

    if truncated or terminated:
        score = info["score"]
        print(f"Final score — A (you): {score[0]}  B (random): {score[1]}")
        obs, _ = env.reset()

env.close()
