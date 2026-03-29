"""
play.py
Run the 3v3 soccer game with random actions.
Close the window or press ESC to quit.

Usage:
    python play.py
"""

import pygame
from football_env import SoccerEnv

env = SoccerEnv(render_mode="human")
env.reset(seed=42)
env.render()  # initialises pygame before the event loop

running = True
while running:
    # ── handle window close / ESC ──────────────────────────────────────────
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False

    # ── random actions for both teams ──────────────────────────────────────
    actions = {
        "team_a": env.action_space["team_a"].sample(),
        "team_b": env.action_space["team_b"].sample(),
    }

    obs, rewards, terminated, truncated, info = env.step(actions)
    env.render()

    if truncated:
        print(f"Final score — A: {info['score'][0]}  B: {info['score'][1]}")
        env.reset()

env.close()