"""
ultimate_match.py
=================
2v2 match — attacker + defender (LEFT, blue)  vs  MF1 + MF2 (RIGHT, red).

Player layout
-------------
  x=0 (left goal)                              x=50 (right goal)

  [0] Team A attacker  — scores at x=50  — trained side, no mirror, action as-is
  [1] Team A defender  — defends  x=0   — OPPOSITE side, mirrored obs, action x negated
  [2] Team B MF1       — scores at x=0  — trained side (team_b), no mirror, action as-is
  [3] Team B MF2       — scores at x=0  — trained side (team_b), no mirror, action as-is

Spawn fix
---------
  SoccerEnv hardcodes pos[1] at x=40 (right half) and pos[2]/pos[3] also at x=40.
  We manually override positions after reset so each player starts on the
  correct side of the field for their role.

Controls  SPACE=pause  R=restart  +/-=speed  Q/ESC=quit
"""

import sys, argparse, os
import numpy as np
import pygame
from stable_baselines3 import PPO
from football_env import SoccerEnv, FIELD_W, FIELD_H

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ATTACKER_PATH = "best_attacker"
DEFENDER_PATH = "best_defender"
MF1_PATH      = "mf1"
MF2_PATH      = "mf2"

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--episodes",  type=int, default=5)
parser.add_argument("--no-render", action="store_true")
parser.add_argument("--fps",       type=int, default=60)
args       = parser.parse_args()
RENDER     = not args.no_render
N_EPISODES = args.episodes
BASE_FPS   = args.fps


# ---------------------------------------------------------------------------
# Spawn positions — correct side for each role
#
#   [0] Attacker : x=10  (left half)   attacks right  ✓ matches env default
#   [1] Defender : x=10  (left half)   defends x=0    ← FIXED (was x=40)
#   [2] MF1      : x=35  (right half)  attacks left   ← FIXED spread out
#   [3] MF2      : x=42  (right half)  attacks left   ← FIXED spread out
# ---------------------------------------------------------------------------
SPAWN = np.array([
    [FIELD_W * 0.20, FIELD_H / 2],   # [0] attacker
    [FIELD_W * 0.15, FIELD_H / 2],   # [1] defender — left half, near own goal
    [FIELD_W * 0.70, FIELD_H * 0.4], # [2] MF1 — right half, slightly high
    [FIELD_W * 0.85, FIELD_H * 0.6], # [3] MF2 — right half, slightly low
], dtype=np.float32)


def reset_with_correct_spawns(env, seed):
    """Reset env then immediately overwrite player positions."""
    obs, info = env.reset(seed=seed)
    env._pos[:] = SPAWN.copy()
    env._vel[:] = 0.0
    return obs, info


# ===========================================================================
# Observation builders
# Attacker/defender copied verbatim from working play_match.py.
# MF obs matches train_midfielders.py get_mf_obs().
# ===========================================================================

def obs_attacker_normal(env, player_idx):
    p, v   = env._pos[player_idx], env._vel[player_idx]
    b, bv  = env._ball_pos, env._ball_vel
    target = np.array([FIELD_W, FIELD_H / 2])
    return np.array([
        p[0] / FIELD_W * 2 - 1,    p[1] / FIELD_H * 2 - 1,
        v[0],                       v[1],
        (b - p)[0] / FIELD_W,      (b - p)[1] / FIELD_H,
        (target - p)[0] / FIELD_W, (target - p)[1] / FIELD_H,
        b[0] / FIELD_W * 2 - 1,    b[1] / FIELD_H * 2 - 1,
        bv[0],                      bv[1],
    ], dtype=np.float32)


def obs_defender_mirrored(env, player_idx):
    p, v   = env._pos[player_idx], env._vel[player_idx]
    b, bv  = env._ball_pos, env._ball_vel
    opp    = min([env._pos[2], env._pos[3]], key=lambda o: np.linalg.norm(o - b))
    p_mx   = FIELD_W - p[0]
    b_mx   = FIELD_W - b[0]
    opp_mx = FIELD_W - opp[0]
    return np.array([
        p_mx / FIELD_W * 2 - 1,          p[1] / FIELD_H * 2 - 1,
        -v[0],                            v[1],
        (b_mx - p_mx) / FIELD_W,         (b[1] - p[1]) / FIELD_H,
        (0.0 - p_mx) / FIELD_W,          0.0,
        b_mx / FIELD_W * 2 - 1,          b[1] / FIELD_H * 2 - 1,
        -bv[0],                           bv[1],
        (FIELD_W - p_mx) / FIELD_W,      0.0,
        (opp_mx - p_mx) / FIELD_W,       (opp[1] - p[1]) / FIELD_H,
    ], dtype=np.float32)


def obs_mf(env, player_idx, mate_idx):
    p, v   = env._pos[player_idx], env._vel[player_idx]
    b, bv  = env._ball_pos, env._ball_vel
    mate   = env._pos[mate_idx]
    mate_v = env._vel[mate_idx]
    OWN    = np.array([FIELD_W, FIELD_H / 2])
    SC     = np.array([0.0,     FIELD_H / 2])
    spd    = np.linalg.norm(bv) + 1e-8
    bto    = OWN - b
    danger = np.dot(bv / spd, bto / (np.linalg.norm(bto) + 1e-8))
    d2b    = np.linalg.norm(b - p)
    return np.array([
        p[0] / FIELD_W * 2 - 1,          p[1] / FIELD_H * 2 - 1,
        np.clip(v[0], -1, 1),             np.clip(v[1], -1, 1),
        (b - p)[0] / FIELD_W,             (b - p)[1] / FIELD_H,
        (OWN - p)[0] / FIELD_W,           (OWN - p)[1] / FIELD_H,
        (SC  - p)[0] / FIELD_W,           (SC  - p)[1] / FIELD_H,
        b[0] / FIELD_W * 2 - 1,           b[1] / FIELD_H * 2 - 1,
        np.clip(bv[0], -1, 1),            np.clip(bv[1], -1, 1),
        mate[0] / FIELD_W * 2 - 1,        mate[1] / FIELD_H * 2 - 1,
        np.clip(mate_v[0], -1, 1),        np.clip(mate_v[1], -1, 1),
        np.clip(d2b / FIELD_W, 0, 1),
        np.clip(danger, -1, 1),
    ], dtype=np.float32)


# ===========================================================================
# HUD
# ===========================================================================
def draw_hud(surf, font_big, font_sm, ep, n_ep, paused, fps, target_fps):
    TW, TH = surf.get_size()
    GRAY = (170, 170, 170)
    BLUE = (80,  140, 240)
    RED  = (240,  70,  70)

    lbl_a = font_sm.render("ATT + DEF", True, BLUE)
    lbl_b = font_sm.render("MF1 + MF2", True, RED)
    surf.blit(lbl_a, (10, TH - lbl_a.get_height() - 6))
    surf.blit(lbl_b, (TW - lbl_b.get_width() - 10, TH - lbl_b.get_height() - 6))

    ep_txt = font_sm.render(f"Ep {ep}/{n_ep}", True, GRAY)
    surf.blit(ep_txt, (TW // 2 - ep_txt.get_width() // 2, TH - ep_txt.get_height() - 6))

    fps_txt = font_sm.render(f"{fps:.0f}/{target_fps} fps", True, GRAY)
    surf.blit(fps_txt, (TW - fps_txt.get_width() - 8, 6))

    ctrl = font_sm.render("SPC=pause  R=restart  +/-=speed  Q=quit", True, GRAY)
    surf.blit(ctrl, (8, 6))

    if paused:
        banner = font_big.render("PAUSED — SPACE to resume", True, (255, 255, 255))
        bx = TW // 2 - banner.get_width() // 2
        by = TH // 2 - banner.get_height() // 2
        bg = pygame.Surface((banner.get_width() + 24, banner.get_height() + 12))
        bg.set_alpha(180)
        bg.fill((0, 0, 0))
        surf.blit(bg, (bx - 12, by - 6))
        surf.blit(banner, (bx, by))


# ===========================================================================
# Main
# ===========================================================================
def run():
    print("\n" + "=" * 56)
    print("  ULTIMATE MATCH  2v2")
    print("  LEFT  blue  [0] Attacker  [1] Defender  → score RIGHT (x=50)")
    print("  RIGHT red   [2] MF1       [3] MF2        → score LEFT  (x=0)")
    print("=" * 56)

    for path in [ATTACKER_PATH, DEFENDER_PATH, MF1_PATH, MF2_PATH]:
        if not os.path.exists(f"{path}.zip"):
            print(f"\nERROR: {path}.zip not found")
            sys.exit(1)

    att = PPO.load(ATTACKER_PATH)
    dfn = PPO.load(DEFENDER_PATH)
    mf1 = PPO.load(MF1_PATH)
    mf2 = PPO.load(MF2_PATH)
    print("  all models loaded\n")

    env = SoccerEnv(render_mode="human" if RENDER else None, n_players=2)

    if RENDER:
        pygame.init()
        pygame.display.set_caption("Ultimate Match  —  Att+Def  vs  MF1+MF2")
        clock    = pygame.time.Clock()
        font_big = pygame.font.SysFont("monospace", 22, bold=True)
        font_sm  = pygame.font.SysFont("monospace", 14)

    total_a = total_b = 0
    cur_fps = BASE_FPS
    paused  = False

    try:
        for ep in range(1, N_EPISODES + 1):
            reset_with_correct_spawns(env, seed=ep)
            done = restart = False
            steps = 0

            while not done and not restart:

                # ── Events ──────────────────────────────────────────────
                if RENDER:
                    for e in pygame.event.get():
                        if e.type == pygame.QUIT:
                            raise KeyboardInterrupt
                        if e.type == pygame.KEYDOWN:
                            if e.key in (pygame.K_q, pygame.K_ESCAPE):
                                raise KeyboardInterrupt
                            if e.key == pygame.K_SPACE:
                                paused = not paused
                            if e.key == pygame.K_r:
                                restart = True
                            if e.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                                cur_fps = min(cur_fps + 10, 200)
                            if e.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                                cur_fps = max(cur_fps - 10, 10)

                if paused:
                    if RENDER:
                        env.render()
                        draw_hud(pygame.display.get_surface(),
                                 font_big, font_sm, ep, N_EPISODES, True,
                                 clock.get_fps(), cur_fps)
                        pygame.display.flip()
                        clock.tick(30)
                    continue

                # ── Actions ──────────────────────────────────────────────
                act_att, _ = att.predict(
                    obs_attacker_normal(env, player_idx=0), deterministic=True)

                act_def_raw, _ = dfn.predict(
                    obs_defender_mirrored(env, player_idx=1), deterministic=True)
                act_def = np.array([-act_def_raw[0], act_def_raw[1]])

                act_mf1, _ = mf1.predict(
                    obs_mf(env, player_idx=2, mate_idx=3), deterministic=True)

                act_mf2, _ = mf2.predict(
                    obs_mf(env, player_idx=3, mate_idx=2), deterministic=True)

                _, _, terminated, truncated, info = env.step({
                    "team_a": np.stack([np.array(act_att), np.array(act_def)]),
                    "team_b": np.stack([np.array(act_mf1), np.array(act_mf2)]),
                })

                done   = terminated or truncated
                steps += 1

                if RENDER:
                    env.render()
                    draw_hud(pygame.display.get_surface(),
                             font_big, font_sm, ep, N_EPISODES, False,
                             clock.get_fps(), cur_fps)
                    pygame.display.flip()
                    clock.tick(cur_fps)

            sc = info["score"]
            total_a += sc[0]
            total_b += sc[1]
            tag = "restarted" if restart else f"{steps} steps"
            print(f"  Ep {ep:>2}: Att+Def {sc[0]} — {sc[1]} MF1+MF2  [{tag}]")

    except KeyboardInterrupt:
        print("\n  aborted")
    finally:
        env.close()
        if RENDER:
            pygame.quit()

    print(f"\n{'='*56}")
    print(f"  FINAL  Att+Def {total_a}  :  {total_b}  MF1+MF2")
    print(f"{'='*56}")


if __name__ == "__main__":
    run()