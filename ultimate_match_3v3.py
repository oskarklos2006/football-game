"""
ultimate_match_3v3.py
=====================
3v3 configurable match. Edit the LINEUP section below to pick your teams.

Controls  SPACE=pause  R=restart  +/-=speed  Q/ESC=quit
"""

import sys, argparse, os
import numpy as np
import pygame
from stable_baselines3 import PPO
from football_env import SoccerEnv, FIELD_W, FIELD_H, PLAYER_RADIUS, BALL_RADIUS

# ===========================================================================
#  LINEUP — edit these to change who plays
#
#  Each team has exactly 3 slots: [slot0, slot1, slot2]
#  Pick from: "attacker"  "midfielder1"  "midfielder2"  "defender"
#
#  Team A plays on the LEFT  (blue)  and scores at the RIGHT goal (x=50)
#  Team B plays on the RIGHT (red)   and scores at the LEFT  goal (x=0)
# ===========================================================================

TEAM_A = ["attacker",    "midfielder2",  "defender"]
TEAM_B = ["defender",    "midfielder2",  "midfielder1"]

# Examples:
# TEAM_A = ["attacker",   "attacker",    "defender"]     # 2 attackers
# TEAM_A = ["defender",   "midfielder1", "midfielder2"]  # mirror of team B
# TEAM_B = ["midfielder1","midfielder1", "midfielder2"]  # 3 midfielders

# ---------------------------------------------------------------------------
# Model paths — change these if your files are named differently
# ---------------------------------------------------------------------------
MODEL_PATHS = {
    "attacker":    "best_attacker",
    "defender":    "best_defender",
    "midfielder1": "mf1",
    "midfielder2": "mf2",
}

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


# ===========================================================================
# Spawn positions for each role on each side
# Team A (left) spawns at low x values
# Team B (right) spawns at high x values
# Three slots per team, spread out vertically
# ===========================================================================

TEAM_A_SPAWNS = np.array([
    [FIELD_W * 0.12, FIELD_H * 0.30],   # slot 0 — high
    [FIELD_W * 0.22, FIELD_H * 0.50],   # slot 1 — centre
    [FIELD_W * 0.12, FIELD_H * 0.70],   # slot 2 — low
], dtype=np.float32)

TEAM_B_SPAWNS = np.array([
    [FIELD_W * 0.88, FIELD_H * 0.70],   # slot 0 — low
    [FIELD_W * 0.78, FIELD_H * 0.50],   # slot 1 — centre
    [FIELD_W * 0.88, FIELD_H * 0.30],   # slot 2 — high
], dtype=np.float32)


def reset_with_spawns(env, seed):
    env.reset(seed=seed)
    for i in range(3):
        env._pos[i]   = TEAM_A_SPAWNS[i]
        env._pos[i+3] = TEAM_B_SPAWNS[i]
    env._vel[:] = 0.0


# ===========================================================================
# Observation builders — one per role × side (normal / mirrored)
# ===========================================================================

def obs_attacker_normal(env, idx):
    """Attacker on LEFT (team_a) — trained side, scores at x=FIELD_W."""
    p, v  = env._pos[idx], env._vel[idx]
    b, bv = env._ball_pos, env._ball_vel
    tgt   = np.array([FIELD_W, FIELD_H / 2])
    return np.array([
        p[0]/FIELD_W*2-1,    p[1]/FIELD_H*2-1,
        v[0],                v[1],
        (b-p)[0]/FIELD_W,    (b-p)[1]/FIELD_H,
        (tgt-p)[0]/FIELD_W,  (tgt-p)[1]/FIELD_H,
        b[0]/FIELD_W*2-1,    b[1]/FIELD_H*2-1,
        bv[0],               bv[1],
    ], dtype=np.float32)


def obs_attacker_mirrored(env, idx):
    """Attacker on RIGHT (team_b) — opposite side, mirror x."""
    p, v  = env._pos[idx], env._vel[idx]
    b, bv = env._ball_pos, env._ball_vel
    p_mx  = FIELD_W - p[0]
    b_mx  = FIELD_W - b[0]
    return np.array([
        p_mx/FIELD_W*2-1,         p[1]/FIELD_H*2-1,
        -v[0],                     v[1],
        (b_mx-p_mx)/FIELD_W,       (b[1]-p[1])/FIELD_H,
        (FIELD_W-p_mx)/FIELD_W,    0.0,
        b_mx/FIELD_W*2-1,          b[1]/FIELD_H*2-1,
        -bv[0],                    bv[1],
    ], dtype=np.float32)


def obs_defender_normal(env, idx, opp_indices):
    """Defender on RIGHT (team_b) — trained side, defends x=FIELD_W."""
    p, v  = env._pos[idx], env._vel[idx]
    b, bv = env._ball_pos, env._ball_vel
    opp   = min([env._pos[i] for i in opp_indices], key=lambda o: np.linalg.norm(o - b))
    own   = np.array([FIELD_W, FIELD_H / 2])
    tgt   = np.array([0.0,     FIELD_H / 2])
    return np.array([
        p[0]/FIELD_W*2-1,    p[1]/FIELD_H*2-1,
        v[0],                v[1],
        (b-p)[0]/FIELD_W,    (b-p)[1]/FIELD_H,
        (tgt-p)[0]/FIELD_W,  (tgt-p)[1]/FIELD_H,
        b[0]/FIELD_W*2-1,    b[1]/FIELD_H*2-1,
        bv[0],               bv[1],
        (own-p)[0]/FIELD_W,  (own-p)[1]/FIELD_H,
        (opp-p)[0]/FIELD_W,  (opp-p)[1]/FIELD_H,
    ], dtype=np.float32)


def obs_defender_mirrored(env, idx, opp_indices):
    """Defender on LEFT (team_a) — opposite side, mirror x."""
    p, v   = env._pos[idx], env._vel[idx]
    b, bv  = env._ball_pos, env._ball_vel
    opp    = min([env._pos[i] for i in opp_indices], key=lambda o: np.linalg.norm(o - b))
    p_mx   = FIELD_W - p[0]
    b_mx   = FIELD_W - b[0]
    opp_mx = FIELD_W - opp[0]
    return np.array([
        p_mx/FIELD_W*2-1,          p[1]/FIELD_H*2-1,
        -v[0],                      v[1],
        (b_mx-p_mx)/FIELD_W,        (b[1]-p[1])/FIELD_H,
        (0.0-p_mx)/FIELD_W,         0.0,
        b_mx/FIELD_W*2-1,           b[1]/FIELD_H*2-1,
        -bv[0],                     bv[1],
        (FIELD_W-p_mx)/FIELD_W,     0.0,
        (opp_mx-p_mx)/FIELD_W,      (opp[1]-p[1])/FIELD_H,
    ], dtype=np.float32)


def obs_mf_normal(env, idx, mate_idx):
    """Midfielder on RIGHT (team_b) — trained side, scores at x=0."""
    p, v   = env._pos[idx], env._vel[idx]
    b, bv  = env._ball_pos, env._ball_vel
    mate   = env._pos[mate_idx]
    mate_v = env._vel[mate_idx]
    OWN    = np.array([FIELD_W, FIELD_H / 2])
    SC     = np.array([0.0,     FIELD_H / 2])
    spd    = np.linalg.norm(bv) + 1e-8
    danger = np.dot(bv/spd, (OWN-b)/(np.linalg.norm(OWN-b)+1e-8))
    d2b    = np.linalg.norm(b - p)
    d2m    = np.linalg.norm(mate - p)
    closer = 1.0 if d2b < np.linalg.norm(b - mate) else 0.0
    return np.array([
        p[0]/FIELD_W*2-1,           p[1]/FIELD_H*2-1,
        np.clip(v[0],-1,1),         np.clip(v[1],-1,1),
        (b-p)[0]/FIELD_W,           (b-p)[1]/FIELD_H,
        (OWN-p)[0]/FIELD_W,         (OWN-p)[1]/FIELD_H,
        (SC-p)[0]/FIELD_W,          (SC-p)[1]/FIELD_H,
        b[0]/FIELD_W*2-1,           b[1]/FIELD_H*2-1,
        np.clip(bv[0],-1,1),        np.clip(bv[1],-1,1),
        mate[0]/FIELD_W*2-1,        mate[1]/FIELD_H*2-1,
        np.clip(mate_v[0],-1,1),    np.clip(mate_v[1],-1,1),
        np.clip(d2b/FIELD_W,0,1),
        np.clip(d2m/FIELD_W,0,1),
        np.clip(danger,-1,1),
        closer,
    ], dtype=np.float32)


def obs_mf_mirrored(env, idx, mate_idx):
    """Midfielder on LEFT (team_a) — opposite side, mirror x."""
    p, v   = env._pos[idx], env._vel[idx]
    b, bv  = env._ball_pos, env._ball_vel
    mate   = env._pos[mate_idx]
    mate_v = env._vel[mate_idx]
    # Mirror x
    pm  = np.array([FIELD_W-p[0],    p[1]])
    bm  = np.array([FIELD_W-b[0],    b[1]])
    mm  = np.array([FIELD_W-mate[0], mate[1]])
    vm  = np.array([-v[0],    v[1]])
    bvm = np.array([-bv[0],   bv[1]])
    mvm = np.array([-mate_v[0], mate_v[1]])
    OWN = np.array([FIELD_W, FIELD_H / 2])
    SC  = np.array([0.0,     FIELD_H / 2])
    spd    = np.linalg.norm(bvm) + 1e-8
    danger = np.dot(bvm/spd, (OWN-bm)/(np.linalg.norm(OWN-bm)+1e-8))
    d2b    = np.linalg.norm(bm - pm)
    d2m    = np.linalg.norm(mm - pm)
    closer = 1.0 if d2b < np.linalg.norm(bm - mm) else 0.0
    return np.array([
        pm[0]/FIELD_W*2-1,          pm[1]/FIELD_H*2-1,
        np.clip(vm[0],-1,1),        np.clip(vm[1],-1,1),
        (bm-pm)[0]/FIELD_W,         (bm-pm)[1]/FIELD_H,
        (OWN-pm)[0]/FIELD_W,        (OWN-pm)[1]/FIELD_H,
        (SC-pm)[0]/FIELD_W,         (SC-pm)[1]/FIELD_H,
        bm[0]/FIELD_W*2-1,          bm[1]/FIELD_H*2-1,
        np.clip(bvm[0],-1,1),       np.clip(bvm[1],-1,1),
        mm[0]/FIELD_W*2-1,          mm[1]/FIELD_H*2-1,
        np.clip(mvm[0],-1,1),       np.clip(mvm[1],-1,1),
        np.clip(d2b/FIELD_W,0,1),
        np.clip(d2m/FIELD_W,0,1),
        np.clip(danger,-1,1),
        closer,
    ], dtype=np.float32)


# ===========================================================================
# Action dispatcher
# Given a role name, team side, player index, and mate index,
# returns (observation, needs_mirror) so the caller knows to negate action[0]
# ===========================================================================

def get_obs(env, role, side, player_idx, mate_idx, opp_indices):
    """
    side : "A" (left, attacks right) or "B" (right, attacks left)
    Returns (obs, mirror) where mirror=True means negate action[0]
    """
    if role == "attacker":
        if side == "A":
            return obs_attacker_normal(env, player_idx), False
        else:
            return obs_attacker_mirrored(env, player_idx), True

    elif role == "defender":
        if side == "A":
            # team_a defender defends x=0 — opposite side for this model
            return obs_defender_mirrored(env, player_idx, opp_indices), True
        else:
            # team_b defender defends x=FIELD_W — trained side
            return obs_defender_normal(env, player_idx, opp_indices), False

    elif role in ("midfielder1", "midfielder2"):
        if side == "A":
            return obs_mf_mirrored(env, player_idx, mate_idx), True
        else:
            return obs_mf_normal(env, player_idx, mate_idx), False

    raise ValueError(f"Unknown role: {role}")


# ===========================================================================
# HUD
# ===========================================================================

def draw_hud(surf, font_big, font_sm, ep, n_ep, paused, fps, target_fps,
             team_a_roles, team_b_roles):
    TW, TH = surf.get_size()
    GRAY = (170, 170, 170)
    BLUE = (80,  140, 240)
    RED  = (240,  70,  70)

    label_a = " + ".join(r[:3].upper() for r in team_a_roles)
    label_b = " + ".join(r[:3].upper() for r in team_b_roles)

    lbl_a = font_sm.render(label_a, True, BLUE)
    lbl_b = font_sm.render(label_b, True, RED)
    surf.blit(lbl_a, (10, TH - lbl_a.get_height() - 6))
    surf.blit(lbl_b, (TW - lbl_b.get_width() - 10, TH - lbl_b.get_height() - 6))

    ep_txt = font_sm.render(f"Ep {ep}/{n_ep}", True, GRAY)
    surf.blit(ep_txt, (TW//2 - ep_txt.get_width()//2, TH - ep_txt.get_height() - 6))

    fps_txt = font_sm.render(f"{fps:.0f}/{target_fps} fps", True, GRAY)
    surf.blit(fps_txt, (TW - fps_txt.get_width() - 8, 6))

    ctrl = font_sm.render("SPC=pause  R=restart  +/-=speed  Q=quit", True, GRAY)
    surf.blit(ctrl, (8, 6))

    if paused:
        banner = font_big.render("PAUSED — SPACE to resume", True, (255, 255, 255))
        bx = TW//2 - banner.get_width()//2
        by = TH//2 - banner.get_height()//2
        bg = pygame.Surface((banner.get_width()+24, banner.get_height()+12))
        bg.set_alpha(180)
        bg.fill((0, 0, 0))
        surf.blit(bg, (bx-12, by-6))
        surf.blit(banner, (bx, by))


# ===========================================================================
# Main
# ===========================================================================

def run():
    # Validate lineup
    valid = set(MODEL_PATHS.keys())
    for role in TEAM_A + TEAM_B:
        if role not in valid:
            print(f"ERROR: unknown role '{role}'. Choose from: {sorted(valid)}")
            sys.exit(1)

    # Check all needed model files exist
    needed = set(TEAM_A + TEAM_B)
    for role in needed:
        path = MODEL_PATHS[role] + ".zip"
        if not os.path.exists(path):
            print(f"ERROR: {path} not found")
            sys.exit(1)

    # Load models (cache — only load each unique model once)
    models = {role: PPO.load(MODEL_PATHS[role]) for role in needed}

    print("\n" + "="*60)
    print("  3v3 MATCH")
    print(f"  LEFT  blue  {' | '.join(TEAM_A)}  → score x=50")
    print(f"  RIGHT red   {' | '.join(TEAM_B)}  → score x=0")
    print("="*60)
    for role in needed:
        print(f"  loaded  {MODEL_PATHS[role]}.zip  ({role})")
    print()

    env = SoccerEnv(render_mode="human" if RENDER else None, n_players=3)

    if RENDER:
        pygame.init()
        pygame.display.set_caption(f"3v3  {'+'.join(TEAM_A)}  vs  {'+'.join(TEAM_B)}")
        clock    = pygame.time.Clock()
        font_big = pygame.font.SysFont("monospace", 22, bold=True)
        font_sm  = pygame.font.SysFont("monospace", 13)

    # Player indices: team_a = 0,1,2  team_b = 3,4,5
    A_IDX = [0, 1, 2]
    B_IDX = [3, 4, 5]

    # For each MF we need a "mate" — the other MF on the same team if one exists,
    # otherwise the nearest teammate
    def find_mate(team_indices, my_idx, env):
        others = [i for i in team_indices if i != my_idx]
        if not others:
            return my_idx
        return min(others, key=lambda i: abs(i - my_idx))

    total_a = total_b = 0
    cur_fps = BASE_FPS
    paused  = False

    try:
        for ep in range(1, N_EPISODES + 1):
            reset_with_spawns(env, seed=ep)
            done = restart = False
            steps = 0

            while not done and not restart:

                # Events
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
                                 clock.get_fps(), cur_fps, TEAM_A, TEAM_B)
                        pygame.display.flip()
                        clock.tick(30)
                    continue

                # Get actions for all 6 players
                actions_a = []
                for slot, role in enumerate(TEAM_A):
                    idx  = A_IDX[slot]
                    mate = find_mate(A_IDX, idx, env)
                    obs, mirror = get_obs(env, role, "A", idx, mate, B_IDX)
                    act, _ = models[role].predict(obs, deterministic=True)
                    if mirror:
                        act = np.array([-act[0], act[1]])
                    actions_a.append(act)

                actions_b = []
                for slot, role in enumerate(TEAM_B):
                    idx  = B_IDX[slot]
                    mate = find_mate(B_IDX, idx, env)
                    obs, mirror = get_obs(env, role, "B", idx, mate, A_IDX)
                    act, _ = models[role].predict(obs, deterministic=True)
                    if mirror:
                        act = np.array([-act[0], act[1]])
                    actions_b.append(act)

                _, _, terminated, truncated, info = env.step({
                    "team_a": np.stack(actions_a),
                    "team_b": np.stack(actions_b),
                })

                done   = terminated or truncated
                steps += 1

                if RENDER:
                    env.render()
                    draw_hud(pygame.display.get_surface(),
                             font_big, font_sm, ep, N_EPISODES, False,
                             clock.get_fps(), cur_fps, TEAM_A, TEAM_B)
                    pygame.display.flip()
                    clock.tick(cur_fps)

            sc = info["score"]
            total_a += sc[0]
            total_b += sc[1]
            tag = "restarted" if restart else f"{steps} steps"
            a_lbl = "+".join(r[:3] for r in TEAM_A)
            b_lbl = "+".join(r[:3] for r in TEAM_B)
            print(f"  Ep {ep:>2}: {a_lbl} {sc[0]} — {sc[1]} {b_lbl}  [{tag}]")

    except KeyboardInterrupt:
        print("\n  aborted")
    finally:
        env.close()
        if RENDER:
            pygame.quit()

    a_lbl = "+".join(r[:3] for r in TEAM_A)
    b_lbl = "+".join(r[:3] for r in TEAM_B)
    print(f"\n{'='*60}")
    print(f"  FINAL  {a_lbl} {total_a}  :  {total_b}  {b_lbl}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run()