import sys
import numpy as np
import pygame
from stable_baselines3 import PPO
from football_env import SoccerEnv, FIELD_W, FIELD_H

# ==========================================
# MODELS
# ==========================================
ATTACKER_PATH = "best_attacker"   # trained: Team A left side, scores at x=FIELD_W
DEFENDER_PATH = "best_defender"   # trained: Team B right side, defends x=FIELD_W

N_EPISODES = 5
RENDER     = True

# ---------------------------------------------------------------------------
# Who plays where:
#
#   x=0 (left goal)                         x=50 (right goal)
#   Team A defends                           Team B defends
#
#   player 0 = Team A ATTACKER  → scores at x=50  → trained side, NO mirror
#   player 1 = Team A DEFENDER  → defends  x=0    → OPPOSITE side, needs mirror
#   player 2 = Team B ATTACKER  → scores at x=0   → OPPOSITE side, needs mirror
#   player 3 = Team B DEFENDER  → defends  x=50   → trained side, NO mirror
#
# Mirroring = flip x coordinates so the model sees the world
# as if it were on its trained side.
# ---------------------------------------------------------------------------


def mirror_x(val, field_w=FIELD_W):
    """Flip a normalised x value: left becomes right and vice versa."""
    return -val


def get_obs_attacker_normal(env, player_idx):
    """
    Attacker obs — trained scoring at x=FIELD_W (right goal).
    Used for Team A attacker (player 0). No mirroring needed.
    """
    p  = env._pos[player_idx]
    v  = env._vel[player_idx]
    b  = env._ball_pos
    bv = env._ball_vel
    target = np.array([FIELD_W, FIELD_H / 2])   # scores right

    return np.array([
        p[0] / FIELD_W * 2 - 1,   p[1] / FIELD_H * 2 - 1,
        v[0],                      v[1],
        (b - p)[0] / FIELD_W,     (b - p)[1] / FIELD_H,
        (target - p)[0] / FIELD_W, (target - p)[1] / FIELD_H,
        b[0] / FIELD_W * 2 - 1,   b[1] / FIELD_H * 2 - 1,
        bv[0],                     bv[1],
    ], dtype=np.float32)


def get_obs_attacker_mirrored(env, player_idx):
    """
    Attacker obs — mirrored for Team B attacker (player 2) who scores at x=0.
    Flip all x values so the model thinks it is on its trained (left) side
    shooting toward x=FIELD_W.
    """
    p  = env._pos[player_idx]
    v  = env._vel[player_idx]
    b  = env._ball_pos
    bv = env._ball_vel

    # Mirrored positions: flip x around field center
    p_mx  =  FIELD_W - p[0]
    b_mx  =  FIELD_W - b[0]
    target_mx = FIELD_W   # scoring goal appears on the right after mirror

    return np.array([
        p_mx / FIELD_W * 2 - 1,              p[1] / FIELD_H * 2 - 1,
        -v[0],                                v[1],          # vx flipped
        (b_mx - p_mx) / FIELD_W,             (b[1] - p[1]) / FIELD_H,
        (target_mx - p_mx) / FIELD_W,        0.0,           # target y offset = 0
        b_mx / FIELD_W * 2 - 1,              b[1] / FIELD_H * 2 - 1,
        -bv[0],                               bv[1],         # bvx flipped
    ], dtype=np.float32)


def get_obs_defender_normal(env, player_idx):
    """
    Defender obs — trained defending x=FIELD_W (right wall).
    Used for Team B defender (player 3). No mirroring needed.
    Closest opponent = Team A players (indices 0, 1).
    """
    p  = env._pos[player_idx]
    v  = env._vel[player_idx]
    b  = env._ball_pos
    bv = env._ball_vel

    # Pick nearest Team A player as the "attacker" reference
    opp_indices = [0, 1]
    opp = min([env._pos[i] for i in opp_indices],
              key=lambda o: np.linalg.norm(o - b))

    target = np.array([0.0,     FIELD_H / 2])   # scores left
    own    = np.array([FIELD_W, FIELD_H / 2])   # defends right

    return np.array([
        p[0] / FIELD_W * 2 - 1,   p[1] / FIELD_H * 2 - 1,
        v[0],                      v[1],
        (b - p)[0] / FIELD_W,     (b - p)[1] / FIELD_H,
        (target - p)[0] / FIELD_W, (target - p)[1] / FIELD_H,
        b[0] / FIELD_W * 2 - 1,   b[1] / FIELD_H * 2 - 1,
        bv[0],                     bv[1],
        (own - p)[0] / FIELD_W,   (own - p)[1] / FIELD_H,
        (opp - p)[0] / FIELD_W,   (opp - p)[1] / FIELD_H,
    ], dtype=np.float32)


def get_obs_defender_mirrored(env, player_idx):
    """
    Defender obs — mirrored for Team A defender (player 1) who defends x=0.
    Flip all x values so the model thinks it is on its trained (right) side
    defending x=FIELD_W.
    Closest opponent = Team B players (indices 2, 3).
    """
    p  = env._pos[player_idx]
    v  = env._vel[player_idx]
    b  = env._ball_pos
    bv = env._ball_vel

    # Pick nearest Team B player as the "attacker" reference
    opp_indices = [2, 3]
    opp = min([env._pos[i] for i in opp_indices],
              key=lambda o: np.linalg.norm(o - b))

    # Mirror all x coordinates
    p_mx   = FIELD_W - p[0]
    b_mx   = FIELD_W - b[0]
    opp_mx = FIELD_W - opp[0]

    # After mirror: target (scoring goal) is at x=FIELD_W, own goal at x=0
    # But from the model's trained perspective it thinks:
    #   own goal  = x=FIELD_W  (right)
    #   score     = x=0        (left)
    target_mx = 0.0      # scores left after mirror → appears as left to model
    own_mx    = FIELD_W  # defends right after mirror

    return np.array([
        p_mx / FIELD_W * 2 - 1,              p[1] / FIELD_H * 2 - 1,
        -v[0],                                v[1],
        (b_mx - p_mx) / FIELD_W,             (b[1] - p[1]) / FIELD_H,
        (target_mx - p_mx) / FIELD_W,        0.0,
        b_mx / FIELD_W * 2 - 1,              b[1] / FIELD_H * 2 - 1,
        -bv[0],                               bv[1],
        (own_mx - p_mx) / FIELD_W,           0.0,
        (opp_mx - p_mx) / FIELD_W,           (opp[1] - p[1]) / FIELD_H,
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Match
# ---------------------------------------------------------------------------
def run_match():
    print("\n>>> 2v2 MATCH")
    print("    Team A (left):  player0=attacker  player1=defender(mirrored)")
    print("    Team B (right): player2=attacker(mirrored)  player3=defender")

    attacker_model = PPO.load(ATTACKER_PATH)
    defender_model = PPO.load(DEFENDER_PATH)

    env = SoccerEnv(render_mode="human" if RENDER else None, n_players=2)

    if RENDER:
        pygame.init()
        clock = pygame.time.Clock()

    a_goals, b_goals = 0, 0

    try:
        for ep in range(N_EPISODES):
            env.reset(seed=ep)
            done = False

            while not done:
                if RENDER:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            env.close()
                            sys.exit()

                # Team A
                # player 0: attacker, trained side → normal obs
                obs_a0 = get_obs_attacker_normal(env, player_idx=0)
                act_a0, _ = attacker_model.predict(obs_a0, deterministic=True)

                # player 1: defender, opposite side → mirrored obs
                obs_a1 = get_obs_defender_mirrored(env, player_idx=1)
                act_a1_raw, _ = defender_model.predict(obs_a1, deterministic=True)
                act_a1 = np.array([-act_a1_raw[0], act_a1_raw[1]])  # flip action x back

                # Team B
                # player 2: attacker, opposite side → mirrored obs
                obs_b0 = get_obs_attacker_mirrored(env, player_idx=2)
                act_b0_raw, _ = attacker_model.predict(obs_b0, deterministic=True)
                act_b0 = np.array([-act_b0_raw[0], act_b0_raw[1]])  # flip action x back

                # player 3: defender, trained side → normal obs
                obs_b1 = get_obs_defender_normal(env, player_idx=3)
                act_b1, _ = defender_model.predict(obs_b1, deterministic=True)

                actions_a = np.stack([act_a0, act_a1])   # shape (2, 2)
                actions_b = np.stack([act_b0, act_b1])   # shape (2, 2)

                _, _, terminated, truncated, info = env.step({
                    "team_a": actions_a,
                    "team_b": actions_b
                })

                done = terminated or truncated

                if RENDER:
                    env.render()
                    clock.tick(60)

            score = info["score"]
            a_goals += score[0]
            b_goals += score[1]
            print(f"Ep {ep+1}: Team A {score[0]} - {score[1]} Team B")

        print("\n" + "="*30)
        print(f" FINAL:  Team A {a_goals} - {b_goals} Team B")
        print("="*30)

    finally:
        env.close()
        if RENDER:
            pygame.quit()


if __name__ == "__main__":
    run_match()