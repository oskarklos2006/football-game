import sys
import numpy as np
import pygame
from stable_baselines3 import PPO
from football_env import SoccerEnv, FIELD_W, FIELD_H

# ---------------------------------------------------------------------------
# Model paths — one attacker and one defender model, each reused twice.
# ---------------------------------------------------------------------------
ATTACKER_PATH = "best_attacker"
DEFENDER_PATH = "best_defender"

N_EPISODES = 5
RENDER     = True

# ---------------------------------------------------------------------------
# Player layout and the mirroring problem.
#
#   x=0 (left goal)                         x=50 (right goal)
#
#   player 0  Team A attacker  → scores at x=50  → trained side, no mirror
#   player 1  Team A defender  → defends  x=0    → opposite side, needs mirror
#   player 2  Team B attacker  → scores at x=0   → opposite side, needs mirror
#   player 3  Team B defender  → defends  x=50   → trained side, no mirror
#
# Both models were trained on a specific side of the field. When a player is
# deployed on the opposite side we mirror all x coordinates in the observation
# so the model still perceives the world as if it were on its trained side,
# then flip the x component of its output action back to real coordinates.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Observation builders — four variants (attacker/defender × normal/mirrored).
# Y coordinates and velocities are never flipped; only x changes.
# ---------------------------------------------------------------------------

def get_obs_attacker_normal(env, player_idx):
    # Team A attacker (player 0): trained to score at x=FIELD_W — no transform needed.
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


def get_obs_attacker_mirrored(env, player_idx):
    # Team B attacker (player 2): scores at x=0, but model expects to score at x=FIELD_W.
    # Flip x so the model sees a reflected field where its goal is still on the right.
    p, v   = env._pos[player_idx], env._vel[player_idx]
    b, bv  = env._ball_pos, env._ball_vel
    p_mx   = FIELD_W - p[0]
    b_mx   = FIELD_W - b[0]
    return np.array([
        p_mx / FIELD_W * 2 - 1,         p[1] / FIELD_H * 2 - 1,
        -v[0],                           v[1],
        (b_mx - p_mx) / FIELD_W,        (b[1] - p[1]) / FIELD_H,
        (FIELD_W - p_mx) / FIELD_W,     0.0,
        b_mx / FIELD_W * 2 - 1,         b[1] / FIELD_H * 2 - 1,
        -bv[0],                          bv[1],
    ], dtype=np.float32)


def get_obs_defender_normal(env, player_idx):
    # Team B defender (player 3): trained to defend x=FIELD_W — no transform needed.
    # Uses the nearest Team A player as the "attacker" reference in the obs vector.
    p, v   = env._pos[player_idx], env._vel[player_idx]
    b, bv  = env._ball_pos, env._ball_vel
    opp    = min([env._pos[i] for i in [0, 1]],
                 key=lambda o: np.linalg.norm(o - b))
    target = np.array([0.0,     FIELD_H / 2])
    own    = np.array([FIELD_W, FIELD_H / 2])
    return np.array([
        p[0] / FIELD_W * 2 - 1,    p[1] / FIELD_H * 2 - 1,
        v[0],                       v[1],
        (b - p)[0] / FIELD_W,      (b - p)[1] / FIELD_H,
        (target - p)[0] / FIELD_W, (target - p)[1] / FIELD_H,
        b[0] / FIELD_W * 2 - 1,    b[1] / FIELD_H * 2 - 1,
        bv[0],                      bv[1],
        (own - p)[0] / FIELD_W,    (own - p)[1] / FIELD_H,
        (opp - p)[0] / FIELD_W,    (opp - p)[1] / FIELD_H,
    ], dtype=np.float32)


def get_obs_defender_mirrored(env, player_idx):
    # Team A defender (player 1): defends x=0, but model expects to defend x=FIELD_W.
    # Flip x so the model sees its goal on the right, then mirror the opponent too.
    # Uses the nearest Team B player as the "attacker" reference.
    p, v   = env._pos[player_idx], env._vel[player_idx]
    b, bv  = env._ball_pos, env._ball_vel
    opp    = min([env._pos[i] for i in [2, 3]],
                 key=lambda o: np.linalg.norm(o - b))
    p_mx   = FIELD_W - p[0]
    b_mx   = FIELD_W - b[0]
    opp_mx = FIELD_W - opp[0]
    return np.array([
        p_mx / FIELD_W * 2 - 1,         p[1] / FIELD_H * 2 - 1,
        -v[0],                           v[1],
        (b_mx - p_mx) / FIELD_W,        (b[1] - p[1]) / FIELD_H,
        (0.0 - p_mx) / FIELD_W,         0.0,
        b_mx / FIELD_W * 2 - 1,         b[1] / FIELD_H * 2 - 1,
        -bv[0],                          bv[1],
        (FIELD_W - p_mx) / FIELD_W,     0.0,
        (opp_mx - p_mx) / FIELD_W,      (opp[1] - p[1]) / FIELD_H,
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Match loop — loads both models once, then runs N_EPISODES.
# Mirrored players have their action x flipped back to real coordinates
# before being passed to the environment.
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

                # Player 0: trained side — use action as-is.
                obs_a0 = get_obs_attacker_normal(env, player_idx=0)
                act_a0, _ = attacker_model.predict(obs_a0, deterministic=True)

                # Player 1: mirrored obs — flip action x back to real coordinates.
                obs_a1 = get_obs_defender_mirrored(env, player_idx=1)
                act_a1_raw, _ = defender_model.predict(obs_a1, deterministic=True)
                act_a1 = np.array([-act_a1_raw[0], act_a1_raw[1]])

                # Player 2: mirrored obs — flip action x back to real coordinates.
                obs_b0 = get_obs_attacker_mirrored(env, player_idx=2)
                act_b0_raw, _ = attacker_model.predict(obs_b0, deterministic=True)
                act_b0 = np.array([-act_b0_raw[0], act_b0_raw[1]])

                # Player 3: trained side — use action as-is.
                obs_b1 = get_obs_defender_normal(env, player_idx=3)
                act_b1, _ = defender_model.predict(obs_b1, deterministic=True)

                _, _, terminated, truncated, info = env.step({
                    "team_a": np.stack([act_a0, act_a1]),
                    "team_b": np.stack([act_b0, act_b1]),
                })

                done = terminated or truncated
                if RENDER:
                    env.render()
                    clock.tick(60)

            score    = info["score"]
            a_goals += score[0]
            b_goals += score[1]
            print(f"Ep {ep+1}: Team A {score[0]} - {score[1]} Team B")

        print("\n" + "=" * 30)
        print(f" FINAL:  Team A {a_goals} - {b_goals} Team B")
        print("=" * 30)

    finally:
        env.close()
        if RENDER:
            pygame.quit()


if __name__ == "__main__":
    run_match()