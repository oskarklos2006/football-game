"""
train_midfielders.py
====================
Two midfielders trained to pass, move, and contribute to both attack and defence.

  MF1 (mf1.zip)  — attack-minded, weights from best_attacker.zip
  MF2 (mf2.zip)  — defence-minded, weights from best_defender.zip

Both play as team_b (right side, attack toward x=0).

Pass reward anti-farming rules
-------------------------------
A pass is only rewarded when ALL of these are true:
  1. Ball was last near the SENDER (not just both players near the ball)
  2. Ball has travelled at least MIN_PASS_DIST units since it left the sender
  3. Ball arrives within PASS_RADIUS of the RECEIVER
  4. Sender and receiver are at least MIN_PLAYER_SEP apart (no piggyback farming)
  5. One reward per possession exchange — the sender flag resets only after
     the ball clearly leaves both zones

Obs layout (22 features)
-------------------------
  0-1  : own position (normalised [-1,1])
  2-3  : own velocity (clipped)
  4-5  : vector to ball
  6-7  : vector to own goal (x=FIELD_W)
  8-9  : vector to scoring goal (x=0)
  10-11: ball absolute position
  12-13: ball velocity (clipped)
  14-15: teammate position (normalised)
  16-17: teammate velocity (clipped)
  18   : distance to ball (normalised)
  19   : distance to teammate (normalised)
  20   : ball heading danger toward own goal
  21   : am I closer to ball than teammate (1/0)
"""

import os, shutil
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from football_env import SoccerEnv, FIELD_W, FIELD_H, PLAYER_RADIUS, BALL_RADIUS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ATTACKER_PATH = "best_attacker"
DEFENDER_PATH = "best_defender"
MF1_NAME      = "mf1"
MF2_NAME      = "mf2"

# ---------------------------------------------------------------------------
# Training schedule
# ---------------------------------------------------------------------------
N_ENVS             = 8
BOOTSTRAP_STEPS    = 500_000
STEPS_ROUND        = 600_000
ALTERNATING_ROUNDS = 6

TEAMMATE_FROZEN_SCHEDULE = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9]

ANNEAL_STAGES = [
    (0.30, 2e-4, 0.04),
    (0.60, 1e-4, 0.02),
    (0.85, 5e-5, 0.008),
    (1.00, 2e-5, 0.002),
]

NET_ARCH     = [256, 256]
ATT_OBS_SIZE = 12
DEF_OBS_SIZE = 16
MF_OBS_SIZE  = 22

# ---------------------------------------------------------------------------
# Passing geometry constants
# ---------------------------------------------------------------------------
# Ball is "near" a player within this radius
PASS_RADIUS = PLAYER_RADIUS + BALL_RADIUS + 3.5

# A pass is only valid if the ball travels at least this far
MIN_PASS_DIST = 6.0

# Players must be at least this far apart for a pass reward to fire
# (prevents two players standing together and farming short taps)
MIN_PLAYER_SEP = 8.0

# Ideal separation between the two MFs — Gaussian reward peaks here
IDEAL_SEP = 14.0

# ---------------------------------------------------------------------------
# Spawn positions — used in both training and match
# ---------------------------------------------------------------------------
SPAWN = np.array([
    [FIELD_W * 0.20, FIELD_H / 2],        # [0] attacker
    [FIELD_W * 0.15, FIELD_H / 2],        # [1] defender
    [FIELD_W * 0.70, FIELD_H * 0.40],     # [2] MF1
    [FIELD_W * 0.85, FIELD_H * 0.60],     # [3] MF2
], dtype=np.float32)


# ===========================================================================
# Opponent obs — must match exactly what the trained models expect
# Copied verbatim from working play_match.py
# ===========================================================================

def get_attacker_obs(env):
    p, v  = env._pos[0], env._vel[0]
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


def get_defender_obs(env):
    # Mirrored — model trained to defend x=FIELD_W, here defends x=0
    # Caller must negate action[0]
    p, v   = env._pos[1], env._vel[1]
    b, bv  = env._ball_pos, env._ball_vel
    opp    = min([env._pos[2], env._pos[3]], key=lambda o: np.linalg.norm(o - b))
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


# ===========================================================================
# Midfielder obs (22 features)
# ===========================================================================

def get_mf_obs(env, player_idx, mate_idx):
    p, v   = env._pos[player_idx], env._vel[player_idx]
    b, bv  = env._ball_pos, env._ball_vel
    mate   = env._pos[mate_idx]
    mate_v = env._vel[mate_idx]

    OWN_GOAL = np.array([FIELD_W, FIELD_H / 2])
    SC_GOAL  = np.array([0.0,     FIELD_H / 2])

    spd      = np.linalg.norm(bv) + 1e-8
    bto_own  = OWN_GOAL - b
    danger   = np.dot(bv / spd, bto_own / (np.linalg.norm(bto_own) + 1e-8))
    d2ball   = np.linalg.norm(b - p)
    d2mate   = np.linalg.norm(mate - p)
    closer   = 1.0 if d2ball < np.linalg.norm(b - mate) else 0.0

    return np.array([
        p[0]/FIELD_W*2-1,              p[1]/FIELD_H*2-1,
        np.clip(v[0], -1, 1),          np.clip(v[1], -1, 1),
        (b-p)[0]/FIELD_W,              (b-p)[1]/FIELD_H,
        (OWN_GOAL-p)[0]/FIELD_W,       (OWN_GOAL-p)[1]/FIELD_H,
        (SC_GOAL-p)[0]/FIELD_W,        (SC_GOAL-p)[1]/FIELD_H,
        b[0]/FIELD_W*2-1,              b[1]/FIELD_H*2-1,
        np.clip(bv[0], -1, 1),         np.clip(bv[1], -1, 1),
        mate[0]/FIELD_W*2-1,           mate[1]/FIELD_H*2-1,
        np.clip(mate_v[0], -1, 1),     np.clip(mate_v[1], -1, 1),
        np.clip(d2ball / FIELD_W, 0, 1),
        np.clip(d2mate / FIELD_W, 0, 1),
        np.clip(danger, -1, 1),
        closer,
    ], dtype=np.float32)


# ===========================================================================
# Weight transplant
# ===========================================================================

def transplant_weights(new_model, donor_path, donor_obs_size):
    donor    = PPO.load(donor_path)
    donor_sd = donor.policy.state_dict()
    new_sd   = new_model.policy.state_dict()

    copied = expanded = skipped = 0
    for key in new_sd:
        if key not in donor_sd:
            skipped += 1
            continue
        d, n = donor_sd[key], new_sd[key]
        if d.shape == n.shape:
            new_sd[key] = d.clone()
            copied += 1
        elif (key.endswith(".weight")
              and d.shape[1] == donor_obs_size
              and n.shape[1] == MF_OBS_SIZE):
            rows = min(d.shape[0], n.shape[0])
            w = torch.zeros_like(n)
            w[:rows, :donor_obs_size] = d[:rows]
            w[:rows, donor_obs_size:] = torch.randn(
                rows, MF_OBS_SIZE - donor_obs_size) * 0.01
            new_sd[key] = w
            expanded += 1
        else:
            skipped += 1

    new_model.policy.load_state_dict(new_sd)
    print(f"    transplant {donor_path}: {copied} copied  {expanded} expanded  {skipped} skipped")


# ===========================================================================
# MidfielderEnv
# ===========================================================================

class MidfielderEnv(gym.Env):
    """
    Reward breakdown
    ----------------
    r_pass_sent     +2.0   valid pass sent to teammate (distance + separation gated)
    r_pass_received +1.0   valid pass received from teammate
    r_separation    +0.04  Gaussian centred at IDEAL_SEP — keeps MFs spread
    r_progress      +0.4   ball moved toward x=0 this tick
    r_shot          +0.3   kick aimed toward scoring goal
    r_goal_scored   +3.0   team_b scores
    r_goal_conceded -3.0   team_a scores
    r_approach      +0.02  gentle pull toward loose ball (always >= 0)

    Pass anti-farming rules (all must hold for r_pass to fire):
      - ball travelled >= MIN_PASS_DIST since leaving sender's zone
      - players >= MIN_PLAYER_SEP apart at time of pass
      - one reward per possession transfer (state machine prevents re-triggering)
    """

    def __init__(self, learner_role, teammate_path=None,
                 frozen_fraction=0.0, total_steps_ref=None):
        super().__init__()
        assert learner_role in ("mf1", "mf2")
        self._role        = learner_role
        self._tm_path     = teammate_path
        self._frozen_frac = frozen_fraction
        self._steps_ref   = total_steps_ref
        self._env         = SoccerEnv(n_players=2)

        self._my_idx   = 2 if learner_role == "mf1" else 3
        self._mate_idx = 3 if learner_role == "mf1" else 2

        self._cached_att = None
        self._cached_def = None
        self._cached_tm  = None
        self._use_frozen = False

        # Pass state machine
        # _possession: "me", "mate", or None
        # _pass_origin: position where ball left the sender's zone
        # _pass_rewarded: True once reward has fired for this possession exchange
        self._possession     = None
        self._pass_origin    = None
        self._pass_rewarded  = False

        self._prev_ball_x = FIELD_W / 2
        self._was_contact = False

        self.observation_space = spaces.Box(-2.0, 2.0, shape=(MF_OBS_SIZE,), dtype=np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, shape=(2,),           dtype=np.float32)

    def _lazy_load(self):
        if self._cached_att is None and os.path.exists(f"{ATTACKER_PATH}.zip"):
            self._cached_att = PPO.load(ATTACKER_PATH)
        if self._cached_def is None and os.path.exists(f"{DEFENDER_PATH}.zip"):
            self._cached_def = PPO.load(DEFENDER_PATH)

    def reset(self, seed=None, options=None):
        self._lazy_load()
        self._env.reset(seed=seed, options=options)

        # Fix spawn positions — SoccerEnv defaults are wrong for this layout
        self._env._pos[:] = SPAWN.copy()
        self._env._vel[:] = 0.0

        self._possession    = None
        self._pass_origin   = None
        self._pass_rewarded = False
        self._prev_ball_x   = self._env._ball_pos[0]
        self._was_contact   = False

        self._use_frozen = (
            self._tm_path is not None
            and os.path.exists(f"{self._tm_path}.zip")
            and np.random.rand() < self._frozen_frac
        )
        if self._use_frozen and self._cached_tm is None:
            self._cached_tm = PPO.load(self._tm_path)

        return get_mf_obs(self._env, self._my_idx, self._mate_idx), {}

    def step(self, my_action):
        # ── Teammate ─────────────────────────────────────────────────────
        tm_obs = get_mf_obs(self._env, self._mate_idx, self._my_idx)
        if self._use_frozen and self._cached_tm is not None:
            tm_act, _ = self._cached_tm.predict(tm_obs, deterministic=False)
        else:
            tm_act = self._env.action_space["team_b"].sample()[0]

        # ── Opponents ────────────────────────────────────────────────────
        if self._cached_att:
            att_act, _ = self._cached_att.predict(
                get_attacker_obs(self._env), deterministic=False)
        else:
            att_act = self._env.action_space["team_a"].sample()[0]

        if self._cached_def:
            def_raw, _ = self._cached_def.predict(
                get_defender_obs(self._env), deterministic=False)
            def_act = np.array([-def_raw[0], def_raw[1]])
        else:
            def_act = self._env.action_space["team_a"].sample()[0]

        # ── Step ─────────────────────────────────────────────────────────
        if self._role == "mf1":
            tb = np.stack([np.array(my_action), np.array(tm_act)])
        else:
            tb = np.stack([np.array(tm_act), np.array(my_action)])

        _, env_rew, terminated, truncated, info = self._env.step({
            "team_a": np.stack([np.array(att_act), np.array(def_act)]),
            "team_b": tb,
        })

        if self._steps_ref is not None:
            self._steps_ref[0] += 1

        # ── State ────────────────────────────────────────────────────────
        b      = info["ball_pos"]
        p      = self._env._pos[self._my_idx]
        mate_p = self._env._pos[self._mate_idx]
        bv     = self._env._ball_vel
        bspd   = np.linalg.norm(bv)

        d2ball  = np.linalg.norm(b - p)
        d2mate  = np.linalg.norm(p - mate_p)
        in_cont = d2ball < (PLAYER_RADIUS + BALL_RADIUS + 0.6)
        edge    = in_cont and not self._was_contact
        self._was_contact = in_cont

        near_me   = d2ball                       < PASS_RADIUS
        near_mate = np.linalg.norm(b - mate_p)   < PASS_RADIUS

        # ── Pass state machine ────────────────────────────────────────────
        #
        # State transitions:
        #   None / unknown  →  ball enters my zone   → possession = "me",
        #                                               record pass_origin
        #   "me"            →  ball leaves my zone   → freeze pass_origin
        #   "me" (departed) →  ball enters mate zone → check distance + sep
        #                                             → award r_pass_sent
        #                                             → possession = "mate"
        #   "mate"          →  ball enters my zone   → check distance + sep
        #                                             → award r_pass_received
        #                                             → possession = "me"
        #
        # _pass_rewarded prevents double-firing within one possession exchange.

        r_pass = 0.0

        if near_me and self._possession != "me":
            # Ball just arrived in my zone — I now have possession
            self._possession    = "me"
            self._pass_origin   = b.copy()
            self._pass_rewarded = False

        elif self._possession == "me" and not near_me:
            # Ball left my zone — freeze the origin so distance is measured
            # from where it departed, not where it was first touched
            if self._pass_origin is None:
                self._pass_origin = b.copy()

        if (self._possession == "me"
                and not self._pass_rewarded
                and near_mate
                and not near_me):
            # Ball is near mate but not near me anymore
            # Check: did it travel far enough? are we far enough apart?
            if self._pass_origin is not None:
                dist_travelled = np.linalg.norm(b - self._pass_origin)
                if dist_travelled >= MIN_PASS_DIST and d2mate >= MIN_PLAYER_SEP:
                    r_pass = 2.0
                    self._pass_rewarded = True
                    self._possession    = "mate"
                    self._pass_origin   = b.copy()

        elif (self._possession == "mate"
              and not self._pass_rewarded
              and near_me
              and not near_mate):
            # Received a pass from teammate — same distance + separation check
            if self._pass_origin is not None:
                dist_travelled = np.linalg.norm(b - self._pass_origin)
                if dist_travelled >= MIN_PASS_DIST and d2mate >= MIN_PLAYER_SEP:
                    r_pass = 1.0
                    self._pass_rewarded = False   # ready for next exchange
                    self._possession    = "me"
                    self._pass_origin   = b.copy()

        # If neither player is near the ball, possession is contested — reset
        if not near_me and not near_mate:
            self._possession    = None
            self._pass_origin   = None
            self._pass_rewarded = False

        # ── Separation reward ─────────────────────────────────────────────
        # Gaussian peak at IDEAL_SEP, sigma=6 — soft incentive to stay spread
        r_sep = 0.04 * float(np.exp(-((d2mate - IDEAL_SEP) ** 2) / (2 * 6.0 ** 2)))

        # ── Ball progress ─────────────────────────────────────────────────
        r_progress = 0.0
        if bspd > 0.05:
            dx = self._prev_ball_x - b[0]   # positive = toward x=0
            r_progress = float(np.clip(dx / FIELD_W, 0.0, 1.0)) * 0.4
        self._prev_ball_x = b[0]

        # ── Shot direction ────────────────────────────────────────────────
        r_shot = 0.0
        SC_GOAL = np.array([0.0, FIELD_H / 2])
        if edge and bspd > 0.4:
            btg = SC_GOAL - b
            dm  = np.dot(bv, btg / (np.linalg.norm(btg) + 1e-8))
            if dm > 0.3:
                r_shot = 0.3

        # ── Goal signals ──────────────────────────────────────────────────
        r_goal = float(env_rew["team_b"]) * 3.0

        # ── Approach (always >= 0) ────────────────────────────────────────
        # Only fires when ball is loose (nobody near it) so it doesn't
        # compete with possession / passing rewards
        r_approach = 0.0
        if not near_me and not near_mate:
            r_approach = 0.02 * float(max(0.0, 1.0 - d2ball / FIELD_W))

        reward = r_pass + r_sep + r_progress + r_shot + r_goal + r_approach
        return (get_mf_obs(self._env, self._my_idx, self._mate_idx),
                reward, terminated, truncated, info)


# ===========================================================================
# Callback
# ===========================================================================

class TrainingCallback(BaseCallback):
    def __init__(self, total_steps, name):
        super().__init__(verbose=1)
        self.total_steps = total_steps
        self.name        = name
        self.best        = -float("inf")
        self._stage      = -1

    def _on_rollout_end(self):
        if self.model.ep_info_buffer:
            avg = np.mean([e["r"] for e in self.model.ep_info_buffer])
            if avg > self.best:
                self.best = avg
                self.model.save(f"best_{self.name}")
                print(f"  [{self.name}] checkpoint  avg={avg:.3f}")

        prog  = self.num_timesteps / self.total_steps
        stage = next((i for i, (th, _, _) in enumerate(ANNEAL_STAGES)
                      if prog <= th), len(ANNEAL_STAGES) - 1)
        _, lr, ent = ANNEAL_STAGES[stage]
        self.model.policy.optimizer.param_groups[0]["lr"] = lr
        self.model.ent_coef = ent
        if stage != self._stage:
            print(f"  [{self.name}] stage {stage+1}  lr={lr}  ent={ent}")
            self._stage = stage

    def _on_step(self):
        return True


# ===========================================================================
# Model factory
# ===========================================================================

def build_model(env, seed=42):
    return PPO(
        "MlpPolicy", env,
        verbose=1,
        n_steps=512,
        batch_size=256,
        n_epochs=10,
        learning_rate=2e-4,
        ent_coef=0.04,
        policy_kwargs={"net_arch": NET_ARCH},
        seed=seed,
    )


# ===========================================================================
# Phase 0 — bootstrap each MF independently with transplanted weights
# ===========================================================================

def phase0_bootstrap(steps_ref):
    print("\n" + "="*60)
    print("PHASE 0  bootstrap")
    print("="*60)

    configs = [
        ("mf1", MF1_NAME, ATTACKER_PATH, ATT_OBS_SIZE, 42),
        ("mf2", MF2_NAME, DEFENDER_PATH, DEF_OBS_SIZE,  7),
    ]

    for role, name, donor_path, donor_obs, seed in configs:
        print(f"\n  >> {role.upper()}  donor={donor_path}")
        env = SubprocVecEnv([
            lambda r=role: Monitor(MidfielderEnv(
                learner_role=r,
                frozen_fraction=0.0,
                total_steps_ref=steps_ref,
            ))
        ] * N_ENVS)

        model = build_model(env, seed=seed)

        if os.path.exists(f"{donor_path}.zip"):
            transplant_weights(model, donor_path, donor_obs)
        else:
            print(f"  WARNING: {donor_path}.zip not found — random init")

        cb = TrainingCallback(BOOTSTRAP_STEPS, f"p0_{role}")
        model.learn(BOOTSTRAP_STEPS, callback=cb, progress_bar=True)

        best = f"best_p0_{role}.zip"
        target = f"{name}.zip"
        if os.path.exists(best):
            shutil.copy(best, target)
        else:
            model.save(name)
        print(f"  >> saved {target}")
        env.close()


# ===========================================================================
# Alternating self-play rounds
# ===========================================================================

def alternating_rounds(steps_ref):
    print("\n" + "="*60)
    print(f"SELF-PLAY  {ALTERNATING_ROUNDS} alternating rounds")
    print("="*60)

    for r in range(ALTERNATING_ROUNDS):
        if r % 2 == 0:
            learner, lname, tmname = "mf1", MF1_NAME, MF2_NAME
        else:
            learner, lname, tmname = "mf2", MF2_NAME, MF1_NAME

        frac = TEAMMATE_FROZEN_SCHEDULE[r]
        tag  = f"{learner}_r{r+1}"
        print(f"\n>> Round {r+1}/{ALTERNATING_ROUNDS}  "
              f"updating {learner.upper()}  frozen_tm={frac}")

        env = SubprocVecEnv([
            lambda lr=learner, tn=tmname: Monitor(MidfielderEnv(
                learner_role=lr,
                teammate_path=tn,
                frozen_fraction=frac,
                total_steps_ref=steps_ref,
            ))
        ] * N_ENVS)

        model = PPO.load(lname, env=env)
        cb    = TrainingCallback(STEPS_ROUND, tag)
        model.learn(STEPS_ROUND, callback=cb, progress_bar=True)

        best = f"best_{tag}.zip"
        if os.path.exists(best):
            shutil.copy(best, f"{lname}.zip")
            print(f"  >> updated {lname}.zip")
        else:
            print(f"  >> no improvement, {lname}.zip unchanged")
        env.close()


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    for p in [ATTACKER_PATH, DEFENDER_PATH]:
        if not os.path.exists(f"{p}.zip"):
            print(f"ERROR: {p}.zip not found"); exit(1)

    # Delete old broken checkpoints if present
    for name in [MF1_NAME, MF2_NAME]:
        if os.path.exists(f"{name}.zip"):
            print(f"  found {name}.zip — deleting for clean retrain")
            os.remove(f"{name}.zip")

    steps_ref = [0]
    phase0_bootstrap(steps_ref)
    alternating_rounds(steps_ref)

    print("\n" + "="*60)
    print(f"DONE  →  {MF1_NAME}.zip   {MF2_NAME}.zip")
    print("="*60)