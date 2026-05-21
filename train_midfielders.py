"""
train_midfielders.py  (CPU-optimised, reward-fixed)
====================================================
Two midfielder models (MF1 = forward, MF2 = defensive) trained to cooperate
against your existing best_attacker + best_defender team.

Key changes vs previous version
---------------------------------
1. REWARD FIX   — the -90 ep_rew_mean was almost entirely goals conceded while
                  the MFs are still random. Fixed by:
                  (a) curriculum: ball starts near OWN half so MFs learn to
                      defend first before being thrown into full-field play.
                  (b) smaller net [128,128] — half the params, 2× faster
                      forward passes on CPU, and transplant covers all layers.
                  (c) idle penalty replaced with a gentle approach reward
                      (always positive or zero, never a permanent drain).
                  (d) goal conceded penalty halved to -5 so the gradient
                      signal isn't dominated by a single event.

2. SPEED        — estimated ~1.1 h on 8 CPU cores instead of ~4+ h:
                  • net_arch [128,128] (was [256,256,128])
                  • n_steps 256 (was 512)  — more frequent updates
                  • batch_size 128 (was 256)
                  • n_epochs 6 (was 10)
                  • phase0 150k steps each (was 300k)
                  • 4 alternating rounds of 250k each (was 6 × 800k)

3. TRANSPLANT   — now matches [128,128] donor architecture correctly.

Obs layout (20 features, unchanged)
-------------------------------------
  0-1  : own normalised position
  2-3  : own velocity (clipped)
  4-5  : vector to ball
  6-7  : vector to own goal  (x=50)
  8-9  : vector to scoring goal (x=0)
  10-11: ball absolute position
  12-13: ball velocity (clipped)
  14-15: teammate normalised position
  16-17: teammate velocity (clipped)
  18   : distance to ball (normalised, [0,1])
  19   : ball-heading-danger toward own goal ([-1,1])
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
# Training schedule  ── tuned for CPU speed
# ---------------------------------------------------------------------------
N_ENVS             = 8
BOOTSTRAP_STEPS    = 150_000   # per midfielder
STEPS_ROUND        = 250_000   # per alternating round
ALTERNATING_ROUNDS = 4         # 2 updates per MF

# Frozen teammate fraction ramps up each round so early rounds explore more
TEAMMATE_FROZEN_SCHEDULE = [0.1, 0.3, 0.6, 0.8]

# LR / entropy anneal within each phase
ANNEAL_STAGES = [
    (0.40, 2e-4, 0.03),
    (0.70, 1e-4, 0.01),
    (1.00, 3e-5, 0.003),
]

# Curriculum: for the first N bootstrap steps, spawn ball in OWN half so the
# MF must learn to defend before dealing with full-field situations.
CURRICULUM_STEPS    = 80_000
CURRICULUM_BALL_X   = FIELD_W * 0.75   # ball starts at 75 % toward own goal

# Network size — smaller = faster on CPU, transplant still covers all layers
NET_ARCH = [128, 128]

# ---------------------------------------------------------------------------
# Obs sizes (must match the donor model architectures)
# ---------------------------------------------------------------------------
ATT_OBS_SIZE = 12   # best_attacker was trained on 12 features
MF_OBS_SIZE  = 20   # our midfielders use 20 features

# ---------------------------------------------------------------------------
# Observation builders
# ---------------------------------------------------------------------------

def get_attacker_obs(env):
    p, vel = env._pos[0], env._vel[0]
    b, bv  = env._ball_pos, env._ball_vel
    tgt    = np.array([FIELD_W, FIELD_H / 2])
    return np.array([
        p[0]/FIELD_W*2-1, p[1]/FIELD_H*2-1,
        vel[0], vel[1],
        (b-p)[0]/FIELD_W, (b-p)[1]/FIELD_H,
        (tgt-p)[0]/FIELD_W, (tgt-p)[1]/FIELD_H,
        b[0]/FIELD_W*2-1, b[1]/FIELD_H*2-1,
        bv[0], bv[1],
    ], dtype=np.float32)


def get_defender_obs(env):
    p, vel = env._pos[1], env._vel[1]
    b, bv  = env._ball_pos, env._ball_vel
    opp    = min([env._pos[2], env._pos[3]], key=lambda o: np.linalg.norm(o-b))
    own    = np.array([FIELD_W, FIELD_H/2])
    tgt    = np.array([0.0,     FIELD_H/2])
    spd    = np.linalg.norm(bv) + 1e-8
    bto    = own - b
    danger = np.dot(bv/spd, bto/(np.linalg.norm(bto)+1e-8))
    return np.array([
        p[0]/FIELD_W*2-1, p[1]/FIELD_H*2-1,
        vel[0], vel[1],
        (b-p)[0]/FIELD_W, (b-p)[1]/FIELD_H,
        (tgt-p)[0]/FIELD_W, (tgt-p)[1]/FIELD_H,
        b[0]/FIELD_W*2-1, b[1]/FIELD_H*2-1,
        bv[0], bv[1],
        (own-p)[0]/FIELD_W, (own-p)[1]/FIELD_H,
        (opp-p)[0]/FIELD_W, (opp-p)[1]/FIELD_H,
    ], dtype=np.float32)


def get_mf_obs(env, player_idx, mate_idx):
    p, vel   = env._pos[player_idx], env._vel[player_idx]
    b, bv    = env._ball_pos, env._ball_vel
    mate     = env._pos[mate_idx]
    mate_v   = env._vel[mate_idx]
    own_goal = np.array([FIELD_W, FIELD_H/2])
    sc_goal  = np.array([0.0,     FIELD_H/2])
    spd      = np.linalg.norm(bv) + 1e-8
    bto_own  = own_goal - b
    danger   = np.dot(bv/spd, bto_own/(np.linalg.norm(bto_own)+1e-8))
    d2ball   = np.linalg.norm(b - p)
    return np.array([
        p[0]/FIELD_W*2-1,           p[1]/FIELD_H*2-1,
        np.clip(vel[0],-1,1),       np.clip(vel[1],-1,1),
        (b-p)[0]/FIELD_W,           (b-p)[1]/FIELD_H,
        (own_goal-p)[0]/FIELD_W,    (own_goal-p)[1]/FIELD_H,
        (sc_goal-p)[0]/FIELD_W,     (sc_goal-p)[1]/FIELD_H,
        b[0]/FIELD_W*2-1,           b[1]/FIELD_H*2-1,
        np.clip(bv[0],-1,1),        np.clip(bv[1],-1,1),
        mate[0]/FIELD_W*2-1,        mate[1]/FIELD_H*2-1,
        np.clip(mate_v[0],-1,1),    np.clip(mate_v[1],-1,1),
        np.clip(d2ball/FIELD_W,0,1),
        np.clip(danger,-1,1),
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Weight transplant  ── attacker [128,128] → midfielder [128,128]
# The only layer with a size difference is the first one (12 → 20 inputs).
# All other layers copy verbatim; the 8 extra input columns get tiny noise.
# ---------------------------------------------------------------------------

def transplant_weights(new_model: PPO, donor_path: str) -> None:
    """
    Copy weights from donor (attacker, any hidden width) into new_model.

    The only structural difference we need to handle is the input layer:
    donor has ATT_OBS_SIZE inputs, new_model has MF_OBS_SIZE inputs.
    All other layers are copied only when shapes match exactly, so a
    donor with [256,256] and a new model with [128,128] will copy nothing
    from the hidden layers — but that is fine: we still get the input-layer
    knowledge (which features matter) transplanted, and the smaller hidden
    layers train from scratch very quickly on CPU.
    """
    donor    = PPO.load(donor_path)
    donor_sd = donor.policy.state_dict()
    new_sd   = new_model.policy.state_dict()

    copied, expanded, skipped = 0, 0, 0
    for key in new_sd:
        if key not in donor_sd:
            skipped += 1
            continue
        d, n = donor_sd[key], new_sd[key]

        if d.shape == n.shape:
            # Identical shape — copy verbatim (hidden layers, heads, etc.)
            new_sd[key] = d.clone()
            copied += 1

        elif (key.endswith(".weight")
              and d.shape[1] == ATT_OBS_SIZE
              and n.shape[1] == MF_OBS_SIZE):
            # First linear layer: donor is [H_donor, 12], ours is [H_new, 20].
            # Only copy the 12 original input columns; the row count may differ.
            # If row counts differ we copy only the rows that fit (min of the two).
            rows = min(d.shape[0], n.shape[0])
            w = torch.zeros_like(n)
            w[:rows, :ATT_OBS_SIZE] = d[:rows]
            w[:rows, ATT_OBS_SIZE:] = torch.randn(rows, MF_OBS_SIZE - ATT_OBS_SIZE) * 0.01
            new_sd[key] = w
            expanded += 1

        elif (key.endswith(".bias")
              and d.shape[0] != n.shape[0]):
            # Bias of the first layer may differ in size — skip, keep random init
            skipped += 1

        else:
            skipped += 1

    new_model.policy.load_state_dict(new_sd)
    print(f"  transplant: {copied} copied, {expanded} expanded, {skipped} skipped")


# ---------------------------------------------------------------------------
# MidfielderEnv
# ---------------------------------------------------------------------------
class MidfielderEnv(gym.Env):
    """
    Single-agent wrapper. The learner controls one midfielder (team_b).
    The teammate, attacker, and defender are all handled internally.

    Reward design (all components positive or zero except goal conceded):
      r_goal_scored  : +5.0   (halved from 10 — still dominant but not crushing)
      r_goal_conceded: -5.0   (halved — prevents gradient collapse when losing badly)
      r_kick         : +0.05 base, +0.4 bonus if aimed at goal
      r_pass         : +1.0 completed, +0.5 received
      r_approach     : +0.03 * (1 - norm_dist) — always positive, fades with distance
                       replaces the idle PENALTY, so every step has a non-negative floor
      r_spread       : small positive when >8 units from teammate
      r_ball_advance : +0.3 proportional to forward ball movement this tick
      r_role         : Gaussian positional bonus (MF1 ahead, MF2 behind ball)
    """

    def __init__(self, learner_role, teammate_path=None,
                 frozen_fraction=0.0, total_steps_ref=None):
        super().__init__()
        assert learner_role in ("mf1", "mf2")
        self._role         = learner_role
        self._tm_path      = teammate_path
        self._frozen_frac  = frozen_fraction
        self._steps_ref    = total_steps_ref
        self._env          = SoccerEnv(n_players=2)

        # player_idx: own index in env._pos; mate_idx: teammate's index
        self._my_idx   = 2 if learner_role == "mf1" else 3
        self._mate_idx = 3 if learner_role == "mf1" else 2

        self._cached_att = None
        self._cached_def = None
        self._cached_tm  = None
        self._use_frozen = False

        self._was_contact   = False
        self._prev_ball_x   = FIELD_W / 2
        self._near_me_prev  = False
        self._near_tm_prev  = False

        self.observation_space = spaces.Box(-2, 2, shape=(MF_OBS_SIZE,), dtype=np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def _lazy_load(self):
        if self._cached_att is None and os.path.exists(f"{ATTACKER_PATH}.zip"):
            self._cached_att = PPO.load(ATTACKER_PATH)
        if self._cached_def is None and os.path.exists(f"{DEFENDER_PATH}.zip"):
            self._cached_def = PPO.load(DEFENDER_PATH)

    def reset(self, seed=None, options=None):
        self._lazy_load()

        # Curriculum: spawn ball in own half until we've seen enough steps
        ball_opts = None
        if (self._steps_ref is not None
                and self._steps_ref[0] < CURRICULUM_STEPS):
            ball_opts = {"ball_pos": [CURRICULUM_BALL_X,
                                      self._env.np_random.uniform(4, FIELD_H-4)
                                      if hasattr(self._env, 'np_random') else FIELD_H/2]}

        obs_d, info = self._env.reset(seed=seed, options=ball_opts)

        self._was_contact  = False
        self._near_me_prev = False
        self._near_tm_prev = False
        self._prev_ball_x  = self._env._ball_pos[0]

        # Decide teammate policy for this episode
        if (self._tm_path is not None
                and os.path.exists(f"{self._tm_path}.zip")
                and np.random.rand() < self._frozen_frac):
            if self._cached_tm is None:
                self._cached_tm = PPO.load(self._tm_path)
            self._use_frozen = True
        else:
            self._use_frozen = False

        return get_mf_obs(self._env, self._my_idx, self._mate_idx), info

    def step(self, my_action):
        # ── teammate action ──────────────────────────────────────────────
        tm_obs = get_mf_obs(self._env, self._mate_idx, self._my_idx)
        if self._use_frozen and self._cached_tm is not None:
            tm_act, _ = self._cached_tm.predict(tm_obs, deterministic=False)
        else:
            tm_act = self._env.action_space["team_b"].sample()[0]

        # ── opponent actions ─────────────────────────────────────────────
        att_act = (self._cached_att.predict(get_attacker_obs(self._env),
                                            deterministic=False)[0]
                   if self._cached_att else
                   self._env.action_space["team_a"].sample()[0])

        def_act = (self._cached_def.predict(get_defender_obs(self._env),
                                            deterministic=False)[0]
                   if self._cached_def else
                   self._env.action_space["team_a"].sample()[0])

        # ── assemble and step ────────────────────────────────────────────
        if self._role == "mf1":
            tb = np.stack([np.array(my_action), np.array(tm_act)])
        else:
            tb = np.stack([np.array(tm_act), np.array(my_action)])

        obs_d, env_rew, terminated, truncated, info = self._env.step({
            "team_a": np.stack([np.array(att_act), np.array(def_act)]),
            "team_b": tb,
        })

        if self._steps_ref is not None:
            self._steps_ref[0] += 1

        # ── reward shaping ───────────────────────────────────────────────
        b         = info["ball_pos"]
        p         = self._env._pos[self._my_idx]
        mate_p    = self._env._pos[self._mate_idx]
        bv        = self._env._ball_vel
        bspd      = np.linalg.norm(bv)
        d2ball    = np.linalg.norm(b - p)
        in_cont   = d2ball < PLAYER_RADIUS + BALL_RADIUS + 0.6
        edge      = in_cont and not self._was_contact
        self._was_contact = in_cont

        SC_GOAL = np.array([0.0, FIELD_H/2])

        # Goal signals — halved magnitude so one conceded goal ≠ wiping out
        # 10 minutes of shaping rewards
        r_goal = env_rew["team_b"] * 5.0

        # Kick reward (edge-triggered — no farming)
        r_kick = 0.0
        if edge and bspd > 0.4:
            r_kick = 0.05
            btg  = SC_GOAL - b
            dm   = np.dot(bv, btg / (np.linalg.norm(btg)+1e-8))
            if dm > 0.4:
                r_kick += 0.4

        # Pass reward
        near_me = d2ball < PLAYER_RADIUS + BALL_RADIUS + 3.0
        near_tm = np.linalg.norm(b - mate_p) < PLAYER_RADIUS + BALL_RADIUS + 3.0
        r_pass  = 0.0
        if near_tm and self._near_me_prev and not self._near_tm_prev:
            r_pass = 1.0   # we just passed to teammate
        elif near_me and self._near_tm_prev and not self._near_me_prev:
            r_pass = 0.5   # received pass from teammate
        self._near_me_prev = near_me
        self._near_tm_prev = near_tm

        # Approach reward — always >= 0, replaces idle penalty
        # Peaks at 0.03 when right on the ball, decays linearly to 0 at far end
        r_approach = 0.03 * max(0.0, 1.0 - d2ball / FIELD_W)

        # Spread bonus — tiny, fires every tick, pushes MFs apart passively
        mate_dist = np.linalg.norm(p - mate_p)
        r_spread  = 0.008 if mate_dist > 8.0 else 0.0

        # Ball advance toward x=0 this tick
        r_advance = 0.0
        if bspd > 0.05:
            dx = self._prev_ball_x - b[0]   # positive = toward x=0
            r_advance = np.clip(dx / FIELD_W, 0, 1) * 0.3
        self._prev_ball_x = b[0]

        # Spatial role: Gaussian bonus for being in the "right" zone
        if self._role == "mf1":
            ideal_x = b[0] - 5.0   # ahead of ball toward scoring goal
        else:
            ideal_x = b[0] + 7.0   # behind ball, covering own half
        r_role = np.exp(-(p[0] - ideal_x)**2 / 12.0) * 0.03

        reward = r_goal + r_kick + r_pass + r_approach + r_spread + r_advance + r_role
        return get_mf_obs(self._env, self._my_idx, self._mate_idx), reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------
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
                print(f"  [{self.name}] checkpoint  avg_rew={avg:.2f}")

        prog = self.num_timesteps / self.total_steps
        stage = next((i for i,(th,_,_) in enumerate(ANNEAL_STAGES) if prog <= th),
                     len(ANNEAL_STAGES)-1)
        _, lr, ent = ANNEAL_STAGES[stage]
        self.model.policy.optimizer.param_groups[0]["lr"] = lr
        self.model.ent_coef = ent
        if stage != self._stage:
            print(f"  [{self.name}] anneal → lr={lr} ent={ent}")
            self._stage = stage

    def _on_step(self): return True


# ---------------------------------------------------------------------------
# Model factory  ── smaller net for CPU speed
# ---------------------------------------------------------------------------
def build_model(env, seed=42):
    return PPO(
        "MlpPolicy", env,
        verbose=1,
        n_steps=256,        # was 512 — halves buffer fill time
        batch_size=128,     # was 256
        n_epochs=6,         # was 10
        learning_rate=2e-4,
        ent_coef=0.03,
        policy_kwargs={"net_arch": NET_ARCH},
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Phase 0 — bootstrap with attacker weight transplant
# ---------------------------------------------------------------------------
def phase0_bootstrap(steps_ref):
    print("\n" + "="*60)
    print("PHASE 0  bootstrap  (~150k steps each, ~15 min total)")
    print("="*60)

    for role, name, seed in [("mf1", MF1_NAME, 42), ("mf2", MF2_NAME, 7)]:
        print(f"\n  >> {role.upper()}")
        env = SubprocVecEnv([
            lambda r=role: Monitor(MidfielderEnv(
                learner_role=r,
                frozen_fraction=0.0,
                total_steps_ref=steps_ref,
            ))
        ] * N_ENVS)

        model = build_model(env, seed=seed)

        if os.path.exists(f"{ATTACKER_PATH}.zip"):
            transplant_weights(model, ATTACKER_PATH)
        else:
            print("  WARNING: attacker not found, random init")

        model.learn(BOOTSTRAP_STEPS,
                    callback=TrainingCallback(BOOTSTRAP_STEPS, f"p0_{role}"),
                    progress_bar=True)

        best = f"best_p0_{role}.zip"
        shutil.copy(best if os.path.exists(best) else f"{name}.zip", f"{name}.zip")
        if not os.path.exists(best):
            model.save(name)
        print(f"  >> saved {name}.zip")
        env.close()


# ---------------------------------------------------------------------------
# Alternating rounds
# ---------------------------------------------------------------------------
def alternating_rounds(steps_ref):
    print("\n" + "="*60)
    print("SELF-PLAY  4 alternating rounds  (~250k each, ~45 min total)")
    print("="*60)

    for r in range(ALTERNATING_ROUNDS):
        if r % 2 == 0:
            learner, lname, tmname = "mf1", MF1_NAME, MF2_NAME
        else:
            learner, lname, tmname = "mf2", MF2_NAME, MF1_NAME

        frac = TEAMMATE_FROZEN_SCHEDULE[r]
        tag  = f"{learner}_r{r+1}"
        print(f"\n>> Round {r+1}/{ALTERNATING_ROUNDS}  updating {learner.upper()}"
              f"  frozen_tm_frac={frac}")

        env = SubprocVecEnv([
            lambda lr=learner, tn=tmname: Monitor(MidfielderEnv(
                learner_role=lr,
                teammate_path=tn,
                frozen_fraction=frac,
                total_steps_ref=steps_ref,
            ))
        ] * N_ENVS)

        model = PPO.load(lname, env=env)
        model.learn(STEPS_ROUND,
                    callback=TrainingCallback(STEPS_ROUND, tag),
                    progress_bar=True)

        best = f"best_{tag}.zip"
        if os.path.exists(best):
            shutil.copy(best, f"{lname}.zip")
            print(f"  >> updated {lname}.zip")
        else:
            print(f"  >> no improvement, {lname}.zip unchanged")
        env.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for p in [ATTACKER_PATH, DEFENDER_PATH]:
        if not os.path.exists(f"{p}.zip"):
            print(f"ERROR: {p}.zip not found"); exit(1)

    steps_ref = [0]

    if os.path.exists(f"{MF1_NAME}.zip") and os.path.exists(f"{MF2_NAME}.zip"):
        print(">> mf1.zip and mf2.zip exist — skipping phase 0")
    else:
        phase0_bootstrap(steps_ref)

    alternating_rounds(steps_ref)

    print("\n" + "="*60)
    print(f"DONE  →  {MF1_NAME}.zip   {MF2_NAME}.zip")
    print("="*60)