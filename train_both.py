import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from football_env import SoccerEnv, FIELD_W, FIELD_H, PLAYER_RADIUS, BALL_RADIUS

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LOAD_A_PATH  = "good_attacker"   # loaded once at the start
LOAD_B_PATH  = "good_defender"   # loaded once at the start
SAVE_A_PATH  = "best_attacker"   # saved and reloaded every round
SAVE_B_PATH  = "best_defender"   # saved and reloaded every round

N_ENVS  = 8
ROUNDS  = 1
STEPS   = 250_000

OBS_SIZE_A = 12
OBS_SIZE_B = 16

# Defender positioning constants
GOAL_Y0        = (FIELD_H - 10.0) / 2
GOAL_Y1        = GOAL_Y0 + 10.0
GOAL_CENTER_Y  = FIELD_H / 2
SHADOW_GAP_MIN = 6.0
SHADOW_GAP_MAX = 22.0
GK_X           = FIELD_W * 0.92
TRAJECTORY_STEPS  = 12
ATTACK_MODE_RATIO = 2.0

# ---------------------------------------------------------------------------
# Annealing
# ---------------------------------------------------------------------------
ANNEALING = [
    (3.0e-4, 0.05, 0.97, 0.93),
    (1.5e-4, 0.02, 0.98, 0.94),
    (5.0e-5, 0.01, 0.99, 0.95),
    (1.0e-5, 0.003, 0.995, 0.95),
]


# ---------------------------------------------------------------------------
# Observation builders
# ---------------------------------------------------------------------------
def get_obs_attacker(env):
    p, v   = env._pos[0], env._vel[0]
    b, bv  = env._ball_pos, env._ball_vel
    target = np.array([FIELD_W, FIELD_H / 2])
    return np.array([
        p[0] / FIELD_W * 2 - 1, p[1] / FIELD_H * 2 - 1,
        v[0], v[1],
        (b - p)[0] / FIELD_W,   (b - p)[1] / FIELD_H,
        (target - p)[0] / FIELD_W, (target - p)[1] / FIELD_H,
        b[0] / FIELD_W * 2 - 1, b[1] / FIELD_H * 2 - 1,
        bv[0], bv[1],
    ], dtype=np.float32)


def get_obs_defender(env):
    p, v   = env._pos[1], env._vel[1]
    opp    = env._pos[0]
    b, bv  = env._ball_pos, env._ball_vel
    target = np.array([0.0,     FIELD_H / 2])
    own    = np.array([FIELD_W, FIELD_H / 2])
    return np.array([
        p[0] / FIELD_W * 2 - 1, p[1] / FIELD_H * 2 - 1,
        v[0], v[1],
        (b - p)[0] / FIELD_W,   (b - p)[1] / FIELD_H,
        (target - p)[0] / FIELD_W, (target - p)[1] / FIELD_H,
        b[0] / FIELD_W * 2 - 1, b[1] / FIELD_H * 2 - 1,
        bv[0], bv[1],
        (own - p)[0] / FIELD_W,  (own - p)[1] / FIELD_H,
        (opp - p)[0] / FIELD_W,  (opp - p)[1] / FIELD_H,
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Defender positioning helpers
# ---------------------------------------------------------------------------
def predict_ball_intercept(ball_pos, ball_vel, steps=TRAJECTORY_STEPS, friction=0.94):
    pos, vel = ball_pos.copy(), ball_vel.copy()
    for _ in range(steps):
        pos = pos + vel
        vel = vel * friction
        pos[0] = np.clip(pos[0], BALL_RADIUS, FIELD_W - BALL_RADIUS)
        pos[1] = np.clip(pos[1], BALL_RADIUS, FIELD_H - BALL_RADIUS)
    return pos


def get_shot_target_y(ball_pos, ball_vel):
    if ball_vel[0] <= 0:
        return None
    dx = FIELD_W - ball_pos[0]
    if dx <= 0:
        return None
    return ball_pos[1] + ball_vel[1] * (dx / ball_vel[0])


def compute_shadow_ideal(ball_pos, ball_vel, player_pos):
    ball_speed        = np.linalg.norm(ball_vel)
    ball_x_norm       = ball_pos[0] / FIELD_W
    own_goal          = np.array([FIELD_W, GOAL_CENTER_Y])
    ball_heading_goal = ball_vel[0] > 0.15 and ball_speed > 0.1
    shot_target_y     = get_shot_target_y(ball_pos, ball_vel) if ball_heading_goal else None

    if ball_heading_goal and shot_target_y is not None:
        if GOAL_Y0 - 3.0 <= shot_target_y <= GOAL_Y1 + 3.0:
            intercept_x  = np.clip(max(player_pos[0] + 1.0, ball_pos[0] + 1.0),
                                   ball_pos[0], GK_X - 1.0)
            return np.array([intercept_x,
                             np.clip(shot_target_y, GOAL_Y0 - 1.0, GOAL_Y1 + 1.0)]), "A"
        else:
            fb   = predict_ball_intercept(ball_pos, ball_vel)
            ftg  = own_goal - fb
            return fb + (ftg / (np.linalg.norm(ftg) + 1e-8)) * 4.0, "B"
    else:
        btg  = own_goal - ball_pos
        gap  = SHADOW_GAP_MIN + (1.0 - ball_x_norm) * (SHADOW_GAP_MAX - SHADOW_GAP_MIN)
        return ball_pos + (btg / (np.linalg.norm(btg) + 1e-8)) * gap, "C"


# ---------------------------------------------------------------------------
# Attacker environment
# ---------------------------------------------------------------------------
class AttackerEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.env = SoccerEnv(n_players=1)
        self.observation_space = spaces.Box(low=-2, high=2,
                                            shape=(OBS_SIZE_A,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=(2,), dtype=np.float32)
        self.opp_model = None

    def _load_opponent(self):
        # Always load the latest saved best_defender
        if os.path.exists(f"{SAVE_B_PATH}.zip"):
            self.opp_model = PPO.load(SAVE_B_PATH)

    def reset(self, seed=None, options=None):
        _, info = self.env.reset(seed=seed, options=options)
        self._load_opponent()
        return get_obs_attacker(self.env), info

    def step(self, action):
        act_a = action.reshape(1, 2)

        if self.opp_model:
            obs_b    = get_obs_defender(self.env)
            act_b, _ = self.opp_model.predict(obs_b, deterministic=False)
        else:
            act_b = self.env.action_space["team_b"].sample()

        _, env_rews, terminated, truncated, info = self.env.step({
            "team_a": act_a,
            "team_b": act_b.reshape(1, 2)
        })

        reward        = 0.0
        ball_pos      = self.env._ball_pos.copy()
        ball_vel      = self.env._ball_vel.copy()
        att_pos       = self.env._pos[0].copy()
        dist_att_ball = np.linalg.norm(att_pos - ball_pos)
        touch_range   = PLAYER_RADIUS + BALL_RADIUS + 2.0

        if env_rews["team_a"] > 0:
            reward += 20.0
        if env_rews["team_b"] > 0:
            reward -= 15.0

        # lightweight shoot anchor — keeps shooting skill alive
        if dist_att_ball < touch_range and ball_vel[0] > 0.1:
            reward += abs(ball_vel[0]) * 0.8

        return get_obs_attacker(self.env), reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Defender environment
# ---------------------------------------------------------------------------
class DefenderEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.env = SoccerEnv(n_players=1)
        self.observation_space = spaces.Box(low=-2, high=2,
                                            shape=(OBS_SIZE_B,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=(2,), dtype=np.float32)
        self.opp_model = None
        self._kicked   = False

    def _load_opponent(self):
        # Always load the latest saved best_attacker
        if os.path.exists(f"{SAVE_A_PATH}.zip"):
            self.opp_model = PPO.load(SAVE_A_PATH)

    def reset(self, seed=None, options=None):
        _, info = self.env.reset(seed=seed, options=options)
        self._load_opponent()
        self._kicked = False
        return get_obs_defender(self.env), info

    def step(self, action):
        act_b = action.reshape(1, 2)

        if self.opp_model:
            obs_a    = get_obs_attacker(self.env)
            act_a, _ = self.opp_model.predict(obs_a, deterministic=False)
        else:
            act_a = self.env.action_space["team_a"].sample()

        _, env_rews, terminated, truncated, info = self.env.step({
            "team_a": act_a.reshape(1, 2),
            "team_b": act_b
        })

        reward     = 0.0
        ball_pos   = self.env._ball_pos.copy()
        ball_vel   = self.env._ball_vel.copy()
        player_pos = self.env._pos[1].copy()
        att_pos    = self.env._pos[0].copy()

        dist_p_ball   = np.linalg.norm(player_pos - ball_pos)
        dist_att_ball = np.linalg.norm(att_pos - ball_pos)
        px            = player_pos[0]
        touch_range   = PLAYER_RADIUS + BALL_RADIUS + 2.0

        opp_is_far   = dist_att_ball >= dist_p_ball * ATTACK_MODE_RATIO
        in_touch     = dist_p_ball < touch_range
        counter_mode = opp_is_far and not self._kicked
        if env_rews["team_b"] > 0 or env_rews["team_a"] > 0:
            self._kicked = False
        if not self._kicked and in_touch and counter_mode and ball_vel[0] < -0.1:
            self._kicked = True

        if env_rews["team_b"] > 0:
            reward += 20.0
        if env_rews["team_a"] > 0:
            reward -= 35.0

        # lightweight shadow anchor — keeps positioning skill alive
        shadow_ideal, mode = compute_shadow_ideal(ball_pos, ball_vel, player_pos)
        shadow_ideal[0] = np.clip(shadow_ideal[0], PLAYER_RADIUS + 1, GK_X - 1.0)
        shadow_ideal[1] = np.clip(shadow_ideal[1], PLAYER_RADIUS + 1, FIELD_H - PLAYER_RADIUS - 1)
        dist_shadow = np.linalg.norm(player_pos - shadow_ideal)
        sigma = {"A": 2.0, "B": 4.0, "C": 5.0}[mode]
        reward += np.exp(-(dist_shadow ** 2) / (2 * sigma ** 2)) * 0.6

        if px > GK_X:
            gk_depth = (px - GK_X) / (FIELD_W - GK_X)
            reward  -= 1.0 + gk_depth * 1.5

        if counter_mode:
            if in_touch and ball_vel[0] < -0.1:
                reward += abs(ball_vel[0]) * 3.0
            elif not in_touch:
                reward += np.exp(-dist_p_ball / 6.0) * 0.3
        else:
            if in_touch and ball_vel[0] < -0.1:
                forward_factor = max(0.0, 1.0 - (px / (FIELD_W * 0.5)))
                reward += abs(ball_vel[0]) * 1.5 * (0.5 + forward_factor)

        return get_obs_defender(self.env), reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Annealing callback
# ---------------------------------------------------------------------------
class AnnealCallback(BaseCallback):
    def __init__(self, total_steps, lr_start, ent_start, lr_end, ent_end):
        super().__init__(verbose=0)
        self._total = total_steps
        self._lr_s  = lr_start;  self._lr_e  = lr_end
        self._ent_s = ent_start; self._ent_e = ent_end

    def _on_step(self):
        p   = self.num_timesteps / self._total
        lr  = self._lr_s  + (self._lr_e  - self._lr_s)  * p
        ent = self._ent_s + (self._ent_e - self._ent_s) * p
        self.model.policy.optimizer.param_groups[0]["lr"] = lr
        self.model.ent_coef = ent
        return True


# ---------------------------------------------------------------------------
# Generic train helper
# ---------------------------------------------------------------------------
def train_role(env_class, load_path, save_path, round_idx):
    lr, ent, gamma, gae = ANNEALING[round_idx]
    next_lr, next_ent   = ANNEALING[min(round_idx + 1, len(ANNEALING) - 1)][:2]

    env = SubprocVecEnv([lambda: Monitor(env_class())] * N_ENVS)

    if os.path.exists(f"{save_path}.zip"):
        # subsequent rounds — load from save_path (already refined)
        print(f"    Loading checkpoint: {save_path}.zip")
        model = PPO.load(save_path, env=env)
    elif os.path.exists(f"{load_path}.zip"):
        # first round — load from the input model
        print(f"    Loading base model: {load_path}.zip")
        model = PPO.load(load_path, env=env)
    else:
        print(f"    No model found at {load_path}.zip — creating new")
        model = PPO("MlpPolicy", env, verbose=1,
                    learning_rate=lr, ent_coef=ent,
                    gamma=gamma, gae_lambda=gae,
                    n_steps=2048, batch_size=64,
                    n_epochs=10, clip_range=0.2)

    # Apply this round's hyperparams
    model.learning_rate = lr
    model.ent_coef      = ent
    model.gamma         = gamma
    model.gae_lambda    = gae
    model.policy.optimizer.param_groups[0]["lr"] = lr

    model.learn(
        total_timesteps = STEPS,
        callback = AnnealCallback(STEPS, lr, ent, next_lr, next_ent),
        progress_bar = True,
    )

    model.save(save_path)
    env.close()
    print(f"    Saved → {save_path}.zip")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  CO-TRAINING")
    print(f"  Load attacker : {LOAD_A_PATH}.zip  →  save as {SAVE_A_PATH}.zip")
    print(f"  Load defender : {LOAD_B_PATH}.zip  →  save as {SAVE_B_PATH}.zip")
    print(f"  {ROUNDS} rounds × {STEPS:,} steps × 2 roles = "
          f"{ROUNDS * STEPS * 2:,} total steps")
    print("=" * 60)

    # Verify input models exist before starting
    for path in [LOAD_A_PATH, LOAD_B_PATH]:
        if not os.path.exists(f"{path}.zip"):
            print(f"\nERROR: {path}.zip not found. "
                  f"Make sure both good_attacker.zip and good_defender.zip are present.")
            exit(1)

    for r in range(ROUNDS):
        lr, ent, gamma, gae = ANNEALING[r]
        print(f"\n{'='*60}")
        print(f"  ROUND {r+1}/{ROUNDS}  |  lr={lr}  ent={ent}  gamma={gamma}")
        print(f"{'='*60}")

        print(f"\n  [{r+1}A] Training ATTACKER vs frozen defender")
        train_role(AttackerEnv, LOAD_A_PATH, SAVE_A_PATH, r)

        print(f"\n  [{r+1}B] Training DEFENDER vs updated attacker")
        train_role(DefenderEnv, LOAD_B_PATH, SAVE_B_PATH, r)

    print("\n" + "=" * 60)
    print("  DONE")
    print(f"  {SAVE_A_PATH}.zip  ←  fine-tuned attacker")
    print(f"  {SAVE_B_PATH}.zip  ←  fine-tuned defender")
    print("=" * 60)