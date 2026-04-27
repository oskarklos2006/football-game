import os
import shutil
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from football_env import SoccerEnv, FIELD_W, FIELD_H

# ---------------------------------------------------------------------------
# Top-level hyperparameters and training schedule knobs.
# MIXING_SCHEDULE controls how often each self-play round faces the frozen
# opponent (0.0 = always random, 1.0 = always frozen previous self).
# ANNEAL_STAGES steps down LR and entropy coefficient as training matures,
# indexed by fraction of total timesteps elapsed.
# ---------------------------------------------------------------------------
MODEL_NAME    = "team_a"
OPPONENT_PATH = "opponent"
N_ENVS        = 8
STEPS_PHASE0  = 3_000_000
STEPS_ROUND   = 1_000_000
SELF_PLAY_ROUNDS = 5

MIXING_SCHEDULE = [0.0, 0.1, 0.3, 0.5, 0.7]

CURRICULUM_STEPS  = 500_000
CURRICULUM_BALL_X = FIELD_W - 8.0

ANNEAL_STAGES = [
    (0.25, 1.5e-4, 0.020),
    (0.50, 1.0e-4, 0.010),
    (0.75, 5.0e-5, 0.005),
    (1.00, 1.0e-5, 0.0005),
]


# ---------------------------------------------------------------------------
# Observation builder for one team.
# Returns a flat float32 vector: per-player (pos, vel, vec-to-ball,
# vec-to-goal) followed by ball absolute position and velocity.
# All spatial values are normalised to [-1, 1] or [-2, 2] range.
# ---------------------------------------------------------------------------
def get_relative_obs(positions, velocities, ball_pos, ball_vel, team_idx, n):
    own = list(range(n)) if team_idx == 0 else list(range(n, 2 * n))
    parts = []
    target_goal = np.array([FIELD_W if team_idx == 0 else 0, FIELD_H / 2])

    for i in own:
        p = positions[i]
        to_ball = ball_pos - p
        to_goal = target_goal - p
        parts += [
            p[0] / FIELD_W * 2 - 1, p[1] / FIELD_H * 2 - 1,
            velocities[i][0],        velocities[i][1],
            to_ball[0] / FIELD_W,    to_ball[1] / FIELD_H,
            to_goal[0] / FIELD_W,    to_goal[1] / FIELD_H,
        ]

    parts += [
        ball_pos[0] / FIELD_W * 2 - 1, ball_pos[1] / FIELD_H * 2 - 1,
        ball_vel[0], ball_vel[1],
    ]
    return np.array(parts, dtype=np.float32)


# ---------------------------------------------------------------------------
# Single-agent wrapper around SoccerEnv.
# The agent controls team_a; team_b is either a loaded frozen model or a
# random policy, selected at reset time based on frozen_fraction.
# During the curriculum phase the ball starts near the opponent's goal so the
# agent can discover kicking and scoring before tackling full-field play.
# ---------------------------------------------------------------------------
class TeamAEnv(gym.Env):
    def __init__(self, render_mode=None, opponent_path=None, frozen_fraction=0.0, total_steps_ref=None):
        super().__init__()
        self._env             = SoccerEnv(render_mode=render_mode, n_players=1)
        self._opponent_path   = opponent_path
        self._frozen_fraction = frozen_fraction
        self._total_steps_ref = total_steps_ref
        self._last_obs_b      = None
        self._steps_since_reset = 0
        self._was_in_contact  = False
        self._cached_opponent = None

        self.observation_space = spaces.Box(low=-2, high=2, shape=(12,), dtype=np.float32)
        self.action_space      = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def _get_my_obs(self, obs_dict, team_idx):
        return get_relative_obs(self._env._pos, self._env._vel,
                                self._env._ball_pos, self._env._ball_vel,
                                team_idx, 1)

    def reset(self, seed=None, options=None):
        # Curriculum: keep the ball close to the scoring end until the agent
        # has accumulated enough experience to handle the full field.
        ball_opts = None
        if self._total_steps_ref and self._total_steps_ref[0] < CURRICULUM_STEPS:
            ball_opts = {"ball_pos": [CURRICULUM_BALL_X, FIELD_H / 2]}

        obs, info = self._env.reset(seed=seed, options=ball_opts)
        self._last_obs_b        = self._get_my_obs(obs, 1)
        self._steps_since_reset = 0
        self._was_in_contact    = False

        # Decide at reset time whether this episode uses the frozen opponent.
        # The cached model is loaded once and reused to avoid repeated disk I/O.
        if self._opponent_path and os.path.exists(f"{self._opponent_path}.zip"):
            if self._cached_opponent is None:
                self._cached_opponent = PPO.load(self._opponent_path)
            self._use_frozen = np.random.rand() < self._frozen_fraction
        else:
            self._use_frozen = False

        return self._get_my_obs(obs, 0), info

    def step(self, action_a):
        action_a_2d = np.array(action_a).reshape(1, 2)

        if self._use_frozen:
            action_b, _ = self._cached_opponent.predict(self._last_obs_b, deterministic=False)
        else:
            action_b = self._env.action_space["team_b"].sample()

        obs, env_rewards, terminated, truncated, info = self._env.step({
            "team_a": action_a_2d,
            "team_b": action_b.reshape(1, 2),
        })

        self._last_obs_b = self._get_my_obs(obs, 1)
        if self._total_steps_ref:
            self._total_steps_ref[0] += 1

        from football_env import PLAYER_RADIUS, BALL_RADIUS, GOAL_Y0, GOAL_Y1
        GOAL_CENTER = np.array([FIELD_W, FIELD_H / 2])
        ball_pos    = info["ball_pos"]
        player_pos  = self._env._pos[0]
        ball_speed  = np.linalg.norm(self._env._ball_vel)
        dist_to_ball = np.linalg.norm(ball_pos - player_pos)

        # Sparse goal reward — this is the primary training signal.
        r_goal = env_rewards["team_a"] * 10.0
        if r_goal != 0 or ball_speed > 0.1:
            self._steps_since_reset = 0
        else:
            self._steps_since_reset += 1

        # Anti-camping penalty: nudge agent toward a stationary ball if it
        # has been idle for too long. Scales with distance so nearby is fine.
        r_fetch = 0.0
        if self._steps_since_reset > 40 and ball_speed < 0.1:
            r_fetch = -(dist_to_ball / 25.0) * 0.02

        # Contact rewards: small bonus for any touch that moves the ball,
        # larger bonus if the kick is directed toward the goal.
        # Edge-triggered (rising edge of contact) so held contact isn't farmed.
        r_kick     = 0.0
        in_contact = dist_to_ball < (PLAYER_RADIUS + BALL_RADIUS + 0.6)
        contact_edge = in_contact and not self._was_in_contact
        self._was_in_contact = in_contact

        if contact_edge:
            if ball_speed > 0.4:
                r_kick = 0.1
                ball_to_goal  = GOAL_CENTER - ball_pos
                btg_dist      = np.linalg.norm(ball_to_goal) + 1e-8
                direction_match = np.dot(self._env._ball_vel, ball_to_goal / btg_dist)
                if direction_match > 0.4:
                    r_kick += 0.7
            else:
                r_kick = 0.02

        # Positional progress reward: dense signal proportional to how close
        # the ball is to the goal. Squared so the reward grows sharply near goal.
        r_prog = 0.0
        if ball_speed > 0.1:
            dist_g = np.linalg.norm(ball_pos - GOAL_CENTER)
            r_prog = (1.0 - np.clip(dist_g / FIELD_W, 0, 1)) ** 2 * 0.03

        reward = r_goal + r_kick + r_fetch + r_prog
        return self._get_my_obs(obs, 0), reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# PPO callback that handles two jobs:
# 1. Checkpointing — saves the model whenever mean episode reward improves.
# 2. LR/entropy annealing — steps through ANNEAL_STAGES as training progresses.
# ---------------------------------------------------------------------------
class TrainingCallback(BaseCallback):
    def __init__(self, total_steps, round_name):
        super().__init__(verbose=1)
        self.total_steps  = total_steps
        self.round_name   = round_name
        self.best_rew     = -float("inf")
        self._last_stage  = -1

    def _on_rollout_end(self):
        if len(self.model.ep_info_buffer) > 0:
            avg = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
            if avg > self.best_rew:
                self.best_rew = avg
                self.model.save(f"best_{self.round_name}")

        progress = self.num_timesteps / self.total_steps
        stage = next(
            (i for i, (th, _, _) in enumerate(ANNEAL_STAGES) if progress <= th),
            len(ANNEAL_STAGES) - 1,
        )
        _, lr, ent = ANNEAL_STAGES[stage]
        self.model.policy.optimizer.param_groups[0]["lr"] = lr
        self.model.ent_coef = ent
        if stage > self._last_stage:
            print(f"  [{self.round_name}] Stage {stage + 1} | LR: {lr} Ent: {ent}")
            self._last_stage = stage

    def _on_step(self):
        return True


# ---------------------------------------------------------------------------
# Model factory — shared architecture for all training phases.
# ---------------------------------------------------------------------------
def build_model(env):
    return PPO(
        "MlpPolicy", env,
        verbose=1,
        n_steps=512, batch_size=256, n_epochs=10,
        learning_rate=1.5e-4, ent_coef=0.02,
        policy_kwargs={"net_arch": [256, 256]},
    )


# ---------------------------------------------------------------------------
# Entry point.
# Phase 0 (commented out): train from scratch against a random opponent to
# establish a baseline policy before self-play begins.
# Self-play rounds: copy the current best model as the frozen opponent, then
# continue training with a gradually increasing fraction of frozen opponents.
# After each round the best checkpoint replaces the live model file.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    steps_ref = [3_000_000]

    # PHASE 0 — uncomment to train from scratch.
    # print("\n>>> STARTING PHASE 0")
    # env = SubprocVecEnv([lambda: Monitor(TeamAEnv(total_steps_ref=steps_ref))] * N_ENVS)
    # model = build_model(env)
    # model.learn(total_timesteps=STEPS_PHASE0, callback=TrainingCallback(STEPS_PHASE0, "phase0"), progress_bar=True)
    # if os.path.exists("best_phase0.zip"): shutil.copy("best_phase0.zip", f"{MODEL_NAME}.zip")
    # env.close()

    for r in range(2, SELF_PLAY_ROUNDS + 1):
        round_n = f"round_{r}"
        shutil.copy(f"{MODEL_NAME}.zip", f"{OPPONENT_PATH}.zip")
        print(f"\n>>> {round_n.upper()} | Opponent Frac: {MIXING_SCHEDULE[r - 1]}")

        env = SubprocVecEnv([
            lambda: Monitor(TeamAEnv(
                opponent_path=OPPONENT_PATH,
                frozen_fraction=MIXING_SCHEDULE[r - 1],
                total_steps_ref=steps_ref,
            ))
        ] * N_ENVS)

        model = PPO.load(MODEL_NAME, env=env)
        model.learn(
            total_timesteps=STEPS_ROUND,
            callback=TrainingCallback(STEPS_ROUND, round_n),
            progress_bar=True,
        )

        if os.path.exists(f"best_{round_n}.zip"):
            shutil.copy(f"best_{round_n}.zip", f"{MODEL_NAME}.zip")

        env.close()