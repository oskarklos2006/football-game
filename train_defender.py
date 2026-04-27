import os
import shutil
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from football_env import SoccerEnv, FIELD_W, FIELD_H, PLAYER_RADIUS, BALL_RADIUS

# ---------------------------------------------------------------------------
# Paths and schedule.
# The attacker is frozen throughout — it never updates here.
# SELF_PLAY_PATH is a snapshot of the defender copied at the start of each
# round so the learner can face its own previous self as the opponent.
# MIXING_SCHEDULE[r] is the probability of drawing that frozen-self opponent
# instead of the attacker, increasing each round to harden the opposition.
# ---------------------------------------------------------------------------
ATTACKER_PATH  = "best_attacker"
DEFENDER_NAME  = "best_defender"
SELF_PLAY_PATH = "defender_selfplay_opponent"
N_ENVS         = 8
STEPS_ROUND    = 500_000
SELF_PLAY_ROUNDS = 3

MIXING_SCHEDULE = [0.0, 0.2, 0.5, 0.7]

# LR and entropy step down as training progresses within each round.
ANNEAL_STAGES = [
    (0.25, 1.0e-4, 0.010),
    (0.50, 5.0e-5, 0.005),
    (1.00, 1.0e-5, 0.0005),
]

# After touching the ball the defender enters counter-attack mode for at most
# MAX_COUNTER_STEPS ticks before reverting to defensive positioning.
MAX_COUNTER_STEPS = 150
DANGER_ZONE_X     = FIELD_W * 0.3


# ---------------------------------------------------------------------------
# Observation builders — one per role.
# The defender observation is symmetric: team_idx controls which end of the
# field is "own goal" so the same function works for both sides during
# self-play (the mirrored defender opponent uses team_idx=0).
# ---------------------------------------------------------------------------
def get_defender_obs(positions, velocities, ball_pos, ball_vel, team_idx, n):
    player_idx = n if team_idx == 1 else 0

    OWN_GOAL     = np.array([0.0     if team_idx == 1 else FIELD_W, FIELD_H / 2])
    SCORING_GOAL = np.array([FIELD_W if team_idx == 1 else 0.0,     FIELD_H / 2])

    p   = positions[player_idx]
    vel = velocities[player_idx]
    to_ball     = ball_pos - p
    to_own_goal = OWN_GOAL - p

    ball_speed        = np.linalg.norm(ball_vel) + 1e-8
    ball_to_own_goal  = OWN_GOAL - ball_pos
    botg_dist         = np.linalg.norm(ball_to_own_goal) + 1e-8
    ball_to_sg_dist   = np.linalg.norm(SCORING_GOAL - ball_pos) + 1e-8

    # Positive = ball moving toward own goal, negative = moving away.
    ball_heading_danger = np.dot(ball_vel / ball_speed, ball_to_own_goal / botg_dist)

    # Binary flag so the agent knows whether it's already behind the ball.
    ball_in_own_half = 1.0 if (
        ball_pos[0] < FIELD_W / 2 if team_idx == 1 else ball_pos[0] > FIELD_W / 2
    ) else 0.0

    return np.array([
        p[0] / FIELD_W * 2 - 1,          p[1] / FIELD_H * 2 - 1,
        np.clip(vel[0], -1, 1),           np.clip(vel[1], -1, 1),
        to_ball[0] / FIELD_W,             to_ball[1] / FIELD_H,
        to_own_goal[0] / FIELD_W,         to_own_goal[1] / FIELD_H,
        ball_pos[0] / FIELD_W * 2 - 1,   ball_pos[1] / FIELD_H * 2 - 1,
        np.clip(ball_vel[0], -1, 1),      np.clip(ball_vel[1], -1, 1),
        np.clip(botg_dist / FIELD_W, 0, 1),
        np.clip(ball_to_sg_dist / FIELD_W, 0, 1),
        np.clip(ball_heading_danger, -1, 1),
        ball_in_own_half,
    ], dtype=np.float32)


def get_attacker_obs(positions, velocities, ball_pos, ball_vel):
    # 12-feature vector matching the frozen attacker model's input shape.
    p   = positions[0]
    vel = velocities[0]
    SCORING_GOAL = np.array([FIELD_W, FIELD_H / 2])
    to_ball = ball_pos - p
    to_goal = SCORING_GOAL - p
    return np.array([
        p[0] / FIELD_W * 2 - 1, p[1] / FIELD_H * 2 - 1,
        vel[0], vel[1],
        to_ball[0] / FIELD_W,   to_ball[1] / FIELD_H,
        to_goal[0] / FIELD_W,   to_goal[1] / FIELD_H,
        ball_pos[0] / FIELD_W * 2 - 1, ball_pos[1] / FIELD_H * 2 - 1,
        ball_vel[0], ball_vel[1],
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# DefenderEnv — the learner is always team_b.
# Opponent selection happens at reset: with probability frozen_fraction the
# episode uses a mirrored copy of the defender's previous self (16-feature
# obs, team_idx=0); otherwise it uses the frozen attacker (12-feature obs).
# The two opponent types need different observation functions, which is why
# the step method branches on _use_selfplay_opp before calling predict.
# ---------------------------------------------------------------------------
class DefenderEnv(gym.Env):
    def __init__(self, render_mode=None, self_play_opponent_path=None,
                 frozen_fraction=0.0, total_steps_ref=None):
        super().__init__()
        self._env              = SoccerEnv(render_mode=render_mode, n_players=1)
        self._self_play_path   = self_play_opponent_path
        self._frozen_fraction  = frozen_fraction
        self._total_steps_ref  = total_steps_ref

        self._was_in_contact    = False
        self._ball_touched_by_def = False
        self._counter_state     = "defend"
        self._steps_in_counter  = 0
        self._prev_ball_x       = 0.0

        # Attacker is loaded once at construction; self-play opponent is loaded
        # lazily on first use so the file can be updated between rounds.
        self._cached_attacker = (PPO.load(ATTACKER_PATH)
                                 if os.path.exists(f"{ATTACKER_PATH}.zip") else None)
        self._cached_selfplay  = None
        self._use_selfplay_opp = False

        self.observation_space = spaces.Box(low=-2, high=2, shape=(16,), dtype=np.float32)
        self.action_space      = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        obs_dict, info = self._env.reset(seed=seed, options=options)
        self._was_in_contact      = False
        self._ball_touched_by_def = False
        self._counter_state       = "defend"
        self._steps_in_counter    = 0
        self._prev_ball_x         = self._env._ball_pos[0]

        self._use_selfplay_opp = (
            self._self_play_path is not None
            and np.random.rand() < self._frozen_fraction
        )

        return (get_defender_obs(self._env._pos, self._env._vel,
                                 self._env._ball_pos, self._env._ball_vel, 1, 1), info)

    def step(self, action_b):
        # Pick the right opponent model and matching observation shape.
        if self._use_selfplay_opp:
            if self._cached_selfplay is None:
                self._cached_selfplay = PPO.load(self._self_play_path)
            opp_obs   = get_defender_obs(self._env._pos, self._env._vel,
                                         self._env._ball_pos, self._env._ball_vel, 0, 1)
            opp_model = self._cached_selfplay
        else:
            opp_obs   = get_attacker_obs(self._env._pos, self._env._vel,
                                         self._env._ball_pos, self._env._ball_vel)
            opp_model = self._cached_attacker

        if opp_model:
            action_a, _ = opp_model.predict(opp_obs, deterministic=False)
        else:
            action_a = self._env.action_space["team_a"].sample()

        obs_dict, env_rewards, terminated, truncated, info = self._env.step({
            "team_a": action_a.reshape(1, 2),
            "team_b": np.array(action_b).reshape(1, 2),
        })

        if self._total_steps_ref:
            self._total_steps_ref[0] += 1

        ball_pos    = info["ball_pos"]
        player_pos  = self._env._pos[1]
        dist_to_ball = np.linalg.norm(ball_pos - player_pos)
        in_contact   = dist_to_ball < (PLAYER_RADIUS + BALL_RADIUS + 0.6)

        # Rising edge of contact flips the state machine into counter mode.
        if in_contact and not self._was_in_contact:
            self._ball_touched_by_def = True
            self._counter_state       = "counter"
            self._steps_in_counter    = 0
        self._was_in_contact = in_contact

        # Counter mode expires when the attacker regains possession or time runs out.
        if self._counter_state == "counter":
            self._steps_in_counter += 1
            att_dist = np.linalg.norm(ball_pos - self._env._pos[0])
            if (att_dist < (PLAYER_RADIUS + BALL_RADIUS + 0.8)
                    or self._steps_in_counter > MAX_COUNTER_STEPS):
                self._counter_state = "defend"

        # Sparse goal signals.
        r_goal          = env_rewards["team_b"] * 10.0
        # Extra bonus for scoring after the defender touched the ball — rewards
        # completing a full defensive sequence (intercept → counter → score).
        r_counter_bonus = 10.0 if (env_rewards["team_b"] > 0
                                   and self._ball_touched_by_def) else 0.0
        # Reward touching the ball inside the danger zone to encourage active defending.
        r_intercept     = 0.5 if (in_contact and ball_pos[0] < DANGER_ZONE_X) else 0.0

        # In counter mode: reward ball moving forward and defender staying close.
        r_carry = 0.0
        if self._counter_state == "counter":
            r_carry += np.clip((ball_pos[0] - self._prev_ball_x) / FIELD_W, 0, 1) * 2.0
            r_carry += (1.0 - np.clip(dist_to_ball / FIELD_W, 0, 1)) * 0.05

        # In defend mode: Gaussian reward for holding a sensible depth
        # relative to the ball — not too deep, not too high.
        r_position = 0.0
        if self._counter_state == "defend":
            ideal_x    = np.clip(ball_pos[0] * 0.5, 5, 20)
            r_position = np.exp(-(abs(player_pos[0] - ideal_x) ** 2) / 10) * 0.05

        self._prev_ball_x = ball_pos[0]
        reward = r_goal + r_counter_bonus + r_intercept + r_carry + r_position

        return (get_defender_obs(self._env._pos, self._env._vel,
                                 ball_pos, self._env._ball_vel, 1, 1),
                reward, terminated, truncated, info)


# ---------------------------------------------------------------------------
# Callback: saves a checkpoint whenever mean episode reward improves and
# steps down LR/entropy through ANNEAL_STAGES as the round progresses.
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

    def _on_step(self):
        return True


# ---------------------------------------------------------------------------
# Entry point.
# Each round: snapshot the current defender as the self-play opponent, then
# train against a mix of that snapshot and the frozen attacker.
# The best checkpoint from the round replaces the live defender file so the
# next round's opponent snapshot is always the strongest version so far.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    steps_ref = [1_500_000]

    if not os.path.exists(f"{DEFENDER_NAME}.zip"):
        print(f"Error: {DEFENDER_NAME}.zip required to skip Phase 0.")
        exit()

    for r in range(1, SELF_PLAY_ROUNDS + 1):
        shutil.copy(f"{DEFENDER_NAME}.zip", f"{SELF_PLAY_PATH}.zip")
        frac = MIXING_SCHEDULE[r]
        print(f"\n>>> ROUND {r} | Self-Play Probability: {frac}")

        env = SubprocVecEnv([
            lambda: Monitor(DefenderEnv(
                self_play_opponent_path=SELF_PLAY_PATH,
                frozen_fraction=frac,
                total_steps_ref=steps_ref,
            ))
        ] * N_ENVS)

        model = PPO.load(DEFENDER_NAME, env=env)
        model.learn(
            total_timesteps=STEPS_ROUND,
            callback=TrainingCallback(STEPS_ROUND, f"round_{r}"),
            progress_bar=True,
        )

        if os.path.exists(f"best_round_{r}.zip"):
            shutil.copy(f"best_round_{r}.zip", f"{DEFENDER_NAME}.zip")
            print(f">>> Round {r} Complete: updated {DEFENDER_NAME}.zip")

        env.close()