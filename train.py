import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from football_env import SoccerEnv, FIELD_W, FIELD_H, PLAYER_RADIUS, BALL_RADIUS

# ---------------------------------------------------------------------------
# Wrapper — exposes single-agent (team_a) interface to SB3
# ---------------------------------------------------------------------------
# KEY FIX 1: action_space must be flat Box(2,) not Box(1,2).
#   SB3 always outputs a 1D action vector. We reshape it inside step().
#
# KEY FIX 2: opponent_holder is a list([model|None]) so all envs share
#   the same reference. When you update opponent_holder[0] between training
#   rounds, every env automatically uses the new opponent — no rebuild needed.
#
# KEY FIX 3: reward shaping teaches the right behaviour:
#   - Small continuous reward for closing distance to ball  → agent stays active
#   - Small continuous reward for pushing ball toward right → agent learns direction
#   - Big sparse reward for scoring                         → actual goal
#   - Small penalty for opponent scoring                    → avoid own goals
#   No "touch bonus" — that teaches camping, not playing.
# ---------------------------------------------------------------------------

class TeamAEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, opponent_holder=None):
        super().__init__()
        self._env             = SoccerEnv(render_mode=render_mode, n_players=1)
        self._opponent_holder = opponent_holder
        self._last_obs_b      = None
        self._prev_ball_x     = FIELD_W / 2   # track ball progress between steps

        # SB3 needs a flat 1D action space
        self.action_space      = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = self._env.observation_space["team_a"]
        self.render_mode       = render_mode

    def reset(self, seed=None, options=None):
        obs, info = self._env.reset(seed=seed, options=options)
        self._last_obs_b  = obs["team_b"]
        self._prev_ball_x = self._env._ball_pos[0]
        return obs["team_a"], info

    def step(self, action_a):
        # Reshape SB3's flat (2,) → env's expected (1, 2)
        action_a_2d = np.array(action_a, dtype=np.float32).reshape(1, 2)

        # Opponent: use loaded model or sample randomly
        opponent = self._opponent_holder[0] if self._opponent_holder else None
        if opponent is not None:
            action_b, _ = opponent.predict(self._last_obs_b, deterministic=False)
        else:
            action_b = self._env.action_space["team_b"].sample()

        obs, env_rewards, terminated, truncated, info = self._env.step({
            "team_a": action_a_2d,
            "team_b": action_b,
        })
        self._last_obs_b = obs["team_b"]

        ball_pos   = info["ball_pos"]
        player_pos = self._env._pos[0]

        # --- reward shaping ---

        # 1. Distance to ball: reward for being close (normalized 0→1)
        dist_to_ball = np.linalg.norm(ball_pos - player_pos)
        r_proximity  = 1.0 - np.clip(dist_to_ball / (FIELD_W * 0.5), 0, 1)

        # 2. Ball progress toward opponent goal (right side):
        #    reward delta — only reward actual forward movement this step
        ball_x_delta  = ball_pos[0] - self._prev_ball_x
        r_ball_fwd    = np.clip(ball_x_delta / FIELD_W, -0.5, 0.5)
        self._prev_ball_x = ball_pos[0]

        # 3. Sparse goal reward (dominates everything else)
        r_goal = env_rewards["team_a"] * 10.0   # +10 score, -10 concede

        # Combined — keep shaping small so sparse goal stays dominant
        reward = r_goal + 0.02 * r_proximity + 0.1 * r_ball_fwd

        return obs["team_a"], reward, terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

MODEL_NAME   = "team_a"
TOTAL_STEPS  = 500_000   # increase to 1-2M for better play
N_ENVS       = 4

# Shared opponent reference — update this between self-play rounds
# without rebuilding the vec env
opponent_holder = [None]

if os.path.exists(f"{MODEL_NAME}.zip"):
    print(f"Loading '{MODEL_NAME}.zip' as frozen opponent.")
    opponent_holder[0] = PPO.load(MODEL_NAME)


# KEY FIX 4: use a factory function, not a lambda.
#   [lambda: Env()] * N creates one lambda repeated N times — they all
#   share the same closure. A factory call creates N independent envs.
def make_env():
    return Monitor(TeamAEnv(opponent_holder=opponent_holder))

env = DummyVecEnv([make_env] * N_ENVS)


# KEY FIX 5: when resuming, pass hyperparams via custom_objects.
#   Setting model.learning_rate = x after PPO.load() has no effect —
#   SB3 captures the schedule at load time, not as a live attribute.
if os.path.exists(f"{MODEL_NAME}.zip"):
    print(f"Resuming training from '{MODEL_NAME}.zip'.")
    model = PPO.load(
        MODEL_NAME,
        env=env,
        custom_objects={
            "learning_rate": 3e-4,
            "clip_range":    0.2,
        },
    )
else:
    print("Starting fresh.")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        # --- rollout ---
        n_steps=2048,       # steps per env per update; 2048*4 = 8192 samples/update
        batch_size=256,
        n_epochs=10,
        # --- optimisation ---
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,      # entropy bonus keeps exploration alive
        clip_range=0.2,
        # --- policy ---
        use_sde=True,
        policy_kwargs={
            "squash_output": True,
            "net_arch": [256, 256],
        },
    )

print(f"Training for {TOTAL_STEPS:,} steps...")
model.learn(total_timesteps=TOTAL_STEPS, progress_bar=True)
model.save(MODEL_NAME)
print(f"Saved '{MODEL_NAME}.zip'.")
env.close()


# ---------------------------------------------------------------------------
# SELF-PLAY LOOP (optional but recommended after initial training)
# ---------------------------------------------------------------------------
# Run this block in a second script or uncomment and loop here.
# Each round: load best model as frozen opponent, train agent against it.
# This prevents the agent from overfit to random play and forces it to
# learn genuine strategy.
#
# ROUNDS = 5
# for round_i in range(1, ROUNDS + 1):
#     print(f"\n=== Self-play round {round_i}/{ROUNDS} ===")
#     opponent_holder[0] = PPO.load(MODEL_NAME)   # freeze current best
#     env = DummyVecEnv([make_env] * N_ENVS)
#     model.set_env(env)
#     model.learn(total_timesteps=TOTAL_STEPS, reset_num_timesteps=False, progress_bar=True)
#     model.save(f"{MODEL_NAME}_r{round_i}")
#     model.save(MODEL_NAME)  # overwrite "best" for next round
#     env.close()