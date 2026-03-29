"""
football_env.py
2v2 Soccer environment built on Gymnasium.

Coordinate system:
  x: 0 (left goal) → FIELD_W (right goal)
  y: 0 (top)       → FIELD_H (bottom)

Team A attacks right (+x), defends left.
Team B attacks left  (-x), defends right.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ---------------------------------------------------------------------------
# Field constants
# ---------------------------------------------------------------------------
FIELD_W = 100.0       # field width  (x-axis)
FIELD_H = 60.0        # field height (y-axis)
GOAL_H  = 20.0        # goal opening
GOAL_Y0 = (FIELD_H - GOAL_H) / 2   # top edge of goal
GOAL_Y1 = GOAL_Y0 + GOAL_H         # bottom edge of goal

PLAYER_RADIUS = 1.5
BALL_RADIUS   = 0.8
MAX_SPEED     = 6.0       # units per step
KICK_POWER    = 12.0      # ball speed on kick
FRICTION      = 0.92      # ball velocity multiplier each step
MAX_STEPS     = 1000

N_PLAYERS = 4             # 2 per team


# ---------------------------------------------------------------------------
# Observation helper
# ---------------------------------------------------------------------------
def _obs_for_team(agent_positions, agent_velocities, ball_pos, ball_vel,
                  team_idx: int) -> np.ndarray:
    """
    Build a flat observation vector for one agent on `team_idx`.

    Contents (all values normalised to [-1, 1]):
      own_player_0  x, y, vx, vy          (4)
      own_player_1  x, y, vx, vy          (4)
      opp_player_0  x, y, vx, vy          (4)
      opp_player_1  x, y, vx, vy          (4)
      ball          x, y, vx, vy          (4)
      ─────────────────────────────────────
      total:  20 floats per agent
    """
    if team_idx == 0:
        own  = [0, 1]
        opp  = [2, 3]
    else:
        own  = [2, 3]
        opp  = [0, 1]

    def norm_pos(p):
        return np.array([p[0] / FIELD_W * 2 - 1,
                         p[1] / FIELD_H * 2 - 1], dtype=np.float32)

    def norm_vel(v):
        return np.array([v[0] / MAX_SPEED,
                         v[1] / MAX_SPEED], dtype=np.float32)

    obs = []
    for i in own + opp:
        obs.append(norm_pos(agent_positions[i]))
        obs.append(norm_vel(agent_velocities[i]))
    obs.append(norm_pos(ball_pos))
    ball_vel_norm = np.array([ball_vel[0] / KICK_POWER,
                              ball_vel[1] / KICK_POWER], dtype=np.float32)
    obs.append(ball_vel_norm)
    return np.concatenate(obs)          # shape (20,)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class SoccerEnv(gym.Env):
    """
    2v2 soccer environment.

    Action space (per team, 2 agents):
        Each agent gets a 3-tuple:
            move_x  ∈ [-1, 1]   (left / right)
            move_y  ∈ [-1, 1]   (up   / down)
            kick    ∈ [-1, 1]   (kick if > 0 and ball is close)

    Observation space (per team):
        Box(20,) — see _obs_for_team for layout.

    Step input:
        actions: dict with keys 'team_a' and 'team_b'
            each value is np.ndarray of shape (2, 3)

    Step output:
        obs    : dict with 'team_a' and 'team_b', each np.ndarray (20,)
        rewards: dict with 'team_a' and 'team_b' scalars
        done   : bool
        info   : dict
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None,
                 reward_cfg: dict | None = None):
        super().__init__()
        self.render_mode = render_mode

        # ── reward weights (fully configurable) ────────────────────────────
        default_rewards = dict(
            goal_scored   =  10.0,  # team scores
            goal_conceded = -10.0,  # team concedes
            ball_to_goal  =   0.1,  # reward proportional to ball → opp goal
            touch_ball    =   0.05, # reward for touching the ball
        )
        self.reward_cfg = {**default_rewards, **(reward_cfg or {})}

        # ── spaces ─────────────────────────────────────────────────────────
        single_action = spaces.Box(
            low=np.array([-1, -1, -1], dtype=np.float32),
            high=np.array([ 1,  1,  1], dtype=np.float32),
        )
        # 2 agents per team, each with 3 continuous actions
        team_action = spaces.Box(
            low=np.full((2, 3), -1, dtype=np.float32),
            high=np.full((2, 3),  1, dtype=np.float32),
        )
        self.action_space = spaces.Dict({
            "team_a": team_action,
            "team_b": team_action,
        })

        single_obs = spaces.Box(
            low=-1, high=1, shape=(20,), dtype=np.float32
        )
        self.observation_space = spaces.Dict({
            "team_a": single_obs,
            "team_b": single_obs,
        })

        # ── internal state ──────────────────────────────────────────────────
        self._pos = np.zeros((N_PLAYERS, 2), dtype=np.float32)
        self._vel = np.zeros((N_PLAYERS, 2), dtype=np.float32)
        self._ball_pos = np.zeros(2, dtype=np.float32)
        self._ball_vel = np.zeros(2, dtype=np.float32)
        self._step_count = 0
        self._score = [0, 0]

        # renderer (lazy init)
        self._renderer = None

    # ── reset ───────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_positions()
        self._step_count = 0
        obs = self._get_obs()
        return obs, {}

    def _reset_positions(self):
        """Place players and ball at kickoff positions."""
        cx, cy = FIELD_W / 2, FIELD_H / 2
        # Team A: left side
        self._pos[0] = [cx - 20, cy - 8]
        self._pos[1] = [cx - 20, cy + 8]
        # Team B: right side
        self._pos[2] = [cx + 20, cy - 8]
        self._pos[3] = [cx + 20, cy + 8]
        self._vel[:] = 0
        self._ball_pos = np.array([cx, cy], dtype=np.float32)
        self._ball_vel = np.zeros(2, dtype=np.float32)

    # ── step ────────────────────────────────────────────────────────────────
    def step(self, actions):
        a_actions = np.clip(actions["team_a"], -1, 1)  # (2, 3)
        b_actions = np.clip(actions["team_b"], -1, 1)  # (2, 3)
        all_actions = np.concatenate([a_actions, b_actions], axis=0)  # (4, 3)

        # 1. Move players
        for i in range(N_PLAYERS):
            move = all_actions[i, :2]
            speed = np.linalg.norm(move)
            if speed > 1.0:
                move = move / speed          # normalise to unit disk
            self._pos[i] += move * MAX_SPEED
            self._pos[i] = np.clip(self._pos[i],
                                   [PLAYER_RADIUS, PLAYER_RADIUS],
                                   [FIELD_W - PLAYER_RADIUS, FIELD_H - PLAYER_RADIUS])

        # 2. Kick ball (agent with kick > 0 and close enough wins priority)
        kick_range = PLAYER_RADIUS + BALL_RADIUS + 0.5
        for i in range(N_PLAYERS):
            if all_actions[i, 2] > 0:
                diff = self._ball_pos - self._pos[i]
                dist = np.linalg.norm(diff)
                if dist < kick_range and dist > 1e-6:
                    direction = diff / dist
                    self._ball_vel = direction * KICK_POWER * all_actions[i, 2]
                    break           # first kicker wins; could make competitive

        # 3. Move ball + friction
        self._ball_pos += self._ball_vel
        self._ball_vel *= FRICTION

        # 4. Ball wall bounces (top / bottom)
        if self._ball_pos[1] <= BALL_RADIUS:
            self._ball_pos[1] = BALL_RADIUS
            self._ball_vel[1] = abs(self._ball_vel[1])
        if self._ball_pos[1] >= FIELD_H - BALL_RADIUS:
            self._ball_pos[1] = FIELD_H - BALL_RADIUS
            self._ball_vel[1] = -abs(self._ball_vel[1])

        # 5. Ball-player collisions (simple push)
        for i in range(N_PLAYERS):
            diff = self._ball_pos - self._pos[i]
            dist = np.linalg.norm(diff)
            min_dist = PLAYER_RADIUS + BALL_RADIUS
            if dist < min_dist and dist > 1e-6:
                push = diff / dist * (min_dist - dist)
                self._ball_pos += push

        # 6. Check goals and compute rewards
        reward_a, reward_b = self._compute_rewards()
        goal_scored = self._check_goal()
        terminated = goal_scored is not None

        if goal_scored == "team_a":
            reward_a += self.reward_cfg["goal_scored"]
            reward_b += self.reward_cfg["goal_conceded"]
            self._score[0] += 1
        elif goal_scored == "team_b":
            reward_b += self.reward_cfg["goal_scored"]
            reward_a += self.reward_cfg["goal_conceded"]
            self._score[1] += 1

        if terminated:
            self._reset_positions()
            terminated = False          # continue playing after goal

        # 7. Time limit
        self._step_count += 1
        truncated = self._step_count >= MAX_STEPS

        obs = self._get_obs()
        rewards = {"team_a": float(reward_a), "team_b": float(reward_b)}
        info = {"score": self._score.copy(), "step": self._step_count}

        return obs, rewards, terminated, truncated, info

    # ── reward shaping ──────────────────────────────────────────────────────
    def _compute_rewards(self) -> tuple[float, float]:
        """Dense reward shaping (called every step, before goal check)."""
        r = self.reward_cfg
        bx = self._ball_pos[0]

        # Ball proximity to opponent goal (normalised to [0,1])
        # Team A's goal is at x=0, Team B's goal is at x=FIELD_W
        ball_a_progress = bx / FIELD_W            # A wants this high
        ball_b_progress = 1.0 - bx / FIELD_W      # B wants this high

        reward_a = r["ball_to_goal"] * ball_a_progress
        reward_b = r["ball_to_goal"] * ball_b_progress

        # Touch bonus: reward any agent touching the ball
        kick_range = PLAYER_RADIUS + BALL_RADIUS + 0.5
        for i in range(2):          # Team A agents
            if np.linalg.norm(self._ball_pos - self._pos[i]) < kick_range:
                reward_a += r["touch_ball"]
        for i in range(2, 4):      # Team B agents
            if np.linalg.norm(self._ball_pos - self._pos[i]) < kick_range:
                reward_b += r["touch_ball"]

        return reward_a, reward_b

    def _check_goal(self):
        """
        Returns 'team_a' if A scored, 'team_b' if B scored, else None.
        Team A attacks right (x = FIELD_W).
        Team B attacks left  (x = 0).
        """
        bx, by = self._ball_pos
        in_goal_y = GOAL_Y0 <= by <= GOAL_Y1

        if bx >= FIELD_W - BALL_RADIUS and in_goal_y:
            return "team_a"
        if bx <= BALL_RADIUS and in_goal_y:
            return "team_b"
        return None

    # ── observations ────────────────────────────────────────────────────────
    def _get_obs(self):
        return {
            "team_a": _obs_for_team(
                self._pos, self._vel, self._ball_pos, self._ball_vel, 0),
            "team_b": _obs_for_team(
                self._pos, self._vel, self._ball_pos, self._ball_vel, 1),
        }

    # ── render ──────────────────────────────────────────────────────────────
    def render(self):
        if self.render_mode is None:
            return
        try:
            import pygame
        except ImportError:
            print("Install pygame to render: pip install pygame")
            return

        SCALE = 8           # pixels per unit
        W, H = int(FIELD_W * SCALE), int(FIELD_H * SCALE)

        if self._renderer is None:
            pygame.init()
            if self.render_mode == "human":
                self._renderer = pygame.display.set_mode((W, H))
                pygame.display.set_caption("2v2 Soccer")
            else:
                self._renderer = pygame.Surface((W, H))
            self._clock = pygame.time.Clock()

        surf = self._renderer
        surf.fill((34, 139, 34))        # grass green

        # Field lines
        WHITE = (255, 255, 255)
        pygame.draw.rect(surf, WHITE,
                         (0, 0, W, H), 2)
        pygame.draw.line(surf, WHITE,
                         (W // 2, 0), (W // 2, H), 2)
        pygame.draw.circle(surf, WHITE,
                           (W // 2, H // 2), int(8 * SCALE), 2)

        # Goals
        gy0, gy1 = int(GOAL_Y0 * SCALE), int(GOAL_Y1 * SCALE)
        pygame.draw.rect(surf, WHITE, (0, gy0, int(4 * SCALE), gy1 - gy0), 2)
        pygame.draw.rect(surf, WHITE,
                         (W - int(4 * SCALE), gy0, int(4 * SCALE), gy1 - gy0), 2)

        # Players
        COLORS = [(30, 100, 220), (30, 100, 220),   # Team A — blue
                  (220, 50, 50),  (220, 50, 50)]     # Team B — red
        for i in range(N_PLAYERS):
            cx = int(self._pos[i, 0] * SCALE)
            cy = int(self._pos[i, 1] * SCALE)
            pygame.draw.circle(surf, COLORS[i], (cx, cy),
                               int(PLAYER_RADIUS * SCALE))
            pygame.draw.circle(surf, WHITE, (cx, cy),
                               int(PLAYER_RADIUS * SCALE), 1)

        # Ball
        bx = int(self._ball_pos[0] * SCALE)
        by = int(self._ball_pos[1] * SCALE)
        pygame.draw.circle(surf, (240, 240, 240),
                           (bx, by), int(BALL_RADIUS * SCALE))
        pygame.draw.circle(surf, (50, 50, 50),
                           (bx, by), int(BALL_RADIUS * SCALE), 1)

        # Score
        font = pygame.font.SysFont("monospace", 22, bold=True)
        score_txt = font.render(
            f"A {self._score[0]} : {self._score[1]} B", True, WHITE)
        surf.blit(score_txt, (W // 2 - score_txt.get_width() // 2, 6))

        if self.render_mode == "human":
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])
        else:
            import numpy as np
            return np.transpose(
                pygame.surfarray.array3d(surf), axes=(1, 0, 2))

    def close(self):
        if self._renderer is not None:
            try:
                import pygame
                pygame.quit()
            except Exception:
                pass
            self._renderer = None
