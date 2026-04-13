"""
football_env.py
6v6 Soccer environment built on Gymnasium.

Coordinate system:
  x: 0 (left goal) - FIELD_W right goal
  y: 0 (top)       - FIELD_H bottom

Team A attacks right (+x), defends left.
Team B attacks left  (-x), defends right.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Field constants
FIELD_W = 50.0
FIELD_H = 30.0
GOAL_H  = 10.0
GOAL_Y0 = (FIELD_H - GOAL_H) / 2
GOAL_Y1 = GOAL_Y0 + GOAL_H

PLAYER_RADIUS = 1.3
BALL_RADIUS   = 0.8
MAX_SPEED     = 1.3       
KICK_POWER    = 1.0
FRICTION      = 0.93
MAX_STEPS     = 1000

N_PLAYERS = 12            # 6 per team


# Observation helper
def _obs_for_team(agent_positions, agent_velocities, ball_pos, ball_vel,
                  team_idx: int) -> np.ndarray:
    """
    Flat observation vector for the whole team (all values in [-1, 1]):
      own_player_0..5  x, y, vx, vy   (6x4 = 24)
      opp_player_0..5  x, y, vx, vy   (6x4 = 24)
      ball             x, y, vx, vy   (4)
      total: 52 floats
    """
    if team_idx == 0:
        own = [0, 1, 2, 3, 4, 5]
        opp = [6, 7, 8, 9, 10, 11]
    else:
        own = [6, 7, 8, 9, 10, 11]
        opp = [0, 1, 2, 3, 4, 5]

    def norm_pos(p):
        return np.array([p[0] / FIELD_W * 2 - 1,
                         p[1] / FIELD_H * 2 - 1], dtype=np.float32)

    def norm_vel(v):
        return np.clip(
            np.array([v[0] / MAX_SPEED, v[1] / MAX_SPEED], dtype=np.float32),
            -1, 1
        )

    obs = []
    for i in own + opp:
        obs.append(norm_pos(agent_positions[i]))
        obs.append(norm_vel(agent_velocities[i]))
    obs.append(norm_pos(ball_pos))
    ball_vel_norm = np.clip(
        np.array([ball_vel[0] / KICK_POWER,
                  ball_vel[1] / KICK_POWER], dtype=np.float32),
        -1, 1
    )
    obs.append(ball_vel_norm)
    return np.concatenate(obs)   


# Environment
class SoccerEnv(gym.Env):
    """
    6v6 soccer environment.

    Action space (per team, 3 agents):
        np.ndarray shape (6, 3):
            move_x  in [-1, 1]
            move_y  in [-1, 1]
            kick    in [-1, 1]   (kicks if > 0 and ball is close)

    Step input:
        actions: dict  {'team_a': ndarray (3,3), 'team_b': ndarray (3,3)}

    Step output:
        obs    : dict  {'team_a': ndarray (28,), 'team_b': ndarray (28,)}
        rewards: dict  {'team_a': float, 'team_b': float}
        terminated, truncated, info
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, reward_cfg: dict | None = None):
        super().__init__()
        self.render_mode = render_mode

        default_rewards = dict(
            goal_scored   =  10.0,
            goal_conceded = -10.0,
            ball_to_goal  =   0.1,
            touch_ball    =   0.05,
        )
        self.reward_cfg = {**default_rewards, **(reward_cfg or {})}

        team_action = spaces.Box(
            low=np.full((6, 3), -1, dtype=np.float32),
            high=np.full((6, 3),  1, dtype=np.float32),
        )
        self.action_space = spaces.Dict({
            "team_a": team_action,
            "team_b": team_action,
        })

        single_obs = spaces.Box(low=-1, high=1, shape=(52,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "team_a": single_obs,
            "team_b": single_obs,
        })

        self._pos      = np.zeros((N_PLAYERS, 2), dtype=np.float32)
        self._vel      = np.zeros((N_PLAYERS, 2), dtype=np.float32)
        self._ball_pos = np.zeros(2, dtype=np.float32)
        self._ball_vel = np.zeros(2, dtype=np.float32)
        self._step_count = 0
        self._score = [0, 0]
        self._renderer = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_positions()
        self._step_count = 0
        return self._get_obs(), {}

    def _reset_positions(self):
        cx, cy = FIELD_W / 2, FIELD_H / 2

        # Team A (left side): goalkeeper, 2 defenders, 2 midfielders, 1 forward
        self._pos[0]  = [1.5,      cy]        # goalkeeper (inside left goal)
        self._pos[1]  = [cx - 18,  cy - 6]    # defender
        self._pos[2]  = [cx - 18,  cy + 6]    # defender
        self._pos[3]  = [cx - 10,  cy - 5]    # midfielder
        self._pos[4]  = [cx - 10,  cy + 5]    # midfielder
        self._pos[5]  = [cx - 3,   cy]        # forward

        # Team B (right side): goalkeeper, 2 defenders, 2 midfielders, 1 forward
        self._pos[6]  = [FIELD_W - 1.5, cy]   # goalkeeper (inside right goal)
        self._pos[7]  = [cx + 18,  cy - 6]    # defender
        self._pos[8]  = [cx + 18,  cy + 6]    # defender
        self._pos[9]  = [cx + 10,  cy - 5]    # midfielder
        self._pos[10] = [cx + 10,  cy + 5]    # midfielder
        self._pos[11] = [cx + 3,   cy]        # forward

        self._vel[:] = 0
        self._ball_pos = np.array([cx, cy], dtype=np.float32)
        self._ball_vel = np.zeros(2, dtype=np.float32)

    def step(self, actions):
        a_act = np.clip(actions["team_a"], -1, 1)
        b_act = np.clip(actions["team_b"], -1, 1)
        all_actions = np.concatenate([a_act, b_act], axis=0)  # (12, 3)

        # 1. Move players — inertia smooths out random action flickering
        INERTIA = 0.75
        for i in range(N_PLAYERS):
            move = all_actions[i, :2]
            speed = np.linalg.norm(move)
            if speed > 1.0:
                move = move / speed
            self._vel[i] = INERTIA * self._vel[i] + (1 - INERTIA) * move * MAX_SPEED
            self._pos[i] += self._vel[i]
            self._pos[i] = np.clip(self._pos[i],
                    [PLAYER_RADIUS, PLAYER_RADIUS],
                    [FIELD_W - PLAYER_RADIUS, FIELD_H - PLAYER_RADIUS])

        # 2. Kick — closest eligible player with kick > 0 wins
        kick_range = PLAYER_RADIUS + BALL_RADIUS + 0.5
        best_i, best_dist = -1, kick_range
        for i in range(N_PLAYERS):
            if all_actions[i, 2] > 0:
                dist = np.linalg.norm(self._ball_pos - self._pos[i])
                if dist < best_dist:
                    best_dist = dist
                    best_i = i
        if best_i >= 0:
            diff = self._ball_pos - self._pos[best_i]
            dist = np.linalg.norm(diff)
            if dist > 1e-6:
                self._ball_vel = (diff / dist) * KICK_POWER * all_actions[best_i, 2]

        # 3. Move ball + friction
        self._ball_pos += self._ball_vel
        self._ball_vel *= FRICTION

        # 4. Wall bounces — all four sides
        if self._ball_pos[0] <= BALL_RADIUS:
            self._ball_pos[0] = BALL_RADIUS
            self._ball_vel[0] = abs(self._ball_vel[0])
        if self._ball_pos[0] >= FIELD_W - BALL_RADIUS:
            self._ball_pos[0] = FIELD_W - BALL_RADIUS
            self._ball_vel[0] = -abs(self._ball_vel[0])
        if self._ball_pos[1] <= BALL_RADIUS:
            self._ball_pos[1] = BALL_RADIUS
            self._ball_vel[1] = abs(self._ball_vel[1])
        if self._ball_pos[1] >= FIELD_H - BALL_RADIUS:
            self._ball_pos[1] = FIELD_H - BALL_RADIUS
            self._ball_vel[1] = -abs(self._ball_vel[1])

        # 5. Ball-player collisions
        for i in range(N_PLAYERS):
            diff = self._ball_pos - self._pos[i]
            dist = np.linalg.norm(diff)
            min_dist = PLAYER_RADIUS + BALL_RADIUS
            if dist < min_dist and dist > 1e-6:
                self._ball_pos += diff / dist * (min_dist - dist)

        # 6. Rewards + goal check
        reward_a, reward_b = self._compute_rewards()
        goal = self._check_goal()

        if goal == "team_a":
            reward_a += self.reward_cfg["goal_scored"]
            reward_b += self.reward_cfg["goal_conceded"]
            self._score[0] += 1
            self._reset_positions()
        elif goal == "team_b":
            reward_b += self.reward_cfg["goal_scored"]
            reward_a += self.reward_cfg["goal_conceded"]
            self._score[1] += 1
            self._reset_positions()

        self._step_count += 1
        truncated = self._step_count >= MAX_STEPS

        return (
            self._get_obs(),
            {"team_a": float(reward_a), "team_b": float(reward_b)},
            False,
            truncated,
            {"score": self._score.copy(), "step": self._step_count},
        )

    def _compute_rewards(self):
        r = self.reward_cfg
        bx = self._ball_pos[0]
        reward_a = r["ball_to_goal"] * (bx / FIELD_W)
        reward_b = r["ball_to_goal"] * (1.0 - bx / FIELD_W)

        kick_range = PLAYER_RADIUS + BALL_RADIUS + 0.5
        for i in range(6):
            if np.linalg.norm(self._ball_pos - self._pos[i]) < kick_range:
                reward_a += r["touch_ball"]
        for i in range(6, 12):
            if np.linalg.norm(self._ball_pos - self._pos[i]) < kick_range:
                reward_b += r["touch_ball"]

        return reward_a, reward_b

    def _check_goal(self):
        bx, by = self._ball_pos
        in_goal_y = GOAL_Y0 <= by <= GOAL_Y1
        if bx >= FIELD_W - BALL_RADIUS and in_goal_y:
            return "team_a"
        if bx <= BALL_RADIUS and in_goal_y:
            return "team_b"
        return None

    def _get_obs(self):
        return {
            "team_a": _obs_for_team(self._pos, self._vel,
                                    self._ball_pos, self._ball_vel, 0),
            "team_b": _obs_for_team(self._pos, self._vel,
                                    self._ball_pos, self._ball_vel, 1),
        }

    def render(self):
        if self.render_mode is None:
            return
        try:
            import pygame
        except ImportError:
            print("pip install pygame")
            return

        SCALE  = 16
        GD     = int(3 * SCALE)          # goal depth in pixels
        MB     = int(4 * SCALE)          # top/bottom margin for future fans
        W      = int(FIELD_W * SCALE)    # field width in pixels
        H      = int(FIELD_H * SCALE)    # field height in pixels
        TW     = W + 2 * GD              # total window width
        TH     = H + 2 * MB              # total window height

        if self._renderer is None:
            pygame.init()
            if self.render_mode == "human":
                self._renderer = pygame.display.set_mode((TW, TH))
                pygame.display.set_caption("6v6 Soccer")
            else:
                self._renderer = pygame.Surface((TW, TH))
            self._clock = pygame.time.Clock()
            pr = int(PLAYER_RADIUS * SCALE)
            self._nfont = pygame.font.SysFont("monospace", max(pr, 8), bold=True)
            self._sfont = pygame.font.SysFont("monospace", 20, bold=True)
            # pre-render crowd once — covers all four black strips
            import numpy as np
            self._crowd = pygame.Surface((TW, TH), pygame.SRCALPHA)
            self._crowd.fill((20, 20, 20))
            FAN_COLORS = [
                (220,  50,  50),   # red
                (255, 200,   0),   # yellow
                (255, 255, 255),   # white
                ( 30, 100, 220),   # blue
                (255, 140,   0),   # orange
                (180,   0, 180),   # purple
            ]
            DOT_R, STEP = 6, 12
            rng = np.random.default_rng(42)
            # four strips: top, bottom, left side, right side
            strips = [
                (0,        0,  TW, MB),           # top
                (0,  MB + H,  TW, MB),           # bottom
                (0,       MB,  GD, H),            # left of field
                (GD + W,  MB,  GD, H),            # right of field
            ]
            for (sx, sy, sw, sh) in strips:
                row = 0
                y = sy + STEP // 2
                while y < sy + sh:
                    x = sx + (STEP // 2) + (STEP // 2 if row % 2 else 0)
                    while x < sx + sw:
                        col = FAN_COLORS[rng.integers(0, len(FAN_COLORS))]
                        pygame.draw.circle(self._crowd, col, (x, y), DOT_R)
                        x += STEP
                    y += STEP
                    row += 1

        surf = self._renderer

        # background
        surf.fill((20, 20, 20))

        # blit pre-rendered crowd
        surf.blit(self._crowd, (0, 0))

        WHITE    = (255, 255, 255)
        GOAL_NET = (160, 160, 160)

        gy0 = MB + int(GOAL_Y0 * SCALE)
        gy1 = MB + int(GOAL_Y1 * SCALE)
        gh  = gy1 - gy0

        # left goal
        pygame.draw.rect(surf, GOAL_NET, (0,      gy0, GD, gh))
        pygame.draw.rect(surf, WHITE,    (0,      gy0, GD, gh), 2)

        # right goal
        pygame.draw.rect(surf, GOAL_NET, (GD + W, gy0, GD, gh))
        pygame.draw.rect(surf, WHITE,    (GD + W, gy0, GD, gh), 2)

        # grass stripes — only on the field area
        DARK_GREEN  = (40, 130, 40)
        LIGHT_GREEN = (50, 160, 50)
        stripe_w = W // 10
        for s in range(10):
            col = DARK_GREEN if s % 2 == 0 else LIGHT_GREEN
            pygame.draw.rect(surf, col, (GD + s * stripe_w, MB, stripe_w, H))

        # field border
        pygame.draw.rect(surf, WHITE, (GD, MB, W, H), 2)

        # goal mouth lines
        pygame.draw.line(surf, WHITE, (GD,      gy0), (GD,      gy1), 3)
        pygame.draw.line(surf, WHITE, (GD + W,  gy0), (GD + W,  gy1), 3)

        # halfway line
        pygame.draw.line(surf, WHITE, (GD + W // 2, MB), (GD + W // 2, MB + H), 2)

        # centre circle + spot
        cx_px = GD + W // 2
        cy_px = MB + H // 2
        pygame.draw.circle(surf, WHITE, (cx_px, cy_px), int(6 * SCALE), 2)
        pygame.draw.circle(surf, WHITE, (cx_px, cy_px), 3)

        # penalty areas
        pa_w = int(8 * SCALE)
        pa_h = int(16 * SCALE)
        pa_y = MB + (H - pa_h) // 2
        pygame.draw.rect(surf, WHITE, (GD,             pa_y, pa_w, pa_h), 2)
        pygame.draw.rect(surf, WHITE, (GD + W - pa_w,  pa_y, pa_w, pa_h), 2)

        # players
        TEAM_A = (30, 100, 220)
        TEAM_B = (220, 50,  50)
        pr = int(PLAYER_RADIUS * SCALE)

        for i in range(N_PLAYERS):
            px = GD + int(self._pos[i, 0] * SCALE)
            py = MB + int(self._pos[i, 1] * SCALE)
            color = TEAM_A if i < 6 else TEAM_B
            pygame.draw.circle(surf, color, (px, py), pr)
            pygame.draw.circle(surf, WHITE, (px, py), pr, 1)
            lbl = self._nfont.render(str(i % 6 + 1), True, WHITE)
            surf.blit(lbl, (px - lbl.get_width() // 2,
                            py - lbl.get_height() // 2))

        # ball
        bx = GD + int(self._ball_pos[0] * SCALE)
        by = MB + int(self._ball_pos[1] * SCALE)
        br = int(BALL_RADIUS * SCALE)
        pygame.draw.circle(surf, (240, 240, 240), (bx, by), br)
        pygame.draw.circle(surf, (30, 30, 30), (bx, by), br, 1)

        # scoreboard — centred in the top margin
        stxt = self._sfont.render(f"  A  {self._score[0]} : {self._score[1]}  B  ",
                            True, (220, 220, 220), (20, 20, 20))
        surf.blit(stxt, (TW // 2 - stxt.get_width() // 2,
                         MB // 2 - stxt.get_height() // 2))

        if self.render_mode == "human":
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])
        else:
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