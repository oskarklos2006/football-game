import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ---------------------------------------------------------------------------
# Field / physics constants
# ---------------------------------------------------------------------------
FIELD_W = 50.0
FIELD_H = 30.0
GOAL_H  = 10.0
GOAL_Y0 = (FIELD_H - GOAL_H) / 2
GOAL_Y1 = GOAL_Y0 + GOAL_H

PLAYER_RADIUS = 1.6
BALL_RADIUS   = 1.0
MAX_SPEED     = 1.0
MAX_STEPS     = 1000

FRICTION   = 0.92
KICK_POWER = 1.2
INERTIA    = 0.75
CORNER_R   = 4.0   # visual/physics corner radius — safe for any value >= 0


# ---------------------------------------------------------------------------
# Rounded rectangle boundary helpers
# ---------------------------------------------------------------------------
def _resolve_ball_boundary(ball_pos, ball_vel, r=BALL_RADIUS):
    """
    Constrains ball inside a rounded-rectangle field.
    Uses effective corner radius = max(CORNER_R, r) so it's safe for any CORNER_R.
    """
    pos = ball_pos.copy()
    vel = ball_vel.copy()
    cr  = max(CORNER_R, r + 0.01)   # ensure arc limit is always positive

    in_left   = pos[0] < cr
    in_right  = pos[0] > FIELD_W - cr
    in_top    = pos[1] < cr
    in_bottom = pos[1] > FIELD_H - cr

    if (in_left or in_right) and (in_top or in_bottom):
        cx = cr if in_left else FIELD_W - cr
        cy = cr if in_top  else FIELD_H - cr
        to_ball = pos - np.array([cx, cy])
        dist    = np.linalg.norm(to_ball)
        limit   = cr - r
        if dist > limit and dist > 1e-6:
            normal = to_ball / dist
            pos    = np.array([cx, cy]) + normal * limit
            vel   -= 2 * np.dot(vel, normal) * normal
        elif dist <= 1e-6:
            # Ball exactly on arc center — push toward field center
            to_center = np.array([FIELD_W / 2, FIELD_H / 2]) - pos
            n = to_center / (np.linalg.norm(to_center) + 1e-8)
            pos += n * limit
            vel  = n * np.linalg.norm(vel)
    else:
        if pos[0] < r:
            pos[0] = r
            vel[0] = abs(vel[0])
        elif pos[0] > FIELD_W - r:
            pos[0] = FIELD_W - r
            vel[0] = -abs(vel[0])

        if pos[1] < r:
            pos[1] = r
            vel[1] = abs(vel[1])
        elif pos[1] > FIELD_H - r:
            pos[1] = FIELD_H - r
            vel[1] = -abs(vel[1])

    return pos, vel


def _resolve_player_boundary(pos, vel):
    """
    Constrains a player inside the rounded-rectangle field.
    Uses effective corner radius = max(CORNER_R, PLAYER_RADIUS + 0.01).
    """
    cr    = max(CORNER_R, PLAYER_RADIUS + 0.01)
    r     = PLAYER_RADIUS
    p     = pos.copy()
    v     = vel.copy()

    in_left   = p[0] < cr + r
    in_right  = p[0] > FIELD_W - cr - r
    in_top    = p[1] < cr + r
    in_bottom = p[1] > FIELD_H - cr - r

    if (in_left or in_right) and (in_top or in_bottom):
        cx = cr if (p[0] < FIELD_W / 2) else FIELD_W - cr
        cy = cr if (p[1] < FIELD_H / 2) else FIELD_H - cr
        to_player = p - np.array([cx, cy])
        dist      = np.linalg.norm(to_player)
        limit     = cr - r
        if limit > 0 and dist < limit and dist > 1e-6:
            p = np.array([cx, cy]) + (to_player / dist) * limit
            normal = to_player / dist
            if np.dot(v, normal) < 0:
                v -= np.dot(v, normal) * normal
        elif dist <= 1e-6:
            # Push away from corner center toward field center
            to_center = np.array([FIELD_W / 2, FIELD_H / 2]) - p
            n = to_center / (np.linalg.norm(to_center) + 1e-8)
            p += n * max(limit, 0.1)

    # Always clamp to straight walls as fallback
    p = np.clip(p, [r, r], [FIELD_W - r, FIELD_H - r])
    return p, v


# ---------------------------------------------------------------------------
# Observation builder
# ---------------------------------------------------------------------------
def _make_obs(positions, velocities, ball_pos, ball_vel, team_idx, n):
    own = list(range(n))      if team_idx == 0 else list(range(n, 2 * n))
    opp = list(range(n, 2*n)) if team_idx == 0 else list(range(n))

    def norm_p(p):
        return np.array([p[0] / FIELD_W * 2 - 1,
                         p[1] / FIELD_H * 2 - 1], dtype=np.float32)

    def norm_v(v):
        return np.clip(np.array([v[0] / MAX_SPEED,
                                 v[1] / MAX_SPEED], dtype=np.float32), -1, 1)

    parts = []
    for i in own: parts += [norm_p(positions[i]), norm_v(velocities[i])]
    for i in opp: parts += [norm_p(positions[i]), norm_v(velocities[i])]
    parts.append(norm_p(ball_pos))
    parts.append(np.clip(
        np.array([ball_vel[0] / KICK_POWER,
                  ball_vel[1] / KICK_POWER], dtype=np.float32), -1, 1
    ))
    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# Main environment
# ---------------------------------------------------------------------------
class SoccerEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, n_players=1):
        super().__init__()
        self.render_mode = render_mode
        self.n = n_players

        team_action = spaces.Box(low=-1.0, high=1.0,
                                 shape=(n_players, 2), dtype=np.float32)
        self.action_space = spaces.Dict({"team_a": team_action,
                                         "team_b": team_action})

        obs_size = n_players * 4 * 2 + 4
        single_obs = spaces.Box(low=-1, high=1,
                                shape=(obs_size,), dtype=np.float32)
        self.observation_space = spaces.Dict({"team_a": single_obs,
                                              "team_b": single_obs})

        N = 2 * n_players
        self._pos      = np.zeros((N, 2), dtype=np.float32)
        self._vel      = np.zeros((N, 2), dtype=np.float32)
        self._ball_pos = np.zeros(2, dtype=np.float32)
        self._ball_vel = np.zeros(2, dtype=np.float32)
        self._step_count = 0
        self._score      = [0, 0]
        self._renderer   = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_positions(options=options)
        self._step_count = 0
        return self._get_obs(), {"ball_pos": self._ball_pos.copy()}

    def _reset_positions(self, options=None):
        n = self.n
        margin = PLAYER_RADIUS + 2.0

        for i in range(n):
            self._pos[i] = [
                self.np_random.uniform(margin, FIELD_W / 2 - 2.0),
                self.np_random.uniform(margin, FIELD_H - margin)
            ]
            self._pos[n + i] = [
                self.np_random.uniform(FIELD_W / 2 + 2.0, FIELD_W - margin),
                self.np_random.uniform(margin, FIELD_H - margin)
            ]

        self._vel[:] = 0

        if options is not None and "ball_pos" in options:
            self._ball_pos[:] = options["ball_pos"]
        else:
            self._ball_pos[:] = [
                self.np_random.uniform(margin, FIELD_W - margin),
                self.np_random.uniform(margin, FIELD_H - margin)
            ]

        self._ball_vel[:] = 0

    def step(self, actions):
        n, N = self.n, 2 * self.n
        all_act = np.clip(
            np.concatenate([actions["team_a"], actions["team_b"]], axis=0),
            -1, 1
        )

        # --- move players ---
        for i in range(N):
            move = all_act[i, :2]
            spd  = np.linalg.norm(move)
            if spd > 1.0:
                move = move / spd
            self._vel[i] = (INERTIA * self._vel[i]
                            + (1 - INERTIA) * move * MAX_SPEED)
            self._pos[i] += self._vel[i]
            self._pos[i], self._vel[i] = _resolve_player_boundary(
                self._pos[i], self._vel[i]
            )

        # --- player-player separation ---
        for i in range(N):
            for j in range(i + 1, N):
                diff = self._pos[i] - self._pos[j]
                dist = np.linalg.norm(diff)
                min_dist = PLAYER_RADIUS * 2
                if dist < min_dist and dist > 1e-6:
                    normal  = diff / dist
                    overlap = (min_dist - dist) / 2
                    self._pos[i] = np.clip(
                        self._pos[i] + normal * overlap,
                        [PLAYER_RADIUS, PLAYER_RADIUS],
                        [FIELD_W - PLAYER_RADIUS, FIELD_H - PLAYER_RADIUS]
                    )
                    self._pos[j] = np.clip(
                        self._pos[j] - normal * overlap,
                        [PLAYER_RADIUS, PLAYER_RADIUS],
                        [FIELD_W - PLAYER_RADIUS, FIELD_H - PLAYER_RADIUS]
                    )
                    rel_vel = self._vel[i] - self._vel[j]
                    if np.dot(rel_vel, normal) < 0:
                        impulse = np.dot(rel_vel, normal) * normal * 0.5
                        self._vel[i] -= impulse
                        self._vel[j] += impulse

        # --- ball physics ---
        self._ball_pos += self._ball_vel
        self._ball_vel *= FRICTION

        # --- collision pass: 3 iterations so ball can't stay wedged ---
        for _ in range(3):
            for i in range(N):
                diff = self._ball_pos - self._pos[i]
                d    = np.linalg.norm(diff)
                min_d = PLAYER_RADIUS + BALL_RADIUS
                if 1e-6 < d < min_d:
                    normal = diff / d
                    self._ball_pos = self._pos[i] + normal * min_d
                    base_push = normal * KICK_POWER * 0.6
                    momentum  = self._vel[i] * 0.8
                    new_vel   = base_push + momentum
                    if np.dot(new_vel - self._ball_vel, normal) > 0:
                        self._ball_vel = new_vel

        # --- rounded rectangle boundary ---
        self._ball_pos, self._ball_vel = _resolve_ball_boundary(
            self._ball_pos, self._ball_vel, BALL_RADIUS
        )

        # --- goal check ---
        ra, rb = 0.0, 0.0
        goal = self._check_goal()
        if goal == "team_a":
            ra, rb = 1.0, -1.0
            self._score[0] += 1
            self._reset_positions()
        elif goal == "team_b":
            ra, rb = -1.0, 1.0
            self._score[1] += 1
            self._reset_positions()

        self._step_count += 1
        truncated = self._step_count >= MAX_STEPS

        return (
            self._get_obs(),
            {"team_a": ra, "team_b": rb},
            False,
            truncated,
            {"score": list(self._score), "ball_pos": self._ball_pos.copy()}
        )

    def _check_goal(self):
        bx, by = self._ball_pos
        in_y = GOAL_Y0 <= by <= GOAL_Y1
        if bx >= FIELD_W - BALL_RADIUS and in_y:
            return "team_a"
        if bx <= BALL_RADIUS and in_y:
            return "team_b"
        return None

    def _get_obs(self):
        return {
            "team_a": _make_obs(self._pos, self._vel,
                                self._ball_pos, self._ball_vel, 0, self.n),
            "team_b": _make_obs(self._pos, self._vel,
                                self._ball_pos, self._ball_vel, 1, self.n),
        }

    def render(self):
        if self.render_mode is None:
            return
        try:
            import pygame
        except ImportError:
            print("pip install pygame")
            return

        SCALE = 16
        GD    = int(3 * SCALE)
        MB    = int(4 * SCALE)
        W     = int(FIELD_W * SCALE)
        H     = int(FIELD_H * SCALE)
        TW    = W + 2 * GD
        TH    = H + 2 * MB

        if self._renderer is None:
            pygame.init()
            if self.render_mode == "human":
                self._renderer = pygame.display.set_mode((TW, TH))
                pygame.display.set_caption("Soccer")
            else:
                self._renderer = pygame.Surface((TW, TH))
            self._clock = pygame.time.Clock()
            pr = int(PLAYER_RADIUS * SCALE)
            self._nfont = pygame.font.SysFont("monospace", max(pr, 8), bold=True)
            self._sfont = pygame.font.SysFont("monospace", 20, bold=True)
            self._crowd = pygame.Surface((TW, TH), pygame.SRCALPHA)
            self._crowd.fill((20, 20, 20))
            FAN_COLORS = [
                (220, 50, 50), (255, 200, 0), (255, 255, 255),
                (30, 100, 220), (255, 140, 0), (180, 0, 180),
            ]
            DOT_R, STEP = 6, 12
            rng = np.random.default_rng(42)
            strips = [
                (0, 0, TW, MB), (0, MB + H, TW, MB),
                (0, MB, GD, H), (GD + W, MB, GD, H),
            ]
            for (sx, sy, sw, sh) in strips:
                row, y = 0, sy + STEP // 2
                while y < sy + sh:
                    x = sx + (STEP // 2) + (STEP // 2 if row % 2 else 0)
                    while x < sx + sw:
                        pygame.draw.circle(
                            self._crowd,
                            FAN_COLORS[rng.integers(0, len(FAN_COLORS))],
                            (x, y), DOT_R
                        )
                        x += STEP
                    y += STEP
                    row += 1

        surf = self._renderer
        surf.fill((20, 20, 20))
        surf.blit(self._crowd, (0, 0))

        WHITE, GOAL_NET = (255, 255, 255), (160, 160, 160)
        gy0 = MB + int(GOAL_Y0 * SCALE)
        gy1 = MB + int(GOAL_Y1 * SCALE)
        gh  = gy1 - gy0

        pygame.draw.rect(surf, GOAL_NET, (0, gy0, GD, gh))
        pygame.draw.rect(surf, WHITE,    (0, gy0, GD, gh), 2)
        pygame.draw.rect(surf, GOAL_NET, (GD + W, gy0, GD, gh))
        pygame.draw.rect(surf, WHITE,    (GD + W, gy0, GD, gh), 2)

        # --- grass with rounded corners using alpha mask ---
        CR = int(max(CORNER_R, 1) * SCALE)
        grass_surf = pygame.Surface((W, H), pygame.SRCALPHA)
        grass_surf.fill((0, 0, 0, 0))
        stripe_w = W // 10
        for s in range(10):
            col = (40, 130, 40) if s % 2 == 0 else (50, 160, 50)
            pygame.draw.rect(grass_surf, col, (s * stripe_w, 0, stripe_w, H))
        mask_surf = pygame.Surface((W, H), pygame.SRCALPHA)
        mask_surf.fill((0, 0, 0, 0))
        pygame.draw.rect(mask_surf, (255, 255, 255, 255),
                         pygame.Rect(0, 0, W, H), border_radius=CR)
        grass_surf.blit(mask_surf, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
        surf.blit(grass_surf, (GD, MB))

        # --- rounded field border ---
        field_rect = pygame.Rect(GD, MB, W, H)
        pygame.draw.rect(surf, WHITE, field_rect, 2, border_radius=CR)

        # --- goal lines ---
        pygame.draw.line(surf, WHITE, (GD,     gy0), (GD,     gy1), 3)
        pygame.draw.line(surf, WHITE, (GD + W, gy0), (GD + W, gy1), 3)

        # --- centre line + circle ---
        pygame.draw.line(surf, WHITE, (GD + W // 2, MB), (GD + W // 2, MB + H), 2)
        cx_px, cy_px = GD + W // 2, MB + H // 2
        pygame.draw.circle(surf, WHITE, (cx_px, cy_px), int(6 * SCALE), 2)
        pygame.draw.circle(surf, WHITE, (cx_px, cy_px), 3)

        # --- penalty areas ---
        pa_w = int(8 * SCALE)
        pa_h = int(16 * SCALE)
        pa_y = MB + (H - pa_h) // 2
        pygame.draw.rect(surf, WHITE, (GD,            pa_y, pa_w, pa_h), 2)
        pygame.draw.rect(surf, WHITE, (GD + W - pa_w, pa_y, pa_w, pa_h), 2)

        # --- players ---
        TEAM_A, TEAM_B = (30, 100, 220), (220, 50, 50)
        pr = int(PLAYER_RADIUS * SCALE)
        for i in range(2 * self.n):
            px    = GD + int(self._pos[i, 0] * SCALE)
            py    = MB + int(self._pos[i, 1] * SCALE)
            color = TEAM_A if i < self.n else TEAM_B
            pygame.draw.circle(surf, color, (px, py), pr)
            pygame.draw.circle(surf, WHITE, (px, py), pr, 1)
            lbl = self._nfont.render(str(i % self.n + 1), True, WHITE)
            surf.blit(lbl, (px - lbl.get_width() // 2,
                            py - lbl.get_height() // 2))

        # --- ball ---
        bx = GD + int(self._ball_pos[0] * SCALE)
        by = MB + int(self._ball_pos[1] * SCALE)
        br = int(BALL_RADIUS * SCALE)
        pygame.draw.circle(surf, (240, 240, 240), (bx, by), br)
        pygame.draw.circle(surf, (30, 30, 30), (bx, by), br, 1)

        # --- scoreboard ---
        stxt = self._sfont.render(
            f"  A  {self._score[0]} : {self._score[1]}  B  ",
            True, (220, 220, 220), (20, 20, 20)
        )
        surf.blit(stxt, (TW // 2 - stxt.get_width() // 2,
                         MB // 2 - stxt.get_height() // 2))

        if self.render_mode == "human":
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(pygame.surfarray.array3d(surf), axes=(1, 0, 2))

    def close(self):
        if self._renderer is not None:
            try:
                import pygame
                pygame.quit()
            except Exception:
                pass
            self._renderer = None