import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ---------------------------------------------------------------------------
# Field dimensions and physics tuning knobs.
# GOAL_Y0/Y1 center the goal vertically on the field.
# ---------------------------------------------------------------------------
FIELD_W = 50.0
FIELD_H = 30.0
GOAL_H  = 10.0
GOAL_Y0 = (FIELD_H - GOAL_H) / 2
GOAL_Y1 = GOAL_Y0 + GOAL_H

PLAYER_RADIUS = 1.6
BALL_RADIUS   = 1.0
MAX_SPEED     = 0.8
MAX_STEPS     = 1000

# FRICTION slows the ball each tick; KICK_POWER is the base impulse applied on
# contact; INERTIA blends old and new velocity so players feel "heavy".
FRICTION   = 0.94
KICK_POWER = 1.2
INERTIA    = 0.75
CORNER_R   = 4.0

# If the ball hasn't moved more than STALE_DX/DY within STALE_STEPS ticks,
# it gets teleported to break stuck-in-corner situations.
STALE_STEPS = 60
STALE_DX    = 3.0
STALE_DY    = 2.0


# ---------------------------------------------------------------------------
# Boundary resolution — rounded rectangle walls.
# The field corners are arcs, not hard 90° angles, so we need to handle the
# case where the ball enters a corner quadrant separately from flat-wall bounces.
# ---------------------------------------------------------------------------
def _resolve_ball_boundary(ball_pos, ball_vel, r=BALL_RADIUS):
    pos = ball_pos.copy()
    vel = ball_vel.copy()
    cr = max(CORNER_R, r + 0.1)
    limit = cr - r

    # Detect which corner quadrant (if any) the ball is in.
    kx = 1 if pos[0] < cr else (-1 if pos[0] > FIELD_W - cr else 0)
    ky = 1 if pos[1] < cr else (-1 if pos[1] > FIELD_H - cr else 0)

    if kx != 0 and ky != 0:
        # Inside a corner arc: push ball back to arc surface and reflect velocity.
        cx = cr if kx == 1 else FIELD_W - cr
        cy = cr if ky == 1 else FIELD_H - cr
        center = np.array([cx, cy])
        to_ball = pos - center
        dist = np.linalg.norm(to_ball)

        if dist > limit:
            normal = to_ball / (dist + 1e-8)
            pos = center + normal * limit
            vel -= 2 * np.dot(vel, normal) * normal
    else:
        # Flat walls: clamp position and flip the perpendicular velocity component.
        if pos[0] < r:
            pos[0] = r; vel[0] = abs(vel[0])
        elif pos[0] > FIELD_W - r:
            pos[0] = FIELD_W - r; vel[0] = -abs(vel[0])
        if pos[1] < r:
            pos[1] = r; vel[1] = abs(vel[1])
        elif pos[1] > FIELD_H - r:
            pos[1] = FIELD_H - r; vel[1] = -abs(vel[1])

    return pos, vel


def _resolve_player_boundary(pos, vel):
    # Same corner logic as the ball, but players slide along the curve instead
    # of bouncing — we cancel only the outward velocity component.
    r = PLAYER_RADIUS
    p = pos.copy()
    v = vel.copy()
    cr = max(CORNER_R, r + 0.1)
    limit = cr - r

    kx = 1 if p[0] < cr else (-1 if p[0] > FIELD_W - cr else 0)
    ky = 1 if p[1] < cr else (-1 if p[1] > FIELD_H - cr else 0)

    if kx != 0 and ky != 0:
        cx = cr if kx == 1 else FIELD_W - cr
        cy = cr if ky == 1 else FIELD_H - cr
        center = np.array([cx, cy])
        to_p = p - center
        dist = np.linalg.norm(to_p)

        if dist > limit:
            normal = to_p / (dist + 1e-8)
            p = center + normal * limit
            if np.dot(v, normal) > 0:
                v -= np.dot(v, normal) * normal

    p = np.clip(p, [r, r], [FIELD_W - r, FIELD_H - r])
    return p, v


# ---------------------------------------------------------------------------
# Observation builder — produces a fixed-size float32 vector for one team.
# All values are normalised to roughly [-1, 1] so the policy network sees a
# consistent scale regardless of field size or speed constants.
# ---------------------------------------------------------------------------
def _make_obs(positions, velocities, ball_pos, ball_vel, team_idx, n):
    p_idx = 0 if team_idx == 0 else 1
    o_idx = 1 if team_idx == 0 else 0

    # Each team attacks the far end of the field from their starting half.
    target_goal = np.array([FIELD_W if team_idx == 0 else 0.0, FIELD_H / 2])
    own_goal    = np.array([0.0 if team_idx == 0 else FIELD_W, FIELD_H / 2])

    p   = positions[p_idx]
    v   = velocities[p_idx]
    opp = positions[o_idx]

    # 16 features: agent pos, agent vel, vec-to-ball, vec-to-target-goal,
    # vec-to-own-goal, ball abs pos, ball vel, vec-to-opponent.
    obs = [
        p[0] / FIELD_W * 2 - 1, p[1] / FIELD_H * 2 - 1,
        v[0] / MAX_SPEED,        v[1] / MAX_SPEED,
        (ball_pos - p)[0] / FIELD_W, (ball_pos - p)[1] / FIELD_H,
        (target_goal - p)[0] / FIELD_W, (target_goal - p)[1] / FIELD_H,
        (own_goal - p)[0] / FIELD_W,    (own_goal - p)[1] / FIELD_H,
        ball_pos[0] / FIELD_W * 2 - 1,  ball_pos[1] / FIELD_H * 2 - 1,
        ball_vel[0] / KICK_POWER,        ball_vel[1] / KICK_POWER,
        (opp - p)[0] / FIELD_W,          (opp - p)[1] / FIELD_H,
    ]

    return np.array(obs, dtype=np.float32)


# ---------------------------------------------------------------------------
# SoccerEnv — a two-team, configurable-n-players Gymnasium environment.
# Action space: continuous 2D movement vector per player in [-1, 1].
# Observation space: 16-feature vector per team (see _make_obs above).
# Rewards: +1 / -1 on goal scored / conceded, 0 otherwise.
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

        obs_size   = n_players * 4 * 2 + 4
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

        self._stale_anchor      = np.zeros(2, dtype=np.float32)
        self._stale_anchor_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_positions(options=options)
        self._step_count = 0
        self._stale_anchor[:]   = self._ball_pos
        self._stale_anchor_step = 0
        return self._get_obs(), {"ball_pos": self._ball_pos.copy()}

    def _reset_positions(self, options=None):
        # Place teams on their respective halves; ball goes to centre or random.
        self._pos[0] = [FIELD_W * 0.2, FIELD_H / 2]
        self._pos[1] = [FIELD_W * 0.8, FIELD_H / 2]
        self._vel[:]  = 0

        margin = PLAYER_RADIUS + 2.0
        if options and "ball_pos" in options:
            self._ball_pos[:] = options["ball_pos"]
        else:
            if self.np_random.random() < 0.5:
                self._ball_pos[:] = [FIELD_W / 2,
                                     self.np_random.uniform(margin, FIELD_H - margin)]
            else:
                self._ball_pos[:] = [
                    self.np_random.uniform(margin, FIELD_W - margin),
                    self.np_random.uniform(margin, FIELD_H - margin),
                ]
        self._ball_vel[:] = 0

    def _teleport_ball(self):
        # Find a spot that isn't occupied by any player and drop the ball there.
        margin = BALL_RADIUS + PLAYER_RADIUS + 2.0
        for _ in range(50):
            x = self.np_random.uniform(margin, FIELD_W - margin)
            y = self.np_random.uniform(margin, FIELD_H - margin)
            candidate = np.array([x, y], dtype=np.float32)
            min_dist  = min(np.linalg.norm(candidate - self._pos[i])
                            for i in range(2 * self.n))
            if min_dist >= PLAYER_RADIUS + BALL_RADIUS + 1.0:
                break
        self._ball_pos[:] = candidate
        self._ball_vel[:] = 0.0

    def _check_stalemate(self):
        # Every STALE_STEPS ticks, compare ball position to the saved anchor.
        # If it hasn't moved enough in either axis, teleport it to shake things up.
        steps_since = self._step_count - self._stale_anchor_step
        if steps_since < STALE_STEPS:
            return

        dx = abs(self._ball_pos[0] - self._stale_anchor[0])
        dy = abs(self._ball_pos[1] - self._stale_anchor[1])

        if dx <= STALE_DX and dy <= STALE_DY:
            self._teleport_ball()

        self._stale_anchor[:]   = self._ball_pos
        self._stale_anchor_step = self._step_count

    def step(self, actions):
        n, N = self.n, 2 * self.n
        all_act = np.clip(
            np.concatenate([actions["team_a"], actions["team_b"]], axis=0),
            -1, 1
        )

        # Apply inertia-blended movement then resolve wall collisions per player.
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

        # Prevent players from overlapping: push overlapping pairs apart and
        # exchange their velocity components along the collision normal.
        for i in range(N):
            for j in range(i + 1, N):
                diff     = self._pos[i] - self._pos[j]
                dist     = np.linalg.norm(diff)
                min_dist = PLAYER_RADIUS * 2
                if dist < min_dist and dist > 1e-6:
                    normal  = diff / dist
                    overlap = (min_dist - dist) / 2
                    self._pos[i] = np.clip(self._pos[i] + normal * overlap,
                                           [PLAYER_RADIUS, PLAYER_RADIUS],
                                           [FIELD_W - PLAYER_RADIUS, FIELD_H - PLAYER_RADIUS])
                    self._pos[j] = np.clip(self._pos[j] - normal * overlap,
                                           [PLAYER_RADIUS, PLAYER_RADIUS],
                                           [FIELD_W - PLAYER_RADIUS, FIELD_H - PLAYER_RADIUS])
                    rel_vel = self._vel[i] - self._vel[j]
                    if np.dot(rel_vel, normal) < 0:
                        impulse = np.dot(rel_vel, normal) * normal * 0.5
                        self._vel[i] -= impulse
                        self._vel[j] += impulse

        # Advance ball position and apply friction.
        self._ball_pos += self._ball_vel
        self._ball_vel *= FRICTION

        # Three collision sub-steps so the ball can't tunnel through a player
        # or stay wedged between two of them across a single tick.
        for _ in range(3):
            for i in range(N):
                diff  = self._ball_pos - self._pos[i]
                d     = np.linalg.norm(diff)
                min_d = PLAYER_RADIUS + BALL_RADIUS
                if 1e-6 < d < min_d:
                    normal         = diff / d
                    self._ball_pos = self._pos[i] + normal * min_d
                    base_push      = normal * KICK_POWER * 0.6
                    momentum       = self._vel[i] * 0.8
                    new_vel        = base_push + momentum
                    if np.dot(new_vel - self._ball_vel, normal) > 0:
                        self._ball_vel = new_vel

        self._ball_pos, self._ball_vel = _resolve_ball_boundary(
            self._ball_pos, self._ball_vel, BALL_RADIUS
        )

        # Check for a goal, assign rewards, and reset if one was scored.
        ra, rb = 0.0, 0.0
        goal   = self._check_goal()
        if goal == "team_a":
            ra, rb = 1.0, -1.0
            self._score[0] += 1
            self._reset_positions()
            self._stale_anchor[:]   = self._ball_pos
            self._stale_anchor_step = self._step_count
        elif goal == "team_b":
            ra, rb = -1.0, 1.0
            self._score[1] += 1
            self._reset_positions()
            self._stale_anchor[:]   = self._ball_pos
            self._stale_anchor_step = self._step_count
        else:
            self._check_stalemate()

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
        # A goal counts when the ball crosses the end line within the goal mouth.
        bx, by = self._ball_pos
        in_y   = GOAL_Y0 <= by <= GOAL_Y1
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

    # ---------------------------------------------------------------------------
    # Renderer — pygame-based, supports both "human" (live window) and
    # "rgb_array" (returns an H×W×3 numpy array for video recording).
    # The crowd backdrop and grass stripes are drawn once and cached.
    # ---------------------------------------------------------------------------
    def render(self):
        if self.render_mode is None:
            return
        try:
            import pygame
        except ImportError:
            print("pip install pygame")
            return

        SCALE = 16
        GD    = int(3 * SCALE)   # goal depth (pixels)
        MB    = int(4 * SCALE)   # margin/border around the pitch
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

            # Build the static crowd backdrop (coloured dots around the pitch).
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

        # Draw goal boxes on both sides.
        pygame.draw.rect(surf, GOAL_NET, (0, gy0, GD, gh))
        pygame.draw.rect(surf, WHITE,    (0, gy0, GD, gh), 2)
        pygame.draw.rect(surf, GOAL_NET, (GD + W, gy0, GD, gh))
        pygame.draw.rect(surf, WHITE,    (GD + W, gy0, GD, gh), 2)

        # Draw striped grass with rounded corners via an alpha mask.
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

        field_rect = pygame.Rect(GD, MB, W, H)
        pygame.draw.rect(surf, WHITE, field_rect, 2, border_radius=CR)

        pygame.draw.line(surf, WHITE, (GD,     gy0), (GD,     gy1), 3)
        pygame.draw.line(surf, WHITE, (GD + W, gy0), (GD + W, gy1), 3)

        # Centre line and kick-off circle.
        pygame.draw.line(surf, WHITE, (GD + W // 2, MB), (GD + W // 2, MB + H), 2)
        cx_px, cy_px = GD + W // 2, MB + H // 2
        pygame.draw.circle(surf, WHITE, (cx_px, cy_px), int(6 * SCALE), 2)
        pygame.draw.circle(surf, WHITE, (cx_px, cy_px), 3)

        # Penalty areas on each end.
        pa_w = int(8 * SCALE)
        pa_h = int(16 * SCALE)
        pa_y = MB + (H - pa_h) // 2
        pygame.draw.rect(surf, WHITE, (GD,            pa_y, pa_w, pa_h), 2)
        pygame.draw.rect(surf, WHITE, (GD + W - pa_w, pa_y, pa_w, pa_h), 2)

        # Draw players as coloured circles with a jersey number label.
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

        # Draw the ball.
        bx = GD + int(self._ball_pos[0] * SCALE)
        by = MB + int(self._ball_pos[1] * SCALE)
        br = int(BALL_RADIUS * SCALE)
        pygame.draw.circle(surf, (240, 240, 240), (bx, by), br)
        pygame.draw.circle(surf, (30, 30, 30),    (bx, by), br, 1)

        # Scoreboard banner centred at the top.
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