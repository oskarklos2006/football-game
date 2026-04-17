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
    """
    2-team soccer environment.

    Physics:
      - Momentum-based kick: ball receives base_impulse + player_momentum.
        Sprint direction matters; agent can aim by choosing approach angle.
      - FRICTION = 0.92: ball travels realistically far across the field.
      - KICK_POWER = 1.2: balanced against higher friction.
      - Single collision pass: kick and separation are merged into one loop
        per player so the separation push cannot cancel a just-applied kick.
      - GOAL_CENTER for reward calculations uses FIELD_W - BALL_RADIUS,
        matching the actual goal trigger boundary exactly.

    Reward API:
      step() returns raw ±1 for goal/concede in the rewards dict.
      All shaping is done in the wrapper (train.py).
    """
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
        # Define the 'safe' area for spawning (avoiding the very edge of the walls)
        margin = PLAYER_RADIUS + 2.0

        for i in range(n):
            # Team A: Randomly spawn on the LEFT half (x: margin to FIELD_W/2)
            self._pos[i] = [
                self.np_random.uniform(margin, FIELD_W / 2 - 2.0),
                self.np_random.uniform(margin, FIELD_H - margin)
            ]

            # Team B: Randomly spawn on the RIGHT half (x: FIELD_W/2 to FIELD_W - margin)
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
            self._pos[i] = np.clip(
                self._pos[i] + self._vel[i],
                [PLAYER_RADIUS, PLAYER_RADIUS],
                [FIELD_W - PLAYER_RADIUS, FIELD_H - PLAYER_RADIUS]
            )

        # --- single collision pass: kick + separation merged ---
        #
        # FIX vs original: previously there were two separate loops —
        # one for kicking, one for separation. The second loop could push
        # the ball back into a wall after the kick, cancelling the impulse.
        # Now each player is handled atomically: if overlapping, apply
        # momentum kick AND resolve separation in the same pass.
        # The second post-ball-physics separation loop is removed entirely.
        #
        # Kick model: base_push (away from player) + player momentum transfer.
        # Only applied when the new velocity adds speed in the away direction,
        # preventing a stationary player from killing a moving ball.
        for i in range(N):
            diff = self._ball_pos - self._pos[i]
            d    = np.linalg.norm(diff)
            min_d = PLAYER_RADIUS + BALL_RADIUS
            if 1e-6 < d < min_d:
                normal = diff / d

                # Separate first so ball is exactly at contact boundary
                self._ball_pos = self._pos[i] + normal * min_d

                # Apply momentum-based kick
                base_push = normal * KICK_POWER * 0.6
                momentum  = self._vel[i] * 0.8
                new_vel   = base_push + momentum

                if np.dot(new_vel - self._ball_vel, normal) > 0:
                    self._ball_vel = new_vel

        # --- ball physics ---
        self._ball_pos += self._ball_vel
        self._ball_vel *= FRICTION

        # Wall bounces — applied after movement so ball never clips through
        for axis, limit in [(0, FIELD_W), (1, FIELD_H)]:
            if self._ball_pos[axis] <= BALL_RADIUS:
                self._ball_pos[axis] = BALL_RADIUS
                self._ball_vel[axis] = abs(self._ball_vel[axis])
            elif self._ball_pos[axis] >= limit - BALL_RADIUS:
                self._ball_pos[axis] = limit - BALL_RADIUS
                self._ball_vel[axis] = -abs(self._ball_vel[axis])

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

        stripe_w = W // 10
        for s in range(10):
            col = (40, 130, 40) if s % 2 == 0 else (50, 160, 50)
            pygame.draw.rect(surf, col, (GD + s * stripe_w, MB, stripe_w, H))

        pygame.draw.rect(surf, WHITE, (GD, MB, W, H), 2)
        pygame.draw.line(surf, WHITE, (GD, gy0), (GD, gy1), 3)
        pygame.draw.line(surf, WHITE, (GD + W, gy0), (GD + W, gy1), 3)
        pygame.draw.line(surf, WHITE, (GD + W // 2, MB), (GD + W // 2, MB + H), 2)

        cx_px, cy_px = GD + W // 2, MB + H // 2
        pygame.draw.circle(surf, WHITE, (cx_px, cy_px), int(6 * SCALE), 2)
        pygame.draw.circle(surf, WHITE, (cx_px, cy_px), 3)

        pa_w = int(8 * SCALE)
        pa_h = int(16 * SCALE)
        pa_y = MB + (H - pa_h) // 2
        pygame.draw.rect(surf, WHITE, (GD, pa_y, pa_w, pa_h), 2)
        pygame.draw.rect(surf, WHITE, (GD + W - pa_w, pa_y, pa_w, pa_h), 2)

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

        bx = GD + int(self._ball_pos[0] * SCALE)
        by = MB + int(self._ball_pos[1] * SCALE)
        br = int(BALL_RADIUS * SCALE)
        pygame.draw.circle(surf, (240, 240, 240), (bx, by), br)
        pygame.draw.circle(surf, (30, 30, 30), (bx, by), br, 1)

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