"""Space Invaders game logic written from scratch (pure Python + NumPy).

The game runs on a 168x168 pixel world so that pixel observations can be
downsampled to 84x84 by an exact 2x2 max-pool (no object is ever lost).

Every frame the game records *events* (enemy killed, player hit, shot missed,
...). The Gymnasium environment turns these events into rewards ("triggers").
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import numpy as np

W = H = 168  # world size in pixels
MAX_ROWS, MAX_COLS, MAX_ENEMY_BULLETS = 6, 8, 4  # fixed size of the state vector for all levels

# 8x6 invader sprites, two animation frames
_SPRITES = [np.array([[c == "#" for c in row] for row in rows]) for rows in (
    ["..#..#..", ".######.", "##.##.##", "########", ".#....#.", "#......#"],
    ["..#..#..", "#.####.#", "##.##.##", "########", "..#..#..", ".#....#."],
)]

EVENT_NAMES = (
    "score",          # points gained this frame
    "enemy_killed",   # number of invaders destroyed
    "ufo_killed",     # UFO destroyed
    "player_hit",     # player lost a life
    "shot_fired",     # player fired a bullet
    "shot_missed",    # player bullet left the screen or hit a shield
    "bullet_dodged",  # enemy bullet passed the player's row close to the player
    "wave_cleared",   # all invaders destroyed
    "game_over",      # last life lost or invaders landed
)


@dataclass
class GameConfig:
    # invader formation
    inv_rows: int = 5
    inv_cols: int = 6
    inv_w: int = 8
    inv_h: int = 6
    inv_gap_x: int = 8
    inv_gap_y: int = 6
    inv_start_y: int = 24
    inv_step_x: int = 2          # pixels per formation move
    inv_step_down: int = 6       # pixels when the formation hits a wall
    inv_move_interval: int = 10  # frames between moves with a full formation
    inv_min_interval: int = 1    # fastest speed (few invaders left)
    inv_fire_prob: float = 0.03  # probability per frame that some invader fires
    max_enemy_bullets: int = 3
    enemy_bullet_speed: int = 2
    # player
    player_y: int = 150
    player_w: int = 9
    player_h: int = 5
    player_speed: int = 2
    bullet_speed: int = 4
    lives: int = 3
    respawn_frames: int = 30     # invulnerable frames after being hit
    # shields
    n_shields: int = 3
    shield_y: int = 128
    shield_rows: int = 3
    shield_cols: int = 6
    shield_cell: int = 3
    # UFO
    ufo_prob: float = 0.002
    ufo_y: int = 12
    ufo_w: int = 10
    ufo_h: int = 4
    ufo_speed: int = 1
    ufo_value: int = 100
    # waves
    wave_speedup: int = 1        # each new wave moves faster by this many frames
    wave_drop: int = 4           # and starts lower by this many pixels
    row_values: tuple = (30, 20, 20, 10, 10, 10, 10, 10)

    def to_dict(self):
        return asdict(self)


DIFFICULTY = {
    # level -> overrides of GameConfig
    0: dict(inv_rows=3, inv_cols=5, inv_move_interval=14, inv_fire_prob=0.015, max_enemy_bullets=2),
    1: dict(),  # default "normal" game
    2: dict(inv_rows=5, inv_cols=8, inv_gap_x=6, inv_move_interval=8, inv_fire_prob=0.05,
            max_enemy_bullets=4, enemy_bullet_speed=3),
}


def make_config(level: int | None = 1, **overrides) -> GameConfig:
    params = dict(DIFFICULTY.get(level, {})) if level is not None else {}
    params.update(overrides)
    return GameConfig(**params)


class SpaceInvadersGame:
    """One game of Space Invaders. Call ``reset`` then ``step(move, fire)`` once per frame."""

    def __init__(self, config: GameConfig | None = None, seed: int | None = None):
        self.cfg = config or GameConfig()
        self.rng = np.random.default_rng(seed)
        self.reset()

    # ------------------------------------------------------------------ setup
    def seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        c = self.cfg
        self.score = 0
        self.lives = c.lives
        self.wave = 0
        self.frame = 0
        self.over = False
        self.player_x = (W - c.player_w) // 2
        self.invuln = 0
        self.bullet = None            # [x, y] of the player bullet or None
        self.enemy_bullets = []       # list of [x, y]
        self.ufo = None               # [x, direction] or None
        self._init_shields()
        self._init_wave()
        return self

    def _init_shields(self):
        c = self.cfg
        self.shields = np.ones((c.n_shields, c.shield_rows, c.shield_cols), dtype=bool)
        sw = c.shield_cols * c.shield_cell
        spacing = W / c.n_shields
        self.shield_x = [int(spacing * (i + 0.5) - sw / 2) for i in range(c.n_shields)]

    def _init_wave(self):
        c = self.cfg
        self.alive = np.ones((c.inv_rows, c.inv_cols), dtype=bool)
        fw = c.inv_cols * c.inv_w + (c.inv_cols - 1) * c.inv_gap_x
        self.form_x = (W - fw) // 2
        self.form_y = c.inv_start_y + min(self.wave * c.wave_drop, 24)
        self.form_dir = 1
        self.move_timer = 0
        self.anim = 0  # animation frame for rendering

    # ------------------------------------------------------------- geometry
    @property
    def cell_w(self):
        return self.cfg.inv_w + self.cfg.inv_gap_x

    @property
    def cell_h(self):
        return self.cfg.inv_h + self.cfg.inv_gap_y

    def invader_rect(self, r, k):
        return (self.form_x + k * self.cell_w, self.form_y + r * self.cell_h, self.cfg.inv_w, self.cfg.inv_h)

    def _alive_bounds(self):
        rows = np.where(self.alive.any(axis=1))[0]
        cols = np.where(self.alive.any(axis=0))[0]
        x0 = self.form_x + cols[0] * self.cell_w
        x1 = self.form_x + cols[-1] * self.cell_w + self.cfg.inv_w
        y1 = self.form_y + rows[-1] * self.cell_h + self.cfg.inv_h
        return x0, x1, y1

    def _move_interval(self):
        c = self.cfg
        frac = self.alive.sum() / self.alive.size
        base = max(c.inv_min_interval, c.inv_move_interval - self.wave * c.wave_speedup)
        return max(c.inv_min_interval, int(round(base * frac)))

    # ------------------------------------------------------------ collisions
    def _hit_invader(self, x, y):
        c = self.cfg
        dx, dy = x - self.form_x, y - self.form_y
        if dx < 0 or dy < 0:
            return None
        k, r = dx // self.cell_w, dy // self.cell_h
        if k >= c.inv_cols or r >= c.inv_rows:
            return None
        if dx % self.cell_w >= c.inv_w or dy % self.cell_h >= c.inv_h:
            return None
        return (r, k) if self.alive[r, k] else None

    def _hit_shield(self, x, y):
        c = self.cfg
        ry = y - c.shield_y
        if ry < 0 or ry >= c.shield_rows * c.shield_cell:
            return False
        for i, sx in enumerate(self.shield_x):
            rx = x - sx
            if 0 <= rx < c.shield_cols * c.shield_cell:
                r, k = ry // c.shield_cell, rx // c.shield_cell
                if self.shields[i, r, k]:
                    self.shields[i, r, k] = False
                    return True
        return False

    # ------------------------------------------------------------------ step
    def step(self, move: int, fire: bool) -> dict:
        """Advance one frame. move in {-1, 0, 1}. Returns the events of this frame."""
        c = self.cfg
        ev = dict.fromkeys(EVENT_NAMES, 0)
        if self.over:
            return ev
        self.frame += 1

        # --- player
        self.player_x = int(np.clip(self.player_x + move * c.player_speed, 0, W - c.player_w))
        if self.invuln > 0:
            self.invuln -= 1
        if fire and self.bullet is None and self.invuln == 0:
            self.bullet = [self.player_x + c.player_w // 2, c.player_y - 4]
            ev["shot_fired"] += 1

        # --- player bullet (move pixel by pixel so nothing is skipped)
        if self.bullet is not None:
            for _ in range(c.bullet_speed):
                self.bullet[1] -= 1
                bx, by = self.bullet
                if by < 0:
                    self.bullet = None
                    ev["shot_missed"] += 1
                    break
                hit = self._hit_invader(bx, by)
                if hit is not None:
                    r, k = hit
                    self.alive[r, k] = False
                    pts = c.row_values[min(r, len(c.row_values) - 1)]
                    ev["score"] += pts
                    ev["enemy_killed"] += 1
                    self.bullet = None
                    break
                if self.ufo is not None and self.ufo[0] <= bx < self.ufo[0] + c.ufo_w and c.ufo_y <= by < c.ufo_y + c.ufo_h:
                    ev["score"] += c.ufo_value
                    ev["ufo_killed"] += 1
                    self.ufo = None
                    self.bullet = None
                    break
                if self._hit_shield(bx, by):
                    ev["shot_missed"] += 1
                    self.bullet = None
                    break

        # --- wave cleared?
        if not self.alive.any():
            ev["wave_cleared"] += 1
            self.wave += 1
            self.enemy_bullets = []
            self.bullet = None
            self._init_wave()

        # --- invader formation movement
        self.move_timer += 1
        if self.move_timer >= self._move_interval():
            self.move_timer = 0
            self.anim ^= 1
            x0, x1, _ = self._alive_bounds()
            if (self.form_dir > 0 and x1 + c.inv_step_x > W - 2) or (self.form_dir < 0 and x0 - c.inv_step_x < 2):
                self.form_dir *= -1
                self.form_y += c.inv_step_down
            else:
                self.form_x += self.form_dir * c.inv_step_x
            # invaders eat the shields they overlap
            self._erode_shields()

        # --- enemy fire: a random column fires from its lowest alive invader
        if len(self.enemy_bullets) < c.max_enemy_bullets and self.rng.random() < c.inv_fire_prob:
            cols = np.where(self.alive.any(axis=0))[0]
            k = int(self.rng.choice(cols))
            r = int(np.where(self.alive[:, k])[0][-1])
            x, y, w, h = self.invader_rect(r, k)
            self.enemy_bullets.append([x + w // 2, y + h])

        # --- enemy bullets
        px0, px1 = self.player_x, self.player_x + c.player_w
        keep = []
        for b in self.enemy_bullets:
            alive = True
            for _ in range(c.enemy_bullet_speed):
                b[1] += 1
                if b[1] >= H:
                    alive = False
                    break
                if self._hit_shield(b[0], b[1]):
                    alive = False
                    break
                if c.player_y <= b[1] < c.player_y + c.player_h and px0 <= b[0] < px1:
                    alive = False
                    if self.invuln == 0:
                        ev["player_hit"] += 1
                    break
                if b[1] == c.player_y + c.player_h and (px0 - 8 <= b[0] < px1 + 8):
                    ev["bullet_dodged"] += 1  # passed the player's row close by without hitting
            if alive:
                keep.append(b)
        self.enemy_bullets = keep

        if ev["player_hit"]:
            self.lives -= 1
            self.invuln = c.respawn_frames
            self.enemy_bullets = []
            if self.lives <= 0:
                self.over = True

        # --- UFO
        if self.ufo is None:
            if self.rng.random() < c.ufo_prob:
                d = 1 if self.rng.random() < 0.5 else -1
                self.ufo = [-c.ufo_w if d > 0 else W, d]
        else:
            self.ufo[0] += self.ufo[1] * c.ufo_speed
            if self.ufo[0] < -c.ufo_w or self.ufo[0] > W:
                self.ufo = None

        # --- invaders landed
        if self.alive.any():
            _, _, y1 = self._alive_bounds()
            if y1 >= c.player_y:
                self.over = True

        if self.over:
            ev["game_over"] = 1
        self.score += ev["score"]
        return ev

    def _erode_shields(self):
        c = self.cfg
        top, bottom = c.shield_y, c.shield_y + c.shield_rows * c.shield_cell
        rows = np.where(self.alive.any(axis=1))[0]
        for r in rows:
            y0 = self.form_y + r * self.cell_h
            if y0 + c.inv_h <= top or y0 >= bottom:
                continue
            for k in np.where(self.alive[r])[0]:
                x0 = self.form_x + k * self.cell_w
                for i, sx in enumerate(self.shield_x):
                    for sr in range(c.shield_rows):
                        cy = top + sr * c.shield_cell
                        if cy + c.shield_cell <= y0 or cy >= y0 + c.inv_h:
                            continue
                        for sk in range(c.shield_cols):
                            cx = sx + sk * c.shield_cell
                            if cx + c.shield_cell > x0 and cx < x0 + c.inv_w:
                                self.shields[i, sr, sk] = False

    # -------------------------------------------------------- observations
    def draw(self, canvas: np.ndarray, palette: dict):
        """Draw the scene into ``canvas`` (HxW or HxWx3). ``palette`` maps object -> value/colour."""
        c = self.cfg
        canvas[...] = palette["bg"]
        for r in range(c.inv_rows):
            col = palette["inv"][min(r, len(palette["inv"]) - 1)]
            for k in range(c.inv_cols):
                if self.alive[r, k]:
                    x, y, w, h = self.invader_rect(r, k)
                    _sprite(canvas, x, y, _SPRITES[self.anim], col)
        for i, sx in enumerate(self.shield_x):
            for sr, sk in zip(*np.where(self.shields[i])):
                _rect(canvas, sx + sk * c.shield_cell, c.shield_y + sr * c.shield_cell, c.shield_cell, c.shield_cell, palette["shield"])
        if not (self.invuln > 0 and (self.frame // 4) % 2):
            _rect(canvas, self.player_x, c.player_y, c.player_w, c.player_h, palette["player"])
            _rect(canvas, self.player_x + c.player_w // 2 - 1, c.player_y - 2, 3, 2, palette["player"])
        if self.bullet is not None:
            _rect(canvas, self.bullet[0], self.bullet[1], 1, 4, palette["bullet"])
        for b in self.enemy_bullets:
            _rect(canvas, b[0] - 1, b[1] - 3, 2, 4, palette["enemy_bullet"])
        if self.ufo is not None:
            _rect(canvas, self.ufo[0], c.ufo_y, c.ufo_w, c.ufo_h, palette["ufo"])
        return canvas

    def state_vector(self) -> np.ndarray:
        """Compact, fully observable state (used for the 'vector' observation mode)."""
        c = self.cfg
        v = [self.player_x / W,
             1.0 if self.bullet is not None else 0.0,
             (self.bullet[0] / W) if self.bullet is not None else 0.0,
             (self.bullet[1] / H) if self.bullet is not None else 0.0,
             self.form_x / W, self.form_y / H, float(self.form_dir),
             self.move_timer / max(1, self._move_interval()),
             self.lives / c.lives, self.invuln / c.respawn_frames]
        bullets = sorted(self.enemy_bullets, key=lambda b: -b[1])[:MAX_ENEMY_BULLETS]
        for i in range(MAX_ENEMY_BULLETS):
            if i < len(bullets):
                bx, by = bullets[i]
                v += [1.0, (bx - self.player_x) / W, by / H]
            else:
                v += [0.0, 0.0, 0.0]
        v += [1.0, self.ufo[0] / W] if self.ufo is not None else [0.0, 0.0]
        v += list(self.shields.reshape(c.n_shields, -1).mean(axis=1))
        alive = np.zeros((MAX_ROWS, MAX_COLS), dtype=np.float32)
        alive[: c.inv_rows, : c.inv_cols] = self.alive
        v += list(alive.ravel())
        return np.asarray(v, dtype=np.float32)


def _sprite(canvas, x, y, mask, value):
    h, w = mask.shape
    x0, y0 = int(x), int(y)
    if x0 < 0 or y0 < 0 or x0 + w > canvas.shape[1] or y0 + h > canvas.shape[0]:
        return
    canvas[y0:y0 + h, x0:x0 + w][mask] = value


def _rect(canvas, x, y, w, h, value):
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(canvas.shape[1], int(x + w)), min(canvas.shape[0], int(y + h))
    if x1 > x0 and y1 > y0:
        canvas[y0:y1, x0:x1] = value
