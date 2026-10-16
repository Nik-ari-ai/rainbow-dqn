"""Gymnasium-compatible environment around :class:`SpaceInvadersGame`.

* Observations: ``"pixels"`` (stack of 84x84 grayscale frames, uint8) or
  ``"vector"`` (compact float32 state).
* Actions (same as Atari Space Invaders): 0 NOOP, 1 FIRE, 2 RIGHT, 3 LEFT,
  4 RIGHT+FIRE, 5 LEFT+FIRE.
* Rewards come from *triggers*: a weighted sum of game events, see
  ``REWARD_PRESETS``. ``info["events"]`` always contains the raw events.
"""
from __future__ import annotations

from collections import deque
import numpy as np

from .game import SpaceInvadersGame, make_config, EVENT_NAMES, W, H

try:  # Gymnasium is optional: the env also works stand-alone
    import gymnasium as gym
    from gymnasium import spaces
    _Base = gym.Env
except ImportError:  # pragma: no cover - minimal fallback
    gym = None

    class _Base:  # type: ignore
        metadata = {}

    class _Box:
        def __init__(self, low, high, shape, dtype):
            self.low, self.high, self.shape, self.dtype = low, high, shape, dtype

    class _Discrete:
        def __init__(self, n, seed=None):
            self.n = n
            self._rng = np.random.default_rng(seed)

        def sample(self):
            return int(self._rng.integers(self.n))

        def seed(self, s):
            self._rng = np.random.default_rng(s)

    class spaces:  # type: ignore
        Box = _Box
        Discrete = _Discrete


ACTIONS = {0: (0, False), 1: (0, True), 2: (1, False), 3: (-1, False), 4: (1, True), 5: (-1, True)}
ACTION_NAMES = ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"]

REWARD_PRESETS = {
    # classic: reward = game score (clip it to [-1, 1] in the agent like on Atari)
    "score": {"score": 1.0},
    # shaped: extra triggers that give denser feedback
    "shaped": {"enemy_killed": 1.0, "ufo_killed": 1.0, "wave_cleared": 1.0,
               "player_hit": -1.0, "game_over": -1.0,
               "shot_missed": -0.05, "bullet_dodged": 0.05},
}

GRAY = {"bg": 0, "inv": [255, 200, 200, 150, 150, 150, 150, 150], "shield": 90,
        "player": 230, "bullet": 255, "enemy_bullet": 180, "ufo": 120}
RGB = {"bg": (11, 15, 30), "inv": [(247, 37, 133), (76, 201, 240), (76, 201, 240), (57, 255, 136), (57, 255, 136), (57, 255, 136), (57, 255, 136), (57, 255, 136)],
       "shield": (60, 200, 90), "player": (255, 209, 102), "bullet": (255, 255, 255),
       "enemy_bullet": (255, 120, 80), "ufo": (230, 60, 60)}


class SpaceInvadersEnv(_Base):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 15}

    def __init__(self, obs_mode: str = "pixels", frame_skip: int = 4, frame_stack: int | None = None,
                 reward: str | dict = "score", level: int = 1, game_overrides: dict | None = None,
                 max_episode_steps: int = 10_000, noop_max: int = 30, render_mode: str | None = None,
                 seed: int | None = None):
        assert obs_mode in ("pixels", "vector")
        self.obs_mode = obs_mode
        self.frame_skip = frame_skip
        self.frame_stack = frame_stack if frame_stack is not None else (4 if obs_mode == "pixels" else 1)
        self.reward_weights = dict(REWARD_PRESETS[reward]) if isinstance(reward, str) else dict(reward)
        self.level = level
        self.game_overrides = game_overrides or {}
        self.max_episode_steps = max_episode_steps
        self.noop_max = noop_max
        self.render_mode = render_mode
        self.game = SpaceInvadersGame(make_config(level, **self.game_overrides), seed=seed)
        self._rng = np.random.default_rng(seed)
        self._frames: deque = deque(maxlen=self.frame_stack)
        self._canvas = np.zeros((H, W), dtype=np.uint8)
        self.action_space = spaces.Discrete(len(ACTIONS))
        if obs_mode == "pixels":
            shape = (self.frame_stack, 84, 84)
            self.observation_space = spaces.Box(0, 255, shape, np.uint8)
        else:
            d = self.game.state_vector().shape[0]
            shape = (d * self.frame_stack,) if self.frame_stack > 1 else (d,)
            self.observation_space = spaces.Box(-np.inf, np.inf, shape, np.float32)
        self._window = None

    # ----------------------------------------------------------- helpers
    def set_level(self, level: int):
        """Change difficulty (takes effect at the next reset). Used for the curriculum."""
        self.level = level

    def _frame(self):
        if self.obs_mode == "vector":
            return self.game.state_vector()
        self.game.draw(self._canvas, GRAY)
        # exact 2x2 max-pool 168x168 -> 84x84 (thin bullets are never lost)
        return self._canvas.reshape(84, 2, 84, 2).max(axis=(1, 3))

    def _obs(self):
        if self.obs_mode == "pixels":
            return np.stack(self._frames, 0)
        return np.concatenate(self._frames, 0) if self.frame_stack > 1 else self._frames[-1]

    def _reward(self, ev):
        return float(sum(w * ev.get(k, 0) for k, w in self.reward_weights.items()))

    # --------------------------------------------------------------- API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self.game.seed(seed)
            self.action_space.seed(seed)
        if options and "level" in options:
            self.level = options["level"]
        self.game.cfg = make_config(self.level, **self.game_overrides)
        self.game.reset()
        self.t = 0
        # random number of no-op frames so episodes do not all start identically
        for _ in range(int(self._rng.integers(0, self.noop_max + 1)) if self.noop_max else 0):
            self.game.step(0, False)
        first = self._frame()
        zero = np.zeros_like(first)
        for _ in range(self.frame_stack - 1):
            self._frames.append(zero)
        self._frames.append(first)
        return self._obs(), self._info(dict.fromkeys(EVENT_NAMES, 0))

    def step(self, action: int):
        move, fire = ACTIONS[int(action)]
        ev = dict.fromkeys(EVENT_NAMES, 0)
        for _ in range(self.frame_skip):
            e = self.game.step(move, fire)
            for k, v in e.items():
                ev[k] += v
            if self.game.over:
                break
        self.t += 1
        self._frames.append(self._frame())
        terminated = self.game.over
        truncated = (not terminated) and self.t >= self.max_episode_steps
        if self.render_mode == "human":
            self.render()
        return self._obs(), self._reward(ev), terminated, truncated, self._info(ev)

    def _info(self, ev):
        return {"score": self.game.score, "lives": self.game.lives, "wave": self.game.wave,
                "events": ev, "level": self.level}

    def render(self, scale: int = 4, hud: bool = True):
        """RGB frame (H*scale x W*scale x 3). With render_mode='human' shows a pygame window."""
        img = np.zeros((H, W, 3), dtype=np.uint8)
        self.game.draw(img, RGB)
        img = img.repeat(scale, 0).repeat(scale, 1)
        if hud:
            img = _draw_hud(img, f"SCORE {self.game.score}   LIVES {self.game.lives}   WAVE {self.game.wave + 1}")
        if self.render_mode == "human":
            import pygame
            if self._window is None:
                pygame.init()
                self._window = pygame.display.set_mode((img.shape[1], img.shape[0]))
                pygame.display.set_caption("Space Invaders RL")
            surf = pygame.surfarray.make_surface(img.swapaxes(0, 1))
            self._window.blit(surf, (0, 0))
            pygame.display.flip()
            pygame.event.pump()
        return img

    def close(self):
        if self._window is not None:
            import pygame
            pygame.quit()
            self._window = None


def _draw_hud(img, text, extra_lines=()):
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return img
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.text((8, 4), text, fill=(232, 236, 248))
    for i, line in enumerate(extra_lines):
        d.text((8, img.shape[0] - 16 * (len(extra_lines) - i) - 4), line, fill=(255, 209, 102))
    return np.asarray(im)


def make_env(**kwargs):
    """Factory used by the training scripts (``env=custom``)."""
    return SpaceInvadersEnv(**kwargs)


if gym is not None:
    try:
        gym.register(id="SpaceInvadersScratch-v0", entry_point="space_invaders.env:SpaceInvadersEnv")
    except Exception:  # already registered
        pass
