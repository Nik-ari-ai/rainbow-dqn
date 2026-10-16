"""Environment factory: our own game (``custom``) or Atari Space Invaders (``ale``).

Both return the same interface: obs of shape (4, 84, 84) uint8 for pixels
(zero-padded frame stack after reset) and ``info["score"]`` = raw game score.
"""
from __future__ import annotations

from collections import deque
import numpy as np


def make_env(env: str = "custom", obs_mode: str = "pixels", reward: str = "score", level: int = 1,
             max_episode_steps: int = 10_000, seed: int | None = None, render_mode=None):
    if env == "custom":
        from space_invaders import SpaceInvadersEnv
        return SpaceInvadersEnv(obs_mode=obs_mode, reward=reward, level=level,
                                max_episode_steps=max_episode_steps, seed=seed, render_mode=render_mode)
    if env == "ale":
        if obs_mode != "pixels":
            raise ValueError("ALE only supports pixel observations")
        return AtariSpaceInvaders(max_episode_steps=max_episode_steps, seed=seed, render_mode=render_mode)
    raise ValueError(env)


class AtariSpaceInvaders:
    """ALE Space Invaders with the DeepMind / Rainbow-paper preprocessing:
    no sticky actions, up to 30 no-op starts, frame skip 4 with max-pool,
    84x84 grayscale, 4-frame stack, episode limit 108k frames (27k steps)."""

    def __init__(self, max_episode_steps: int = 27_000, seed=None, render_mode=None):
        import gymnasium as gym
        import ale_py
        from gymnasium.wrappers import AtariPreprocessing
        gym.register_envs(ale_py)
        base = gym.make("ALE/SpaceInvaders-v5", frameskip=1, repeat_action_probability=0.0,
                        full_action_space=False, render_mode=render_mode,
                        max_num_frames_per_episode=108_000)
        self.env = AtariPreprocessing(base, noop_max=30, frame_skip=4, screen_size=84,
                                      terminal_on_life_loss=False, grayscale_obs=True)
        self.action_space = self.env.action_space
        self.max_episode_steps = max_episode_steps
        self.frames: deque = deque(maxlen=4)
        self._seed = seed
        self.level = None

    def set_level(self, level):  # no difficulty knobs on Atari
        pass

    def _obs(self):
        return np.stack(self.frames, 0)

    def reset(self, *, seed=None, options=None):
        frame, info = self.env.reset(seed=seed if seed is not None else self._seed)
        self._seed = None
        for _ in range(3):
            self.frames.append(np.zeros_like(frame))
        self.frames.append(frame)
        self.score, self.t = 0, 0
        return self._obs(), {"score": 0, "events": {}, "lives": info.get("lives")}

    def step(self, action):
        frame, r, term, trunc, info = self.env.step(int(action))
        self.frames.append(frame)
        self.score += r
        self.t += 1
        trunc = trunc or (self.t >= self.max_episode_steps)
        return self._obs(), float(r), term, trunc, {"score": self.score, "events": {}, "lives": info.get("lives")}

    def render(self, **_):
        return self.env.render()

    def close(self):
        self.env.close()
