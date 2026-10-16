"""Prioritized n-step replay memory (NumPy only).

Each slot stores ONE frame (the newest frame of the observation the action was
taken in) plus action, reward, done flag and the time step inside the episode.
Stacked observations (history of H frames) are rebuilt on sampling, with zero
padding before the start of an episode. This keeps memory ~4x smaller than
storing full stacks.

A transition at index i becomes sampleable only once the n following frames are
written (or the episode ended earlier), so the n-step target is always valid.
"""
from __future__ import annotations

from collections import deque
import numpy as np


class SumTree:
    """Binary sum tree over ``capacity`` leaves, vectorised with NumPy."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.size = 1
        while self.size < capacity:
            self.size *= 2
        self.tree = np.zeros(2 * self.size, dtype=np.float64)
        self.depth = int(np.log2(self.size))

    @property
    def total(self) -> float:
        return float(self.tree[1])

    def get(self, idx):
        return self.tree[np.asarray(idx) + self.size]

    def update(self, idx, values):
        idx = np.asarray(idx, dtype=np.int64) + self.size
        self.tree[idx] = values
        for _ in range(self.depth):
            idx = np.unique(idx // 2)
            self.tree[idx] = self.tree[2 * idx] + self.tree[2 * idx + 1]

    def find(self, values):
        """Leaf index for each cumulative value (prefix-sum search)."""
        values = np.asarray(values, dtype=np.float64).copy()
        idx = np.ones(len(values), dtype=np.int64)
        for _ in range(self.depth):
            left = 2 * idx
            go_right = values > self.tree[left]
            values -= self.tree[left] * go_right
            idx = left + go_right
        return np.minimum(idx - self.size, self.capacity - 1)


class PrioritizedReplay:
    def __init__(self, capacity: int, frame_shape, frame_dtype=np.uint8, history: int = 4,
                 n_step: int = 3, gamma: float = 0.99, alpha: float = 0.5, eps: float = 1e-6,
                 seed: int | None = None):
        self.capacity = capacity
        self.history = history
        self.n = n_step
        self.gamma = gamma
        self.alpha = alpha  # alpha = 0 -> uniform replay
        self.eps = eps
        self.frames = np.zeros((capacity, *frame_shape), dtype=frame_dtype)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)
        self.ts = np.zeros(capacity, dtype=np.int64)
        self.tree = SumTree(capacity)
        self.max_priority = 1.0  # already raised to alpha
        self.ptr = 0
        self.count = 0
        self.pending: deque = deque()
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.count

    # ------------------------------------------------------------- writing
    def append(self, frame, action: int, reward: float, done: bool, t: int):
        """Store the transition taken at episode step ``t`` from an observation whose newest frame is ``frame``."""
        j = self.ptr
        if self.count == self.capacity:
            # slots j+1..j+H-1 used frame j in their stacks -> no longer valid
            stale = (j + np.arange(1, self.history)) % self.capacity
            self.tree.update(np.append(stale, j), 0.0)
        self.frames[j] = frame
        self.actions[j] = action
        self.rewards[j] = reward
        self.dones[j] = done
        self.ts[j] = t
        self.pending.append(j)
        # the transition written n steps ago now has its n-step successor
        while len(self.pending) > self.n:
            self.tree.update([self.pending.popleft()], self.max_priority)
        if done:  # episode ended: all pending transitions are complete
            idx = list(self.pending)
            self.pending.clear()
            self.tree.update(idx, self.max_priority)
        self.ptr = (j + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def end_episode(self):
        """Mark the last stored transition as terminal (e.g. after resuming training)."""
        if self.count == 0:
            return
        last = (self.ptr - 1) % self.capacity
        self.dones[last] = True
        idx = list(self.pending)
        self.pending.clear()
        if idx:
            self.tree.update(idx, self.max_priority)

    # ------------------------------------------------------------- reading
    def _stack(self, idx):
        """Rebuild (B, H, *frame) observations ending at ``idx``."""
        k = np.arange(self.history - 1, -1, -1)              # H-1 ... 0
        pos = (idx[:, None] - k[None, :]) % self.capacity     # (B, H)
        obs = self.frames[pos]
        valid = (self.ts[idx][:, None] - k[None, :]) >= 0     # before episode start -> zero
        obs = obs * valid.reshape(valid.shape + (1,) * (obs.ndim - 2)).astype(obs.dtype)
        return obs

    def sample(self, batch_size: int, beta: float = 0.4):
        total = self.tree.total
        seg = total / batch_size
        values = (np.arange(batch_size) + self.rng.random(batch_size)) * seg
        idx = self.tree.find(values)
        prio = self.tree.get(idx)
        bad = prio <= 0  # numerical edge cases: resample uniformly among valid leaves
        if bad.any():
            valid = np.nonzero(self.tree.get(np.arange(self.capacity)) > 0)[0]
            idx[bad] = self.rng.choice(valid, bad.sum())
            prio = self.tree.get(idx)

        # n-step discounted return, stopping at the end of the episode
        returns = np.zeros(batch_size, dtype=np.float32)
        alive = np.ones(batch_size, dtype=bool)
        for k in range(self.n):
            p = (idx + k) % self.capacity
            returns += alive * (self.gamma ** k) * self.rewards[p]
            alive &= ~self.dones[p]
        next_idx = (idx + self.n) % self.capacity

        probs = prio / total
        weights = (self.count * probs) ** (-beta)
        weights = (weights / weights.max()).astype(np.float32)
        obs = self._stack(idx)
        next_obs = self._stack(next_idx)
        if self.history == 1:  # vector observations: drop the history axis
            obs, next_obs = obs[:, 0], next_obs[:, 0]
        return {"obs": obs, "actions": self.actions[idx], "returns": returns,
                "next_obs": next_obs, "nonterminal": alive.astype(np.float32),
                "weights": weights, "idx": idx}

    def update_priorities(self, idx, priorities):
        p = (np.asarray(priorities, dtype=np.float64) + self.eps) ** self.alpha
        self.tree.update(idx, p)
        self.max_priority = max(self.max_priority, float(p.max()))

    # ---------------------------------------------------------- save/load
    def state_dict(self):
        return {k: getattr(self, k) for k in ("frames", "actions", "rewards", "dones", "ts", "max_priority", "ptr", "count")} | {
            "tree": self.tree.tree, "pending": list(self.pending)}

    def load_state_dict(self, d):
        for k in ("frames", "actions", "rewards", "dones", "ts"):
            getattr(self, k)[...] = d[k]
        self.max_priority, self.ptr, self.count = float(d["max_priority"]), int(d["ptr"]), int(d["count"])
        self.tree.tree[...] = d["tree"]
        self.pending = deque(int(i) for i in d["pending"])

    def save(self, path):
        np.savez(path, **{k: np.asarray(v) for k, v in self.state_dict().items()})

    def load(self, path):
        with np.load(path) as d:
            self.load_state_dict({k: d[k] for k in d.files})
