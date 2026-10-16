"""Running evaluation episodes for any policy (trained agent, random, ...)."""
from __future__ import annotations

import numpy as np


def run_episodes(policy, env, n_episodes: int = 10, seed: int = 10_000, max_steps: int | None = None,
                 on_step=None):
    """policy(obs) -> action. Returns a list of per-episode dicts.

    Seeds are ``seed, seed+1, ...`` so different policies are compared on the same games.
    """
    results = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        ret, steps, events = 0.0, 0, {}
        done = False
        while not done:
            a = policy(obs)
            obs, r, term, trunc, info = env.step(a)
            ret += r
            steps += 1
            for k, v in info.get("events", {}).items():
                events[k] = events.get(k, 0) + v
            if on_step is not None:
                on_step(env, a, r, info)
            done = term or trunc or (max_steps is not None and steps >= max_steps)
        row = {"episode": ep, "seed": seed + ep, "score": info["score"], "return": ret, "length": steps}
        row.update({f"ev_{k}": v for k, v in events.items()})
        if "wave" in info:
            row["waves_cleared"] = info["wave"]
        results.append(row)
    return results


def summarize(rows, key="score"):
    x = np.array([r[key] for r in rows], dtype=float)
    return {"mean": float(x.mean()), "std": float(x.std()), "min": float(x.min()),
            "max": float(x.max()), "median": float(np.median(x)), "n": len(x)}


def random_policy(n_actions: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    return lambda obs: int(rng.integers(n_actions))
