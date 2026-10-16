"""Compare trained agents with a random agent on the same seeds.

  python evaluate.py --models runs/rainbow_pixels/model.pth --episodes 30
  python evaluate.py --models runs/rainbow_pixels/model.pth runs/dqn/model.pth --names Rainbow DQN

Writes results/<name>_episodes.csv, results/summary.csv, results/summary.md and
results/comparison.png (box plot of scores).
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from rainbow.envs import make_env
from rainbow.evaluation import run_episodes, random_policy, summarize


def load_agent(path, device):
    from rainbow import RainbowAgent
    agent, d = RainbowAgent.from_file(path, device)
    return agent, d.get("env_kwargs", {})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="*", default=[], help="model.pth files")
    p.add_argument("--names", nargs="*", default=None)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=12345, help="first evaluation seed (same for all agents)")
    p.add_argument("--level", type=int, default=None, help="override difficulty (generalisation test)")
    p.add_argument("--env", default=None, help="custom | ale (default: from the model file)")
    p.add_argument("--obs", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="results")
    args = p.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    names = args.names or [os.path.basename(os.path.dirname(m)) or f"model{i}" for i, m in enumerate(args.models)]
    agents = [load_agent(m, args.device) for m in args.models]
    env_kw = dict(agents[0][1]) if agents else {}
    env_kw.setdefault("env", "custom")
    env_kw.setdefault("obs_mode", "pixels")
    if args.env:
        env_kw["env"] = args.env
    if args.obs:
        env_kw["obs_mode"] = args.obs
    env_kw.pop("reward", None)  # always report the game score
    level = args.level if args.level is not None else 1

    all_rows, summary = [], []
    entries = [("Random", None, dict(env_kw))]
    for name, (agent, kw) in zip(names, agents):
        kw = dict(kw)
        kw.pop("reward", None)
        kw.setdefault("env", env_kw["env"])
        kw.setdefault("obs_mode", env_kw["obs_mode"])
        entries.append((name, agent, kw))
    for name, agent, kw in entries:
        env = make_env(level=level, **kw)
        if agent is None:
            policy = random_policy(env.action_space.n, seed=0)
        else:
            policy = (lambda ag: (lambda o: ag.act(o, greedy=True)))(agent)
        rows = run_episodes(policy, env, args.episodes, seed=args.seed)
        for r in rows:
            r["agent"] = name
        pd.DataFrame(rows).to_csv(os.path.join(args.out, f"{name}_episodes.csv"), index=False)
        all_rows += rows
        s = summarize(rows)
        extra = {}
        if "ev_shot_fired" in rows[0]:
            fired = sum(r.get("ev_shot_fired", 0) for r in rows)
            kills = sum(r.get("ev_enemy_killed", 0) + r.get("ev_ufo_killed", 0) for r in rows)
            extra["accuracy"] = kills / max(1, fired)
        summary.append({"agent": name, "mean_score": s["mean"], "std_score": s["std"], "median": s["median"],
                        "min": s["min"], "max": s["max"],
                        "mean_length": float(np.mean([r["length"] for r in rows])),
                        "mean_waves": float(np.mean([r.get("waves_cleared", 0) for r in rows])), **extra})
        print(f"{name:>12}: score {s['mean']:8.1f} ± {s['std']:6.1f}  (median {s['median']:.0f}, max {s['max']:.0f})")

    df = pd.DataFrame(summary)
    rnd = df.loc[df.agent == "Random", "mean_score"].iloc[0]
    df["x_random"] = df["mean_score"] / max(rnd, 1e-9)
    df.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        try:
            f.write(df.round(2).to_markdown(index=False))
        except ImportError:  # tabulate not installed
            f.write(df.round(2).to_string(index=False))
    print("\n" + df.round(2).to_string(index=False))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        data = pd.DataFrame(all_rows)
        order = [e[0] for e in entries]
        fig, ax = plt.subplots(figsize=(1.8 + 1.6 * len(order), 4.5))
        ax.boxplot([data.loc[data.agent == n, "score"] for n in order], showmeans=True)
        ax.set_xticks(range(1, len(order) + 1), order)
        ax.set_ylabel("game score")
        ax.set_title(f"{args.episodes} evaluation episodes per agent (same seeds)")
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "comparison.png"), dpi=130)
        print(f"plot: {os.path.join(args.out, 'comparison.png')}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
