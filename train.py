"""Train Rainbow DQN (or an ablation / vanilla DQN) on Space Invaders.

Examples
--------
  # our game, pixels, paper-like Rainbow, 1M agent steps
  python train.py --id rainbow_pixels --steps 1000000

  # quick sanity run on CPU with the compact state vector
  python train.py --id quick --obs vector --steps 50000

  # shaped reward triggers and a difficulty curriculum
  python train.py --id shaped_curr --reward shaped --curriculum "0:0,200000:1,600000:2"

  # baselines / ablations
  python train.py --id dqn --dqn
  python train.py --id no_noisy --no-noisy

  # Atari Space Invaders (needs: pip install ale-py), paper hyper-parameters
  python train.py --id ale_rainbow --env ale --preset atari --steps 2000000

  # resume after a crash / Kaggle session end
  python train.py --id rainbow_pixels --steps 1000000 --resume
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import fields

import numpy as np
import torch

from rainbow import AgentConfig, PrioritizedReplay, RainbowAgent, make_env
from rainbow.evaluation import run_episodes, summarize

PRESETS = {
    # hyper-parameters from the Rainbow paper (Hessel et al., 2018, Table 1)
    "atari": dict(lr=6.25e-5, target_update=8_000, learn_start=20_000, memory=1_000_000, arch="canonical", hidden=512),
    # our game with pixels: smaller buffer and faster target updates for a short budget
    "custom": dict(lr=1e-4, target_update=2_000, learn_start=10_000, memory=200_000, arch="canonical", hidden=512),
    # our game with the state vector: MLP torso
    "vector": dict(lr=1e-4, target_update=2_000, learn_start=5_000, memory=200_000, arch="mlp", hidden=256),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--id", default="rainbow", help="run name; output goes to runs/<id>")
    p.add_argument("--env", choices=["custom", "ale"], default="custom")
    p.add_argument("--obs", choices=["pixels", "vector"], default="pixels")
    p.add_argument("--reward", default="score", help="reward triggers preset: score | shaped (custom env only)")
    p.add_argument("--level", type=int, default=1, help="difficulty 0 easy / 1 normal / 2 hard")
    p.add_argument("--curriculum", default="", help='e.g. "0:0,200000:1,600000:2" (step:level,...)')
    p.add_argument("--preset", choices=list(PRESETS), default=None, help="default: custom/vector/atari by env+obs")
    p.add_argument("--steps", type=int, default=1_000_000, help="agent steps (1 step = 4 game frames)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    # hyper-parameters (None -> take from preset)
    p.add_argument("--lr", type=float)
    p.add_argument("--memory", type=int, help="replay capacity")
    p.add_argument("--learn-start", type=int)
    p.add_argument("--target-update", type=int)
    p.add_argument("--arch", choices=["canonical", "data-efficient", "mlp"])
    p.add_argument("--hidden", type=int)
    p.add_argument("--replay-frequency", type=int, default=4, help="learn every k steps")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-step", type=int, default=3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--atoms", type=int, default=51)
    p.add_argument("--v-min", type=float, default=-10.0)
    p.add_argument("--v-max", type=float, default=10.0)
    p.add_argument("--noisy-std", type=float, default=0.5)
    p.add_argument("--priority-exponent", type=float, default=0.5, help="alpha")
    p.add_argument("--priority-weight", type=float, default=0.4, help="initial beta, annealed to 1")
    p.add_argument("--reward-clip", type=float, default=1.0, help="0 disables clipping")
    p.add_argument("--max-episode-steps", type=int, default=10_000)
    # ablations
    p.add_argument("--no-double", action="store_true")
    p.add_argument("--no-dueling", action="store_true")
    p.add_argument("--no-distributional", action="store_true")
    p.add_argument("--no-noisy", action="store_true")
    p.add_argument("--no-per", action="store_true")
    p.add_argument("--dqn", action="store_true", help="vanilla DQN: all six components off, n=1")
    # evaluation / saving
    p.add_argument("--eval-interval", type=int, default=50_000)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--checkpoint-interval", type=int, default=100_000)
    p.add_argument("--save-memory", action="store_true", help="also save the replay buffer with checkpoints (large)")
    p.add_argument("--resume", action="store_true", help="continue from runs/<id>/checkpoint.pt")
    p.add_argument("--time-limit", type=float, default=0, help="hours; stop cleanly (with checkpoint) before this")
    p.add_argument("--log-interval", type=int, default=10_000)
    p.add_argument("--out", default="runs")
    args = p.parse_args(argv)

    preset = args.preset or ("atari" if args.env == "ale" else ("vector" if args.obs == "vector" else "custom"))
    for k, v in PRESETS[preset].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    args.preset = preset
    if args.dqn:
        args.no_double = args.no_dueling = args.no_distributional = args.no_noisy = args.no_per = True
        args.n_step = 1
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    return args


def parse_curriculum(s):
    if not s:
        return []
    pts = sorted((int(a), int(b)) for a, b in (x.split(":") for x in s.split(",")))
    return pts


def level_at(curriculum, step, default):
    lvl = default
    for s, l in curriculum:
        if step >= s:
            lvl = l
    return lvl


def agent_config(args) -> AgentConfig:
    return AgentConfig(arch=args.arch, hidden=args.hidden, atoms=args.atoms, v_min=args.v_min, v_max=args.v_max,
                       noisy_std=args.noisy_std, gamma=args.gamma, n_step=args.n_step, lr=args.lr,
                       batch_size=args.batch_size, double=not args.no_double, dueling=not args.no_dueling,
                       distributional=not args.no_distributional, noisy=not args.no_noisy, per=not args.no_per)


class CSVLog:
    def __init__(self, path, resume):
        self.path = path
        self.header = None
        if resume and os.path.exists(path):
            with open(path) as f:
                self.header = next(csv.reader(f), None)
        elif os.path.exists(path):
            os.remove(path)  # fresh run: start a new log

    def write(self, row: dict):
        new = self.header is None
        if new:
            self.header = list(row)
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.header, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)


def plot_curves(run_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        return
    ep_path, ev_path = os.path.join(run_dir, "episodes.csv"), os.path.join(run_dir, "eval.csv")
    if not os.path.exists(ev_path):
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if os.path.exists(ep_path):
        ep = pd.read_csv(ep_path)
        ax.plot(ep["step"], ep["score"].rolling(20, min_periods=1).mean(), color="#9AA3C0", lw=1, label="training episodes (avg 20)")
    ev = pd.read_csv(ev_path)
    ax.plot(ev["step"], ev["mean_score"], color="#F72585", lw=2, marker="o", ms=3, label="evaluation (mean)")
    ax.fill_between(ev["step"], ev["mean_score"] - ev["std_score"], ev["mean_score"] + ev["std_score"], color="#F72585", alpha=0.15)
    if "random_score" in ev:
        ax.axhline(ev["random_score"].iloc[0], color="#4CC9F0", ls="--", label="random agent")
    ax.set_xlabel("agent steps")
    ax.set_ylabel("game score")
    ax.set_title(os.path.basename(run_dir))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "learning_curve.png"), dpi=120)
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    run_dir = os.path.join(args.out, args.id)
    os.makedirs(run_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    curriculum = parse_curriculum(args.curriculum)

    env_kw = dict(env=args.env, obs_mode=args.obs, reward=args.reward, max_episode_steps=args.max_episode_steps)
    env = make_env(level=level_at(curriculum, 0, args.level), seed=args.seed, **env_kw)
    eval_env = make_env(level=args.level, seed=args.seed + 1, **env_kw)
    pixel = args.obs == "pixels"
    obs, info = env.reset(seed=args.seed)
    obs_shape = obs.shape
    n_actions = env.action_space.n

    agent = RainbowAgent(obs_shape, n_actions, agent_config(args), args.device, pixel_input=pixel)
    memory = PrioritizedReplay(args.memory, obs_shape[1:] if pixel else obs_shape,
                               np.uint8 if pixel else np.float32, history=obs_shape[0] if pixel else 1,
                               n_step=args.n_step, gamma=args.gamma,
                               alpha=0.0 if args.no_per else args.priority_exponent, seed=args.seed)

    # random-agent reference on the evaluation seeds (computed once)
    from rainbow.evaluation import random_policy
    rand_rows = run_episodes(random_policy(n_actions, args.seed), eval_env, args.eval_episodes)
    random_score = summarize(rand_rows)["mean"]

    start_step, episode, best = 1, 0, -float("inf")
    learn_start = args.learn_start
    ckpt_path = os.path.join(run_dir, "checkpoint.pt")
    mem_path = os.path.join(run_dir, "memory.npz")
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        agent.load_state_dict(ck["agent"])
        start_step, episode, best = ck["step"] + 1, ck["episode"], ck["best"]
        if os.path.exists(mem_path):
            memory.load(mem_path)
            memory.end_episode()  # the interrupted episode must not be glued to the new one
            print(f"resumed from step {ck['step']} with replay memory ({len(memory)} transitions)")
        else:
            learn_start = start_step + args.learn_start  # refill the buffer before learning again
            print(f"resumed from step {ck['step']} without replay memory: refilling for {args.learn_start} steps")
        obs, info = env.reset(seed=args.seed + episode)
    else:
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(vars(args) | {"agent": agent.cfg.to_dict(), "obs_shape": list(obs_shape)}, f, indent=2)

    ep_log = CSVLog(os.path.join(run_dir, "episodes.csv"), args.resume)
    ev_log = CSVLog(os.path.join(run_dir, "eval.csv"), args.resume)
    print(f"run {args.id}: env={args.env} obs={args.obs} reward={args.reward} preset={args.preset} device={args.device}")
    print(f"components: double={agent.cfg.double} per={agent.cfg.per} dueling={agent.cfg.dueling} "
          f"n_step={agent.cfg.n_step} distributional={agent.cfg.distributional} noisy={agent.cfg.noisy}")
    print(f"random agent score on eval seeds: {random_score:.1f}")

    def save_checkpoint(step):
        torch.save({"agent": agent.state_dict(), "step": step, "episode": episode, "best": best,
                    "args": vars(args)}, ckpt_path)
        if args.save_memory:
            memory.save(mem_path)

    t0 = time.time()
    t_ep, ep_ret, ep_events = 0, 0.0, {}
    recent, last_stats = [], {}
    agent.train()
    step = start_step
    for step in range(start_step, args.steps + 1):
        agent.steps = step
        if step % args.replay_frequency == 0:
            agent.reset_noise()
        action = agent.act(obs)
        next_obs, reward, term, trunc, info = env.step(action)
        ep_ret += reward
        for k, v in info.get("events", {}).items():
            ep_events[k] = ep_events.get(k, 0) + v
        r = float(np.clip(reward, -args.reward_clip, args.reward_clip)) if args.reward_clip > 0 else reward
        done = term or trunc
        memory.append(obs[-1] if pixel else obs, action, r, done, t_ep)
        obs = next_obs
        t_ep += 1

        if done:
            episode += 1
            level = getattr(env, "level", None)
            row = {"step": step, "episode": episode, "score": info["score"], "return": round(ep_ret, 3),
                   "length": t_ep, "level": level, "time_min": round((time.time() - t0) / 60, 2)}
            row.update({f"ev_{k}": v for k, v in ep_events.items()})
            ep_log.write(row)
            recent = (recent + [info["score"]])[-20:]
            if curriculum and hasattr(env, "set_level"):
                env.set_level(level_at(curriculum, step, args.level))
            obs, info = env.reset()
            t_ep, ep_ret, ep_events = 0, 0.0, {}

        if step >= learn_start:
            beta = min(1.0, args.priority_weight + (1 - args.priority_weight) * step / args.steps)
            if step % args.replay_frequency == 0:
                last_stats = agent.learn(memory, beta)
            if step % args.target_update == 0:
                agent.update_target()

        if step % args.log_interval == 0:
            sps = (step - start_step + 1) / (time.time() - t0)
            avg = np.mean(recent) if recent else float("nan")
            print(f"step {step:>9,} | {sps:6.0f} steps/s | episodes {episode:5d} | avg score (20) {avg:7.1f} | "
                  f"loss {last_stats.get('loss', float('nan')):.3f} | Q {last_stats.get('q', float('nan')):.2f} | "
                  f"eps {agent.epsilon():.3f} | ETA {(args.steps - step) / sps / 3600:.1f} h", flush=True)

        if step % args.eval_interval == 0 or step == args.steps:
            agent.eval()
            rows = run_episodes(lambda o: agent.act(o, greedy=True), eval_env, args.eval_episodes)
            agent.train()
            s = summarize(rows)
            ev_log.write({"step": step, "mean_score": s["mean"], "std_score": s["std"], "min_score": s["min"],
                          "max_score": s["max"], "mean_return": float(np.mean([r_["return"] for r_ in rows])),
                          "mean_length": float(np.mean([r_["length"] for r_ in rows])),
                          "random_score": random_score, "time_min": round((time.time() - t0) / 60, 2)})
            print(f"  eval @ {step:,}: score {s['mean']:.1f} ± {s['std']:.1f} (random {random_score:.1f})", flush=True)
            if s["mean"] > best:
                best = s["mean"]
                torch.save(agent.state_dict() | {"eval_score": best, "step": step, "env_kwargs": env_kw},
                           os.path.join(run_dir, "model.pth"))
            plot_curves(run_dir)

        if args.checkpoint_interval and step % args.checkpoint_interval == 0:
            save_checkpoint(step)

        if args.time_limit and (time.time() - t0) / 3600 > args.time_limit:
            print(f"time limit reached at step {step:,}; saving checkpoint and stopping")
            save_checkpoint(step)
            break

    save_checkpoint(step)
    torch.save(agent.state_dict() | {"step": step, "env_kwargs": env_kw}, os.path.join(run_dir, "final.pth"))
    plot_curves(run_dir)
    print(f"done. best eval score {best:.1f}. files in {run_dir}")


if __name__ == "__main__":
    main()
