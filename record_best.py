"""Record a video of the best agent playing its best game.

Plays N candidate games without recording (deterministic: CPU + fixed seeds),
then replays the highest-scoring game and records it with score, action and triggers on screen.

  python record_best.py --model runs/rainbow_pixels/model.pth --candidates 20 --out results/rainbow_best_game.mp4
"""
import argparse
import numpy as np

from demo import Recorder
from rainbow import RainbowAgent
from rainbow.evaluation import run_episodes
from space_invaders import SpaceInvadersEnv


def play(agent, seed, level, recorder=None):
    agent.rng = np.random.default_rng(seed)  # makes the rare eval-epsilon actions reproducible
    env = SpaceInvadersEnv(obs_mode="pixels", level=level, reward="shaped", seed=seed)
    return run_episodes(lambda o: agent.act(o, greedy=True), env, 1, seed=seed, on_step=recorder)[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="runs/rainbow_pixels/model.pth")
    p.add_argument("--candidates", type=int, default=20)
    p.add_argument("--first-seed", type=int, default=1000)
    p.add_argument("--level", type=int, default=1)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--title", default="RAINBOW DQN")
    p.add_argument("--out", default="results/rainbow_best_game.mp4")
    a = p.parse_args()

    agent, meta = RainbowAgent.from_file(a.model, "cpu")
    print(f"model from step {meta.get('step')}, eval score {meta.get('eval_score')}", flush=True)
    scores = {}
    for s in range(a.first_seed, a.first_seed + a.candidates):
        r = play(agent, s, a.level)
        scores[s] = r["score"]
        print(f"seed {s}: score {r['score']}, steps {r['length']}, waves {r.get('waves_cleared', 0)}", flush=True)
    best = max(scores, key=scores.get)
    print(f"best seed {best} with score {scores[best]}; mean of candidates {np.mean(list(scores.values())):.0f}", flush=True)

    rec = Recorder(a.title)
    r = play(agent, best, a.level, rec)
    print(f"recorded replay: score {r['score']} (expected {scores[best]}), {len(rec.frames)} frames", flush=True)
    # hold the final frame for 2 seconds
    rec.frames += [rec.frames[-1]] * (2 * a.fps)
    import imageio.v2 as imageio
    imageio.mimsave(a.out, rec.frames, fps=a.fps, macro_block_size=8, quality=8)
    print(f"saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
