"""Record a demo video of an agent playing our Space Invaders.

  python demo.py --model runs/rainbow_pixels/model.pth --out demo.mp4
  python demo.py --model runs/rainbow_pixels/model.pth --compare-random --out demo_vs_random.mp4
  python demo.py --random --out random.gif

The video shows the score, the chosen action and the reward triggers as they fire.
MP4 needs ``imageio-ffmpeg``; otherwise use a .gif file name.
"""
from __future__ import annotations

import argparse

import numpy as np

from space_invaders import SpaceInvadersEnv, ACTION_NAMES
from space_invaders.env import _draw_hud

TRIGGER_LABELS = {"enemy_killed": "+ enemy killed", "ufo_killed": "+ UFO!", "player_hit": "- player hit",
                  "wave_cleared": "+ wave cleared", "bullet_dodged": "+ dodged", "shot_missed": "- missed"}


class Recorder:
    def __init__(self, title):
        self.title = title
        self.frames = []
        self.popups = []  # (text, frames_left)

    def __call__(self, env, action, reward, info):
        for k, label in TRIGGER_LABELS.items():
            if info["events"].get(k):
                self.popups.append([label, 12])
        self.popups = [p for p in self.popups if p[1] > 0][-4:]
        img = env.render(scale=3, hud=False)
        lines = [p[0] for p in self.popups]
        for p in self.popups:
            p[1] -= 1
        head = f"{self.title}   SCORE {info['score']}   LIVES {info['lives']}   {ACTION_NAMES[action]}"
        self.frames.append(_draw_hud(img, head, lines))


def play(policy, title, episodes, seed, level, max_steps):
    env = SpaceInvadersEnv(obs_mode="pixels", level=level, reward="shaped", noop_max=0)
    rec = Recorder(title)
    from rainbow.evaluation import run_episodes
    rows = run_episodes(policy, env, episodes, seed=seed, max_steps=max_steps, on_step=rec)
    return rec.frames, rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help="model.pth of a trained agent")
    p.add_argument("--random", action="store_true", help="record a random agent instead")
    p.add_argument("--compare-random", action="store_true", help="side by side: agent | random")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--level", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=3000, help="cap per episode to keep the video short")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--out", default="demo.mp4")
    args = p.parse_args(argv)

    rng = np.random.default_rng(0)
    rand_policy = lambda o: int(rng.integers(6))
    videos = []
    if args.model and not args.random:
        from rainbow import RainbowAgent
        agent, _ = RainbowAgent.from_file(args.model)
        if not agent.pixel_input:
            raise SystemExit("demo.py expects a pixel-based model (use the same env settings as in training)")
        videos.append(play(lambda o: agent.act(o, greedy=True), "RAINBOW", args.episodes, args.seed, args.level, args.max_steps))
    if args.random or args.compare_random or not args.model:
        videos.append(play(rand_policy, "RANDOM", args.episodes, args.seed, args.level, args.max_steps))

    for frames, rows in videos:
        print(f"{frames and len(frames)} frames, scores: {[r['score'] for r in rows]}")

    if len(videos) == 2:  # side by side, pad the shorter clip with its last frame
        a, b = videos[0][0], videos[1][0]
        n = max(len(a), len(b))
        a += [a[-1]] * (n - len(a))
        b += [b[-1]] * (n - len(b))
        sep = np.full((a[0].shape[0], 8, 3), 40, np.uint8)
        frames = [np.concatenate([x, sep, y], 1) for x, y in zip(a, b)]
    else:
        frames = videos[0][0]

    import imageio.v2 as imageio
    if args.out.endswith(".gif"):
        imageio.mimsave(args.out, frames[::2], duration=2 / args.fps, loop=0)
    else:
        imageio.mimsave(args.out, frames, fps=args.fps, macro_block_size=8)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
