"""Play our Space Invaders yourself (needs pygame: pip install pygame).

Controls: LEFT / RIGHT arrows to move, SPACE to fire, ESC to quit.
Used to measure the human reference score and to check the reward triggers.

  python play.py --level 1
"""
from __future__ import annotations

import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--level", type=int, default=1)
    p.add_argument("--fps", type=int, default=15, help="agent steps per second (1 step = 4 frames)")
    args = p.parse_args()

    import pygame
    from space_invaders import SpaceInvadersEnv

    env = SpaceInvadersEnv(obs_mode="vector", level=args.level, reward="shaped", render_mode="human", noop_max=0)
    env.reset()
    clock = pygame.time.Clock()
    total, running = 0.0, True
    scores = []
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False
        keys = pygame.key.get_pressed()
        left, right, fire = keys[pygame.K_LEFT], keys[pygame.K_RIGHT], keys[pygame.K_SPACE]
        action = {(0, 0, 0): 0, (0, 0, 1): 1, (0, 1, 0): 2, (1, 0, 0): 3, (0, 1, 1): 4, (1, 0, 1): 5}.get((left, right, fire), 0)
        _, r, term, trunc, info = env.step(action)
        total += r
        fired = [k for k, v in info["events"].items() if v and k not in ("score", "shot_fired")]
        if fired:
            print("triggers:", ", ".join(fired), f"| shaped return so far {total:.2f}")
        if term or trunc:
            scores.append(info["score"])
            print(f"GAME OVER - score {info['score']}  (games: {len(scores)}, mean {sum(scores) / len(scores):.0f})")
            env.reset()
            total = 0.0
        clock.tick(args.fps)
    env.close()


if __name__ == "__main__":
    main()
