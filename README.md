# Space Invaders from Scratch × Rainbow DQN

RL course project, Innopolis University: Sergey Peshkov, Anton Kropotov, Arina Nikolaeva.

We wrote our own Space Invaders (pure Python/NumPy) as a Gymnasium environment with configurable
**reward triggers** and **difficulty levels**. Then we trained a **Rainbow DQN** agent on it, with all six
components implemented from scratch in PyTorch.

```
space_invaders/        the game and the environment
  game.py              rules, events (triggers), drawing, state vector
  env.py               Gymnasium env: pixels (4x84x84) or vector obs, 6 actions, reward presets, levels 0-2
rainbow/               the agent
  model.py             Nature CNN / MLP torso, NoisyLinear, dueling C51 head
  memory.py            prioritized n-step replay (sum tree), frame stacks rebuilt on sampling
  agent.py             Rainbow update + switches for ablations / vanilla DQN
  envs.py              our game or the original Atari game (ale-py)
  evaluation.py        run episodes, summaries, random policy
train.py               training with logs, evaluation, checkpoints, resume, curriculum
evaluate.py            trained agent(s) vs random on the same seeds -> table + box plot
demo.py                demo video (agent | random side by side, triggers on screen)
play.py                play it yourself (pygame)
notebooks/project.ipynb      shared notebook: environment, algorithm, worked example, results
notebooks/kaggle_train.ipynb training on a Kaggle / Colab GPU
tests/test_all.py      self-checks
```

## Quick start

```bash
pip install -r requirements.txt
python tests/test_all.py                                   # env, replay memory, agent
python train.py --id quick --obs vector --steps 50000      # ~minutes on CPU, checks the whole pipeline
python train.py --id rainbow_pixels --steps 1000000        # main run (GPU recommended)
python evaluate.py --models runs/rainbow_pixels/model.pth --names Rainbow --episodes 30
python demo.py --model runs/rainbow_pixels/model.pth --compare-random --out demo_vs_random.mp4
```

No GPU? Use `notebooks/kaggle_train.ipynb`: zip this folder, upload it to Kaggle as a dataset and run it with a GPU.

## Experiments the code supports

| Question | Command |
|---|---|
| Rainbow vs random | `train.py --id rainbow_pixels` then `evaluate.py` |
| Rainbow vs vanilla DQN | `train.py --id dqn --dqn` |
| Which component matters? | `--no-double`, `--no-per`, `--no-dueling`, `--n-step 1`, `--no-distributional`, `--no-noisy` |
| Do reward triggers help? | `--reward shaped` vs `--reward score` |
| Does a curriculum help? | `--curriculum "0:0,200000:1"` |
| Pixels vs state vector | `--obs vector` |
| Generalisation | `evaluate.py --level 2` on an agent trained on level 1 |
| Original Atari game | `--env ale --preset atari` (needs `pip install ale-py`) |

## Notes
* 1 agent step = 4 game frames. Rewards are clipped to [−1, 1] during training, as in DQN/Rainbow on Atari;
  logs and evaluation always report the real game score.
* `--time-limit H` stops cleanly with a checkpoint. `--resume` continues from it. Add `--save-memory` to keep the
  replay buffer between sessions, otherwise the buffer is refilled first.
* Rainbow paper reference for Atari Space Invaders (200M frames): DQN 1 692, Rainbow 18 789. Our game has
  different rules, so those numbers only show the order of magnitude, not a target.
