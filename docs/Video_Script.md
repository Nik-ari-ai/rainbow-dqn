# Proposal video — script (~950 words ≈ 6.5 min at 145 words/min; if you go over 7 min, skip the italic optional demo and shorten slide 11)

Speakers follow the assignment table: **Part 1 — Sergey**, **Part 2 — participant 3**, **Part 3 — participant 2**.
Speak at a normal pace (~145 words/min). Slide numbers match `RL_Proposal_Rainbow_SpaceInvaders.pdf`.

---

## Slide 1 — Title (Part 1 speaker, ~10 s)
Hello! We are Sergey Peshkov, Anton Kropotov and Arina Nikolaeva. Our assigned algorithm is Rainbow DQN, and we will train it on Space Invaders — in our own re-implementation of the arcade game.

## Slide 2 — Outline (~10 s)
The talk has three parts: the algorithm, the environment, and why this combination makes sense, together with our training and evaluation plan.

---

## PART 1 — ALGORITHM (≈ 1.5 min)

### Slide 3 — Rainbow DQN at a glance (~35 s)
Rainbow DQN is a **value-based, off-policy, model-free** algorithm for **discrete actions**.
It builds on DQN. DQN uses a convolutional network to estimate Q of s and a — the expected return of taking action a in state s — and the agent simply picks the action with the highest Q. Two tricks make it stable: experience replay, where the network learns from random past transitions, and a target network, a slowly updated copy used to compute the learning target.
Rainbow combines six independent improvements of DQN into one agent. Its key difference: instead of one number per action, it learns the whole distribution of returns, and acts on the mean of that distribution.

### Slide 4 — Six components (~35 s)
The six components are:
Double Q-learning — the online network chooses the next action and the target network evaluates it, which removes overestimation.
Prioritized replay — transitions with a large error are replayed more often.
Dueling network — separate streams for the value of the state and the advantage of each action.
Multi-step returns — we use three real rewards before bootstrapping, so reward spreads faster.
Distributional learning — 51 atoms describe the probability of each possible return.
And Noisy Nets — learned noise in the network replaces epsilon-greedy exploration.

### Slide 5 — One learning step (~25 s)
One step looks like this. The agent acts, stores the transition in the replay buffer, and samples a batch by priority. For each sample we build the target: three rewards plus the discounted target distribution at the next state. We project it back onto the 51 atoms and minimise the cross-entropy to our prediction. The loss becomes the new priority. Every 2,000 updates the target network is refreshed.

---

## PART 2 — ENVIRONMENT (≈ 2.5 min)

### Slide 6 — Our Space Invaders (~40 s)
Our environment is Space Invaders. We did not just take the Atari ROM — we wrote the game ourselves in Python and NumPy, with the same rules: rows of invaders move side to side, speed up and shoot back, shields erode, and a UFO sometimes flies over for bonus points.
It follows the standard Gymnasium interface, runs about 1,200 steps per second without rendering, and reports every game event — a kill, a hit, a miss, a cleared wave. It also has three difficulty levels. The game is stochastic: enemy fire is random, and each episode starts with a random number of no-op frames.

*(Optional: play 5–10 seconds of `rainbow_best_game.mp4` here.)*

### Slide 7 — Observation and action space (~35 s)
The observation is what a human player sees: four stacked grayscale frames of 84 by 84 pixels. One frame only shows positions; four consecutive frames also show the direction and speed of the invaders and bullets, so the state becomes approximately Markov.
The action space is discrete with six actions, the same as on Atari: no-op, fire, right, left, and right or left while firing. One agent step lasts four game frames.

### Slide 8 — Reward, goal and episode end (~40 s)
The reward is the game score: 10 points for invaders in the bottom rows, 20 in the middle, 30 for the top row, and 100 for the UFO. As in DQN on Atari, we clip the reward to minus one to one during training, so every kill gives plus one. Because we own the game, we can also switch on a shaped reward: a penalty for being hit or missing, and a bonus for dodging and clearing a wave.
The goal is to maximise the score without losing all three lives.
An episode ends when all three lives are lost, when the invaders reach the player's row, or after a time limit of 10,000 steps. After a wave is cleared, a new and faster wave starts.

### Slide 9 — What the agent should learn (~25 s)
From pixels only, the agent has to learn four skills: aim — shoot only when an invader is above, because just one bullet can be on screen; dodge enemy bullets and use the shields; prioritise the lowest invaders and the UFO; and plan ahead, because the reward arrives only after the bullet reaches its target.

---

## PART 3 — MOTIVATION AND PLAN (≈ 1.5 min)

### Slide 10 — Why this environment (~25 s)
Why Space Invaders? First, it is a classic benchmark: it is one of the Atari games on which DQN and Rainbow were evaluated in the original papers. Second, the difficulty is right — pixel input, moving targets, random fire and delayed reward, but still trainable on one GPU in hours. Third, our own implementation gives us full control over rewards, difficulty and speed. And the result is easy to judge: the score and a video show what the agent has learned.

### Slide 11 — Why Rainbow fits, and challenges (~30 s)
Rainbow suits this task because the action space is small and discrete, which is exactly what Q-learning needs, and it was designed for pixel input. Prioritized replay reuses rare events like a UFO hit, the distributional head models the random enemy fire, and multi-step returns bridge the delay between a shot and a kill.
The challenges: our compute budget is about 25 times smaller than in the paper, results vary a lot between episodes, clipped rewards treat all invaders as equal, and shaped rewards could be exploited by the agent.

### Slide 12 — Training and evaluation plan (~30 s)
We train for 2 million agent steps — 8 million frames — on an RTX 5070, about two hours per run. Every 50,000 steps we play 10 evaluation games and keep the best model. For the final evaluation, every agent plays the same 30 games, and we compare mean and median score, cleared waves and shooting accuracy against a random agent and vanilla DQN. If time allows, we will add ablations, the shaped reward and a difficulty curriculum.
We already ran a pilot to check the pipeline: on 30 test games Rainbow scores about 2,070 points, vanilla DQN about 1,460, and a random agent about 370.

### Slide 13 — References (~5 s)
Thank you for your attention!
