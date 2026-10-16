"""Quick self-checks. Run from the project root:  python tests/test_all.py

Parts that need PyTorch are skipped automatically if it is not installed.
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from space_invaders import SpaceInvadersEnv  # noqa: E402
from rainbow.memory import PrioritizedReplay, SumTree  # noqa: E402


def test_env():
    for mode in ("pixels", "vector"):
        for level in (0, 1, 2):
            env = SpaceInvadersEnv(obs_mode=mode, level=level, seed=0)
            obs, info = env.reset(seed=0)
            assert obs.shape == env.observation_space.shape, (obs.shape, env.observation_space.shape)
            done, steps = False, 0
            while not done:
                obs, r, term, trunc, info = env.step(env.action_space.sample())
                done = term or trunc
                steps += 1
            assert info["score"] >= 0 and steps > 10
    # vector size is the same for all levels (needed for the curriculum)
    sizes = {SpaceInvadersEnv(obs_mode="vector", level=l).reset()[0].shape for l in (0, 1, 2)}
    assert len(sizes) == 1
    print("env OK")


def test_memory():
    t = SumTree(5)
    t.update([0, 1, 2, 3, 4], [1, 2, 3, 4, 0])
    assert t.total == 10 and list(t.find([0.5, 1.5, 5.9, 9.99])) == [0, 1, 2, 3]
    cap = 50
    m = PrioritizedReplay(cap, (1,), np.int64, history=4, n_step=3, gamma=0.5, seed=0)
    rng, g, log = np.random.default_rng(1), 0, {}
    for _ in range(30):
        L = int(rng.integers(1, 9))
        for t in range(L):
            m.append(np.array([g]), g, float(g), t == L - 1, t)
            log[g] = (t, t == L - 1)
            g += 1
    for _ in range(100):
        b = m.sample(16)
        for i in range(16):
            gid = int(b["actions"][i])
            t = log[gid][0]
            assert list(b["obs"][i, :, 0]) == [gid - k if t - k >= 0 else 0 for k in (3, 2, 1, 0)]
            R, alive = 0.0, 1.0
            for k in range(3):
                R += 0.5 ** k * (gid + k)
                if log[gid + k][1]:
                    alive = 0.0
                    break
            assert abs(b["returns"][i] - R) < 1e-4 and b["nonterminal"][i] == alive
    with tempfile.TemporaryDirectory() as d:
        m.save(os.path.join(d, "m.npz"))
        m2 = PrioritizedReplay(cap, (1,), np.int64, seed=0)
        m2.load(os.path.join(d, "m.npz"))
        assert m2.tree.total == m.tree.total
    print("memory OK")


def test_torch():
    try:
        import torch
    except ImportError:
        print("torch not installed: skipping agent tests")
        return
    from rainbow.agent import RainbowAgent, AgentConfig, project_distribution
    # C51 projection keeps probability mass and the mean (when inside the support)
    support = torch.linspace(-10, 10, 51)
    p = torch.softmax(torch.randn(8, 51), 1)
    r = torch.tensor([0.0, 1.0, -1.0, 0.5, 0.0, 2.0, 0.3, -0.7])
    nt = torch.tensor([1.0, 1, 1, 0, 1, 1, 0, 1])
    m = project_distribution(p * 0 + torch.softmax(torch.randn(8, 51) * 0.1, 1), r, nt, 0.5, support, -10, 10)
    assert torch.allclose(m.sum(1), torch.ones(8), atol=1e-5)
    pz = torch.softmax(torch.randn(8, 51) * 0.1, 1)
    m = project_distribution(pz, r, nt, 0.5, support, -10, 10)
    expected = r + nt * 0.5 * (pz * support).sum(1)
    assert torch.allclose((m * support).sum(1), expected, atol=1e-4)
    # terminal transition: all mass on the reward
    m = project_distribution(pz[:1], torch.tensor([1.0]), torch.tensor([0.0]), 0.99, support, -10, 10)
    assert abs((m * support).sum().item() - 1.0) < 1e-5

    # all component combinations build and learn on both observation types
    for mode, arch in (("vector", "mlp"), ("pixels", "data-efficient")):
        env = SpaceInvadersEnv(obs_mode=mode, seed=0)
        obs, _ = env.reset(seed=0)
        for flags in ({}, {"distributional": False}, {"noisy": False, "dueling": False, "double": False}):
            cfg = AgentConfig(arch=arch, hidden=64, batch_size=8, **flags)
            agent = RainbowAgent(obs.shape, 6, cfg, "cpu", pixel_input=(mode == "pixels"))
            mem = PrioritizedReplay(500, obs.shape[1:] if mode == "pixels" else obs.shape,
                                    np.uint8 if mode == "pixels" else np.float32,
                                    history=4 if mode == "pixels" else 1, n_step=3)
            o, t = env.reset()[0], 0
            for _ in range(200):
                a = agent.act(o)
                o2, rew, te, tr, _ = env.step(a)
                mem.append(o[-1] if mode == "pixels" else o, a, float(np.clip(rew, -1, 1)), te or tr, t)
                o, t = (env.reset()[0], 0) if (te or tr) else (o2, t + 1)
            for _ in range(5):
                stats = agent.learn(mem, 0.4)
            assert np.isfinite(stats["loss"])
            agent.update_target()
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, "model.pth")
                torch.save(agent.state_dict(), path)
                agent2, _ = RainbowAgent.from_file(path)
                agent2.act(o, greedy=True)
    print("agent OK")


if __name__ == "__main__":
    test_env()
    test_memory()
    test_torch()
    print("all tests passed")
