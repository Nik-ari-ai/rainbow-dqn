"""Rainbow DQN. Torch-dependent parts are imported lazily so that the replay
memory, environments and evaluation helpers work without PyTorch."""
from .memory import PrioritizedReplay
from .envs import make_env


def __getattr__(name):
    if name in ("RainbowAgent", "AgentConfig", "project_distribution"):
        from . import agent
        return getattr(agent, name)
    raise AttributeError(name)


__all__ = ["RainbowAgent", "AgentConfig", "project_distribution", "PrioritizedReplay", "make_env"]
