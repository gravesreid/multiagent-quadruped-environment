import gym
from gym import spaces
import numpy
import torch
from copy import copy
from mqe.envs.wrappers.empty_wrapper import EmptyWrapper
from isaacgym.torch_utils import *

class Go1DoubleObjectWrapper(EmptyWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.num_npcs = getattr(self.env.cfg.env, 'num_npcs', 2)  # Configurable number of NPCs

        # Adjust observation and action spaces for the two objects
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(20 + 2 * self.cfg.env.num_npcs + self.num_agents,), dtype=float)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=float)
        self.action_scale = torch.tensor([[[2, 0.5, 0.5],],], device="cuda").repeat(self.num_envs, self.num_agents, 1)

        # Reward scale for moving the objects
 