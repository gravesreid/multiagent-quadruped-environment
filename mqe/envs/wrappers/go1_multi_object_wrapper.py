import gym
from gym import spaces
import numpy
import torch
from copy import copy
from mqe.envs.wrappers.empty_wrapper import EmptyWrapper

from isaacgym.torch_utils import *
class MultiObjectWrapper(EmptyWrapper):

    def __init__(self, env):
        super().__init__(env)

        # Defining the observation space- the shape is important- it is determined by base requirements for the observation space,
        # the number of agents, and the number of NPCs
        self.observation_space = spaces.Box(
                                            low=-float('inf'), high=float('inf'), shape=(18 + self.num_agents + 
                                            2*self.cfg.env.num_npcs,), dtype=float
                                            )
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=float)
        self.action_scale = torch.tensor([[[2, 0.5, 0.5],],], device="cuda").repeat(self.num_envs, self.num_agents, 1)


        self.reward_buffer = {
            "ball approach reward": 0,
            "contact punishment": 0,
            "angle punishment": 0,
            "success reward": 0,
            "step count": 0
        }
    
    def _init_extras(self, obs):
        pass

    def reset(self):
        obs_buff = self.env.reset()

        if getattr(self, "gate_pos", None) is None:
            self._init_extras(obs_buff)



        #obs = torch.cat([base_info, torch.flip(base_info, [1]), padding, padding, padding, padding2], dim=2)
        obs = torch.zeros([self.env.num_envs, self.num_agents, 18 + 2*self.cfg.env.num_npcs + 1], device=self.env.device)


        return obs
    
    def step(self, action):
        action = torch.clip(action, -1, 1)
        obs_buf, _, termination, info = self.env.step((action * self.action_scale).reshape(-1, self.action_space.shape[0]))
        
        if getattr(self, "gate_pos", None) is None:
            self._init_extras(obs_buf)


        #obs = torch.cat([self.obs_ids, base_info, torch.flip(base_info, [1]), padding, padding, padding, padding2], dim=2)
        obs = torch.zeros([self.env.num_envs, self.num_agents, 18 + 2*self.cfg.env.num_npcs + 1], device=self.env.device)


        self.reward_buffer["step count"] += 1
        reward = torch.zeros([self.env.num_envs, self.num_agents], device=self.env.device)


        return obs, reward, termination, info

