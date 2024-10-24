import gym
from gym import spaces
import numpy
import torch
from copy import copy
from mqe.envs.wrappers.empty_wrapper import EmptyWrapper

from isaacgym.torch_utils import *
class MinimalWrapper(EmptyWrapper):

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
        #print(f"obs: {obs}")
        pass

    def reset(self):
        print(f"shape of observation space: {self.observation_space.shape}")
        print(f'shape of action space: {self.action_space.shape}')
        print(f'shape of action scale: {self.action_scale.shape}')
        obs_buff = self.env.reset()
        print(f"obs_buff: {obs_buff}")
        obs_buff_attributes = dir(obs_buff.cfgs)
        print(f"obs_buff attributes: {obs_buff_attributes}")
        base_pos_dim = obs_buff.base_pos.shape
        print(f"base_pos_dim: {base_pos_dim}")
        base_quat_dim = obs_buff.base_quat.shape
        print(f"base_quat_dim: {base_quat_dim}")
        dof_pos_dim = obs_buff.dof_pos.shape
        print(f"dof_pos_dim: {dof_pos_dim}")
        dof_vel_dim = obs_buff.dof_vel.shape
        print(f"dof_vel_dim: {dof_vel_dim}")
        lin_vel_dim = obs_buff.lin_vel.shape
        print(f"lin_vel_dim: {lin_vel_dim}")
        ang_vel_dim = obs_buff.ang_vel.shape
        print(f"ang_vel_dim: {ang_vel_dim}")
        projected_gravity_dim = obs_buff.projected_gravity.shape
        print(f"projected_gravity_dim: {projected_gravity_dim}")
        base_rpy_dim = obs_buff.base_rpy.shape
        print(f"base_rpy_dim: {base_rpy_dim}")
        last_action_dim = obs_buff.last_action.shape
        print(f"last_action_dim: {last_action_dim}")
        last_last_action_dim = obs_buff.last_last_action.shape
        print(f"last_last_action_dim: {last_last_action_dim}")
        env_info_dim = obs_buff.env_info
        print(f"env_info_dim: {env_info_dim}")

        if getattr(self, "gate_pos", None) is None:
            self._init_extras(obs_buff)

        print(f'self.root_states_npc shape: {self.root_states_npc.shape}')
        ball_pos = self.root_states_npc[:, :3].reshape(self.num_envs, 3) - self.env_origins
        ball_pos = ball_pos.unsqueeze(1).repeat(1, self.num_agents, 1)
        print(f"ball_pos: {ball_pos}")
        ball_vel = self.root_states_npc[:, 7:10].reshape(self.num_envs, 3).unsqueeze(1).repeat(1, self.num_agents, 1)

        padding = torch.zeros([self.num_envs, self.num_agents, 3], device=self.env.device)
        print(f"padding: {padding}")

        print(f'root_states_npc shape: {self.root_states_npc.shape}')

        base_pos = obs_buff.base_pos
        print(f"base_pos: {base_pos}")
        base_rpy = obs_buff.base_rpy
        print(f"base_rpy: {base_rpy}")
        base_info = torch.cat([base_pos, base_rpy], dim=1).reshape([self.env.num_envs, self.num_agents, -1])[:, :2, :]
        print(f"base_info: {base_info}")
        obs = torch.cat([base_info, torch.flip(base_info, [1]), ball_pos, ball_vel, padding], dim=2)
        print(f"obs: {obs}")
        print(f"obs shape: {obs.shape}")


        return obs
    
    def step(self, action):
        action = torch.clip(action, -1, 1)
        obs_buf, _, termination, info = self.env.step((action * self.action_scale).reshape(-1, self.action_space.shape[0]))
        
        if getattr(self, "gate_pos", None) is None:
            self._init_extras(obs_buf)

        ball_pos = self.root_states_npc[:,:3].reshape(self.num_envs, 3) - self.env_origins
        ball_pos = ball_pos.unsqueeze(1).repeat(1, self.num_agents, 1)
        ball_vel = self.root_states_npc[:, 7:10].reshape(self.num_envs, 3).unsqueeze(1).repeat(1, self.num_agents, 1)
        ball_acc = self.root_states_npc[:, 10:12].reshape(self.num_envs, 2).unsqueeze(1).repeat(1, self.num_agents, 1)

        base_pos = obs_buf.base_pos
        base_rpy = obs_buf.base_rpy
        base_info = torch.cat([base_pos, base_rpy], dim=1).reshape([self.env.num_envs, self.env.num_agents, -1])[:, :2, :]
        obs = torch.cat([self.obs_ids, base_info, torch.flip(base_info, [1]), ball_pos, ball_vel, ball_acc], dim=2)

        approach_angle = torch.atan2(ball_pos[:, 0, 1] - base_pos[:, 1], ball_pos[:, 0, 0] - base_pos[:, 0])

        self.reward_buffer["step count"] += 1
        reward = torch.zeros([self.env.num_envs, self.num_agents], device=self.env.device)

        if self.ball_approach_reward_scale != 0:
            distance_to_ball = torch.norm(base_pos[:, :2] - ball_pos[:, 0, :2], p=2, dim=1)
            if not hasattr(self, "last_distance_to_ball"):
                self.last_distance_to_ball = copy(distance_to_ball)
            ball_approach_reward = (self.last_distance_to_ball - distance_to_ball).reshape(self.num_envs, -1).sum(dim=1, keepdim=True)
            ball_approach_reward[self.env.reset_ids] = 0
            ball_approach_reward *= self.ball_approach_reward_scale
            reward += ball_approach_reward.repeat(1, self.env.num_agents)
            self.last_distance_to_ball = copy(distance_to_ball)
            self.reward_buffer["ball approach reward"] += torch.sum(ball_approach_reward).cpu()
        # contact punishment
        if self.contact_punishment_scale != 0:
            collide_reward = self.contact_punishment_scale * self.env.collide_buf
            reward += collide_reward.unsqueeze(1).repeat(1, self.num_agents)
            self.reward_buffer["contact punishment"] += torch.sum(collide_reward).cpu()

        # angle punishment
        if self.angle_punishment_scale != 0:
            angle_punishment = self.angle_punishment_scale * torch.abs(approach_angle).unsqueeze(1).repeat(1, self.num_agents)
            reward += angle_punishment
            self.reward_buffer["angle punishment"] += torch.sum(angle_punishment).cpu()

        return obs, reward, termination, info
