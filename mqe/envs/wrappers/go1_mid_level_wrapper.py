import gym
from gym import spaces
import numpy
import torch
from copy import copy
from mqe.envs.wrappers.empty_wrapper import EmptyWrapper
import random

class Go1MidLevelWrapper(EmptyWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(18 + self.num_agents,), dtype=float)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=float)
        self.action_scale = torch.tensor([[[2, 0.5, 0.5],],], device="cuda").repeat(self.num_envs, self.num_agents, 1)
        self.num_steps = 0

        
        self.obs_ids = torch.eye(self.num_agents, dtype=torch.float32, device=self.device).repeat(self.num_envs, 1).reshape(self.num_envs, self.num_agents, -1)

        # for hard setting of reward scales (not recommended)
        
        # self.target_reward_scale = 1
        # self.success_reward_scale = 0
        # self.lin_vel_x_reward_scale = 0
        # self.approach_frame_punishment_scale = 0
        # self.agent_distance_punishment_scale = 0
        # self.contact_punishment_scale = -10
        # self.lin_vel_y_punishment_scale = 0
        # self.command_value_punishment_scale = 0

        self.reward_buffer = {
            "target reward": 0,
            "success reward": 0,
            # "approach frame punishment": 0,
            "agent distance punishment": 0,
            # "command lin_vel.y punishment": 0,
            # "command value punishment": 0,
            "contact punishment": 0,
            # "lin_vel.x reward": 0,
            "step count": 0
        }

    def _init_extras(self, obs):
        return

    def reset(self):
        obs_buf = self.env.reset()

        if getattr(self, "point_pos", None) is None:
            self._init_extras(obs_buf)

        ball_pos = self.root_states_npc[:, :3].reshape(self.num_envs, 3) - self.env_origins
        ball_pos = ball_pos.unsqueeze(1).repeat(1, self.num_agents, 1)

        ball_vel = self.root_states_npc[:, 7:10].reshape(self.num_envs, 3).unsqueeze(1).repeat(1, self.num_agents, 1)

        base_pos = obs_buf.base_pos
        base_rpy = obs_buf.base_rpy
        base_info = torch.cat([base_pos, base_rpy], dim=1).reshape([self.env.num_envs, self.env.num_agents, -1])
        obs = torch.cat([self.obs_ids, base_info, torch.flip(base_info, [1]),
                        ], dim=2)
        base_info = torch.cat([base_pos, base_rpy], dim=1).reshape([self.env.num_envs, self.env.num_agents, -1])[:, :2, :]
        obs = torch.cat([self.obs_ids, base_info, torch.flip(base_info, [1]), ball_pos, ball_vel], dim=2)
        #obs = 0
        return obs

    def step(self, action):
        self.num_steps += 1
        action = torch.clip(action, -1, 1)
        obs_buf, x, termination, info = self.env.step((action * self.action_scale).reshape(-1, self.action_space.shape[0]))
       # print(f'termination: {termination}')
       # print(f'termination shape: {termination.shape}')
      #  print(f'Obs: {obs_buf}')
      #  print(f'info: {info}')
      #  print(f'x: {x}')
    

        if getattr(self, "point_pos", None) is None:
            self._init_extras(obs_buf)
        
        ball_pos = self.root_states_npc[:, :3].reshape(self.num_envs, 3) - self.env_origins
        ball_pos = ball_pos.unsqueeze(1).repeat(1, self.num_agents, 1)

        ball_vel = self.root_states_npc[:, 7:10].reshape(self.num_envs, 3).unsqueeze(1).repeat(1, self.num_agents, 1)
        
        base_pos = obs_buf.base_pos
        base_rpy = obs_buf.base_rpy
        base_info = torch.cat([base_pos, base_rpy], dim=1).reshape([self.env.num_envs, self.env.num_agents, -1])
        obs = torch.cat([self.obs_ids, base_info, torch.flip(base_info, [1]), ball_pos, ball_vel], dim=2)
        self.reward_buffer["step count"] += 1
        reward = torch.zeros([self.env.num_envs, self.env.num_agents], device=self.env.device)
       
        # approach reward
        if self.target_reward_scale != 0:
            distance_to_taget = torch.norm(base_pos.squeeze()[:2] - ball_pos.squeeze()[:2])
            #print(f'distance_to_taget: {distance_to_taget}')
       
            if not hasattr(self, "last_distance_to_taget"):
                self.last_distance_to_taget = copy(distance_to_taget)
          
            target_reward = (self.last_distance_to_taget - distance_to_taget).reshape(self.num_envs, -1).sum(dim=1, keepdim=True)
            target_reward[self.env.reset_ids] = 0
       
            target_reward *= self.target_reward_scale
            reward += target_reward.repeat(1, self.env.num_agents)
       
            self.last_distance_to_target = copy(distance_to_taget)
       
            self.reward_buffer["target reward"] += torch.sum(target_reward).cpu()
       
        # contact punishment
        if self.contact_punishment_scale != 0:
            collide_reward = self.contact_punishment_scale * self.env.collide_buf
            reward += collide_reward.unsqueeze(1).repeat(1, self.num_agents)
            self.reward_buffer["contact punishment"] += torch.sum(collide_reward).cpu()
       
        # success reward
        if self.success_reward_scale != 0:
            success_reward = torch.zeros([self.env.num_envs * self.env.num_agents], device="cuda")
            # if distance to target is less than 0.25, give success reward
            #if self.last_distance_to_taget < 0.1:
            #    success_reward = self.success_reward_scale
            #success_reward[base_pos[:, 0] > ball_pos.squeeze()[:,0] + 0.25] = self.success_reward_scale
            reward += success_reward.reshape([self.env.num_envs, self.env.num_agents])
            self.reward_buffer["success reward"] += torch.sum(success_reward).cpu()
       
        # approach frame punishment
        if self.approach_frame_punishment_scale != 0:
            dis_to_left_frame = ((base_pos[:, :2] - self.frame_left) ** 2).sum(dim=1).reshape(self.num_envs, -1)
            dis_to_right_frame = ((base_pos[:, :2] - self.frame_right) ** 2).sum(dim=1).reshape(self.num_envs, -1)
       
            approach_left = self.approach_frame_punishment_scale / dis_to_left_frame[dis_to_left_frame < 0.04]
            approach_right = self.approach_frame_punishment_scale / dis_to_right_frame[dis_to_left_frame < 0.04]
            reward[dis_to_left_frame < 0.04] += approach_left
            reward[dis_to_right_frame < 0.04] += approach_right
            self.reward_buffer["approach frame punishment"] += torch.sum(approach_left).cpu()
            self.reward_buffer["approach frame punishment"] += torch.sum(approach_right).cpu()
       
        # agent distance punishment
        if self.agent_distance_punishment_scale != 0:
            agent_dis = (base_pos[:, :2] - torch.flip(base_pos[:, :2].reshape(self.num_envs, self.num_agents, 2), dims=[1,]).reshape(-1, 2)) ** 2
            agent_dis = agent_dis.sum(dim=1).reshape(self.num_envs, -1)
            agent_distance_punishment = self.agent_distance_punishment_scale  / agent_dis[agent_dis < 0.25]
            reward[agent_dis < 0.25] += agent_distance_punishment
            self.reward_buffer["agent distance punishment"] += torch.sum(agent_distance_punishment).cpu()
       
        # command lin_vel.y punishment
        if self.lin_vel_y_punishment_scale != 0:
            v_y_punishment = self.lin_vel_y_punishment_scale * action[:, :, 1] ** 2
            reward += v_y_punishment
            self.reward_buffer["command lin_vel.y punishment"] += torch.sum(v_y_punishment).cpu()
       
        # command value punishment
        if self.command_value_punishment_scale != 0:
            command_value_punishment = self.command_value_punishment_scale * torch.clip(action ** 2 - 1, 0, 1).sum(dim=2)
            reward += command_value_punishment
            self.reward_buffer["command value punishment"] += torch.sum(command_value_punishment).cpu()
       
        # lin_vel.x reward
        if self.lin_vel_x_reward_scale != 0:
            v_x_reward = self.lin_vel_x_reward_scale * obs_buf.lin_vel[:, 0].reshape(self.num_envs, self.num_agents)
            reward += v_x_reward
            self.reward_buffer["lin_vel.x reward"] += torch.sum(v_x_reward).cpu()
       
        reward = reward.sum(dim=1).unsqueeze(1).repeat(1, self.num_agents)
        #obs, reward = 0, 0
        #print(f'termination: {termination}')
        return obs, reward, termination, info