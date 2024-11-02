import gym
from gym import spaces
import numpy
import torch
from copy import copy
from mqe.envs.wrappers.empty_wrapper import EmptyWrapper

from isaacgym import gymapi, gymutil
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
        self.goal_pos = None


        self.reward_buffer = {
            "goal approach reward": 0,
            "contact punishment": 0,
            "angle punishment": 0,
            "success reward": 0,
            "step count": 0
        }
    
    def _init_extras(self, obs):
        pass
    
    def draw_spheres(self, point):
        self.env.gym.clear_lines(self.env.viewer)
        num_lines = 5  # Number of lines per sphere
        line_length = 0.12  # Length of each line segment

        center_pose = gymapi.Transform()
        print(f'point shape: {point.shape}')
        print(f'point: {point}')
        center_pose.p = gymapi.Vec3(point.squeeze()[0], point.squeeze()[1], 0.25)

        # Draw random lines around each point to simulate a sphere
        for _ in range(num_lines):
            # Generate a random direction for the line
            direction = torch.randn(3, device=self.env.device)
            direction = direction / torch.norm(direction) * (line_length / 2)  # Normalize and scale

            start_pose = gymapi.Transform()
            end_pose = gymapi.Transform()

            start_pose.p = gymapi.Vec3(
                center_pose.p.x + direction[0].item(),
                center_pose.p.y + direction[1].item(),
                center_pose.p.z
            )
            end_pose.p = gymapi.Vec3(
                center_pose.p.x - direction[0].item(),
                center_pose.p.y - direction[1].item(),
                center_pose.p.z
            )

            # Define the color as a Vec3 object (e.g., red)
            color = gymapi.Vec3(1.0, 0.0, 0.0)

            # Draw the line in each environment
            for env in self.env.envs:
                gymutil.draw_line(start_pose.p, end_pose.p, color, self.env.gym, self.env.viewer, env)

    def reset(self):
        '''
        Reset function for the environment. Returns observation of shape (num_envs, num_agents, obs_dim)'''
        obs_buff = self.env.reset()

        if getattr(self, "gate_pos", None) is None:
            self._init_extras(obs_buff)

        # make random goal position in range of x = [2,7] , y = [-2,5]
        goal_pos_x = torch.rand([self.env.num_envs, 1], device=self.env.device) * 5 + 2
        goal_pos_y = torch.rand([self.env.num_envs, 1], device=self.env.device) * 4 - 2
        self.goal_pos = torch.cat([goal_pos_x, goal_pos_y], dim=1)
        print(f'goal_pos shape: {self.goal_pos.shape}')
        print(f'goal_pos: {self.goal_pos}')
        #self.env.gym.clear_lines(self.env.viewer)
        
        #self.draw_spheres(self.goal_pos)

        base_pos = obs_buff.base_pos # position of agents- shape (num_envs, 3)
        base_rpy = obs_buff.base_rpy # roll, pitch, yaw of agents- shape (num_envs, 3)
        base_info = torch.cat([base_pos, base_rpy], dim=1).reshape([self.env.num_envs, self.num_agents, -1])
        npc_states = self.root_states_npc[:,:2]  # npc states- shape (num_envs*num_npcs, 2)
        npc_states = npc_states.view(self.env.num_envs, self.cfg.env.num_npcs, 2) # reshaping npc states to (num_envs, num_npcs, 2)
        npc_1_state = npc_states[:,0,:]
        npc_2_state = npc_states[:,1,:]
        npc_3_state = npc_states[:,2,:]
        npc_4_state = npc_states[:,3,:]
        npc_5_state = npc_states[:,4,:]
        padding = torch.zeros([self.env.num_envs, self.num_agents, 6], device=self.env.device)
        
        # obs needs to be of shape (num_envs, num_agents, obs_dim) (18 + 2*num_npcs + 1) (29)
        obs = torch.cat([self.obs_ids, base_info, torch.flip(base_info, [1]),
                         npc_1_state.unsqueeze(1), npc_2_state.unsqueeze(1), npc_3_state.unsqueeze(1),
                         npc_4_state.unsqueeze(1), npc_5_state.unsqueeze(1), padding], dim=2)
        return obs
    
    def step(self, action):
        # Clear previous lines
        #self.env.gym.clear_lines(self.env.viewer)
        
        #self.draw_spheres(self.goal_pos)
        action = torch.clip(action, -1, 1)
        obs_buf, _, termination, info = self.env.step((action * self.action_scale).reshape(-1, self.action_space.shape[0]))

        if termination.any():
            goal_pos_x = torch.rand([self.env.num_envs, 1], device=self.env.device) * 5 + 2
            goal_pos_y = torch.rand([self.env.num_envs, 1], device=self.env.device) * 4 - 2
            self.goal_pos = torch.cat([goal_pos_x, goal_pos_y], dim=1)
            #print(f'new goal_pos: {self.goal_pos}')

        
        if getattr(self, "gate_pos", None) is None:
            self._init_extras(obs_buf)


        base_pos = obs_buf.base_pos # position of agents- shape (num_envs, 3)
        base_rpy = obs_buf.base_rpy # roll, pitch, yaw of agents- shape (num_envs, 3)
        base_info = torch.cat([base_pos, base_rpy], dim=1).reshape([self.env.num_envs, self.num_agents, -1])
        npc_states = self.root_states_npc[:,:2]  # npc states- shape (num_envs*num_npcs, 2)
        npc_states = npc_states.view(self.env.num_envs, self.cfg.env.num_npcs, 2) # reshaping npc states to (num_envs, num_npcs, 2)
        npc_1_state = npc_states[:,0,:]
        npc_2_state = npc_states[:,1,:]
        npc_3_state = npc_states[:,2,:]
        npc_4_state = npc_states[:,3,:]
        npc_5_state = npc_states[:,4,:]
        padding = torch.zeros([self.env.num_envs, self.num_agents, 6], device=self.env.device)



        # obs needs to be of shape (num_envs, num_agents, obs_dim) (18 + 2*num_npcs + 1) (29)
        obs = torch.cat([self.obs_ids, base_info, torch.flip(base_info, [1]),
                         npc_1_state.unsqueeze(1), npc_2_state.unsqueeze(1), npc_3_state.unsqueeze(1),
                         npc_4_state.unsqueeze(1), npc_5_state.unsqueeze(1), padding], dim=2)
        #obs = torch.zeros([self.env.num_envs, self.num_agents, 18 + 2*self.cfg.env.num_npcs + 1], device=self.env.device)

        self.reward_buffer["step count"] += 1
        reward = torch.zeros([self.env.num_envs, self.num_agents], device=self.env.device)

        approach_angle = torch.atan2(self.goal_pos[0, 1] - base_pos[:, 1], self.goal_pos[0, 0] - base_pos[:, 0])

        if self.goal_approach_reward_scale != 0:
            distance_to_goal = torch.norm(base_pos[:, :2] - self.goal_pos, p=2, dim=1)
            if not hasattr(self, "last_distance_to_goal"):
                self.last_distance_to_goal = distance_to_goal
            goal_approach_reward = (self.last_distance_to_goal - distance_to_goal).reshape(self.num_envs, -1).sum(dim=1)
            goal_approach_reward[self.env.reset_ids] = 0
            goal_approach_reward = (goal_approach_reward * self.goal_approach_reward_scale).unsqueeze(1)
            reward += goal_approach_reward.repeat(1, self.num_agents)
            self.last_distance_to_goal = distance_to_goal
            self.reward_buffer["goal approach reward"] += torch.sum(goal_approach_reward).cpu()

        # success reward
        if self.success_reward_scale != 0:
            success_reward = torch.zeros([self.env.num_envs*self.env.num_agents], device=self.env.device)
            success_reward[distance_to_goal < 0.1] = self.success_reward_scale
            reward += success_reward.reshape([self.env.num_envs, self.env.num_agents])
            self.reward_buffer["success reward"] += torch.sum(success_reward).cpu()

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

