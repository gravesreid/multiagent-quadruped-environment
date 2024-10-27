import gym
from gym import spaces
import numpy
import torch
from copy import copy
from mqe.envs.wrappers.empty_wrapper import EmptyWrapper

from isaacgym import gymapi, gymutil
from isaacgym.torch_utils import *


class Go1HighLevelWrapper(EmptyWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(20 + self.num_agents,), dtype=float)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=float)
        self.action_scale = torch.tensor([[[2, 0.5, 0.5],],], device="cuda").repeat(self.num_envs, self.num_agents, 1)

        # for hard setting of reward scales (not recommended)
        
        self.box_x_movement_reward_scale = 1

        self.reward_buffer = {
            "box movement reward": 0,
            "step count": 0,
            "target_reward": 0,
        } 

        self.target_points = []

    def _init_extras(self, obs):

        self.gate_pos = obs.env_info["gate_deviation"]
        self.gate_pos[:, 0] += self.BarrierTrack_kwargs["init"]["block_length"] + self.BarrierTrack_kwargs["gate"]["block_length"] / 2
        self.gate_pos = self.gate_pos.unsqueeze(1).repeat(1, self.num_agents, 1)
        self.gate_distance = self.gate_pos.reshape(-1, 2)[:, 0]

    def reset(self):
        obs_buf = self.env.reset()

        if getattr(self, "gate_pos", None) is None:
            self._init_extras(obs_buf)

        box_pos = self.root_states_npc[:, :3] - self.env.env_origins
        
        base_pos = obs_buf.base_pos
        base_rpy = obs_buf.base_rpy
        base_info = torch.cat([base_pos, base_rpy], dim=1).reshape([self.env.num_envs, self.env.num_agents, -1])
        obs = torch.cat([self.obs_ids, base_info, torch.flip(base_info, [1]),
                         self.gate_pos, box_pos[:, :2].unsqueeze(1).repeat(1, self.num_agents, 1),
                         self.root_states_npc[:, 3:7].unsqueeze(1).repeat(1, self.num_agents, 1)], dim=2)

        self.last_box_pos = None

        return obs

    def step(self, action, target_points=None):

        # Define position range for x and y
        init_npc_base_pos_range = dict(
            x=[2, 6],
            y=[-2.25, 2.25],
        )

        rand_vector = torch.rand((1, 2, 3), device=self.env.device)
        rand_vector[:, :, 0] = rand_vector[:, :, 0] * (init_npc_base_pos_range['x'][1] - init_npc_base_pos_range['x'][0]) + init_npc_base_pos_range['x'][0]
        rand_vector[:, :, 1] = rand_vector[:, :, 1] * (init_npc_base_pos_range['y'][1] - init_npc_base_pos_range['y'][0]) + init_npc_base_pos_range['y'][0]
        rand_vector[:, :, 2] = 0.12
        target_points = rand_vector

        print('target_points: ', target_points)
        print('target_points shape: ', target_points.shape)


        import numpy as np

        # Draw lines between the points in rand_vector
        for i in range(rand_vector.shape[1] - 1):
            start_point = rand_vector[:, i, :]
            end_point = rand_vector[:, i + 1, :]
            start_pose = gymapi.Transform()
            end_pose = gymapi.Transform()
            start_pose.p = gymapi.Vec3(start_point[0, 0].item(), start_point[0, 1].item(), start_point[0, 2].item())
            end_pose.p = gymapi.Vec3(end_point[0, 0].item(), end_point[0, 1].item(), end_point[0, 2].item())
            
            print('start_pose: ', start_pose)
            print('startpose.p: ', start_pose.p)

            start_x = round(start_pose.p.x, 2)
            start_y = round(start_pose.p.y, 2)
            start_z = round(start_pose.p.z, 2)
            end_x = round(end_pose.p.x, 2)
            end_y = round(end_pose.p.y, 2)
            end_z = round(end_pose.p.z, 2)
            verts = np.array([[start_x, start_y, start_z],
                              [end_x, end_y, end_z]], dtype=np.float32)
            colors = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)  # RGB values for red color

            # verts[0][0] = (start_x, start_y, start_z)
            print('verts[0]: ', verts[0])
            print('verts[0][0]: ', verts[0][0])
            
            gymutil.draw_line(verts[0], verts[1], colors[0], self.env.gym, self.env.viewer, self.env.sim)
 

        

        # # Set the properties of the geometry (position and color)

        # pose = gymapi.Transform()

        # pose.p = gymapi.Vec3(x, y, z)  # Replace x, y, z with point coordinates

        

        # color = gymapi.Vec3(1, 0, 0)  # RGB values for red color

        

        # # Draw the sphere at the specified location

        # gym.draw_sphere_geom(env, sphere_geom, pose, color)





        action = torch.clip(action, -1, 1)
        #print(f'Action: {action}')
        obs_buf, _, termination, info = self.env.step((action * self.action_scale).reshape(-1, self.action_space.shape[0]))
        #print(f'Obs: {obs_buf}')
        #print(f'Termination: {termination}')
        #print(f'Info: {info}')

        box_pos = self.root_states_npc[:, :3] - self.env.env_origins
        #print(f'Box Pos: {box_pos}')
        
        if getattr(self, "gate_pos", None) is None:
            self._init_extras(obs_buf)
        
        base_pos = obs_buf.base_pos
        #print(f'Base Pos: {base_pos}')
        base_rpy = obs_buf.base_rpy
        #print(f'Base RPY: {base_rpy}')
        base_info = torch.cat([base_pos, base_rpy], dim=1).reshape([self.env.num_envs, self.env.num_agents, -1])
        obs = torch.cat([self.obs_ids, base_info, torch.flip(base_info, [1]),
                         self.gate_pos, box_pos[:, :2].unsqueeze(1).repeat(1, self.num_agents, 1),
                         self.root_states_npc[:, 3:7].unsqueeze(1).repeat(1, self.num_agents, 1)], dim=2)

        self.reward_buffer["step count"] += 1
        self.reward_buffer["target_reward"] += 0
        reward = torch.zeros([self.env.num_envs, 1], device=self.env.device)

        if self.box_x_movement_reward_scale != 0:
            if self.last_box_pos != None:
                x_movement = (box_pos - self.last_box_pos)[:, 0]
                x_movement[self.env.reset_ids] = 0
                box_x_movement_reward = self.box_x_movement_reward_scale * x_movement
                reward[:, 0] += box_x_movement_reward
                self.reward_buffer["box movement reward"] += torch.sum(box_x_movement_reward).cpu()
        
        reward = reward.repeat(1, self.num_agents)

        self.last_box_pos = copy(box_pos)

        return obs, reward, termination, info
