import gym
from gym import spaces
import numpy as np
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


        # simulated path that rl policy spits out
        self.target_points = [np.array([[2, 4, 0.25], [2, 2, 0.25]]),
                      np.array([[3, 4, 0.25], [3, 2, 0.25]]),
                      np.array([[4, 4, 0.25], [4, 2, 0.25]]),
                      np.array([[5, 4, 0.25], [5, 2, 0.25]])]
        


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

    def step(self, action):

        target_points = self.target_points

        def draw_spheres(target_points):
            self.env.gym.clear_lines(self.env.viewer)
            num_lines = 5  # Number of lines per sphere
            line_length = 0.12  # Length of each line segment

            for points_pair in target_points:
                for point in points_pair:
                    center_pose = gymapi.Transform()
                    center_pose.p = gymapi.Vec3(point[0], point[1], point[2])
                    
                    # Draw random lines around each point to simulate a sphere
                    for _ in range(num_lines):
                        # Generate a random direction for the line
                        direction = torch.randn(3).to(self.env.device)
                        direction = direction / torch.norm(direction) * (line_length / 2)  # Normalize and scale

                        start_pose = gymapi.Transform()
                        end_pose = gymapi.Transform()

                        start_pose.p = gymapi.Vec3(
                            center_pose.p.x + direction[0].item(),
                            center_pose.p.y + direction[1].item(),
                            center_pose.p.z + direction[2].item()
                        )
                        end_pose.p = gymapi.Vec3(
                            center_pose.p.x - direction[0].item(),
                            center_pose.p.y - direction[1].item(),
                            center_pose.p.z - direction[2].item()
                        )

                        # Define the color as a Vec3 object (e.g., red)
                        color = gymapi.Vec3(1.0, 0.0, 0.0)

                        # Draw the line in each environment
                        for env in self.env.envs:
                            gymutil.draw_line(start_pose.p, end_pose.p, color, self.env.gym, self.env.viewer, env)

        # Draw spheres at target points
        draw_spheres(target_points)



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
