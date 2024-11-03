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

        # Define the observation space dimension
        # base_info (6) + flipped base_info (6) + target_point (3) = 15
        self.observation_space = spaces.Box(
            low=-float('inf'), high=float('inf'),
            shape=(15,),
            dtype=float
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=float)
        self.action_scale = torch.tensor([[[2, 0.5, 0.5]]], device=self.env.device).repeat(self.num_envs, self.num_agents, 1)

        # Reward buffer
        self.reward_buffer = {
            "target approach reward": 0,
            "success reward": 0,
            "orientation reward": 0,
            "step count": 0
        }

        # Define target points
        self.target_points = torch.stack([
            torch.tensor([[np.random.uniform(2, 7), np.random.uniform(3, 5), 0.25]], device=self.env.device)
            for _ in range(6)
        ])
        self.current_target_index = 0
        self.target_point = self.target_points[self.current_target_index]

        # Initialize last_distance_to_target
        self.last_distance_to_target = None

        # Reward scale attributes
        self.target_approach_reward_scale = 1.0
        self.success_reward_scale = 10.0
        self.orientation_reward_scale = 0.5  # Adjust this value as needed

    def _init_extras(self, obs):
        pass

    def draw_spheres(self, target_points):
        self.env.gym.clear_lines(self.env.viewer)
        num_lines = 20  # Increased number of lines for better visualization
        line_length = 0.12  # Length of each line segment

        for points_pair in target_points:
            for point in points_pair:
                center_pose = gymapi.Transform()
                center_pose.p = gymapi.Vec3(point[0].item(), point[1].item(), point[2].item())

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

    def reset(self):
        obs_buf = self.env.reset()

        if getattr(self, "gate_pos", None) is None:
            self._init_extras(obs_buf)

        base_pos = obs_buf.base_pos
        base_rpy = obs_buf.base_rpy
        base_info = torch.cat([base_pos, base_rpy], dim=1).reshape([self.env.num_envs, self.num_agents, -1])[:, :2, :]

        # Include target point in observations
        target_point_obs = self.target_point.unsqueeze(0).repeat(self.env.num_envs, self.num_agents, 1)

        obs = torch.cat([base_info, torch.flip(base_info, [1]), target_point_obs], dim=2)

        # Initialize last_distance_to_target
        self.last_distance_to_target = torch.norm(
            base_pos[:, :2] - self.target_point[:, :2], p=2, dim=1
        ).detach()

        return obs

    def step(self, action):
        self.env.gym.clear_lines(self.env.viewer)
        self.draw_spheres([self.target_point])

        self.reward_buffer["step count"] += 1
        action = torch.clip(action, -1, 1)
        obs_buf, _, termination, info = self.env.step(
            (action * self.action_scale).reshape(-1, self.action_space.shape[0])
        )

        base_pos = obs_buf.base_pos  # Agent's position
        base_rpy = obs_buf.base_rpy  # Agent's orientation (roll, pitch, yaw)

        # Compute distance to target
        distance_to_target = torch.norm(
            base_pos[:, :2] - self.target_point[:, :2], p=2, dim=1
        )

        reward = torch.zeros([self.env.num_envs, self.num_agents], device=self.env.device)

        # Reward for moving closer to the target
        distance_reduction = self.last_distance_to_target - distance_to_target
        approach_reward = distance_reduction * self.target_approach_reward_scale
        reward += approach_reward.unsqueeze(1)
        self.reward_buffer["target approach reward"] += torch.sum(approach_reward).cpu()
        self.last_distance_to_target = distance_to_target.detach()

        # **Compute orientation reward**
        # Direction vector from agent to target
        target_direction = self.target_point[:, :2] - base_pos[:, :2]  # Shape: (num_envs, 2)
        target_direction_norm = target_direction / torch.norm(target_direction, p=2, dim=1, keepdim=True)  # Normalize

        # Agent's heading vector
        agent_yaw = base_rpy[:, 2]  # Yaw angle
        agent_heading = torch.stack([torch.cos(agent_yaw), torch.sin(agent_yaw)], dim=1)  # Shape: (num_envs, 2)

        # Cosine of the angle between agent's heading and target direction
        cos_theta = torch.sum(agent_heading * target_direction_norm, dim=1)
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

        # Angle between agent's heading and target direction
        angle = torch.acos(cos_theta)  # Shape: (num_envs,)

        # Orientation reward (1 when facing target, 0 when facing away)
        orientation_reward = (torch.pi - angle) / torch.pi  # Normalize to [0, 1]
        orientation_reward = orientation_reward * self.orientation_reward_scale

        reward += orientation_reward.unsqueeze(1)
        self.reward_buffer["orientation reward"] += torch.sum(orientation_reward).cpu()

        # Check if agent is close enough to switch to the next target
        reached_target = distance_to_target < 0.5
        if reached_target.any():
            self.current_target_index += 1
            if self.current_target_index < len(self.target_points):
                self.target_point = self.target_points[self.current_target_index]
                self.last_distance_to_target = torch.norm(
                    base_pos[:, :2] - self.target_point[:, :2], p=2, dim=1
                ).detach()
            else:
                success_reward = self.success_reward_scale * reached_target.float()
                reward += success_reward.unsqueeze(1)
                self.reward_buffer["success reward"] += torch.sum(success_reward).cpu()

        # Process obs_buf into obs
        if getattr(self, "gate_pos", None) is None:
            self._init_extras(obs_buf)

        # Extract and process observations
        base_pos = obs_buf.base_pos
        base_rpy = obs_buf.base_rpy
        base_info = torch.cat([base_pos, base_rpy], dim=1).reshape(
            [self.env.num_envs, self.num_agents, -1]
        )[:, :2, :]

        # Include target point in observations
        target_point_obs = self.target_point.unsqueeze(0).repeat(self.env.num_envs, self.num_agents, 1)

        obs = torch.cat(
            [base_info, torch.flip(base_info, [1]), target_point_obs], dim=2
        )

        return obs, reward, termination, info
