import gym
from gym import spaces
import numpy as np
import torch
from copy import copy
from mqe.envs.wrappers.empty_wrapper import EmptyWrapper
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from isaacgym import gymapi, gymutil
from isaacgym.torch_utils import *


class Go1HighLevelWrapper(EmptyWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(20 + self.num_agents,), dtype=float)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=float)
        self.action_scale = torch.tensor([[[2, 0.5, 0.5],]], device="cuda").repeat(self.num_envs, self.num_agents, 1)

        # Reward buffer
        self.reward_buffer = {
            "box movement reward": 0,
            "step count": 0,
            "target_reward": 0,
            "trajectory_reward": 0,
            "success_reward": 0,
        }

        # Simulated path that rl policy spits out
        self.target_points = torch.tensor([
            [[2, 5, 0.25], [2, 2, 0.25]],
            [[3, 5, 0.25], [3, 2, 0.25]],
            [[4, 4.5, 0.25], [4, 2.75, 0.25]],
            [[5, 4, 0.25], [5, 3.25, 0.25]],
            [[6, 4.25, 0.25], [6, 3.25, 0.25]],
            [[7, 3.5, 0.25], [7, 2, 0.25]]
        ], device=self.env.device)

        # Add buffer for the first and last array
        self.target_points = torch.cat((self.target_points[:1], self.target_points, self.target_points[-1:]), dim=0)

        # Extract control points for each path
        control_points_path1 = self.target_points[:, 0, :]
        control_points_path2 = self.target_points[:, 1, :]

        # Precompute interpolated paths
        self.interpolated_path1 = self.interpolate_catmull_rom(control_points_path1)
        self.interpolated_path2 = self.interpolate_catmull_rom(control_points_path2)

        self.agent_path = torch.empty((0, 2), device=self.env.device)

    def draw_spheres(self, target_points):
        self.env.gym.clear_lines(self.env.viewer)
        num_lines = 5  # Number of lines per sphere
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

    def interpolate_catmull_rom(self, points, num_points=100):
        def catmull_rom(p0, p1, p2, p3, t):
            """Compute a point on a Catmull-Rom spline."""
            return 0.5 * (
                (2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3
            )

        # Ensure that we have enough points by padding start and end
        if len(points) < 2:
            return points  # No interpolation needed for fewer than 2 points

        if len(points) == 2:
            # Linear interpolation if only two points are provided
            return torch.linspace(points[0], points[1], num_points, device=self.env.device)

        path_points = []
        for i in range(len(points) - 3):
            for t in torch.linspace(0, 1, num_points, device=self.env.device):
                point = catmull_rom(points[i], points[i + 1], points[i + 2], points[i + 3], t)
                path_points.append(point)

        return torch.stack(path_points)

    def draw_smooth_path(self, interpolated_path, color):
        # Draw lines between consecutive points on the interpolated path
        for i in range(len(interpolated_path) - 1):
            start_pose = gymapi.Transform()
            end_pose = gymapi.Transform()

            start_pose.p = gymapi.Vec3(*interpolated_path[i].tolist())
            end_pose.p = gymapi.Vec3(*interpolated_path[i + 1].tolist())

            for env in self.env.envs:
                gymutil.draw_line(start_pose.p, end_pose.p, color, self.env.gym, self.env.viewer, env)

    def chamfer_distance(self, A, B):
        """
        Computes the Chamfer distance between two sets of points A and B using PyTorch.
        A and B are tensors of shape [N, D] and [M, D] respectively,
        where N and M are the number of points, and D is the dimensionality (e.g., 2 for x, y or 3 for x, y, z).
        """
        # Compute pairwise distances
        A_sq = torch.sum(A ** 2, dim=1, keepdim=True)  # Shape: [N, 1]
        B_sq = torch.sum(B ** 2, dim=1, keepdim=True).T  # Shape: [1, M]
        dist_matrix = A_sq - 2 * torch.matmul(A, B.T) + B_sq  # Shape: [N, M]

        # For each point in A, find the minimum distance to a point in B
        min_dist_A_to_B = torch.min(dist_matrix, dim=1)[0]  # Shape: [N]
        min_dist_B_to_A = torch.min(dist_matrix, dim=0)[0]  # Shape: [M]

        # Average Chamfer distance
        chamfer_dist = torch.mean(min_dist_A_to_B) + torch.mean(min_dist_B_to_A)

        return chamfer_dist

    # The rest of the methods would follow the same pattern of tensor conversion where applicable.

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
        # initialize last target distance
        current_pos_tensor = obs_buf.base_pos[:, :2]  # Shape: [num_envs * num_agents, 2]
        target_points_for_envs = self.target_points[-1, 0, :2].unsqueeze(0).repeat(current_pos_tensor.shape[0], 1)
        self.last_target_distance = torch.norm(current_pos_tensor - target_points_for_envs, dim=1)
        self.last_target_distance = self.last_target_distance.view(self.env.num_envs, self.num_agents)
    

        return obs

    def step(self, action):
        # Clear previous lines
        #self.env.gym.clear_lines(self.env.viewer)
        
        #self.draw_spheres(self.target_points)
        #self.draw_smooth_path(self.interpolated_path1, gymapi.Vec3(1.0, 0.0, 0.0))  # Path 1 in magenta
        #self.draw_smooth_path(self.interpolated_path2, gymapi.Vec3(1.0, 0.0, 0.0))  # Path 2 in cyan



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
        current_pos = base_pos[:, :2]
        self.agent_path = torch.cat([self.agent_path, current_pos], dim=0)
        max_length = len(self.interpolated_path1)
        if self.agent_path.shape[0] >= max_length:
            self.agent_path = self.agent_path[-max_length:, :]
        #print(f'Base Pos: {base_pos}')
        base_rpy = obs_buf.base_rpy
        #print(f'Base RPY: {base_rpy}')
        base_info = torch.cat([base_pos, base_rpy], dim=1).reshape([self.env.num_envs, self.env.num_agents, -1])
        obs = torch.cat([self.obs_ids, base_info, torch.flip(base_info, [1]),
                         self.gate_pos, box_pos[:, :2].unsqueeze(1).repeat(1, self.num_agents, 1),
                         self.root_states_npc[:, 3:7].unsqueeze(1).repeat(1, self.num_agents, 1)], dim=2)

        self.reward_buffer["step count"] += 1

        reward = torch.zeros([self.env.num_envs, self.num_agents], device=self.env.device)



       #  Calculate trajectory reward
        if self.target_path_reward_scale != 0:
            if len(self.agent_path) >= len(self.target_points):
                interp_points = torch.cat((self.interpolated_path1[:, :2], self.interpolated_path2[:, :2]), dim=0)
                agent_path_tensor = self.agent_path.reshape(-1, 2)
                distance = self.chamfer_distance(agent_path_tensor, interp_points)
                self.reward_buffer["trajectory_reward"] += -distance.item()
                reward += -distance

        # Target reward with last distance decrease reward
        if self.target_reward_scale != 0:
            current_pos_tensor = obs_buf.base_pos[:, :2]  # Shape: [num_envs * num_agents, 2]
            target_points_for_envs = self.target_points[-1, 0, :2].unsqueeze(0).repeat(current_pos_tensor.shape[0], 1)
            current_target_distance = torch.norm(current_pos_tensor - target_points_for_envs, dim=1)
            current_target_distance = current_target_distance.view(self.env.num_envs, self.num_agents)  # Shape: [num_envs, num_agents, 1]

            # Compute change in distance
            distance_reduction = self.last_target_distance - current_target_distance  # Positive if agent moved closer

            # Apply scaling factor
            target_reward = self.target_reward_scale * distance_reduction

            # Update rewards
            reward += target_reward

            # Update last_target_distance
            self.last_target_distance = current_target_distance.detach()

            # Update reward buffer
            self.reward_buffer["target_reward"] += torch.sum(target_reward).cpu()

            # Check for success and apply success reward
        if self.success_reward_scale != 0:
            # Create a mask of agents that have reached the target
            success_mask = current_target_distance < 0.1  # Shape: [num_envs, num_agents], dtype: torch.bool

            # Apply success reward to those agents
            success_reward = self.success_reward_scale * success_mask.float()  # Convert bool to float
            reward += success_reward

            # Update reward buffer
            self.reward_buffer["success_reward"] += torch.sum(success_reward).cpu()


        # Compute target reward per agent
        #if self.target_reward_scale != 0:
        #    current_pos_tensor = current_pos[:, :2]
        #    self.target_points = self.target_points.to(self.env.device)
        #    target_points_for_envs = self.target_points[-1, 0, :2]  # Shape: [2]
        #    target_points_for_envs = target_points_for_envs.unsqueeze(0).repeat(current_pos_tensor.shape[0], 1)
        #    target_distance = torch.norm(current_pos_tensor - target_points_for_envs, dim=1)
        #    target_reward = -self.target_path_reward_scale * target_distance
        #    target_reward = target_reward.view(self.env.num_envs, self.num_agents)
        #    reward += target_reward
        #    self.reward_buffer["target_reward"] += torch.sum(target_reward).cpu()

        # Adjust box movement reward to per agent
        if self.box_x_movement_reward_scale != 0:
            if self.last_box_pos is not None:
                x_movement = (box_pos - self.last_box_pos)[:, 0]
                x_movement[self.env.reset_ids] = 0
                box_x_movement_reward = self.box_x_movement_reward_scale * x_movement
                box_x_movement_reward = box_x_movement_reward.unsqueeze(1).repeat(1, self.num_agents)
                reward += box_x_movement_reward
                self.reward_buffer["box movement reward"] += torch.sum(box_x_movement_reward).cpu()

        # Flatten reward if necessary
       # reward = reward.view(-1, 1)
        #print(f'Reward shape: {reward.shape}')

        self.last_box_pos = copy(box_pos)

        return obs, reward, termination, info