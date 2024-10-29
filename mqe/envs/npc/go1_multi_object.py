from mqe import LEGGED_GYM_ROOT_DIR
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch

from mqe import LEGGED_GYM_ROOT_DIR

from mqe.envs.go1.go1 import Go1
class MultiObjectGo1Object(Go1):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        # Remove this line
        # self.npc_collision = getattr(cfg.asset, "npc_collision", False)
        
        self.num_npcs = len(cfg.assets.npc_assets)

    def _step_npc(self):
        return

    def _prepare_npc(self):
        self.init_state_npc = getattr(self.cfg.init_state, "init_states_npc")
        self.asset_npcs = []  # List to hold different NPC assets and their configurations

        for npc_asset_cfg in self.cfg.assets.npc_assets:
            asset_path_npc = npc_asset_cfg['file_npc'].format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
            asset_root_npc = os.path.dirname(asset_path_npc)
            asset_file_npc = os.path.basename(asset_path_npc)
            asset_options_npc = gymapi.AssetOptions()
            asset_options_npc.fix_base_link = npc_asset_cfg['fix_npc_base_link']
            asset_options_npc.disable_gravity = not npc_asset_cfg['npc_gravity']
            asset_npc = self.gym.load_asset(self.sim, asset_root_npc, asset_file_npc, asset_options_npc)
            # Store the entire NPC configuration along with the loaded asset
            npc_cfg = npc_asset_cfg.copy()
            npc_cfg['asset_npc'] = asset_npc
            self.asset_npcs.append(npc_cfg)

        # Initialize NPC states
        self.base_init_state_npc = torch.stack([
            to_torch(
                init_state_npc.pos + init_state_npc.rot + init_state_npc.lin_vel + init_state_npc.ang_vel,
                device=self.device,
                requires_grad=False
            )
            for init_state_npc in self.init_state_npc
        ], dim=0).repeat(self.num_envs, 1)

    def _create_npc(self, env_handle, env_id):
        npc_handles = []

        for i, npc_cfg in enumerate(self.asset_npcs):
            asset_npc = npc_cfg['asset_npc']
            name_npc = npc_cfg['name_npc']
            npc_collision = npc_cfg.get('npc_collision', False)
            # Set the start pose for each NPC
            init_state_npc = self.init_state_npc[i]
            start_pose_npc = gymapi.Transform()
            start_pose_npc.p = gymapi.Vec3(*init_state_npc.pos)
            start_pose_npc.r = gymapi.Quat(*init_state_npc.rot)

            # Create the NPC actor
            npc_handle = self.gym.create_actor(
                env_handle, asset_npc, start_pose_npc, name_npc, env_id, not npc_collision, 0
            )
            npc_handles.append(npc_handle)

        return npc_handles

