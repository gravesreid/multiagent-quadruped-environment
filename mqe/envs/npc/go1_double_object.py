from mqe import LEGGED_GYM_ROOT_DIR
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
import numpy as np
from mqe.envs.go1.go1 import Go1

class Go1DoubleObject(Go1):

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):

        self.npc_collision = getattr(cfg.asset, "npc_collision", True)
        self.fix_npc_base_link = getattr(cfg.asset, "fix_npc_base_link", False)
        self.npc_gravity = getattr(cfg.asset, "npc_gravity", True)
        
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

  