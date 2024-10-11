import numpy as np
from mqe.utils.helpers import merge_dict
from mqe.envs.go1.go1 import Go1Cfg

class Go1DoubleObjectCfg(Go1Cfg):

    class env(Go1Cfg.env):
        env_name = "go1_double_object"
        num_envs = 1
        num_agents = 2
        num_npcs = 2  # Now there are two NPCs (objects)
        episode_length_s = 15  # episode length in seconds
