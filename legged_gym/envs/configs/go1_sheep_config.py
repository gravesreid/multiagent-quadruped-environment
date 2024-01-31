import numpy as np
from legged_gym.utils.helpers import merge_dict
from legged_gym.envs.go1.go1 import Go1Cfg

class Go1SheepCfg(Go1Cfg):

    class env(Go1Cfg.env):
        env_name = "go1sheep"
        num_envs = 2 # 4096
        num_agents = 2
        num_npcs = 25
        episode_length_s = 5
    
    class asset(Go1Cfg.asset):
        file_npc = "{LEGGED_GYM_ROOT_DIR}/resources/objects/sheep.urdf"
        name_npc = "sheep"
    
    class terrain(Go1Cfg.terrain):

        num_rows = 2 # 20
        num_cols = 1 # 50

        BarrierTrack_kwargs = merge_dict(Go1Cfg.terrain.BarrierTrack_kwargs, dict(
            options = [
                "init",
                "plane",
                "gate",
                "plane",
                "wall",
            ],
            randomize_obstacle_order = False,
            # wall_thickness= 0.2,
            track_width = 5.,
            # track_block_length = 2., # the x-axis distance from the env origin point
            init = dict(
                block_length = 2.0,
                room_size = (1.0, 2.0),
                border_width = 0.00,
                offset = (0, 0),
            ),
            gate = dict(
                block_length = 3.0,
                width = 1.,
                depth = 0.1, # size along the forward axis
                offset = (0, 0),
            ),
            plane = dict(
                block_length = 3.0,
            ),
            wall = dict(
                block_length = 0.1
            ),
            wall_height= 0.5,
            virtual_terrain = False, # Change this to False for real terrain
            no_perlin_threshold = 0.06,
            add_perlin_noise = False
       ))

    class command(Go1Cfg.command):

        class cfg(Go1Cfg.command.cfg):
            vel = True         # lin_vel, ang_vel

    class init_state(Go1Cfg.init_state):
        multi_init_state = True
        init_state_class = Go1Cfg.init_state
        init_states = [
            init_state_class(
                pos = [0.0, 0.0, 0.34],
                rot = [0.0, 0.0, 0.0, 1.0],
                lin_vel = [0.0, 0.0, 0.0],
                ang_vel = [0.0, 0.0, 0.0],
            ),
            init_state_class(
                pos = [0.0, 0.0, 0.34],
                rot = [0.0, 0.0, 0.0, 1.0],
                lin_vel = [0.0, 0.0, 0.0],
                ang_vel = [0.0, 0.0, 0.0],
            ),
        ]

    class termination(Go1Cfg.termination):
        # additional factors that determines whether to terminates the episode
        check_obstacle_conditioned_threshold = False
        termination_terms = [
            "roll",
            "pitch",
            "z_low",
            "z_high",
            "out_of_track",
        ]

    class domain_rand(Go1Cfg.domain_rand):
        # push_robots = True # use for virtual training
        push_robots = False # use for non-virtual training
        init_dof_pos_ratio_range = None
        init_base_pos_range = dict(
            x= [-0.1, 0.1],
            y= [-0.1, 0.1],
        )

    class rewards(Go1Cfg.rewards):
        class scales:
            pass
            # tracking_ang_vel = 0.05
            # world_vel_l2norm = -1.
            # legs_energy_substeps = -1e-5
            # alive = 2.
            # penetrate_depth = -3e-3
            # penetrate_volume = -3e-3
            # exceed_dof_pos_limits = -1e-1
            # exceed_torque_limits_i = -2e-1

    class viewer(Go1Cfg.viewer):
        pos = [0., 3., 5.]  # [m]
        lookat = [4., 3., 0.]  # [m]
