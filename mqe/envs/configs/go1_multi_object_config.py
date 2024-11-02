from mqe.envs.go1.go1 import Go1Cfg
from mqe.utils.helpers import merge_dict

class MultiObjectConfig(Go1Cfg):
    class env(Go1Cfg.env):
        env_name = "go1multiobject"
        num_envs = 1
        num_agents = 1
        num_npcs = 5
        episode_length_s = 10

    class assets(Go1Cfg.asset):
        npc_assets = [
            dict(
                file_npc="{LEGGED_GYM_ROOT_DIR}/resources/objects/urdf_models/models/plastic_banana/model.urdf",
                name_npc="banana",
                npc_collision=False,
                fix_npc_base_link=False,
                npc_gravity=True
            ),
            dict(
                file_npc="{LEGGED_GYM_ROOT_DIR}/resources/objects/urdf_models/models/scissors/model.urdf",
                name_npc="sheep",
                npc_collision=False,
                fix_npc_base_link=False,
                npc_gravity=True
            ),
            dict(
                file_npc="{LEGGED_GYM_ROOT_DIR}/resources/objects/urdf_models/models/green_cup/model.urdf",
                name_npc="box",
                npc_collision=False,
                fix_npc_base_link=False,
                npc_gravity=True
            ),
            dict(
                file_npc="{LEGGED_GYM_ROOT_DIR}/resources/objects/table.urdf",
                name_npc="table",
                npc_collision=False,
                fix_npc_base_link=False,
                npc_gravity=True
            ),
            dict(
                file_npc="{LEGGED_GYM_ROOT_DIR}/resources/objects/urdf_models/models/knife/model.urdf",
                name_npc="lego",
                npc_collision=False,
                fix_npc_base_link=False,
                npc_gravity=True
            )
            # Add more NPC asset configurations here
        ]

    class terrain(Go1Cfg.terrain):
        num_rows = 1
        num_cols = 1

        BarrierTrack_kwargs = merge_dict(Go1Cfg.terrain.BarrierTrack_kwargs, dict(
            options = [
                "init",
                "gate",
                "wall",
            ],
            # wall_thickness= 0.2,
            track_width = 5.0,
            init = dict(
                block_length = 2.0,
                room_size = (1.0, 2.5),
                border_width = 0.0,
                offset = (0, 0),
            ),
            gate = dict(
                block_length = 5.0,
                width = 5,
                depth = 0.1, # size along the forward axis
                offset = (0, 0),
                random = (0, 0),
            ),
            wall = dict(
                block_length = 0.1
            ),
            wall_height= 0.5,
            virtual_terrain = False, # Change this to False for real terrain
            no_perlin_threshold = 0.06,
            add_perlin_noise = False,
       ))
    
    class command(Go1Cfg.command):

        class cfg(Go1Cfg.command.cfg):
            vel = True         # lin_vel, ang_vel

    class init_state(Go1Cfg.init_state):
        multi_init_state = True
        init_state_class = Go1Cfg.init_state
        init_states = [
            init_state_class(
                pos = [0.0, 0.0, 0.42],
                rot = [0.0, 0.0, 0.0, 1.0],
                lin_vel = [0.0, 0.0, 0.0],
                ang_vel = [0.0, 0.0, 0.0],
            ),
        ]
        class init_state_class:
            def __init__(self, pos, rot, lin_vel, ang_vel):
                self.pos = pos
                self.rot = rot
                self.lin_vel = lin_vel
                self.ang_vel = ang_vel

        # Define init_states_npc as a class attribute
        init_states_npc = [
            init_state_class(
                pos=[5, .5, 0.15],
                rot=[0.0, 0.0, 0.0, 1.0],
                lin_vel=[0.0, 0.0, 0.0],
                ang_vel=[0.0, 0.0, 0.0],
            ),
            init_state_class(
                pos=[6, -.5, 0.15],
                rot=[0.0, 0.0, 0.0, 1.0],
                lin_vel=[0.0, 0.0, 0.0],
                ang_vel=[0.0, 0.0, 0.0],
            ),
            init_state_class(
                pos=[4, -1.5, 0.15],
                rot=[0.0, 0.0, 0.0, 1.0],
                lin_vel=[0.0, 0.0, 0.0],
                ang_vel=[0.0, 0.0, 0.0],
            ),
            init_state_class(
                pos=[3, -1, 0.15],
                rot=[0.0, 0.0, 0.0, 1.0],
                lin_vel=[0.0, 0.0, 0.0],
                ang_vel=[0.0, 0.0, 0.0],
            ),
            init_state_class(
                pos=[2, 1, 0.15],
                rot=[0.0, 0.0, 0.0, 1.0],
                lin_vel=[0.0, 0.0, 0.0],
                ang_vel=[0.0, 0.0, 0.0],
            ),
        ]

    class control(Go1Cfg.control):
        control_type = 'C'

    class termination(Go1Cfg.termination):
        # additional factors that determines whether to terminates the episode
        check_obstacle_conditioned_threshold = False
        termination_terms = [
            "roll",
            "pitch",
        ]

    class domain_rand(Go1Cfg.domain_rand):
        # push_robots = True # use for virtual training
        push_robots = False # use for non-virtual training
        init_base_pos_range = dict(
            x= [-0.1, 0.1],
            y= [-0.1, 0.1],
        )
        init_npc_base_pos_range = dict(
            x= [-.1, .1],
            y= [-.1, .1],
        )

    class rewards(Go1Cfg.rewards):
        class scales:
            #box_x_movement_reward_scale = 10
            goal_approach_reward_scale = 10
            contact_punishment_scale = -2
            angle_punishment_scale = -0.1
            success_reward_scale = 50
            # tracking_ang_vel = 0.05
            # world_vel_l2norm = -1.
            # legs_energy_substeps = -1e-5
            # alive = 2.
            # penetrate_depth = -3e-3
            # penetrate_volume = -3e-3
            # exceed_dof_pos_limits = -1e-1
            # exceed_torque_limits_i = -2e-1

    class viewer(Go1Cfg.viewer):
        pos = [0., 6., 5.]  # [m]
        lookat = [4., 6., 0.]  # [m]
        