from configutils import Config
import numpy as np


def get_sim_config():
    cfg = Config('Sim')
    cfg.Object_num = 1
    cfg.direction_num = 16

    cfg.Mass_auxiliary = 0.04
    cfg.Mass_min = 0.1
    cfg.Mass_max = 0.2
    cfg.Mass_categories = 30

    cfg.Friction_auxiliary = 0.2
    cfg.Friction_min = 0.4
    cfg.Friction_max = 0.7
    cfg.Friction_categories = 30

    cfg.Orientation_initialization = [0.0, 1.0, 0.0, 1.0]
    cfg.Auxiliary_initialization_height = 0.14

    ##############################################################
    # CONFIGURATION of CAMERRA AND TABLE
    ##############################################################

    tt = 0.8443340571124678
    cfg.Camera_pose = np.array([[1.,    0.,     -0.,    0.5],
                                [-0.,   0.8/tt,    -0.27/tt,   -0.3],
                                [0.,    0.27/tt,    0.8/tt,    0.8],
                                [0.,    0.,     0.,     1.]])

    cfg.Camera_intrinsic = np.array([[462,      0,      640 / 2],
                                     [0,        462,    360 / 2],
                                     [0,        0,      1]])

    
    cfg.Orientation_vertical = [-1.0, 1.0, 0.0, 0.0]
    cfg.Orientation_horizontal = [-1.0, 0.0, 0.0, 0.0]

    # [X, Y] - The center of the table
    cfg.Table_center = [0.5, 0.0]

    # float - Moving speed of robot
    cfg.Fast_speed = 0.03
    cfg.Slow_speed = 0.01

    cfg.Gripper_prepare_height = 0.19

    ##############################################################
    # CONFIGURATION of COLLISION
    ##############################################################

    # [X, Y] - The intialized position of auxiliary cylinder
    cfg.Auxiliary_initialization = [[0.4, -0.5], [0.4, 0.5]]

    cfg.Gripper_grasp_height = 0.14

    # object position: [X, GRIPPER_PUT_Y, GRIPPER_PUT_HEIGHT]
    cfg.Gripper_place_y = [-0.47, 0.47]
    cfg.Gripper_place_height = 0.132

    # gripper position of collide: [X, GRIPPER_OPEN_Y, GRIPPER_OPEN_HEIGHT]
    cfg.Gripper_open_y = [-0.535, 0.535]
    cfg.Gripper_open_height = 0.14

    # [[MIN_X, MAX_X], [MIN_Y, MAX_Y]] - The region of collision
    cfg.Collide_region = [[0.35, 0.65], [-0.15, 0.15]]

    ##############################################################
    # CONFIGURATION of PUSH
    ##############################################################

    cfg.Push_height = 0.025

    cfg.Push_speed = [0.05, 0.045, 0.04, 0.03]

    return cfg