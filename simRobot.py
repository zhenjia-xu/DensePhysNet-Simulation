#!/usr/bin/env python3
#########################################
####### Written by: Shuran Song #########
#########################################

import sys
sys.path.append('..')
import time
import numpy as np
import threading
import pybullet as p
import pybullet_data
import os
import robotutils as utils


# old_step = p.stepSimulation
# import threading
# lock = threading.Lock()
# def new_step(*args, **kwargs):
#     with lock:
#         return old_step(*args, **kwargs)
# p.stepSimulation = new_step


# --------------------------------------------------
# Setup UR5 robot in simulation
class SimUR5:

    def __init__(self, center, prepare_height, real_time_sim=True, GUI=True):
        self.center = center
        self.prepare_height = prepare_height

        self.__initialized = False
        # # Start PyBullet simulation environment
        if GUI:
            self.__physics_client = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
        else:
            self.__physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        p.setGravity(0, 0, -9.8)

        self.__client_id = p.connect(p.SHARED_MEMORY, key=p.SHARED_MEMORY_KEY)

        self.__plane_id = p.loadURDF("plane.urdf")

        self.__initialized = False
        self.__shutdown_requested = False
        self.__real_time_sim = real_time_sim
        if self.__real_time_sim:
            p.setRealTimeSimulation(1) # does not work with p.DIRECT

        # Add UR5 robot to simulation environment
        robot_pose_trans = [0, 0, 0]
        robot_pose_rot = p.getQuaternionFromEuler([0, 0, np.pi])
        self.__robot_body_id = p.loadURDF(os.path.join("sim_pybullet/descriptions/ur5/ur5.urdf"),
                                          robot_pose_trans, robot_pose_rot)
        self.__robot_tool_idx = 9
        self.__joint_epsilon = 0.000000001  # Joint position threshold for blocking calls (i.e. move until joint difference < epsilon)

        # Get revolute joint indices of robot (skip fixed joints)
        robot_joint_info = [p.getJointInfo(self.__robot_body_id, x) for x in
                            range(p.getNumJoints(self.__robot_body_id))]
        self.__robot_joint_indices = [x[0] for x in robot_joint_info if x[2] == p.JOINT_REVOLUTE]

        self.__robot_jointLowerLimit = [p.getJointInfo(self.__robot_body_id, x)[8] for x in self.__robot_joint_indices]
        self.__robot_jointUpperLimit = [p.getJointInfo(self.__robot_body_id, x)[9] for x in self.__robot_joint_indices]
        self.__robot_jointRanges = [self.__robot_jointUpperLimit[x] - self.__robot_jointLowerLimit[x] for x in
                                    range(len(self.__robot_jointLowerLimit))]

        # Add gripper to UR5 robot
        self.__gripper_body_id = p.loadURDF("sim_pybullet/descriptions/gripper/robotiq_2f_85.urdf")
        p.resetBasePositionAndOrientation(self.__gripper_body_id,[0.5,0.1,0.2],p.getQuaternionFromEuler([0,0,0]))
        self.__robot_tool_joint_idx = 9
        self.__robot_tool_offset = [0,0,0.10]
        p.createConstraint(self.__robot_body_id,self.__robot_tool_joint_idx,self.__gripper_body_id,0,jointType=p.JOINT_FIXED,jointAxis=[0,0,0],parentFramePosition=[0,0,0],childFramePosition=self.__robot_tool_offset,childFrameOrientation=p.getQuaternionFromEuler([0,0,np.pi/2]))

        # Get inner finger joint indices of gripper (skip fixed joints and outer finger joints)
        self.__gripper_joint_indices = [3,4,5,6] # Only inner fingers should be actuated
        self.__gripper_jointLowerLimit = [p.getJointInfo(self.__gripper_body_id,x)[8] for x in self.__gripper_joint_indices]
        self.__gripper_jointUpperLimit = [p.getJointInfo(self.__gripper_body_id,x)[9] for x in self.__gripper_joint_indices]

        # Set friction coefficients for gripper fingers
        for i in range(p.getNumJoints(self.__gripper_body_id)):
            p.changeDynamics(self.__gripper_body_id,i,lateralFriction=1.0,spinningFriction=1.0,rollingFriction=0.001,frictionAnchor=True) #,contactStiffness=0.0,contactDamping=0.0) 


        # Move robot to home configuration
        # self.__robot_home_config = [np.pi,-np.pi/2,np.pi/2,-np.pi/2,-np.pi/2,0]
        self.__robot_home_config = [0.252238382702895, -2.6382404165678888, 1.9867945050793192, -0.919691244323651, -1.5734818198621676, 0.2520320827996964] #[0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0]
        self.move_joints(self.__robot_home_config, blocking=True, speed=1.0)
        self.open_gripper()
        p.stepSimulation()

        # Start thread to handle additional gripper constraints (gripper joint mimic behavior)
        if self.__real_time_sim:
            self.__gripper_motor_joint_idx = 1 # set one joint as the open/close motor joint (other joints should mimic)
            self.__constraints_thread = threading.Thread(target=self.step_constraints)
            self.__constraints_thread.daemon = True
            self.__constraints_thread.start()

        # Done with initialization
        self.__initialized = True

    def step_constraints(self):
        while True:
            gripper_joint_positions = np.array([p.getJointState(self.__gripper_body_id,i)[0] for i in range(p.getNumJoints(self.__gripper_body_id))])
            p.setJointMotorControlArray(self.__gripper_body_id,[6,3,8,5,10],p.POSITION_CONTROL,[gripper_joint_positions[1],-gripper_joint_positions[1],-gripper_joint_positions[1],gripper_joint_positions[1],gripper_joint_positions[1]],positionGains=np.ones(5))
            time.sleep(0.0001)

    # Use position control to enforce hard contraints on gripper behavior (warning: is hacky)
    def update_gripper(self):
        gripper_joint_positions = np.array([p.getJointState(self.__gripper_body_id,i)[0] for i in range(p.getNumJoints(self.__gripper_body_id))])
        p.setJointMotorControlArray(self.__gripper_body_id,[6,3,8,5,10],p.POSITION_CONTROL,[gripper_joint_positions[1],-gripper_joint_positions[1],-gripper_joint_positions[1],gripper_joint_positions[1],gripper_joint_positions[1]],positionGains=np.ones(5))

    # Move robot tool to specified pose
    def move_tool(self, position, orientation, blocking=True, speed=0.03):
        # Use IK to compute target joint configuration
        target_joint_config = p.calculateInverseKinematics(self.__robot_body_id, self.__robot_tool_idx, position,
                                                           orientation,
                                                           lowerLimits=self.__robot_jointLowerLimit,
                                                           upperLimits=self.__robot_jointUpperLimit,
                                                           jointRanges=self.__robot_jointRanges,
                                                           restPoses=self.__robot_home_config)
        # Move joints
        self.move_joints(target_joint_config, blocking, speed)

    # Move robot arm to specified joint configuration
    def move_joints(self, target_joint_config, blocking=True, speed=0.03):
        # Move joints
        p.setJointMotorControlArray(self.__robot_body_id, self.__robot_joint_indices,
                                    p.POSITION_CONTROL, target_joint_config,
                                    positionGains=speed * np.ones(len(self.__robot_joint_indices)))

        # Block call until joints move to target configuration
        if blocking:
            actual_joint_config = [p.getJointState(self.__robot_body_id, x)[0] for x in self.__robot_joint_indices]
            cnt = 0
            while cnt < 300 and not all(
                    [np.abs(actual_joint_config[i] - target_joint_config[i]) < self.__joint_epsilon for i in range(6)]):
                self.update_gripper()
                p.stepSimulation()
                actual_joint_config = [p.getJointState(self.__robot_body_id, x)[0] for x in self.__robot_joint_indices]
                cnt = cnt + 1
                time.sleep(0.0001)
            # if cnt == 300:
            #     raise NotImplementedError()

    # Close gripper via velocity control
    def close_gripper(self, blocking=True):
        p.setJointMotorControl2(self.__gripper_body_id,1,p.VELOCITY_CONTROL,targetVelocity=1,force=100)

        gripper_close_config = [-0.831506050919347, 0.0, 0.8326909014019432, 0.8346822830158388]
        p.setJointMotorControlArray(self.__gripper_body_id, self.__gripper_joint_indices, p.POSITION_CONTROL,
                                    gripper_close_config)
        self.update_gripper()
        # Block call until gripper joints move to target configuration
        if blocking:
            reach_target = False
            in_limit_lower = True
            cnt = 0
            while cnt < 300 and not reach_target : #and in_limit_lower:
                self.update_gripper()
                p.stepSimulation()

                actual_joint_config = [p.getJointState(self.__gripper_body_id, x)[0] for x in
                                       self.__gripper_joint_indices]
                reach_target = all(
                    [np.abs(actual_joint_config[i] - gripper_close_config[i]) < self.__joint_epsilon for i in range(4)])
                in_limit_lower = all([actual_joint_config[i] - self.__gripper_jointLowerLimit[i] > 0 for i in range(4)])
                cnt = cnt + 1
                

    # Open gripper via position control
    def open_gripper(self, blocking=True):
        p.setJointMotorControl2(self.__gripper_body_id,1,p.VELOCITY_CONTROL,targetVelocity=-1,force=100)

        gripper_open_config = [0, 0, 0, 0]
        p.setJointMotorControlArray(self.__gripper_body_id, self.__gripper_joint_indices, p.POSITION_CONTROL,
                                    gripper_open_config,
                                    positionGains=0.03 * np.ones(len(self.__gripper_joint_indices)))

        # Block call until gripper joints move to target configuration
        if blocking:
            cnt = 0
            reach_target = False
            in_limit_upper = True
            in_limit_lower = True
            while cnt < 300 and not reach_target:
                self.update_gripper()
                p.stepSimulation()

                actual_joint_config = [p.getJointState(self.__gripper_body_id, x)[0] for x in
                                       self.__gripper_joint_indices]
                reach_target = all(
                    [np.abs(actual_joint_config[i] - gripper_open_config[i]) < self.__joint_epsilon for i in range(4)])
                cnt = cnt + 1

    # -----------------High Level primitives---------------------------------
    def primitive_gohome(self, speed=0.03):
        self.move_tool(position=self.center + [self.prepare_height],
                       orientation=[-1.0, 1.0, 0.0, 0.0], blocking=True, speed=0.1)
        self.move_tool(position=[0, 0.1, 0.3], orientation=[-1.0, 1.0, 0.0, 0.0], blocking=True, speed=0.1)

   
    # modified by xzj:
    def primitive_push_tilt(self, position, rotation_angle, speed=0.01):
        push_orientation = [1.0, 0.0]
        # tool_rotation_angle = rotation_angle/2
        # tool_orientation = np.asarray([push_orientation[0]*np.cos(tool_rotation_angle) - push_orientation[1]*np.sin(tool_rotation_angle), push_orientation[0]*np.sin(tool_rotation_angle) + push_orientation[1]*np.cos(tool_rotation_angle), 0.0])*np.pi
        # tool_orientation_angle = np.linalg.norm(tool_orientation)
        # tool_orientation_axis = tool_orientation/tool_orientation_angle
        # tmp = np.array([tool_orientation_angle, tool_orientation_axis[0],tool_orientation_axis[1],tool_orientation_axis[2]])
        # tool_orientation_rotm = utils.angle2rotm(tmp)

        # tool_rotation_angle = rotation_angle + np.pi / 2
        tool_rotation_angle = rotation_angle
        tool_o_m = np.dot(utils.euler2rotm([0, 0, tool_rotation_angle]),
                          np.array(p.getMatrixFromQuaternion([-1, 1, 0, 0])).reshape(3, 3))
        tool_o_q = p.getQuaternionFromEuler(utils.rotm2euler(tool_o_m).tolist())

        # Compute push direction and endpoint
        push_direction = np.asarray(
            [push_orientation[0] * np.cos(rotation_angle) - push_orientation[1] * np.sin(rotation_angle),
             push_orientation[0] * np.sin(rotation_angle) + push_orientation[1] * np.cos(rotation_angle), 0.0])
        distance = 0.1
        target_x = position[0] + push_direction[0] * distance
        target_y = position[1] + push_direction[1] * distance
        position_end = np.asarray([target_x, target_y, position[2]])
        push_direction.shape = (3, 1)

        # Compute tilted tool orientation during push
        tilt_axis = np.dot(utils.euler2rotm(np.asarray([0, 0, np.pi / 2]))[:3, :3], push_direction)
        tilt_rotm = utils.angle2rotm([-np.pi / 8, tilt_axis[0], tilt_axis[1], tilt_axis[2]])
        tilted_tool_orientation_rotm = np.dot(tilt_rotm, tool_o_m)
        tool_t_q = p.getQuaternionFromEuler(utils.rotm2euler(tilted_tool_orientation_rotm).tolist())

        # Attempt push
        self.move_tool(position=position[:2] + [self.prepare_height],
                       orientation=tool_o_q, blocking=True, speed=0.1)

        self.move_tool(position, orientation=tool_o_q, blocking=True, speed=0.1)
        self.move_tool(position_end, orientation=tool_t_q, blocking=True, speed=speed)

        self.move_tool(position=position[:2] + [self.prepare_height],
                       orientation=tool_o_q, blocking=True, speed=0.1)

    
    # modified by xzj
    def primitive_topdown_grasp(self, position, rotation_angle):
        
        tool_o_m = np.dot(utils.euler2rotm([0, 0, rotation_angle]),
                          np.array(p.getMatrixFromQuaternion([-1, 1, 0, 0])).reshape(3, 3))
        tool_o_q = p.getQuaternionFromEuler(utils.rotm2euler(tool_o_m).tolist())

        # Approach target location +0.3
        self.move_tool(position=self.center + [self.prepare_height],
                       orientation=tool_o_q, blocking=True, speed=0.1)
        self.move_tool(position=position[:2] + [self.prepare_height],
                       orientation=tool_o_q, blocking=True, speed=0.1)

        self.open_gripper(blocking=True)
        self.move_tool(position, orientation=tool_o_q, blocking=True)
        self.close_gripper(blocking=True)

        #Lift object up 10 cm and check if something has been grasped
        self.move_tool(position=[position[0],position[1],position[2]+ 0.1],
                       orientation=tool_o_q, blocking=True)
        tmp_grasp_success = True#self.check_grasp()
        return tmp_grasp_success



    def primitive_place(self, position, rotation_angle, speed=0.01):
        tool_o_m = np.dot(utils.euler2rotm([0, 0, rotation_angle]),
                          np.array(p.getMatrixFromQuaternion([-1, 1, 0, 0])).reshape(3, 3))
        tool_o_q = p.getQuaternionFromEuler(utils.rotm2euler(tool_o_m).tolist())

        self.move_tool(position=position[:2] + [self.prepare_height],
                       orientation=tool_o_q,
                       speed=speed, blocking=True)

        self.move_tool(position, tool_o_q, blocking=True, speed=speed)
        self.open_gripper(blocking=True)

    def primitive_touch(self, position, rotation_angle):
        tool_o_m = np.dot(utils.euler2rotm([0, 0, rotation_angle]),
                          np.array(p.getMatrixFromQuaternion([-1, 1, 0, 0])).reshape(3, 3))
        tool_o_q = p.getQuaternionFromEuler(utils.rotm2euler(tool_o_m).tolist())

        self.move_tool(position=position[:2] + [self.prepare_height],
                       orientation=tool_o_q, blocking=True)
        self.close_gripper(blocking=True)
        self.move_tool(position, tool_o_q, blocking=True)
        self.open_gripper(blocking=True)
        self.close_gripper(blocking=True)

    ############################################################################################
    def get_gripper_camera(self, imgSize):
        #
        camera_angle = [0, np.pi / 12, 0]
        camera_relative_loc = [-0.1, -0.15]

        girperLocation = p.getLinkState(self.__gripper_body_id, 0)
        camera_t_R = p.getMatrixFromQuaternion(p.getQuaternionFromEuler(camera_angle))
        maxtrix_gri = p.getMatrixFromQuaternion(girperLocation[1])
        maxtrix_cam = np.array(maxtrix_gri) * np.array(camera_t_R)

        camlookat = [maxtrix_cam[2] + girperLocation[0][0], maxtrix_cam[5] + girperLocation[0][1],
                     maxtrix_cam[8] + girperLocation[0][2]]
        camup = [maxtrix_cam[1], maxtrix_cam[4], maxtrix_cam[7]]

        girpper_up = [maxtrix_gri[1], maxtrix_gri[4], maxtrix_gri[7]]
        gripper_fwd = [maxtrix_gri[2], maxtrix_gri[5], maxtrix_gri[8]]
        camera_loc = np.array(girperLocation[0]) + np.array(girpper_up) * camera_relative_loc[0] + np.array(
            gripper_fwd) * camera_relative_loc[1]

        view_matrix = p.computeViewMatrix(camera_loc.tolist(), camlookat, camup)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=1, nearVal=0.01, farVal=10.0)

        img_arr = p.getCameraImage(imgSize, imgSize, view_matrix, proj_matrix,
                                   shadow=1, flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)
        return img_arr