import argparse
import numpy as np
import math
import random
import pybullet as p
from simRobot import SimUR5
from simCamera import simCam
from scene_env import Sim_Scene
from robotutils import get_mask, get_pos, depth_smooth, invRt, get_dist, get_mask_real
from data_logger import DataLogger
import os
from config import get_sim_config
from shapely.geometry import Point, Polygon
from sklearn.externals import joblib
import time

def mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def invRt(Rt):
    # RtInv = [Rt(:,1:3)'  -Rt(:,1:3)'* Rt(:,4)];
    invR = Rt[0:3, 0:3].T
    invT = -1 * np.dot(invR, Rt[0:3, 3])
    invT.shape = (3, 1)
    RtInv = np.concatenate((invR, invT), axis=1)
    RtInv = np.concatenate((RtInv, np.array([0, 0, 0, 1]).reshape(1, 4)), axis=0)
    return RtInv


def get_transform():
    camera_pose = np.array([[1., 0., -0., 0.5],
                            [-0., 0.8, -0.27, -0.3],
                            [0., 0.27, 0.8, 0.8],
                            [0., 0., 0., 1.]])
    view_matrix = invRt(camera_pose).T
    view_matrix = tuple(view_matrix.reshape(16))
    cam_wrd2cam = np.array(view_matrix).reshape(4, 4).T
    fov_w = 34.7 * 2
    imw = 640
    imh = 360
    fov_w = fov_w / 180 * math.pi
    f = imw / 2 / math.tan(fov_w / 2)
    camera_intrisic = np.array([[f, 0, imw / 2],
                                [0, f, imh / 2],
                                [0, 0, 1]])
    return cam_wrd2cam, camera_intrisic


def tranform_points(pts, trasform):
    # pts = [3xN] array
    # trasform: [3x4]
    pts_t = np.dot(trasform[0:3, 0:3], pts) + np.tile(trasform[0:3, 3:], (1, pts.shape[1]))
    return pts_t


def prejct_pts_to_2d(pts, RT_wrd2cam, camera_intrisic):
    # RT_wrd2cam = invRt(RT_cam2wrd)
    pts_c = tranform_points(pts, RT_wrd2cam[0:3, :])

    # rot_algix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0]])
    # pts_c = tranform_points(pts_c, rot_algix)

    coord_2d = np.dot(camera_intrisic, pts_c)
    coord_2d[0:2, :] = coord_2d[0:2, :] / np.tile(coord_2d[2, :], (2, 1))
    coord_2d[2, :] = pts_c[2, :]
    return coord_2d


###########

def get_random(min=-0.1, max=0.1):
    return random.random() * (max - min) + min


def get_sign(x):
    return 1 if x > 0 else -1


def get_onehot(x, length):
    arr = [0] * length
    arr[x] = 1
    return arr


def in_region(pos, region):
    # pos: [X, Y, Z] or [X, Y]
    # region: [[MIN_X, MAX_X], [MIN_Y, MAX_Y]]
    return pos[0] >= region[0][0] and pos[0] <= region[0][1] and \
           pos[1] >= region[1][0] and pos[1] <= region[1][1]


def check_pos(pos1, pos2, lim=0.07):
    dis = np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)
    if dis < 0.1:
        return False
    if np.abs(pos1[0] - pos2[0]) < lim:
        return False
    return True


def get_pos_list(obj_num):
    while True:
        # Here is the boundary of workspace
        pos_list = [[get_random(0.36, 0.64), get_random(-0.13, 0.13), 0.04] for _ in range(obj_num)]
        flag = True
        for i in range(obj_num):
            for j in range(i + 1, obj_num):
                flag &= check_pos(pos_list[i], pos_list[j])
        if flag:
            return pos_list


def can_collide(pos_list, id, collide_region, dir):
    if not in_region(pos_list[id], collide_region):
        return False
    x, y = pos_list[id][0], pos_list[id][1]
    if dir == 1 and y < -0.01:
        return False
    if dir == 0 and y > 0.01:
        return False
    bbox = {
        0: [[x - 0.06, x + 0.06], [-0.2, y + 0.22]],
        1: [[x - 0.06, x + 0.06], [y - 0.22, 0.2]]
    }
    for i, pos in enumerate(pos_list):
        if i == id:
            continue
        if in_region(pos, bbox[dir]):
            return False
    return True


def can_push(pos_list, points_list, id, ang, table_center):
    dx = np.cos(ang)
    dy = np.sin(ang)
    x, y = pos_list[id][0], pos_list[id][1]

    new_x = x + dx * 0.14
    new_y = y + dy * 0.14
    if not(new_x >= 0.36 and new_x <= 0.64 and new_y >= -0.13 and new_y <= 0.13):
        return False
    # if dx * (x - table_center[0]) > 0.03:
    #     return False
    # if dy * (y - table_center[1]) > 0.045:
    #     return False

    cur = np.asarray([x, y])
    vec1 = np.asarray([np.cos(ang), np.sin(ang)]) * 0.18
    vec2 = np.asarray([np.sin(ang), -np.cos(ang)]) * 0.05
    box = Polygon([cur - vec2 + vec1,
                   cur + vec2 + vec1,
                   cur + vec2 - vec1,
                   cur - vec2 - vec1])
    for i, points in enumerate(points_list):
        if i == id:
            continue
        for point in points:
            if Point(point).within(box):
                return False
    return True


def set_friction_and_mass(obj_num, obj_id_list, cfg):
    friction_discrete, mass_discrete = [], []
    for i in range(obj_num):
        friction_discrete.append(np.random.choice(range(cfg.Friction_categories)))
        mass_discrete.append(np.random.choice(range(cfg.Mass_categories)))

    friction_list = [cfg.Friction_min + x / (cfg.Friction_categories - 1) * (cfg.Friction_max - cfg.Friction_min) for x
                     in friction_discrete]
    mass_list = [cfg.Mass_min + x / (cfg.Mass_categories - 1) * (cfg.Mass_max - cfg.Mass_min) for x in mass_discrete]

    # auxiliary
    friction_list.append(cfg.Friction_auxiliary)
    mass_list.append(cfg.Mass_auxiliary)

    for index, (friction, mass) in enumerate(zip(friction_list, mass_list)):
        p.changeDynamics(obj_id_list[index], -1,
                         lateralFriction=friction,
                         mass=mass,
                         spinningFriction=0.5,
                         restitution=1.0)

    # auxiliary
    p.changeDynamics(obj_id_list[-1], -1, spinningFriction=0.5, restitution=0.8)

    return friction_discrete, mass_discrete


def set_position_and_color(obj_num, obj_id_list):
    pos_list = get_pos_list(obj_num)
    for index, pos in enumerate(pos_list):
        color = [1, 1, 1, 1]
        color[random.choice([0, 1, 2])] = 0.8
        color = [0.3, 0.5, 0.7, 1]
        p.changeVisualShape(obj_id_list[index], -1, rgbaColor=color)
        p.resetBasePositionAndOrientation(obj_id_list[index], posObj=pos, ornObj=p.getQuaternionFromEuler([0, 0, 0]))


def generate_data(arg_step=10):
    # argparser
    parser = argparse.ArgumentParser()

    parser.add_argument('--step', type=int, default=arg_step)
    parser.add_argument('--nogui', action='store_true')
    parser.add_argument('--path', type=str, default='./log')
    parser.add_argument('--file_name', type=str, default=None)

    args = parser.parse_args()

    # logger
    if args.file_name is not None:
        mkdir(args.path)
    logger = DataLogger()

    # set robot & camera
    cfg = get_sim_config()

    view_matrix = tuple(invRt(cfg.Camera_pose).T.reshape(16))
    imsize_w, imsize_h = 640, 360
    f = 462
    fov_w = math.atan(imsize_w / 2 / f) * 2 / math.pi * 180
    fov_h = math.atan(imsize_h / 2 / f) * 2 / math.pi * 180

    proj_matrix = p.computeProjectionMatrixFOV(fov=fov_h, aspect=imsize_w / imsize_h, nearVal=0.01, farVal=5.0)

    cam = simCam(imsize=[640, 360],
                 view_matrix=view_matrix,
                 proj_matrix=proj_matrix,
                 z_near=0.01, z_far=5.0)
    robot = SimUR5(center=cfg.Table_center,
                   prepare_height=cfg.Gripper_prepare_height,
                   real_time_sim=False,
                   GUI=not args.nogui)

    obj_num = cfg.Object_num

    # set object
    scene = Sim_Scene(view_matrix=view_matrix,
                      fov_w=fov_w,
                      imsize=[imsize_w, imsize_h],
                      cube_num=cfg.Object_num)

    image_RGB_ini, image_depth_ini, _ = cam.get_data()

    # set property
    friction_discrete, mass_discrete = set_friction_and_mass(obj_num, scene.object_ids, cfg)
    set_position_and_color(obj_num, scene.object_ids)

    # always go home first
    robot.open_gripper(blocking=True)
    robot.close_gripper(blocking=True)

    logger.save_attribute('friction', friction_discrete)
    logger.save_attribute('mass', mass_discrete)
    cnt_dict = {'collide':0, 'push':0}
    all_proposal = {'collide':[], 'push':[]}

    # start action
    for step in range(args.step):
        robot.close_gripper(blocking=True)
        image_RGB, image_depth, _ = cam.get_data()

        mask = get_mask_real(
            image_RGB=image_RGB,
            image_RGB_ini=image_RGB_ini,
            image_depth=image_depth,
            image_depth_ini=image_depth_ini,
            Object_num=cfg.Object_num
        )

        # depth normalization
        tmp_depth = image_depth.copy()
        tmp_depth -= np.min(tmp_depth)
        tmp_depth /= np.max(tmp_depth)

        obj_pos, points_list = [], []
        for index in range(cfg.Object_num):
            cur_pos, points = get_pos((mask == (index + 1)).astype(np.int), image_RGB, image_depth, cfg.Camera_pose,
                                      cfg.Camera_intrinsic)

            obj_pos.append(cur_pos)
            points_list.append(points)
        scene.reset_coord_prev()

        # generate proposal

        # collide
        for i in range(cfg.Object_num):
            for dir in range(2):
                if can_collide(obj_pos, i, cfg.Collide_region, dir):
                    all_proposal['collide'].append({
                        'action_type': 'collide',
                        'obj_id': i,
                        'dir': dir
                    })

        # push
        for i in range(cfg.Object_num):
            for dir in range(cfg.direction_num):
                if can_push(obj_pos, points_list, i, dir / cfg.direction_num * 2 * np.pi, cfg.Table_center):
                    all_proposal['push'].append({
                        'action_type': 'push',
                        'obj_id': i,
                        'dir': dir,
                    })

        np.random.shuffle(all_proposal['push'])
        np.random.shuffle(all_proposal['collide'])

        proposal_list = all_proposal['push'] + all_proposal['collide']
        if cnt_dict['push'] > cnt_dict['collide']:
            proposal_list = all_proposal['collide'] + all_proposal['push']
        if get_random(0, 1) < 0.2:
            np.random.shuffle(proposal_list)


        # choose action randomly
        if len(proposal_list) == 0:
            print('No Proposal')
            exit()
        action = proposal_list[0]
        cnt_dict[action['action_type']] += 1

        # prepare
        cur_pos = obj_pos[action['obj_id']]
        points = points_list[action['obj_id']]

        logger.save_data(step, 'depth', image_depth)
        logger.save_data(step, 'mask', mask)
        logger.save_data(step, 'action_type', action['action_type'])
        logger.save_data(step, 'obj_id', action['obj_id'])

        if action['action_type'] == 'collide':
            dir = action['dir']
            aux_init = cfg.Auxiliary_initialization[dir]

            # set the auxiliary cylinder
            p.resetBasePositionAndOrientation(scene.object_ids[cfg.Object_num],
                                              posObj=aux_init + [cfg.Auxiliary_initialization_height],
                                              ornObj=p.getQuaternionFromEuler([0, 0, np.pi / 2]))

            # start action
            robot.primitive_topdown_grasp(aux_init + [cfg.Gripper_grasp_height],
                                          np.pi / 2)
            robot.primitive_place([cur_pos[0], cfg.Gripper_place_y[dir], cfg.Gripper_place_height],
                                  np.pi / 2,
                                  cfg.Slow_speed)
            robot.close_gripper(True)

            # finish action
            robot.primitive_gohome(speed=cfg.Fast_speed)
            p.resetBasePositionAndOrientation(scene.object_ids[cfg.Object_num],
                                              posObj=cfg.Auxiliary_initialization[0] + [
                                                  cfg.Auxiliary_initialization_height],
                                              ornObj=cfg.Orientation_initialization)

            # save info
            logger.save_data(step, 'action', [1] + [0, 0, 0] + [0] * 30 + [0] + [cur_pos[0], action['dir']])
            logger.save_data(step, 'direction', action['dir'])

        elif action['action_type'] == 'push':
            angle = action['dir'] / cfg.direction_num * 2 * np.pi
            delta = -1 * np.asarray([np.cos(angle), np.sin(angle)]) * (get_dist(points, cur_pos, [np.cos(angle), np.sin(angle)]) + 0.075)
            speed_discrete = np.random.choice(len(cfg.Push_speed))
            push_speed = cfg.Push_speed[speed_discrete]

            start = [cur_pos[0] + delta[0],
                     cur_pos[1] + delta[1],
                     cfg.Push_height]

            robot.primitive_push_tilt(start, angle, push_speed)
            robot.primitive_gohome(speed=cfg.Fast_speed)

            # save info
            logger.save_data(step, 'action', [0] + start + get_onehot(action['dir'], 30) + [push_speed] + [0, 0])
            logger.save_data(step, 'direction', action['dir'])
            logger.save_data(step, 'speed', speed_discrete)

        # calc the optical flow
        flowim, depth_im_prev, depth_im_curr = scene.comput_2dflow(get_mask=False)
        _, flow = depth_smooth(depth_im_prev, flowim)
        flow = flow * (mask == (action['obj_id'] + 1)).astype(np.float32)
        logger.save_data(step, 'flow', flow)

    if args.file_name is not None:
        joblib.dump((logger.attribute_dict, logger.data_dict), os.path.join(args.path, args.file_name), compress=True)

if __name__ == '__main__':
    generate_data()
