#!/usr/bin/env python3

#########################################
####### Written by: Shuran Song #########
#########################################

import numpy as np
import pybullet as p
import math
import trimesh


def invRt(Rt):
    #RtInv = [Rt(:,1:3)'  -Rt(:,1:3)'* Rt(:,4)];
    invR = Rt[0:3,0:3].T
    RtInv = np.concatenate(invR, -1*invR* Rt[:,4], axis=1)
    return RtInv

def prejct_pts_to_2d(pts, RT_wrd2cam, camera_intrisic):
    # transformation from word to virtual camera
    # camera_intrisic for virtual camera [ [f,0,0],[0,f,0],[0,0,1]] f is focal length
    pts_c = tranform_points(pts, RT_wrd2cam[0:3, :])

    rot_algix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0]])
    pts_c = tranform_points(pts_c, rot_algix)

    coord_2d = np.dot(camera_intrisic, pts_c)
    # coord_2d[0:2,:] = coord_2d[(1,0),:]/np.tile(coord_2d[2,:],(2,1))
    coord_2d[0:2, :] = coord_2d[0:2, :] / np.tile(coord_2d[2, :], (2, 1))
    coord_2d[2, :] = pts_c[2, :]
    return coord_2d

def tranform_points(pts, trasform):
    # pts = [3xN] array
    # trasform: [3x4]
    pts_t = np.dot(trasform[0:3,0:3],pts)+np.tile(trasform[0:3,3:],(1,pts.shape[1]))
    return pts_t

def pose2tranform(postion, rotation):
    postion = np.array(postion)
    R = np.array(p.getMatrixFromQuaternion(rotation)).reshape(3,3)
    trasform = np.concatenate((R,postion.reshape(3,1)),axis=1)
        
    # from postion, rotation to transformation matrix 
    return trasform

def check_projection_valid(coord_2d,imh,imw):
    N = coord_2d.shape[1]
    valid_idx = np.ones(N)
    depth_im = np.zeros((imh,imw))
    # sort depth from close to far 
    sort_id = np.argsort(coord_2d[2,:])
    for i in range(N):
        idx = sort_id[i]
        x_i = int(coord_2d[0,idx])
        y_i = int(coord_2d[1,idx])
        if x_i<imw and x_i>=0 and y_i <imh and y_i>=0:
            if depth_im[y_i,x_i] == 0:
                depth_im[y_i,x_i] = coord_2d[2,idx]
            else:
                valid_idx[idx] = 0
        else:
            valid_idx[idx] = 0
    return valid_idx, depth_im

class Sim_Scene:
    def __init__(self, view_matrix, fov_w, imsize, cube_num, auxiliary_num=1):
        self.auxiliary_num = auxiliary_num
        self.cube_center = [[0.5, -0.4, 0.18], [0.7, -0.4, 0.24]]

        # set up camera 
        self.cam_wrd2cam = np.array(view_matrix).reshape(4, 4).T
        self.imw = imsize[0]
        self.imh = imsize[1]
        self.fov_w = fov_w / 180 * math.pi
        self.f = self.imw / 2 / math.tan(self.fov_w / 2)
        self.camera_intrisic = np.array([[self.f, 0, self.imw / 2],
                                         [0, self.f, self.imh / 2],
                                         [0, 0, 1]])
        
        # set up scene 
        self.object_ids, self.object_pts= self.scene_setup(cube_num)

        # transform object point to their current location 
        object_pts_curr = self.get_object_pts_curr()

        self.object_pts_prev = object_pts_curr
        self.coord_2d_prev = prejct_pts_to_2d(object_pts_curr, self.cam_wrd2cam, self.camera_intrisic)
        self.valid_idx_prev, depth_im = check_projection_valid(self.coord_2d_prev,self.imh,self.imw)

    def scene_setup(self, cube_num):
        p.loadURDF("sim_pybullet/descriptions/scene_setup/double_table/table.urdf",[0.5,0,0],p.getQuaternionFromEuler([0,0,-1.57]), useFixedBase=True)
        number_sample = 4000
        object_ids = []
        object_pts = []
        id_list = np.random.choice(range(6))
        for i in range(cube_num):
            id = np.random.choice(id_list)
            cube_id  = p.loadURDF("sim_pybullet/descriptions/shape/{}.urdf".format(id),
                                  [0.6 + 0.15 * i, -0.8, 0.18])
            object_ids.append(cube_id)
            mesh = trimesh.util.concatenate(trimesh.load("sim_pybullet/descriptions/shape/{}.obj".format(id)))
            pts = mesh.sample(number_sample)
            pts = pts.T
            object_pts.append(pts)

        cube_id = p.loadURDF("sim_pybullet/descriptions/clevr/clevr_cylinder.urdf",
                             [0.4, -1.5, 0.14])
        object_ids.append(cube_id)
        mesh = trimesh.load("sim_pybullet/descriptions/clevr/clevr_cylinder.obj")
        pts = mesh.sample(number_sample)
        pts = pts.T
        object_pts.append(pts)

        object_ids = np.array(object_ids)
        object_pts = np.array(object_pts)
        return object_ids, object_pts

    def get_object_pts_curr(self):
        object_pts_curr = []
        for i in range(self.object_ids.size - self.auxiliary_num):
            tmp = p.getBasePositionAndOrientation(self.object_ids[i])
            obj_transfrom =  pose2tranform(tmp[0], tmp[1])
            pts_t = tranform_points(self.object_pts[i].squeeze(),obj_transfrom)
            object_pts_curr.append(pts_t.T)
            # vis_point3d(pts_t)

        object_pts_curr = np.array(object_pts_curr)
        object_pts_curr = object_pts_curr.reshape((int(object_pts_curr.size/3),3))
        object_pts_curr = object_pts_curr.T
        return object_pts_curr


    def get_objectpose(self, objectId_array):
        # input object id array
        object_poses = {}
        for i in range(objectId_array.size):
            tmp = p.getBasePositionAndOrientation(objectId_array[i])
            object_poses[i] = tmp
        #output array of object poses
        return object_poses

    def reset_coord_prev(self):
        object_pts_curr = self.get_object_pts_curr()
        coord_2d_curr = prejct_pts_to_2d(object_pts_curr, self.cam_wrd2cam, self.camera_intrisic)
        valid_idx_curr, depth_im_curr = check_projection_valid(coord_2d_curr,self.imh,self.imw)
        self.coord_2d_prev = coord_2d_curr
        self.valid_idx_prev = valid_idx_curr

    def comput_2dflow(self, get_mask=False):
        # Note: This function assumes total number of point not change
        # same point is always in the same location in the all_points array
        # object_pts is the initial point pose 

        # loop though each object perpare all_points, all_lables for current
         
        object_pts_curr = self.get_object_pts_curr()
        
        # get current 2d
        coord_2d_curr = prejct_pts_to_2d(object_pts_curr,self.cam_wrd2cam,self.camera_intrisic)
        valid_idx_curr, depth_im_curr = check_projection_valid(coord_2d_curr,self.imh,self.imw)
        valid_idx_prev, depth_im_prev = check_projection_valid(self.coord_2d_prev, self.imh, self.imw)

        mask = np.zeros(shape=[self.imh, self.imw])
        for i in range(4000 * (len(self.object_ids) - self.auxiliary_num)):
            pt = self.coord_2d_prev[:, i]
            y = int(pt[0])
            x = int(pt[1])
            if x < 0 or y < 0 or x >= self.imh or y >= self.imw:
                print(i, x, y)
                raise NotImplementedError
            mask[x, y] = i // 4000 + 1

        # compare old 2d localtion get flow 
        flow_uv = coord_2d_curr - self.coord_2d_prev

        # label the flow image 
        # remove the the ones that not visible in the old image or new image 
        flowim = np.zeros((2,self.imh,self.imw))
        for i in range(valid_idx_curr.size):
            if valid_idx_prev[i]>0 and self.valid_idx_prev[i]>0 and np.sum(np.abs(flow_uv[0:2,i]))>0.0001:
                x_i = int(self.coord_2d_prev[0,i])
                y_i = int(self.coord_2d_prev[1,i])
                flowim[0:2, y_i, x_i] = flow_uv[0:2,i]

        # update state 
        self.coord_2d_prev = coord_2d_curr
        self.valid_idx_prev = valid_idx_curr
        if get_mask:
            return flowim, depth_im_prev, depth_im_curr, mask
        else:
            return flowim, depth_im_prev, depth_im_curr

    def resetCube(self, pos=None, orientation=None, friction=None, mass=None):
        obj_num = len(self.object_ids)
        if pos is None:
            pos = [self.cube_center] * obj_num
        if orientation is None:
            orientation = [[-1.0, 1.0, 0.0, 0.0]] * obj_num
        if friction is not None:
            if not isinstance(friction, list):
                friction = [friction] * obj_num
            for id, f in enumerate(friction):
                p.changeDynamics(self.object_ids[id], -1, lateralFriction=f)
        if mass is not None:
            if not isinstance(mass, list):
                mass = [mass] * obj_num
            for id, m in enumerate(mass):
                p.changeDynamics(self.object_ids[id], -1, mass=m)
        for i in range(obj_num):
            p.resetBasePositionAndOrientation(self.object_ids[i], pos[i], orientation[i])

    def get_pointcloud(self,color_img, depth_img):

        # Get depth image size

        im_h = depth_img.shape[0]
        im_w = depth_img.shape[1]

        # Project depth into 3D point cloud in camera coordinates
        pix_x,pix_y = np.meshgrid(np.linspace(0,im_w-1,im_w), np.linspace(0,im_h-1,im_h))
        cam_pts_x = np.multiply(pix_x-self.camera_intrisic[0,2],depth_img/self.camera_intrisic[0,0])
        cam_pts_y = np.multiply(pix_y-self.camera_intrisic[1,2],depth_img/self.camera_intrisic[1,1])
        cam_pts_z = depth_img.copy()
        cam_pts_x.shape = (im_h*im_w,1)
        cam_pts_y.shape = (im_h*im_w,1)
        cam_pts_z.shape = (im_h*im_w,1)

        # Reshape image into colors for 3D point cloud
        rgb_pts_r = color_img[:,:,0]
        rgb_pts_g = color_img[:,:,1]
        rgb_pts_b = color_img[:,:,2]
        rgb_pts_r.shape = (im_h*im_w,1)
        rgb_pts_g.shape = (im_h*im_w,1)
        rgb_pts_b.shape = (im_h*im_w,1)

        cam_pts = np.concatenate((cam_pts_x, cam_pts_y, cam_pts_z), axis=1)
        rgb_pts = np.concatenate((rgb_pts_r, rgb_pts_g, rgb_pts_b), axis=1)

        return cam_pts, rgb_pts