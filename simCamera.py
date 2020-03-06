import pybullet as p
import numpy as np
import math

class simCam(object):
    def __init__(self, imsize, view_matrix, proj_matrix, z_near, z_far):
        self.imsize = imsize
        self.view_matrix = view_matrix
        self.proj_matrix = proj_matrix
        self.z_near = z_near
        self.z_far = z_far

    def get_data(self):
        img_arr = p.getCameraImage(self.imsize[0], self.imsize[1], self.view_matrix, self.proj_matrix,
                                   shadow=0, flags=p.ER_TINY_RENDERER)
        w = img_arr[0]
        h = img_arr[1]
        rgb = img_arr[2]
        rgb_arr = np.array(rgb, dtype=np.uint8).reshape([h, w, 4])
        rgb = rgb_arr[:, :, 0:3]

        d = img_arr[3].astype(np.float32)
        d = self.z_far * self.z_near / (self.z_far - (self.z_far - self.z_near) * d)

        mask = np.zeros([h, w])
        for i, line in enumerate(img_arr[4]):
            for j, pixel in enumerate(line):
                if (pixel >= 0):
                    obUid = pixel & ((1 << 24) - 1)
                    mask[i, j] = max(0, obUid - 3)
        return rgb, d, mask

    def get_image_RGB(self):
        return self.get_data()[0]

    def get_image_depth(self):
        return self.get_data()[1]

    def get_image_mask(self):
        return self.get_data()[2]