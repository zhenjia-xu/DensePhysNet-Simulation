import struct
import math
import numpy as np
import cv2
import skimage.measure
import scipy.ndimage.interpolation

import pdb
from scipy.ndimage.measurements import label

def tranformRT(RT0, RT1):
    # apply RT1 to RT0 
    RT0 = np.matrix(RT0)
    RT1 = np.matrix(RT1)
    RT_out = np.zeros((3,4))

    RT_out = np.matrix(RT1[0:3,0:3])*np.matrix(RT0)
    RT_out[:,3] = RT_out[:,3]+RT1[:,3]
    return RT_out
    
def get_pointcloud(color_img, depth_img, camera_intrinsics):

    # Get depth image size
    im_h = depth_img.shape[0]
    im_w = depth_img.shape[1]

    # Project depth into 3D point cloud in camera coordinates
    pix_x,pix_y = np.meshgrid(np.linspace(0,im_w-1,im_w), np.linspace(0,im_h-1,im_h))
    cam_pts_x = np.multiply(pix_x-camera_intrinsics[0,2],depth_img/camera_intrinsics[0,0])
    cam_pts_y = np.multiply(pix_y-camera_intrinsics[1,2],depth_img/camera_intrinsics[1,1])
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


def get_heightmap(color_im,xyz_im,cam_pose,wkspc_lim,hm_pix_size):

    # Compute heightmap size
    hm_size = np.round(((wkspc_lim[1,1]-wkspc_lim[1,0])/hm_pix_size,(wkspc_lim[0,1]-wkspc_lim[0,0])/hm_pix_size)).astype(int)

    # Get 3D point cloud from RGB-D images
    rgb_pts = color_im.reshape(color_im.shape[0]*color_im.shape[1],3)
    xyz_pts = xyz_im.reshape(xyz_im.shape[0]*xyz_im.shape[1],3)

    # Transform 3D point cloud from camera coordinates to robot coordinates
    xyz_pts = np.transpose(np.dot(cam_pose[0:3,0:3],np.transpose(xyz_pts))+np.tile(cam_pose[0:3,3:],(1,xyz_pts.shape[0])))
    
    # Sort surface points by z value
    sort_z_ind = np.argsort(xyz_pts[:,2])
    xyz_pts = xyz_pts[sort_z_ind]
    rgb_pts = rgb_pts[sort_z_ind]
    
    # Filter out surface points outside heightmap boundaries
    heightmap_valid_ind = np.logical_and(np.logical_and(np.logical_and(np.logical_and(xyz_pts[:,0] >= wkspc_lim[0,0], xyz_pts[:,0] < wkspc_lim[0,1]), xyz_pts[:,1] >= wkspc_lim[1,0]), xyz_pts[:,1] < wkspc_lim[1,1]), xyz_pts[:,2] < wkspc_lim[2,1])
    xyz_pts = xyz_pts[heightmap_valid_ind]
    rgb_pts = rgb_pts[heightmap_valid_ind]

    # Create orthographic top-down-view RGB-D heightmaps
    color_hm_r = np.zeros((hm_size[0],hm_size[1],1),dtype=np.uint8)
    color_hm_g = np.zeros((hm_size[0],hm_size[1],1),dtype=np.uint8)
    color_hm_b = np.zeros((hm_size[0],hm_size[1],1),dtype=np.uint8)
    depth_hm = np.zeros(hm_size)
    heightmap_pix_x = np.floor((xyz_pts[:,0]-wkspc_lim[0,0])/hm_pix_size).astype(int)
    heightmap_pix_y = np.floor((xyz_pts[:,1]-wkspc_lim[1,0])/hm_pix_size).astype(int)
    color_hm_r[heightmap_pix_y,heightmap_pix_x] = rgb_pts[:,[0]]
    color_hm_g[heightmap_pix_y,heightmap_pix_x] = rgb_pts[:,[1]]
    color_hm_b[heightmap_pix_y,heightmap_pix_x] = rgb_pts[:,[2]]
    color_hm = np.concatenate((color_hm_r,color_hm_g,color_hm_b),axis=2)
    depth_hm[heightmap_pix_y,heightmap_pix_x] = xyz_pts[:,2]
    z_bottom = wkspc_lim[2,0]
    depth_hm = depth_hm-z_bottom
    depth_hm[depth_hm < 0] = 0
    depth_hm[depth_hm == -z_bottom] = 0 # np.nan

    return color_hm,depth_hm

# Save a 3D point cloud to a binary .ply file
def pcwrite(xyz_pts, filename, rgb_pts=None):
    assert xyz_pts.shape[1] == 3, 'input XYZ points should be an Nx3 matrix'
    if rgb_pts is None:
        rgb_pts = np.ones(xyz_pts.shape).astype(np.uint8)*255
    assert xyz_pts.shape == rgb_pts.shape, 'input RGB colors should be Nx3 matrix and same size as input XYZ points'

    # Write header for .ply file
    pc_file = open(filename, 'wb')
    pc_file.write(bytearray('ply\n','utf8'))
    pc_file.write(bytearray('format binary_little_endian 1.0\n','utf8'))
    pc_file.write(bytearray('element vertex %d\n' % xyz_pts.shape[0],'utf8'))
    pc_file.write(bytearray('property float x\n','utf8'))
    pc_file.write(bytearray('property float y\n','utf8'))
    pc_file.write(bytearray('property float z\n','utf8'))
    pc_file.write(bytearray('property uchar red\n','utf8'))
    pc_file.write(bytearray('property uchar green\n','utf8'))
    pc_file.write(bytearray('property uchar blue\n','utf8'))
    pc_file.write(bytearray('end_header\n','utf8'))

    # Write 3D points to .ply file
    for i in range(xyz_pts.shape[0]):
        pc_file.write(bytearray(struct.pack("fffccc",xyz_pts[i,0],xyz_pts[i,1],xyz_pts[i,2],rgb_pts[i,0].tostring(),rgb_pts[i,1].tostring(),rgb_pts[i,2].tostring())))
   
    pc_file.close()

# def vis_point3d(pts,rgb=None,ax=None,N=None):
#     if ax == None:
#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')
#
#     sample = np.random.permutation(pts.shape_obj[1])
#     if N==None:
#         N = 1000
#     X = pts[0,sample[1:N]]
#     Y = pts[1,sample[1:N]]
#     Z = pts[2,sample[1:N]]
#     color_sample = rgb[sample[1:N],:].astype(np.float32)/255
#     ax.scatter(X, Y, Z, s=1, c=color_sample)
#     # max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
#
#     # mid_x = (X.max()+X.min()) * 0.5
#     # mid_y = (Y.max()+Y.min()) * 0.5
#     # mid_z = (Z.max()+Z.min()) * 0.5
#     # ax.set_xlim(mid_x - max_range, mid_x + max_range)
#     # ax.set_ylim(mid_y - max_range, mid_y + max_range)
#     # ax.set_zlim(mid_z - max_range, mid_z + max_range)
#
#
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     plt.pause(0.001)
#     return ax

def get_aff_vis(grasp_aff,color_img,grasp_pix,grasp_angle):

    # grasp_aff = np.clip(grasp_aff, 0.0, 1.0) # Assume probability
    grasp_aff_vis = cv2.applyColorMap((grasp_aff*255).astype(np.uint8), cv2.COLORMAP_JET)
    vis = (0.7*cv2.cvtColor(color_img,cv2.COLOR_RGB2BGR).astype(float) + 0.3*grasp_aff_vis.astype(float)).astype(np.uint8)
        
    # Draw grasp icons for each grasp
    num_grasps = grasp_pix.shape[0]
    for i in range(num_grasps):
        vis = cv2.circle(vis,(int(grasp_pix[i,1]),int(grasp_pix[i,0])),7,(0,255,0),2)
        vis = draw_grasp_icon(vis,grasp_pix[i,:],grasp_angle[i,0])

    return vis


def draw_grasp_icon(im,grasp_pix,grasp_angle):
    grasp_icon_pix = np.asarray([[-50.,10.],[-50.,0.],[-50.,-10.],[50.,10.],[50.,0.],[50.,-10.]])
    grasp_icon_pix = np.transpose(np.dot(np.asarray([[np.cos(-grasp_angle*np.pi/180.),-np.sin(-grasp_angle*np.pi/180.)],[np.sin(-grasp_angle*np.pi/180.),np.cos(-grasp_angle*np.pi/180.)]]),np.transpose(grasp_icon_pix)))
    grasp_icon_pix = (grasp_icon_pix+grasp_pix).astype(int)
    im = cv2.line(im,(grasp_icon_pix[0,1],grasp_icon_pix[0,0]),(grasp_icon_pix[2,1],grasp_icon_pix[2,0]),(0,255,0),2)
    im = cv2.line(im,(grasp_icon_pix[1,1],grasp_icon_pix[1,0]),(grasp_icon_pix[4,1],grasp_icon_pix[4,0]),(0,255,0),2)
    im = cv2.line(im,(grasp_icon_pix[3,1],grasp_icon_pix[3,0]),(grasp_icon_pix[5,1],grasp_icon_pix[5,0]),(0,255,0),2)
    return im


def get_grasp_angle(depth_hm,grasp_pix):

    # grasp_width = 100
    # finger_width = 20

    pad_depth_hm = np.pad(depth_hm.copy(),((50,50),(50,50)),mode='constant',constant_values=0.03)

    # Construct grasp kernel filter
    grasp_kern = np.linspace(1,0,50).reshape(50,1)
    grasp_kern = np.tile(grasp_kern,(1,20))
    grasp_kern = np.concatenate((grasp_kern,np.flipud(grasp_kern)),axis=0)
    grasp_kern *= grasp_kern
    grasp_kern = np.concatenate((np.zeros((100,40)),grasp_kern,np.zeros((100,40))),axis=1)

    num_grasps = grasp_pix.shape[0]
    grasp_angle = np.zeros((num_grasps,1))
    for i in range(num_grasps):
        tmp_grasp_pix = grasp_pix[i,:].astype(int)

        mid_crop = pad_depth_hm[(tmp_grasp_pix[0]+50-10):(tmp_grasp_pix[0]+50+10),(tmp_grasp_pix[1]+50-10):(tmp_grasp_pix[1]+50+10)].copy()
        mid_crop[mid_crop == 0] = np.nan
        mid_height = np.nanmedian(mid_crop)
        if np.isnan(mid_height):
            grasp_angle[i] = 0
            continue

        crop_depth_hm = pad_depth_hm[(tmp_grasp_pix[0]+0):(tmp_grasp_pix[0]+100),(tmp_grasp_pix[1]+0):(tmp_grasp_pix[1]+100)]
        crop_scores = mid_height-crop_depth_hm
        crop_scores = np.clip(crop_scores,-0.03,0.03)

        max_rot_score = -np.inf
        max_rot_angle = None

        for rot_angle in [-67.5,-45.,-22.5,0,22.5,45.,67.5,90.]:
            rot_grasp_kern = scipy.ndimage.interpolation.rotate(grasp_kern,angle=-rot_angle,reshape=False,order=0,mode='constant',cval=0.,prefilter=False)

            crop_prod = rot_grasp_kern*crop_scores

            rot_grasp_score = np.sum(crop_prod)

            if rot_grasp_score > max_rot_score:
                max_rot_score = rot_grasp_score
                max_rot_angle = rot_angle

        if max_rot_angle is None:
            pdb.set_trace()

        grasp_angle[i] = max_rot_angle

    return grasp_angle


def get_top_grasps(aff_map,num_top_grasps=3):

    search_radius = 150

    pad_aff_map = np.pad(aff_map.copy(),((search_radius,search_radius),(search_radius,search_radius)),mode='constant',constant_values=0)

    grasp_pix = np.zeros((0,2))
    grasp_conf = np.zeros((0,1))
    for i in range(num_top_grasps):

        # Add top grasp (iff confidence is larger than 0)
        tmp_grasp_conf = np.max(pad_aff_map)
        if tmp_grasp_conf > 0:
            tmp_grasp_pix = np.unravel_index(np.argmax(pad_aff_map),pad_aff_map.shape) # [y,x]
            grasp_pix = np.concatenate((grasp_pix,np.asarray([[tmp_grasp_pix[0]-search_radius,tmp_grasp_pix[1]-search_radius]])),axis=0)
            grasp_conf = np.concatenate((grasp_conf,np.asarray([[tmp_grasp_conf]])),axis=0)

            # Zero out affordances within search radius
            pad_aff_map[(tmp_grasp_pix[0]-search_radius):(tmp_grasp_pix[0]+search_radius),(tmp_grasp_pix[1]-search_radius):(tmp_grasp_pix[1]+search_radius)] = 0

    return grasp_pix,grasp_conf



# Get rotation matrix from euler angles
def euler2rotm(theta):
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])         
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])            
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R


# Checks if a matrix is a valid rotation matrix.
def isRotm(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
 
 
# Calculates rotation matrix to euler angles
def rotm2euler(R) :
 
    assert(isRotm(R))

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])


def angle2rotm(angle_axis, point=None):
    # Copyright (c) 2006-2018, Christoph Gohlke
    angle = angle_axis[0]
    axis = angle_axis[1:]

    sina = math.sin(angle)
    cosa = math.cos(angle)
    axis = axis / np.linalg.norm(axis)

    # Rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(axis, axis) * (1.0 - cosa)
    axis *= sina
    R += np.array([[0.0, -axis[2], axis[1]],
                   [axis[2], 0.0, -axis[0]],
                   [-axis[1], axis[0], 0.0]], dtype=np.float64)
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # Rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M[:3, :3]


def rotm2angle(R):
    # From: euclideanspace.com

    epsilon = 0.01 # Margin to allow for rounding errors
    epsilon2 = 0.1 # Margin to distinguish between 0 and 180 degrees

    assert(isRotm(R))

    if ((abs(R[0][1]-R[1][0])< epsilon) and (abs(R[0][2]-R[2][0])< epsilon) and (abs(R[1][2]-R[2][1])< epsilon)):
        # Singularity found
        # First check for identity matrix which must have +1 for all terms in leading diagonaland zero in other terms
        if ((abs(R[0][1]+R[1][0]) < epsilon2) and (abs(R[0][2]+R[2][0]) < epsilon2) and (abs(R[1][2]+R[2][1]) < epsilon2) and (abs(R[0][0]+R[1][1]+R[2][2]-3) < epsilon2)):
            # this singularity is identity matrix so angle = 0
            return [0,1,0,0] # zero angle, arbitrary axis

        # Otherwise this singularity is angle = 180
        angle = np.pi
        xx = (R[0][0]+1)/2
        yy = (R[1][1]+1)/2
        zz = (R[2][2]+1)/2
        xy = (R[0][1]+R[1][0])/4
        xz = (R[0][2]+R[2][0])/4
        yz = (R[1][2]+R[2][1])/4
        if ((xx > yy) and (xx > zz)): # R[0][0] is the largest diagonal term
            if (xx< epsilon):
                x = 0
                y = 0.7071
                z = 0.7071
            else:
                x = np.sqrt(xx)
                y = xy/x
                z = xz/x
        elif (yy > zz): # R[1][1] is the largest diagonal term
            if (yy< epsilon):
                x = 0.7071
                y = 0
                z = 0.7071
            else:
                y = np.sqrt(yy)
                x = xy/y
                z = yz/y
        else: # R[2][2] is the largest diagonal term so base result on this
            if (zz< epsilon):
                x = 0.7071
                y = 0.7071
                z = 0
            else:
                z = np.sqrt(zz)
                x = xz/z
                y = yz/z
        return [angle,x,y,z] # Return 180 deg rotation

    # As we have reached here there are no singularities so we can handle normally
    s = np.sqrt((R[2][1] - R[1][2])*(R[2][1] - R[1][2]) + (R[0][2] - R[2][0])*(R[0][2] - R[2][0]) + (R[1][0] - R[0][1])*(R[1][0] - R[0][1])) # used to normalise
    if (abs(s) < 0.001):
        s = 1 

    # Prevent divide by zero, should not happen if matrix is orthogonal and should be
    # Caught by singularity test above, but I've left it in just in case
    angle = np.arccos(( R[0][0] + R[1][1] + R[2][2] - 1)/2)
    x = (R[2][1] - R[1][2])/s
    y = (R[0][2] - R[2][0])/s
    z = (R[1][0] - R[0][1])/s
    return [angle,x,y,z]


# Convert from URx rotation format to axis angle
def urx2angle(v):
    angle = np.linalg.norm(v)
    axis = v/angle
    return np.insert(axis,0,angle)


# Convert from angle axis format to URx rotation format
def angle2urx(angle_axis):
    angle_axis[1:4] = angle_axis[1:4]/np.linalg.norm(angle_axis[1:4])
    return angle_axis[0]*np.asarray(angle_axis[1:4])


# Convert from quaternion (w,x,y,z) to rotation matrix
def quat2rotm(quat):
    # From: Christoph Gohlke

    # epsilon for testing whether a number is close to zero
    _EPS = np.finfo(float).eps * 4.0

    q = np.array(quat, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [                0.0,                 0.0,                 0.0, 1.0]])


# Convert from rotation matrix to quaternion (w,x,y,z)
def rotm2quat(matrix,isprecise=False):
    # From: Christoph Gohlke

    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4, ))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 0, 1, 2
            if M[1, 1] > M[0, 0]:
                i, j, k = 1, 2, 0
            if M[2, 2] > M[i, i]:
                i, j, k = 2, 0, 1
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
            q = q[[3, 0, 1, 2]]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array([[m00-m11-m22, 0.0,         0.0,         0.0],
                         [m01+m10,     m11-m00-m22, 0.0,         0.0],
                         [m02+m20,     m12+m21,     m22-m00-m11, 0.0],
                         [m21-m12,     m02-m20,     m10-m01,     m00+m11+m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q


# Estimate rigid transform with SVD (from Nghia Ho)
def get_rigid_transform(A, B):
    assert len(A) == len(B)
    N = A.shape[0]; # Total points
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - np.tile(centroid_A, (N, 1)) # Centre the points
    BB = B - np.tile(centroid_B, (N, 1))
    H = np.dot(np.transpose(AA), BB) # Dot is matrix multiplication for array
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0: # Special reflection case
       Vt[2,:] *= -1
       R = np.dot(Vt.T, U.T)
    t = np.dot(-R, centroid_A.T) + centroid_B.T
    t.shape = (3,1)
    return np.concatenate((np.concatenate((R, t), axis=1),np.array([[0, 0, 0, 1]])), axis=0)


# Get nearest nonzero pixel
def nearest_nonzero_pix(img, y, x):
    r,c = np.nonzero(img)
    min_idx = ((r - y)**2 + (c - x)**2).argmin()
    return r[min_idx], c[min_idx]


def angle_between(v1,v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = v1/np.linalg.norm(v1)
    v2_u = v2/np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

# modified by xzj
def transform_points(pts, transform):
    # pts = [3xN] array
    # trasform: [3x4]
    pts_t = np.dot(transform[0:3,0:3],pts)+np.tile(transform[0:3,3:],(1,pts.shape[1]))
    return pts_t

def get_mask(image_ini, image_RGB):
    h_ini, _, _ = cv2.split(cv2.cvtColor(image_ini, cv2.COLOR_RGB2HSV))
    h_cur, _, _ = cv2.split(cv2.cvtColor(image_RGB, cv2.COLOR_RGB2HSV))
    tmp = np.abs(h_ini - h_cur)
    tmp = np.minimum(tmp, 360-tmp)
    
    mask = (tmp > 30).astype(np.int)
    mask[:,:150] = 0
    return mask

    
def get_pos_real(mask, image_RGB, image_depth, camera_pose, camera_intrinsic):
    xyz_cam, rgb = get_pointcloud(image_RGB, image_depth, camera_intrinsic)
    xyz_wrd = xyz_cam.T
    # rot_algix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0]])

    # # only for sim
    # xyz_wrd = transform_points(xyz_wrd, rot_algix)

    xyz_wrd = transform_points(xyz_wrd, camera_pose)

    list = []
    tmp = np.reshape(xyz_wrd, [3, 360, 640])
    x_coord, y_coord = np.nonzero(mask)
    for c in zip(x_coord, y_coord):
        list.append(tmp[:, c[0], c[1]])
    list = np.asarray(list)
    minx = np.min(list[:, 0])
    maxx = np.max(list[:, 0])
    miny = np.min(list[:, 1])
    maxy = np.max(list[:, 1])

    return [(minx + maxx) / 2, (miny + maxy) / 2], list



def get_pos(mask, image_RGB, image_depth, camera_pose, camera_intrinsic):
    xyz_cam, rgb = get_pointcloud(image_RGB, image_depth, camera_intrinsic)
    xyz_wrd = xyz_cam.T
    rot_algix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0]])
    xyz_wrd = transform_points(xyz_wrd, rot_algix)
    xyz_wrd = transform_points(xyz_wrd, camera_pose)

    list = []
    tmp = np.reshape(xyz_wrd, [3, 360, 640])
    x_coord, y_coord = np.nonzero(mask)
    for c in zip(x_coord, y_coord):
        list.append(tmp[:, c[0], c[1]])
    list = np.asarray(list)
    minx = np.min(list[:, 0])
    maxx = np.max(list[:, 0])
    miny = np.min(list[:, 1])
    maxy = np.max(list[:, 1])

    return [(minx + maxx) / 2, (miny + maxy) / 2], list

def get_length(vec):
    return np.sqrt((vec[0] ** 2) + (vec[1] ** 2))

def get_cos(vec0, vec1):
    return (vec0[0] * vec1[0] + vec0[1] * vec1[1]) / get_length(vec0) / get_length(vec1)

def get_dist(points, center, dir):
    cur_max = 0
    eps = np.pi / 60
    for p in points:
        vec = (p[0] - center[0], p[1] - center[1])
        if get_cos(vec, dir) > np.cos(eps):
            cur_max = max(cur_max, get_length(vec))
    return cur_max

def depth_smooth(depth_img, flow_img=None):
    img_size = depth_img.shape
    dist = np.zeros_like(depth_img)
    depth_img_new = np.zeros_like(depth_img)
    if flow_img is not None:
        flow_img_new = np.zeros_like(flow_img)
    list = np.nonzero(depth_img)
    for x, y in zip(list[0], list[1]):
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                cur_x = x + dx
                cur_y = y + dy
                if cur_x < 0 or cur_x >= img_size[0] or cur_y < 0 or cur_y >= img_size[1]:
                    break
                if depth_img[x, y] > dist[cur_x, cur_y]:
                    dist[cur_x, cur_y] = depth_img[x, y]
                    depth_img_new[cur_x, cur_y] = depth_img[x, y]
                    if flow_img is not None:
                        flow_img_new[:, cur_x, cur_y] = flow_img[:, x, y]
    if flow_img is None:
        return depth_img_new
    else:
        return depth_img_new, flow_img_new

def mask_smooth(mask):
    img_size = mask.shape
    new_mask = np.zeros_like(mask)
    list = np.nonzero(mask)
    for x, y in zip(list[0], list[1]):
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                cur_x = x + dx
                cur_y = y + dy
                if cur_x < 0 or cur_x >= img_size[0] or cur_y < 0 or cur_y >= img_size[1]:
                    break
                new_mask[cur_x, cur_y] = mask[x, y]
    return new_mask

def invRt(Rt):
    #RtInv = [Rt(:,1:3)'  -Rt(:,1:3)'* Rt(:,4)];
    invR = Rt[0:3,0:3].T
    invT = -1*np.dot(invR, Rt[0:3,3])
    invT.shape = (3,1)
    RtInv = np.concatenate((invR, invT),axis=1)
    RtInv = np.concatenate((RtInv,np.array([0,0,0,1]).reshape(1,4)),axis=0)
    return RtInv


def get_mask_real(image_RGB,image_RGB_ini,image_depth,image_depth_ini,Object_num):
    color_diff = np.sum(np.abs(image_RGB-image_RGB_ini.astype(np.float32)),2)
    binary_mask = color_diff>0.6
    binary_mask = np.bitwise_and(binary_mask,image_depth_ini*image_depth>0)
    binary_mask = np.bitwise_and(binary_mask,image_depth<1.5)
    binary_mask = np.bitwise_and(binary_mask,np.sum(image_RGB_ini,2)>1.5)
    binary_mask[:,550:] = 0
    binary_mask[:,0:100] = 0
    labeled, ncomponents = label(binary_mask)
    nnsize = np.zeros(ncomponents+1)
    for i in range(ncomponents+1):
        nnsize[i] = np.sum(labeled==i)
    idxs = np.argsort(-1*nnsize)
    mask = np.zeros((binary_mask.shape[0],binary_mask.shape[1]))

    for i in range(Object_num+1):
        mask[labeled==idxs[i]] = i


    h_cur, _, _ = cv2.split(cv2.cvtColor(image_RGB, cv2.COLOR_RGB2HSV))
    h_list = [-np.inf]
    for i in range(1, Object_num + 1):
        mask_i = (mask == i).astype(np.float32)
        h_i = np.sum(h_cur * mask_i) / np.sum(mask_i)
        h_list.append(h_i)
    idxs = np.argsort(h_list)
    mask_new = np.zeros_like(mask)

    for i in range(1, Object_num + 1):
        mask_new[mask == idxs[i]] = i

    return mask_new






