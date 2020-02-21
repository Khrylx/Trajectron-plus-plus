# Author: Xinshuo
# Email: xinshuow@cs.cmu.edu

import matplotlib as mpl; mpl.use('Agg')
import yaml, torch, os, numpy as np, copy, scipy, matplotlib.pyplot as plt
from torch import nn
from easydict import EasyDict as edict
from numba import jit
from scipy.spatial import ConvexHull, Delaunay

def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)

def Config(filename):
    listfile1 = open(filename, 'r')
    listfile2 = open(filename, 'r')
    cfg = edict(yaml.safe_load(listfile1))
    settings_show = listfile2.read().splitlines()

    return cfg, settings_show

def adjust_learning_rate(optimizer, epoch, gammas, schedule):
    assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
    multiple = 1
    for (gamma, step) in zip(gammas, schedule):
        if (epoch == step):
            multiple = gamma
            break
    all_lrs = []
    for param_group in optimizer.param_groups:
        param_group['lr'] = multiple * param_group['lr']
        all_lrs.append(param_group['lr'])
    return set(all_lrs)

def initialize_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

def FindMatch(list_id, list1, list2):
    """
    :param list_id:
    :param list1:
    :param list2:
    :return:
    """
    index_pair = []
    for index, id in enumerate(list_id):
        index1 = list1.index(id)
        index2 = list2.index(id)
        index_pair.append(index1)
        index_pair.append(index2)

    return index_pair

def LoadModel(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model

@jit
def iou(bb_test, bb_gt):
    """
    Computes IoU between two bboxes in the form [x1,y1,x2,y2]
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1]) + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
    
    return(o)

def obtain_dummy_results(dataset, seq_name, frame_index):
    # conf in gt means if to be considered
    
    # for class in MOT
    # Pedestrian 1
    # Person on vehicle 2
    # Car 3
    # Bicycle 4
    # Motorbike 5
    # Non motorized vehicle 6 
    # Static person 7 
    # Distractor 8
    # Occluder 9
    # Occluder on the ground 10 
    # Occluder full 11
    # Reflection 12

    # for class in KITTI
    # 'Pedestrian': 1
    # 'Car': 2
    # 'Cyclist': 3
    # 'Truck': 4
    # 'Van': 5
    # 'Tram': 6
    # 'Person': 7
    # 'Misc': 8
    # 'DontCare': 9

    if dataset == 'MOT15':                    
        # gt:  frame, id, x1, y1, w, h, conf, x, y, z
        # det: frame, id, x1, y1, w, h, conf, x, y, z
        data_det = np.array([[frame_index, -1, 1., 1., 2., 2., 0., -1, -1, -1,]])        
    elif dataset == 'MOT16':                  
        # gt:  frame, id, x1, y1, w, h, conf, class, visibility
        # det: frame, id, x1, y1, w, h, conf, x, y, z 
        data_det = np.array([[frame_index, -1, 1., 1., 2., 2., 0., -1, -1, -1]])        
    elif dataset == 'MOT17':                  
        # gt:  frame, id, x1, y1, w, h, conf, class, visibility
        # det: frame, id, x1, y1, w, h, conf
        if 'FRCNN' in seq_name:
            data_det = np.array([[frame_index, -1, 1., 1., 2., 2., 0.]])        
        elif 'DPM' in seq_name:
            data_det = np.array([[frame_index, -1, 1., 1., 2., 2., 0., -1, -1, -1]])        
        elif 'SDP' in seq_name:
            data_det = np.array([[frame_index, -1, 1., 1., 2., 2., 0.]])        
        else: assert False, 'error'
    elif dataset == 'MOT19':                  
        # gt:  frame, id, x1, y1, w, h, conf, class, visibility
        # det: frame, id, x1, y1, w, h, conf, x, y, z
        data_det = np.array([[frame_index, -1, 1., 1., 2., 2., 0., -1, -1, -1]])        
    elif dataset == 'KITTI':                 
        # gt:  frame, id, class_str, truncated, occluded, alpha, x1, y1, x2, y2, h, w, l, x, y, z, theta
        # det: frame, id, class_num, truncated, occluded, alpha, x1, y1, x2, y2, h, w, l, x, y, z, theta, conf
        data_det = np.array([[frame_index, -1, 2, -1, -1, -10, 1, 1, 2, 2, 0.1, 0.1, 0.1, 0, 0, 0, -10, 0]])
    else:
        assert False, 'error'

    return data_det

def plot_loss(loss, val_loss, save_path):
    loss = np.array(loss)
    val_loss = np.array(val_loss)
    plt.figure("loss")
    plt.gcf().clear()
    plt.plot(loss[:, 0], label='train')
    plt.plot(val_loss[:, 0], label='valid')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(save_path)

def plot_accu(accu, val_accu, save_path):
    accu = np.array(accu)
    val_accu = np.array(val_accu)
    plt.figure("accu")
    plt.gcf().clear()
    plt.plot(accu[:, 0], label='train')
    plt.plot(val_accu[:, 0], label='valid')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig(save_path)

##################################################### 3d box processing

@jit          # with this 119.5
def poly_area(x,y):
    """ Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

@jit          # with this 207.7
def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2))
    b = np.sqrt(np.sum((corners[1,:] - corners[2,:])**2))
    c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2))
    return a*b*c

@jit          # with this 211.6
def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1,p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0  

# @jit
def polygon_clip(subjectPolygon, clipPolygon):
   """ Clip a polygon with another polygon.

   Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python

   Args:
     subjectPolygon: a list of (x,y) 2d points, any polygon.
     clipPolygon: a list of (x,y) 2d points, has to be *convex*
   Note:
     **points have to be counter-clockwise ordered**

   Return:
     a list of (x,y) vertex point for the intersection polygon.
   """
   def inside(p):
      return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])
 
   def computeIntersection():
      dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
      dp = [ s[0] - e[0], s[1] - e[1] ]
      n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
      n2 = s[0] * e[1] - s[1] * e[0] 
      n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
      return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]
 
   outputList = subjectPolygon
   cp1 = clipPolygon[-1]
 
   for clipVertex in clipPolygon:
      cp2 = clipVertex
      inputList = outputList
      outputList = []
      s = inputList[-1]
 
      for subjectVertex in inputList:
         e = subjectVertex
         if inside(e):
            if not inside(s):
               outputList.append(computeIntersection())
            outputList.append(e)
         elif inside(s):
            outputList.append(computeIntersection())
         s = e
      cp1 = cp2
      if len(outputList) == 0:
          return None
   return(outputList)

# @jit
def iou3d(corners1, corners2):
    ''' Compute 3D bounding box IoU.

    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU

    todo (rqi): add more description on corner points' orders.
    '''
    # corner points are in counter clockwise order
    rect1 = [(corners1[i,0], corners1[i,2]) for i in range(3,-1,-1)]
    rect2 = [(corners2[i,0], corners2[i,2]) for i in range(3,-1,-1)] 
    area1 = poly_area(np.array(rect1)[:,0], np.array(rect1)[:,1])
    area2 = poly_area(np.array(rect2)[:,0], np.array(rect2)[:,1])

    # TODO: has bug here
    try:
      _, inter_area = convex_hull_intersection(rect1, rect2)
    except ValueError:
      # print(rect1)
      # print(rect2)
      # zxc
      inter_area = 0
    except scipy.spatial.qhull.QhullError:
      # print(rect1)
      # print(rect2)
      # zxc
      inter_area = 0
    # print(inter_area)
    # zxc

    iou_2d = inter_area/(area1+area2-inter_area)
    ymax = min(corners1[0,1], corners2[0,1])
    ymin = max(corners1[4,1], corners2[4,1])
    inter_vol = inter_area * max(0.0, ymax-ymin)
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou, iou_2d

# @jit          # with this 214.7 fps
def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],[0,  1,  0],[-s, 0,  c]])

# @jit          # with this 2
def convert_3dbox_to_8corner(bbox3d_input):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
    
        bbox3d_input: h, w, l, x, y, z, theta

        Returns:
            corners_2d: (8,2) array in left image coord.
            corners_3d: (8,3) array in in rect camera coord.
    '''
    # compute rotational matrix around yaw axis
    bbox3d = copy.copy(bbox3d_input)

    R = roty(bbox3d[6])    

    # 3d bounding box dimensions
    l = bbox3d[2]
    w = bbox3d[1]
    h = bbox3d[0]
    
    # 3d bounding box corners
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
    y_corners = [0,0,0,0,-h,-h,-h,-h];
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];
    
    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    #print corners_3d.shape
    corners_3d[0,:] = corners_3d[0,:] + bbox3d[3]
    corners_3d[1,:] = corners_3d[1,:] + bbox3d[4]
    corners_3d[2,:] = corners_3d[2,:] + bbox3d[5]
 
    return np.transpose(corners_3d)

def load_velo_scan(velo_filename):
    scan = np.fromfile(velo_filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    return scan

def in_hull(p, hull):
    if not isinstance(hull, Delaunay): hull = Delaunay(hull)
    return hull.find_simplex(p)>=0

def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,4), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:,0:3], box3d)
    return pc[box3d_roi_inds,:], box3d_roi_inds
