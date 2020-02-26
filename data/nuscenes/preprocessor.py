# Author: Xinshuo
# Email: xinshuow@cs.cmu.edu

import torch, os, numpy as np, copy
from torchvision import transforms
from PIL import Image
from utils.utils import obtain_dummy_results, convert_3dbox_to_8corner, extract_pc_in_box3d, load_velo_scan
from utils.kitti_calib import Calibration
from copy import deepcopy

from xinshuo_miscellaneous import print_log, is_path_exists
from xinshuo_io import load_list_from_folder

np.set_printoptions(suppress=True)

class preprocess(object):
    def __init__(self, data_root, seq_name, parser, log, split='train', phase='training', test_type='forecast', init_frame=1):
        # during the training phase, split of train or val can be used
        # during the testing phase, split of train, val or test can be used

        self.dataset = parser.dataset
        self.past_frames = parser.past_frames
        self.future_frames = parser.future_frames
        self.purturb_ratio = parser.purturb_ratio
        self.traj_scale = parser.traj_scale
        self.seq_name = seq_name
        self.split = split
        self.phase = phase
        self.test_type = test_type
        self.init_frame = init_frame
        self.log = log

        if split in ['train', 'val']:
            subfolder = 'train'
        elif split == 'test':
            subfolder = 'test'
        else: assert False, 'error'

        if parser.dataset in ['KITTI']:
            image_dir = os.path.join(data_root, '{}ing/image_02/{}'.format(subfolder, seq_name))
            det_path  = os.path.join(data_root, 'produced/det/{}ing/pointrcnn_combined/{}.txt'.format(subfolder, seq_name))
            self.lidar_dir = os.path.join(data_root, '{}ing/velodyne/{}'.format(subfolder, seq_name))
            calib_file = os.path.join(data_root, '{}ing/calib/{}.txt'.format(subfolder, seq_name))
            self.calib = Calibration(calib_file)
            self.minimal_pts = 5

            if split in ['train', 'val']:
                label_path = os.path.join(data_root, 'training/label_02/{}.txt'.format(seq_name))
                # label_path = os.path.join(data_root, 'produced/results/training/pointrcnn_car_val_3dtracking/data/{}.txt'.format(seq_name))

            delimiter = ' '
        elif parser.dataset in ['nuscenes']:
            image_dir = os.path.join(data_root, '{}/image_2/{}'.format(split, seq_name))
            det_path  = os.path.join(data_root, 'produced/det/{}/megvii/data/{}.txt'.format(split, seq_name))
            self.lidar_dir = os.path.join(data_root, '{}/velodyne/{}'.format(split, seq_name))
            calib_file = os.path.join(data_root, '{}/calib/{}.txt'.format(split, seq_name))
            self.calib = Calibration(calib_file)
            self.minimal_pts = 5

            if split in ['train', 'val']:
                label_path = os.path.join(data_root, '{}/label_2/{}.txt'.format(split, seq_name))

            delimiter = ' '
        else:
            assert False, 'error'

        # defining the threshold for box confidence
        if parser.dataset in ['KITTI']: self.conf_thres = 0
        elif parser.dataset in ['nuscenes']: self.conf_thres = -1000        # TO DO
        else: assert False, 'error'
        print_log('confidence threshold %f' % (self.conf_thres), log=log)

        # load data
        self.img_list, self.num_images = load_list_from_folder(image_dir)

        if split in ['train', 'val']:
            self.gt = np.genfromtxt(label_path, delimiter=delimiter, dtype=str)
            self.det = np.genfromtxt(det_path, delimiter=delimiter, dtype=str)
        elif split in ['test']:
            self.det = np.genfromtxt(det_path, delimiter=delimiter, dtype=str)

        # warm up
        image_warmup = Image.open(self.img_list[0])
        self.ImageWidth = image_warmup.size[0]
        self.ImageHeight = image_warmup.size[1]
        print_log('%d images loaded, width/height: %d/%d' % (self.num_images, self.ImageWidth, self.ImageHeight), log=log)

    def shuffle_id(self, data):
        # input is a list of data, to shuffle the order
        len_data = data.shape[0]
        order = np.arange(len_data)
        np.random.shuffle(order)
        new_data = data[order]
    
        return new_data

    def GetID(self, data):
        id = []
        for i in range(data.shape[0]):
            id.append(data[i, 1].copy())

        return id

    def TotalFrame(self):
        return len(self.img_list)

    def perturb_data(self, data, ratio):
        raise NotImplementedError("Subclasses should implement this!")

    def CurData(self, frame):
        if self.phase == 'training':
            data = self.gt[self.gt[:, 0] == frame]           # numpy array, N x 10
            data = self.perturb_data(data, self.purturb_ratio)
            data = self.shuffle_id(data)
        else:
            data = self.det[self.det[:, 0] == frame]        

        # train/test KITTI, filter out all other classes except for car
        if self.dataset in ['KITTI', 'nuscenes'] and len(data) > 0:
            data = data[data[:, 2] == 2]

        if self.phase == 'testing':
            if self.dataset in ['KITTI', 'nuscenes'] and len(data) > 0:
                data = data[data[:, 17] > self.conf_thres]           # select only confident prediction

            if len(data) == 0:          # only fill with dummy values when no data
                data = obtain_dummy_results(self.dataset, self.seq_name, frame)

        return data

    def PreData(self, frame):
        DataList = []
        for i in range(self.past_frames):
            if frame - i < 1:              
                data = []

            if self.phase == 'training':    # might be empty due to negative frame and missing detection
                data = self.gt[self.gt[:, 0] == (frame - i)]    
                data = self.perturb_data(data, self.purturb_ratio)
                # data = self.shuffle_id(data)
            else:
                if self.test_type == 'forecast':
                    data = self.gt[self.gt[:, 0] == (frame - i)]
                else:
                    past_trajectory = np.loadtxt(self.result_path)
                    data = past_trajectory[past_trajectory[:, 0] == (frame - i)]

            # train KITTI, filter out all other classes except for car
            if self.dataset in ['KITTI', 'nuscenes'] and len(data) > 0:
                data = data[data[:, 2] == 2]
 
            DataList.append(data)

        return DataList
    
    def FutureData(self, frame):
        DataList = []
        for i in range(self.future_frames + 1):
            if self.phase == 'training':
                data = self.gt[self.gt[:, 0] == (frame + i)]        # might be empty due to out of range
                data = self.perturb_data(data, self.purturb_ratio)
                # data = self.shuffle_id(data)
            else:
                data = self.gt[self.gt[:, 0] == (frame + i)]

            # train KITTI, filter out all other classes except for car
            if self.dataset in ['KITTI', 'nuscenes'] and len(data) > 0:
                data = data[data[:, 2] == 2]
            DataList.append(data)

        return DataList

    def Appearance(self, data):   
        raise NotImplementedError("Subclasses should implement this!")

    def CurMotion(self, data):
        raise NotImplementedError("Subclasses should implement this!")

    def PreMotion(self, DataTuple):
        raise NotImplementedError("Subclasses should implement this!")

    def __call__(self, frame):
        if self.phase == 'training':
            assert frame - self.init_frame >= 0 and frame - self.init_frame <= self.TotalFrame() - 2, 'frame is %d, total is %d' % (frame, self.TotalFrame())
        else:
            assert frame - self.init_frame >= -1 and frame - self.init_frame <= self.TotalFrame() - 2, 'frame is %d, total is %d' % (frame, self.TotalFrame())

        # print('using frame %d' % frame)

        cur_data = self.CurData(frame + 1)
        pre_data = self.PreData(frame)           # list with length of past frames, each item is an array of N x 9
        fut_data = self.FutureData(frame)
        # fut_data[0] = deepcopy(pre_data[0])

        # Note that pre data is in reverse order, frame 10, 9, 8, 7, 6, 5, 4, 3, 2
        # while fut data is in normal order, 11, 12, 13, 14, 15, 16, ...
        # but premotion 2D and 3D make the order normal again, in 2, 3, 4, 5, ...

        # TODO: during training, assumming the box in the previous and current frame both exists!!!!!
        if self.phase == 'training':
            if (len(cur_data) == 0) or (len(pre_data[0]) == 0) or (len(fut_data[0]) == 0):
                return None
        else:
            if (len(pre_data[0]) == 0) or (len(fut_data[0]) == 0):
                return None

        pre_motion_3D = self.PreMotion_3D(pre_data)        # list of array frames x 7
        cur_motion_3D = self.CurMotion_3D(cur_data)        # list of array 1 x 2
    
        future_motion_3D, future_motion_mask = self.FutureMotion_3D(fut_data)  # list of array frames x 3

        pre_id = self.GetID(pre_data[0])
        cur_id = self.GetID(cur_data)

        return cur_motion_3D, pre_motion_3D, future_motion_3D, future_motion_mask, \
               cur_id, pre_id, cur_data, pre_data, fut_data

class preprocess_3d(preprocess):
    def __init__(self, data_root, seq_name, parser, log, split='train', phase='training', test_type='forecast', init_frame=1):
        super(preprocess_3d, self).__init__(data_root, seq_name, parser, log, split, phase, test_type, init_frame)
        class_names = {'Pedestrian': 1, 'Car': 2, 'Cyclist': 3, 'Truck': 4, 'Van': 5, 'Tram': 6, 'Person': 7, \
            'Misc': 8, 'DontCare': 9, 'Traffic_cone': 10, 'Construction_vehicle': 11, 'Barrier': 12, 'Motorcycle': 13, \
            'Bicycle': 14, 'Bus': 15, 'Trailer': 16}
        
        for row_index in range(len(self.gt)):
            self.gt[row_index][2] = class_names[self.gt[row_index][2]]
        
        self.gt = self.gt.astype('float32')
        self.det = self.det.astype('float32')

    def perturb_data(self, data, ratio):
        ''' Randomly shift box center, randomly scale width and height 
            data[index, 10:17] = xmin, ymin, w, h
        '''
        data = copy.copy(data)
        len_data = data.shape[0]
        for data_index in range(len_data):
            r = ratio
            x1, y1, x2, y2, h, w, l, x, y, z, theta = data[data_index, 6:17]

            # purturb 2D box
            im_w, im_h = x2 - x1, y2 - y1
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            cx_new = cx + im_w * r * (np.random.random() * 2 - 1)
            cy_new = cy + im_h * r * (np.random.random() * 2 - 1)
            im_w_new  = im_w + im_w * r * (np.random.random() * 2 - 1) # 0.9 to 1.1
            im_h_new  = im_h + im_h * r * (np.random.random() * 2 - 1) # 0.9 to 1.1

            x1_new = cx_new - im_w_new / 2.0
            x2_new = cx_new + im_w_new / 2.0
            y1_new = cy_new - im_h_new / 2.0
            y2_new = cy_new + im_h_new / 2.0

            # purturb 3D box
            h_new = h + h * r * (np.random.random() * 2 - 1)
            w_new = w + w * r * (np.random.random() * 2 - 1)
            l_new = l + l * r * (np.random.random() * 2 - 1)
            
            x_new = x + l * r * (np.random.random() * 2 - 1)
            y_new = y + h * r * (np.random.random() * 2 - 1)
            z_new = z + w * r * (np.random.random() * 2 - 1)
            theta_new = theta + np.pi * r * (np.random.random() * 2 - 1)
            data[data_index, 6:17] = np.array([x1_new, y1_new, x2_new, y2_new, h_new, w_new, l_new, x_new, y_new, z_new, theta_new])

        return data

    def CurMotion_3D(self, data):                  # TODO: ablation, using only center coordinate or the entire box
        motion = []
        for i in range(data.shape[0]):
            # box_3d = data[i][10:17]
            # box_3d = data[i][13:16]
            # print(data[i])
            box_3d = data[i][[13, 15]]
            # print(box_3d)
            # zxc
            box_3d_torch = torch.from_numpy(box_3d).float()

            motion.append(box_3d_torch)

        return motion

    def PreMotion_3D(self, DataTuple):
        # this function only considers the identify in the most recent previous frame
        motion = []
        most_recent_data = DataTuple[0]
        for i in range(most_recent_data.shape[0]):    # number of objects in the most recent frame
            # box_3d = torch.zeros([self.past_frames, 7])
            # box_3d = torch.zeros([self.past_frames, 3])
            box_3d = torch.zeros([self.past_frames, 2])
            identity = most_recent_data[i, 1]
            # box_3d[self.past_frames-1, :] = torch.from_numpy(most_recent_data[i][10:17]).float()
            # box_3d[self.past_frames-1, :] = torch.from_numpy(most_recent_data[i][13:16]).float()
            box_3d[self.past_frames-1, :] = torch.from_numpy(most_recent_data[i][[13, 15]]).float()
            for j in range(1, self.past_frames):
                past_data = DataTuple[j]              # past_data
                if len(past_data) > 0 and identity in past_data[:, 1]:
                    # found_data = past_data[past_data[:, 1] == identity].squeeze()[10:17]
                    # found_data = past_data[past_data[:, 1] == identity].squeeze()[13:16]
                    found_data = past_data[past_data[:, 1] == identity].squeeze()[[13, 15]]
                    box_3d[self.past_frames-1 - j, :] = torch.from_numpy(found_data).float()
                else:
                    box_3d[self.past_frames-1 - j, :] = box_3d[self.past_frames - j, :]    # if none, copy from previous

            motion.append(box_3d)

        return motion

    def FutureMotion_3D(self, DataTuple):
        # this function only considers the identity in the most recent previous frame
        motion = []
        mask = []
        most_recent_data = DataTuple[0]
        for i in range(most_recent_data.shape[0]):    # number of objects in the most recent frame
            pos_3d = torch.zeros([self.future_frames + 1, 2])
            # pos_3d = torch.zeros([self.future_frames + 1, 3])
            # pos_3d = torch.zeros([self.future_frames + 1, 7])
            mask_i = torch.zeros(self.future_frames + 1)
            identity = most_recent_data[i, 1]
            base_pos = torch.from_numpy(most_recent_data[i][[13, 15]]).float() / self.traj_scale
            # base_pos = torch.from_numpy(most_recent_data[i][13:16]).float() / self.traj_scale
            # base_pos = torch.from_numpy(most_recent_data[i][10:17]).float() / self.traj_scale
            old_pos = base_pos.clone()
            for j in range(self.future_frames + 1):
                cur_data = DataTuple[j]              # cur_data
                if len(cur_data) > 0 and identity in cur_data[:, 1]:
                    found_data = cur_data[cur_data[:, 1] == identity].squeeze()[[13, 15]] / self.traj_scale
                    # found_data = cur_data[cur_data[:, 1] == identity].squeeze()[13:16] / self.traj_scale
                    # found_data = cur_data[cur_data[:, 1] == identity].squeeze()[10:17] / self.traj_scale
                    pos_3d[j, :] = torch.from_numpy(found_data).float()
                    mask_i[j] = 1.0
                else:
                    pos_3d[j, :] = old_pos    # if none, copy from previous
                old_pos = pos_3d[j, :].clone()

            motion.append(pos_3d[1:])
            mask.append(mask_i[1:])

        return motion, mask
