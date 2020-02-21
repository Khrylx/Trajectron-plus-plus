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
        elif parser.dataset in ['NuScenes']:
            image_dir = os.path.join(data_root, '{}/image_2/{}'.format(split, seq_name))
            det_path  = os.path.join(data_root, '{}/det/pointrcnn/{}.txt'.format(split, seq_name))
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
        elif parser.dataset in ['NuScenes']: self.conf_thres = -1000
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
        
        # define the transform, TODO: augment the transform
        self.image_transforms = transforms.Compose([                          
            transforms.Resize((parser.resize_h, parser.resize_w)),
            transforms.ToTensor()
        ])

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
        if self.dataset in ['KITTI'] and len(data) > 0:
            data = data[data[:, 2] == 2]

        if self.phase == 'testing':
            if self.dataset in ['KITTI'] and len(data) > 0:
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
            if self.dataset in ['KITTI'] and len(data) > 0:
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
            if self.dataset in ['KITTI'] and len(data) > 0:
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

        # print(pre_data)
        # print(fut_data)

        # # TODO: why only use the appearance from the current frame

        # pre_crop_3D = self.Appearance_3D(pre_data[0])      # list of array 3 x 84 x 32
        # cur_crop_3D = self.Appearance_3D(cur_data)         # list of array 3 x 84 x 32

        # pre_crop_2D = self.Appearance_2D(pre_data[0])      # list of array 3 x 84 x 32
        # cur_crop_2D = self.Appearance_2D(cur_data)         # list of array 3 x 84 x 32

        pre_motion_3D = self.PreMotion_3D(pre_data)        # list of array frames x 7
        cur_motion_3D = self.CurMotion_3D(cur_data)        # list of array 1 x 2
    
        # pre_motion_2D = self.PreMotion_2D(pre_data)        # list of array frames x 4
        # cur_motion_2D = self.CurMotion_2D(cur_data)        # list of array 1 x 2
    
        future_motion_3D, future_motion_mask = self.FutureMotion_3D(fut_data)  # list of array frames x 3

        # # print(pre_motion_3D[0].shape)
        # # print(pre_motion_2D[0].shape)
        # # print(future_motion_3D[0].shape)
        # # zxc
        # print(pre_motion_3D[0])
        # print(future_motion_3D[0])

        # print(pre_motion_3D[1])
        # print(future_motion_3D[1])

        # print(pre_motion_3D[2])
        # print(future_motion_3D[2])
        # zxc

        pre_id = self.GetID(pre_data[0])
        cur_id = self.GetID(cur_data)

        # return cur_crop_3D, pre_crop_3D, cur_motion_3D, pre_motion_3D, future_motion_3D, future_motion_mask, \
        #        cur_crop_2D, pre_crop_2D, cur_motion_2D, pre_motion_2D, \
        #        cur_id, pre_id, cur_data, pre_data, fut_data

        return cur_motion_3D, pre_motion_3D, future_motion_3D, future_motion_mask, \
               cur_id, pre_id, cur_data, pre_data, fut_data

class preprocess_3d(preprocess):
    def __init__(self, data_root, seq_name, parser, log, split='train', phase='training', test_type='forecast', init_frame=1):
        super(preprocess_3d, self).__init__(data_root, seq_name, parser, log, split, phase, test_type, init_frame)
        class_names = {'Pedestrian': 1, 'Car': 2, 'Cyclist': 3, 'Truck': 4, 'Van': 5, 'Tram': 6, 'Person': 7, 'Misc': 8, 'DontCare': 9}
        
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

    # def Appearance_3D(self, data):             # TODO: add 3d appearance here, ablation, using only 2d appearance, only 3d, or 2d+3d
    #     appearance = []
    #     frame_id = int(data[0, 0]) - 1   

    #     if frame_id >= 0:
    #         lidar_path = os.path.join(self.lidar_dir, '%06d.bin' % frame_id)
    #         if is_path_exists(lidar_path): 
    #             pc_velo = load_velo_scan(lidar_path)                 # 1080 x 1920 x 3
    #             pc_rect = np.zeros_like(pc_velo)
    #             pc_rect[:,0:3] = self.calib.project_velo_to_rect(pc_velo[:,0:3])
    #             pc_rect[:,3] = pc_velo[:,3]
    #         else:
    #             print_log('lidar does not exist', log=self.log, display=False)
    #             pc_rect = None           
    #     else:       
    #         pc_rect = None                              # might be a dummy frame

    #     for i in range(data.shape[0]):
    #         box_3d = data[i, 10:17]
    #         box_3d = convert_3dbox_to_8corner(box_3d)

    #         if pc_rect is not None:
    #             crop, _ = extract_pc_in_box3d(pc_rect, box_3d)   # N x 4   
    #             crop = crop.transpose()                             # 4 x N
    #             if crop.shape[1] < self.minimal_pts:           # ensure that at least 5 points are needed
    #                 crop = np.zeros((4, 100), dtype='float32')
    #         else:
    #             crop = np.zeros((4, 100), dtype='float32')

    #         crop = torch.from_numpy(crop)
    #         appearance.append(crop)

    #     return appearance

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

    # def Appearance_2D(self, data):
    #     appearance = []
    #     frame_id = int(data[0, 0]) - 1   

    #     if frame_id >= 0:
    #         img_path = self.img_list[frame_id]      
    #         img = Image.open(img_path)                 # 1080 x 1920 x 3
    #     else:                                     # might be a dummy frame
    #         img = Image.fromarray(np.zeros((self.ImageHeight, self.ImageWidth, 3), dtype='uint8'), )

    #     for i in range(data.shape[0]):
    #         x1, y1, x2, y2 = data[i, 6:10]
    #         crop = img.crop((int(x1), int(y1), int(x2), int(y2)))     # 205 x 63 x 3
    #         crop = self.image_transforms(crop)            # 3 x 84 x 32
    #         appearance.append(crop)

    #     return appearance

    # def CurMotion_2D(self, data):                  # TODO: ablation, using only center coordinate or the entire box
    #     motion = []
    #     for i in range(data.shape[0]):
    #         box_2d = data[i][6:10]
    #         box_2d_torch = torch.from_numpy(box_2d).float()

    #         motion.append(box_2d_torch)

    #     return motion

    # def PreMotion_2D(self, DataTuple):
    #     # this function only considers the identity in the most recent previous frame

    #     motion = []
    #     most_recent_data = DataTuple[0]
    #     for i in range(most_recent_data.shape[0]):    # number of objects in the most recent frame
    #         box_2d = torch.zeros([self.past_frames, 4])
    #         identity = most_recent_data[i, 1]

    #         box_2d[self.past_frames-1, :] = torch.from_numpy(most_recent_data[i][6:10]).float()
    #         # coordinate[self.past_frames-1, 0], coordinate[self.past_frames-1, 1] = self.CenterCoordinate(most_recent_data[i])
    #         for j in range(1, self.past_frames):
    #             past_data = DataTuple[j]              # past_data

    #             if len(past_data) == 0:
    #                 box_2d[self.past_frames-1 - j, :] = box_2d[self.past_frames - j, :]    # if none, copy from previous
    #             elif identity in past_data[:, 1]:
    #                 found_data = past_data[past_data[:, 1] == identity].squeeze()[6:10]
    #                 box_2d[self.past_frames-1 - j, :] = torch.from_numpy(found_data).float()

    #                 # box_2d[self.past_frames-1 - j, 0], coordinate[self.past_frames-1 - j, 1] = self.CenterCoordinate(
    #                     # past_data[past_data[:, 1] == identity].squeeze())
    #             else:
    #                 box_2d[self.past_frames-1 - j, :] = box_2d[self.past_frames - j, :]    # if none, copy from previous

    #         motion.append(box_2d)

    #     return motion

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
