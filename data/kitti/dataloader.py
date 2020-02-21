# Author: Xinshuo
# Email: xinshuow@cs.cmu.edu

import os, random, numpy as np, copy

from preprocessor import preprocess_3d
from utils.utils import FindMatch
from xinshuo_miscellaneous import print_log

class data_generator(object):
    def __init__(self, parser, log, split='train', phase='training', test_type='forecast'):
        self.past_frames = parser.past_frames
        self.test_type = test_type
        self.sequence = []

        assert phase in ['training', 'testing'], 'error'
        assert split in ['train', 'val', 'test'], 'error'

        self.phase = phase
        self.split = split

        if parser.dataset == 'KITTI':
            data_root = parser.data_root_kitti            
            # seq_train  = ['%04d' % i for i in range(21)]      # all seqs
            seq_train  = ['0000', '0002', '0003', '0004', '0005', '0007', '0009', '0011', '0017', '0020']
            seq_val = ['0001', '0006', '0008', '0010', '0012', '0013', '0014', '0015', '0016', '0018', '0019']
            seq_test  = ['%04d' % i for i in range(29)]
            # seq_train = list(filter(lambda x: x not in seq_val, seq_train))
            # print(seq_train)
            # zxc
            self.init_frame = 0
        else:
            assert False, 'dataset error'

        process_func = preprocess_3d
        self.data_root = data_root

        print_log("\n-------------------------- loading %s data --------------------------" % split, log=log)
        if self.split == 'train':
            self.sequence_to_load = seq_train
        elif self.split == 'val':
            self.sequence_to_load = seq_val
        elif self.split == 'test':
            self.sequence_to_load = seq_test
        else:
            assert False, 'error'

        self.num_total_samples = 0
        self.num_sample_list = []
        for seq_name in self.sequence_to_load:
            print_log("loading sequence {} ...".format(seq_name), log=log)
            preprocessor = process_func(data_root, seq_name, parser, log, self.split, self.phase, self.test_type, self.init_frame)

            num_seq_samples = preprocessor.num_images - parser.past_frames - parser.future_frames + 1
            self.num_total_samples += num_seq_samples
            self.num_sample_list.append(num_seq_samples)
            self.sequence.append(preprocessor)
        
        self.sample_list = list(range(self.num_total_samples))
        self.index = 0
        print_log("------------------------------ done --------------------------------\n", log=log)

    def shuffle(self):
        random.shuffle(self.sample_list)
        
    def get_seq_and_frame(self, index):
        index_tmp = copy.copy(index)
        for seq_index in range(len(self.num_sample_list)):    # 0-indexed
            if index_tmp < self.num_sample_list[seq_index]:
                frame_index = index_tmp + self.past_frames - 1      # from 0-indexed list index to 1-indexed frame index (for mot)
                return seq_index, frame_index
            else:
                index_tmp -= self.num_sample_list[seq_index]

        assert False, 'index is %d, out of range' % (index)

    def get_gt_matrix(self, pre_id, cur_id):
        list_id = [x for x in pre_id if x in cur_id]        # id co-exists in both frames
        index_pair = FindMatch(list_id, pre_id, cur_id)
        gt_matrix = np.zeros([len(pre_id), len(cur_id)])  
        for i in range(len(index_pair) // 2):
            gt_matrix[index_pair[2 * i], index_pair[2 * i + 1]] = 1

        return gt_matrix

    def is_epoch_end(self):
        if self.index >= self.num_total_samples:
            self.index = 0      # reset
            return True
        else:
            return False

    def __call__(self):
        sample_index = self.sample_list[self.index]
        seq_index, frame = self.get_seq_and_frame(sample_index)
        seq = self.sequence[seq_index]
        self.index += 1

        data = seq(frame)
        if data is None:
            return None, seq.seq_name, frame

        # cur_crop_3D, pre_crop_3D, cur_motion_3D, pre_motion_3D, fut_motion_3D, fut_motion_mask, \
            # cur_crop_2D, pre_crop_2D, cur_motion_2D, pre_motion_2D, cur_id, pre_id, cur_data, pre_data, fut_data = data

        cur_motion_3D, pre_motion_3D, fut_motion_3D, fut_motion_mask, \
            cur_id, pre_id, cur_data, pre_data, fut_data = data

        if self.phase == 'training':
            gt_matrix = self.get_gt_matrix(pre_id, cur_id)
        else:
            gt_matrix = None
            
        # return cur_crop_3D, pre_crop_3D, cur_motion_3D, pre_motion_3D, fut_motion_3D, fut_motion_mask, \
        #     cur_crop_2D, pre_crop_2D, cur_motion_2D, pre_motion_2D, pre_data, fut_data, \
        #     gt_matrix, seq.seq_name, frame

        return cur_motion_3D, pre_motion_3D, fut_motion_3D, fut_motion_mask, \
            pre_data, fut_data, gt_matrix, seq.seq_name, frame

if __name__ == '__main__':
    import sys
    from utils.config import Config
    sys.path.append('../')
    np.random.seed(1)
    cfg = Config('config')
    log = open(os.path.join(cfg.log_dir, 'log.txt'), 'w')
    generator = data_generator(cfg, log, split='train', phase='training')
    generator.shuffle()
    data = generator()
