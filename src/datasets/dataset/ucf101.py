from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from .base_dataset import BaseDataset


class UCF101(BaseDataset):
    num_classes = 24

    def __init__(self, root_dir, mode, split=1, K=7, ninput=1, resize=(288, 288)):
        assert split == 1, "We use only the first split of UCF101"
        self.ROOT_DATASET_PATH = root_dir
        pkl_filename = 'UCF101v2-GT.pkl'
        super(UCF101, self).__init__(mode, self.ROOT_DATASET_PATH, pkl_filename, split, K, ninput,
                                             resize_height=resize[0], resize_width=resize[1])

    def imagefile(self, v, i):
        return os.path.join(self.ROOT_DATASET_PATH, 'rgb-images', v, '{:0>5}.jpg'.format(i))

    def flowfile(self, v, i):
        return os.path.join(self.ROOT_DATASET_PATH, 'brox-images', v, '{:0>5}.jpg'.format(i))
