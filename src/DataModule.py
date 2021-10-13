import math 
import random
import pytorch_lightning as pl
import torch
import os
import pickle
import cv2
import numpy as np

from torch.utils import data
from datasets.init_dataset import get_dataset
from ACT_utils.ACT_utils import tubelet_in_out_tubes, tubelet_has_gt
from MOC_utils.gaussian_hm import gaussian_radius, draw_umich_gaussian
from ACT_utils.ACT_aug import apply_distort, apply_expand, crop_image

from pprint import pprint

class UCFDataset(data.Dataset):
    def __init__(self,
                 root_dir, 
                 mode, # train or val
                 pkl_filename = 'UCF101v2-GT.pkl', 
                 K=7, 
                 skip=1,
                 downratio=4,
                 mean=[0.40789654, 0.44719302, 0.47026115],
                 std=[0.28863828, 0.27408164, 0.27809835], 
                 resize=(288, 288), # (h, w)
                 max_objs=128):
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.K = K
        self.skip = skip # TODO implement skiping frames in getitem
        self._resize_height = resize[0]
        self._resize_width = resize[1]
        self.down_ratio = downratio
        self.mean = mean
        self.std = std
        self.max_objs = max_objs

        pkl_file = os.path.join(root_dir, pkl_filename)
        with open(pkl_file, 'rb') as fid:
            pkl = pickle.load(fid, encoding='iso-8859-1')
        for k in pkl:
            setattr(self, ('_' if k != 'labels' else '') + k, pkl[k])
        # labels, _nframes, _train_videos, _test_videos
        # _gttubes, _resolution

        self.num_classes = len(self.labels)

        self._indices = []
        video_list = self._train_videos if mode == 'train' else self._test_videos
        for v in video_list:
            vtubes = sum(self._gttubes[v].values(), [])
            self._indices += [(v, i) for i in range(1, self._nframes[v] + 2 - self.K, self.K)
                              if tubelet_in_out_tubes(vtubes, i, self.K) and tubelet_has_gt(vtubes, i, self.K)]
        
        self.init_aug_params()


    def init_aug_params(self):
        self._mean_values = [104.0136177, 114.0342201, 119.91659325]
        self.distort_param = {
            'brightness_prob': 0.5,
            'brightness_delta': 32,
            'contrast_prob': 0.5,
            'contrast_lower': 0.5,
            'contrast_upper': 1.5,
            'hue_prob': 0.5,
            'hue_delta': 18,
            'saturation_prob': 0.5,
            'saturation_lower': 0.5,
            'saturation_upper': 1.5,
            'random_order_prob': 0.0,
        }
        self.expand_param = {
            'expand_prob': 0.5,
            'max_expand_ratio': 4.0,
        }
        self.batch_samplers = [{
            'sampler': {},
            'max_trials': 1,
            'max_sample': 1,
        }, {
            'sampler': {'min_scale': 0.3, 'max_scale': 1.0, 'min_aspect_ratio': 0.5, 'max_aspect_ratio': 2.0, },
            'sample_constraint': {'min_jaccard_overlap': 0.1, },
            'max_trials': 50,
            'max_sample': 1,
        }, {
            'sampler': {'min_scale': 0.3, 'max_scale': 1.0, 'min_aspect_ratio': 0.5, 'max_aspect_ratio': 2.0, },
            'sample_constraint': {'min_jaccard_overlap': 0.3, },
            'max_trials': 50,
            'max_sample': 1,
        }, {
            'sampler': {'min_scale': 0.3, 'max_scale': 1.0, 'min_aspect_ratio': 0.5, 'max_aspect_ratio': 2.0, },
            'sample_constraint': {'min_jaccard_overlap': 0.5, },
            'max_trials': 50,
            'max_sample': 1,
        }, {
            'sampler': {'min_scale': 0.3, 'max_scale': 1.0, 'min_aspect_ratio': 0.5, 'max_aspect_ratio': 2.0, },
            'sample_constraint': {'min_jaccard_overlap': 0.7, },
            'max_trials': 50,
            'max_sample': 1,
        }, {
            'sampler': {'min_scale': 0.3, 'max_scale': 1.0, 'min_aspect_ratio': 0.5, 'max_aspect_ratio': 2.0, },
            'sample_constraint': {'min_jaccard_overlap': 0.9, },
            'max_trials': 50,
            'max_sample': 1,
        }, {
            'sampler': {'min_scale': 0.3, 'max_scale': 1.0, 'min_aspect_ratio': 0.5, 'max_aspect_ratio': 2.0, },
            'sample_constraint': {'max_jaccard_overlap': 1.0, },
            'max_trials': 50,
            'max_sample': 1,
        }, ]

    def __len__(self):
        return len(self._indices)
    
    def imagefile(self, v, i):
        return os.path.join(self.root_dir, 'rgb-images', v, '{:0>5}.jpg'.format(i))
    
    def flip_video(self, images, frame, v):
        do_mirror = random.getrandbits(1) == 1
        # filp the image
        if do_mirror:
            images = [im[:, ::-1, :] for im in images]

        h, w = self._resolution[v]
        gt_bbox = {}
        for ilabel, tubes in self._gttubes[v].items():
            for t in tubes:
                if frame not in t[:, 0]:
                    continue
                assert frame + self.K - 1 in t[:, 0]
                # copy otherwise it will change the gt of the dataset also
                t = t.copy()
                if do_mirror:
                    # filp the gt bbox
                    xmin = w - t[:, 3]
                    t[:, 3] = w - t[:, 1]
                    t[:, 1] = xmin
                boxes = t[(t[:, 0] >= frame) * (t[:, 0] < frame + self.K), 1:5]

                assert boxes.shape[0] == self.K
                if ilabel not in gt_bbox:
                    gt_bbox[ilabel] = []
                # gt_bbox[ilabel] ---> a list of numpy array, each one is K, x1, x2, y1, y2
                gt_bbox[ilabel].append(boxes)
        return images, gt_bbox

    def make_gttbox(self, frame, v):
        gt_bbox = {}
        for ilabel, tubes in self._gttubes[v].items():
            for t in tubes:
                if frame not in t[:, 0]:
                    continue
                assert frame + self.K - 1 in t[:, 0]
                t = t.copy()
                boxes = t[(t[:, 0] >= frame) * (t[:, 0] < frame + self.K), 1:5]
                assert boxes.shape[0] == self.K
                if ilabel not in gt_bbox:
                    gt_bbox[ilabel] = []
                gt_bbox[ilabel].append(boxes)
        return gt_bbox

    def resize_video(self, images, gt_bbox):
        original_h, original_w = images[0].shape[:2]
        output_h = self._resize_height // self.down_ratio
        output_w =  self._resize_width // self.down_ratio
        # resize the original img and it's GT bbox
        for ilabel in gt_bbox:
            for itube in range(len(gt_bbox[ilabel])):
                gt_bbox[ilabel][itube][:, 0] = gt_bbox[ilabel][itube][:, 0] / original_w * output_w
                gt_bbox[ilabel][itube][:, 1] = gt_bbox[ilabel][itube][:, 1] / original_h * output_h
                gt_bbox[ilabel][itube][:, 2] = gt_bbox[ilabel][itube][:, 2] / original_w * output_w
                gt_bbox[ilabel][itube][:, 3] = gt_bbox[ilabel][itube][:, 3] / original_h * output_h
        images = [cv2.resize(im, (self._resize_width, self._resize_height), interpolation=cv2.INTER_LINEAR) for im in images]
        return images, gt_bbox

    def normalize(self, images):
        data = [np.empty((3, self._resize_height, self._resize_width), dtype=np.float32) for i in range(self.K)]
        mean = np.tile(np.array(self.mean, dtype=np.float32)[:, None, None], (1, 1, 1))
        std = np.tile(np.array(self.std, dtype=np.float32)[:, None, None], (1, 1, 1))
        for i in range(self.K):
            data[i][0:3, :, :] = np.transpose(images[i], (2, 0, 1))
            data[i] = ((data[i] / 255.) - mean) / std
        return data

    def draw_ground_truths(self, gt_bbox):
        output_h = self._resize_height // self.down_ratio
        output_w =  self._resize_width // self.down_ratio

        hm = np.zeros((self.num_classes, output_h, output_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, self.K * 2), dtype=np.float32)
        mov = np.zeros((self.max_objs, self.K * 2), dtype=np.float32)
        index = np.zeros((self.max_objs), dtype=np.int64)
        index_all = np.zeros((self.max_objs, self.K * 2), dtype=np.int64)
        mask = np.zeros((self.max_objs), dtype=np.uint8)

        num_objs = 0
        for ilabel in gt_bbox:
            for itube in range(len(gt_bbox[ilabel])):
                key = self.K // 2
                # key frame's bbox height and width （both on the feature map）
                key_h, key_w = gt_bbox[ilabel][itube][key, 3] - gt_bbox[ilabel][itube][key, 1], gt_bbox[ilabel][itube][key, 2] - gt_bbox[ilabel][itube][key, 0]
                # create gaussian heatmap
                radius = gaussian_radius((math.ceil(key_h), math.ceil(key_w)))
                radius = max(0, int(radius))

                # ground truth bbox's center in key frame
                center = np.array([(gt_bbox[ilabel][itube][key, 0] + gt_bbox[ilabel][itube][key, 2]) / 2, (gt_bbox[ilabel][itube][key, 1] + gt_bbox[ilabel][itube][key, 3]) / 2], dtype=np.float32)
                center_int = center.astype(np.int32)
                assert 0 <= center_int[0] and center_int[0] <= output_w and 0 <= center_int[1] and center_int[1] <= output_h

                # draw ground truth gaussian heatmap at each center location
                draw_umich_gaussian(hm[ilabel], center_int, radius)

                for i in range(self.K):
                    center_all = np.array([(gt_bbox[ilabel][itube][i, 0] + gt_bbox[ilabel][itube][i, 2]) / 2,  (gt_bbox[ilabel][itube][i, 1] + gt_bbox[ilabel][itube][i, 3]) / 2], dtype=np.float32)
                    center_all_int = center_all.astype(np.int32)
                    # wh is ground truth bbox's height and width in i_th frame
                    wh[num_objs, i * 2: i * 2 + 2] = 1. * (gt_bbox[ilabel][itube][i, 2] - gt_bbox[ilabel][itube][i, 0]), 1. * (gt_bbox[ilabel][itube][i, 3] - gt_bbox[ilabel][itube][i, 1])
                    # mov is ground truth movement from i_th frame to key frame
                    mov[num_objs, i * 2: i * 2 + 2] = (gt_bbox[ilabel][itube][i, 0] + gt_bbox[ilabel][itube][i, 2]) / 2 - \
                        center_int[0],  (gt_bbox[ilabel][itube][i, 1] + gt_bbox[ilabel][itube][i, 3]) / 2 - center_int[1]
                    # index_all are all frame's bbox center position
                    index_all[num_objs, i * 2: i * 2 + 2] = center_all_int[1] * output_w + center_all_int[0], center_all_int[1] * output_w + center_all_int[0]
                # index is key frame's boox center position
                index[num_objs] = center_int[1] * output_w + center_int[0]
                # mask indicate how many objects in this tube
                mask[num_objs] = 1
                num_objs = num_objs + 1

        return hm, wh, mov, index, index_all, mask

    def __getitem__(self, id):
        v, frame = self._indices[id]
        
        # Read images
        images = [cv2.imread(self.imagefile(v, frame + i)).astype(np.float32) for i in range(0,self.K,self.skip)]
        
        if self.mode == 'train':
            # apply data augmentation
            images, gt_bbox = self.flip_video(images, frame, v)
            images = apply_distort(images, self.distort_param)
            images, gt_bbox = apply_expand(images, gt_bbox, self.expand_param, self._mean_values)
            images, gt_bbox = crop_image(images, gt_bbox, self.batch_samplers)
        else:
            # no data augmentation or flip when validation
            gt_bbox = self.make_gttbox(frame, v)
        
        # Resize the video
        images, gt_bbox = self.resize_video(images, gt_bbox)
        data = self.normalize(images)

        hm, wh, mov, index, index_all, mask = self.draw_ground_truths(gt_bbox)
        return {'input': data, 'hm': hm, 'mov': mov, 'wh': wh, 'mask': mask, 'index': index, 'index_all': index_all}

    def _draw_bb(self, video, frame, index):
        i = index
        for label in self._gttubes[video]:
            # print(label)
            tubes = self._gttubes[video][label]
            for tube in tubes:
                x = np.where(tube[..., 0] == i)[0]
                if (len(x) != 0): 
                    x = int(x)
                    x1, y1, x2, y2 = tube[x, 1:] 
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
        return frame   

    def save_video(self, index, fps=25, drawbb=True, save_dir='.'):
        video, start_frame = self._indices[index]
        h, w = self._resolution[video]
        save_path = video.split(os.path.sep)[-1] + '_'+ str(index) + '.mp4'
        save_path = os.path.join(save_dir, save_path)
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        for i in range(start_frame, start_frame+self.K, self.skip):
            frame = cv2.imread(self.imagefile(video, i))
            if drawbb:
                frame = self._draw_bb(video, frame, i)      
            out.write(frame)

        out.release()



class VideoDataModule(pl.LightningDataModule):
    def __init__(self,
                 root_dir,
                 pkl_file,
                 K, 
                 resize,
                 batch_size, 
                 num_workers=None, 
                 pin_memory=False):
        super().__init__()
        self.root_dir = root_dir
        self.pkl_file = pkl_file
        self.batch_size = batch_size 
        self.num_workers = os.cpu_count() - 1 if num_workers is None else num_workers
        self.pin_memory = pin_memory

        self.Dataset = get_dataset("ucf101") #ucf101 or hmdb
        self.num_classes = self.Dataset.num_classes
        self.K = K
        self.resize = resize

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
                    UCFDataset(root_dir=self.root_dir,
                         pkl_filename=self.pkl_file,
                         mode='train',
                         K=self.K, 
                         resize=self.resize,
                    ),
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    drop_last=True,
                )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
                    UCFDataset(root_dir=self.root_dir,
                         pkl_filename=self.pkl_file,
                         mode='val',
                         K=self.K, 
                         resize=self.resize,
                    ),
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    drop_last=True,
                )

# TEST CASES
def test_dataset():
    dataset = UCFDataset(root_dir='../data/ucf24',
                         mode='train',
                         pkl_filename='SalsaSpin.pkl',
                         K=7
                )
    print("len of dataset ", len(dataset))
    data = dataset.__getitem__(0)
    print(data.keys()) # 'input', 'hm', 'mov', 'wh', 'mask', 'index', 'index_all']
    print(data['input'][0].shape)
    print(data['hm'].shape)
    print(data['mov'].shape)
    print(data['wh'].shape)

    # save_dir = '../SalsaSpin'
    # os.makedirs(save_dir, exist_ok=True)
    # for i in range(len(dataset)):
    #     dataset.save_video(i, fps=1, save_dir=save_dir, drawbb=True)

if __name__ == '__main__':
    datamodule = VideoDataModule(root_dir='../data/ucf24',
                    pkl_file="SalsaSpin.pkl",
                    K=7,
                    resize=(288, 288),
                    batch_size=2,
                    num_workers=0,
                    pin_memory=False
                    )
    print("Number of classes ", datamodule.num_classes)

    train_dl = datamodule.train_dataloader()
    print("Len of train_dl", len(train_dl))

    for data in train_dl:
        break

    print(data.keys()) # 'input', 'hm', 'mov', 'wh', 'mask', 'index', 'index_all']
    print(data['hm'].shape)
    print(data['mov'].shape)
    print(data['wh'].shape)
    val_dl = datamodule.val_dataloader()
    print("Len of val_dl", len(val_dl))
    for data in val_dl:
        break

    print(data.keys())