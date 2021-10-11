import pytorch_lightning as pl
import torch
import os

from torch.utils import data

from datasets.init_dataset import get_dataset

class VideoDataModule(pl.LightningDataModule):
    def __init__(self,
                 root_dir,
                 K, 
                 resize,
                 batch_size, 
                 num_workers=None, 
                 pin_memory=False):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size 
        self.num_workers = os.cpu_count() - 1 if num_workers is None else num_workers
        self.pin_memory = pin_memory

        self.Dataset = get_dataset("ucf101") #ucf101 or hmdb
        self.num_classes = self.Dataset.num_classes
        self.K = K
        self.resize = resize

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
                    self.Dataset(self.root_dir,
                                'train',
                                K=self.K, 
                                resize=self.resize
                                ),
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    drop_last=True,
                )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
                    self.Dataset(self.root_dir,
                                'val',
                                K=self.K, 
                                resize=self.resize
                                ),
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    drop_last=True,
                )

if __name__ == '__main__':
    datamodule = VideoDataModule(root_dir='../data/ucf24',
                    K=7,
                    resize=(288, 288),
                    batch_size=2,
                    num_workers=0,
                    pin_memory=False
                    )
    print("Number of classes ", datamodule.num_classes)

    train_dl = datamodule.train_dataloader()
    print("Len of train_dl", len(train_dl))
    # data = next(iter(train_dl))

    for data in train_dl:
        break

    print(data.keys()) # 'input', 'hm', 'mov', 'wh', 'mask', 'index', 'index_all']

    val_dl = datamodule.val_dataloader()
    print("Len of val_dl", len(val_dl))
    for data in val_dl:
        break

    print(data.keys())