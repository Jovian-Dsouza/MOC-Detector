import torch
import pytorch_lightning as pl

from model import MOC_Net
from DataModule import VideoDataModule
from MOC_utils.model import load_coco_pretrained_model


# Common params
K = 7
arch = 'resnet_18'

datamodule = VideoDataModule(root_dir='../data/ucf24',
                    K=K,
                    resize=(288, 288),
                    batch_size=4,
                    num_workers=None,
                    pin_memory=False
                    )

model = MOC_Net(arch=arch,
                num_classes=datamodule.num_classes, 
                K = K, 
                lr=5e-4, 
                optimizer='adam'
                )
model = load_coco_pretrained_model(model, arch, print_log=False)

trainer = pl.Trainer(
                    fast_dev_run=False, 
                    max_epochs=10, 
                    precision=32,
                    # datalimits = (0.2, 0.2, 1.0), 
                    # benchmark=True,
                    gpus=-1,
                    progress_bar_refresh_rate=20,
                    )
trainer.fit(model, datamodule=datamodule)