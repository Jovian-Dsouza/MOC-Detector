import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from network.dla import MOC_DLA
from network.resnet import MOC_ResNet
from trainer.losses import MOCLoss
from MOC_utils.model import load_coco_pretrained_model

backbone = {
    'dla': MOC_DLA,
    'resnet': MOC_ResNet
}

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class MOC_Branch(nn.Module):
    def __init__(self, input_channel, arch, head_conv, branch_info, K):
        super(MOC_Branch, self).__init__()
        assert head_conv > 0
        wh_head_conv = 64 if arch == 'resnet' else head_conv

        self.hm = nn.Sequential(
            nn.Conv2d(K * input_channel, head_conv,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, branch_info['hm'],
                      kernel_size=1, stride=1,
                      padding=0, bias=True))
        self.hm[-1].bias.data.fill_(-2.19)

        self.mov = nn.Sequential(
            nn.Conv2d(K * input_channel, head_conv,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, branch_info['mov'],
                      kernel_size=1, stride=1,
                      padding=0, bias=True))
        fill_fc_weights(self.mov)

        self.wh = nn.Sequential(
            nn.Conv2d(input_channel, wh_head_conv,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(wh_head_conv, branch_info['wh'] // K,
                      kernel_size=1, stride=1,
                      padding=0, bias=True))
        fill_fc_weights(self.wh)

    def forward(self, input_chunk):
        output = {}
        output_wh = []
        for feature in input_chunk:
            output_wh.append(self.wh(feature))
        input_chunk = torch.cat(input_chunk, dim=1)
        output_wh = torch.cat(output_wh, dim=1)
        output['hm'] = self.hm(input_chunk)
        output['mov'] = self.mov(input_chunk)
        output['wh'] = output_wh
        return output


class MOC_Net(pl.LightningModule):
    def __init__(self, arch, num_classes, head_conv=256, K=7, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
        arch = arch[:arch.find('_')] if '_' in arch else arch

        branch_info = {'hm': num_classes,
                        'mov': 2 * K,
                        'wh': 2 * K}

        self.K = K
        self.backbone = backbone[arch](num_layers)
        self.branch = MOC_Branch(self.backbone.output_channel, arch, head_conv, branch_info, K)

        # Define the loss function
        self.loss = MOCLoss()


    def forward(self, x):
        chunk = [self.backbone(x[i]) for i in range(self.K)]
        return [self.branch(chunk)]

    def configure_optimizers(self):
        if self.hparams.optimizer == 'sgd':
            return optim.SGD(self.parameters(), self.hparams.lr, momentum = 0.9)
        
        elif self.hparams.optimizer == 'adam':
            return optim.Adam(self.parameters(), self.hparams.lr)
        
        elif self.hparams.optimizer == 'adamax':
            return optim.Adamax(self.parameters(), self.hparams.lr)
    
    def run_epoch(self, phase, batch, batch_idx):
        assert len(batch['input']) == self.K
        
        output = self(batch['input'])[0]
        loss, loss_stats = self.loss(output, batch)
        self.log(f'{phase}_loss', loss, prog_bar=True, logger=True)
        self.log(f'{phase}_loss_hm', loss_stats['loss_hm'], prog_bar=True, logger=True)
        self.log(f'{phase}_loss_mov', loss_stats['loss_mov'], prog_bar=True, logger=True)
        self.log(f'{phase}_loss_wh', loss_stats['loss_wh'], prog_bar=True, logger=True)

        return loss.mean()

    def training_step(self, batch, batch_idx):
        return self.run_epoch("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        self.run_epoch("val", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        self.run_epoch("test", batch, batch_idx)


if __name__ == '__main__':
    num_classes = 24
    K = 7
    arch = 'resnet_18'
    
    head_conv = 256

    model = MOC_Net(arch, num_classes, head_conv, K, lr=0.001, optimizer='adam')
    model = load_coco_pretrained_model(model, arch, print_log=False)


    input_shape = (1, 3, 288, 288)
    x = [torch.randn(input_shape)] * K 

    # y = model.backbone(x) #1, 64, 72, 72
    y = model(x)

    # print(len(y))
    print(y[0].keys())
    hm = y[0]['hm']
    mov = y[0]['mov']
    wh = y[0]['wh']

    print(hm.shape)
    print(mov.shape)
    print(wh.shape)
    
    print(model.hparams)
    model.configure_optimizers()