# From Noise2Sim Repo:

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import pytorch_lightning as pl
from torch import nn, optim
from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio, MetricCollection
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from models.scheduler import RampedLR


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, activation, mid_channels=None, use_bn=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if use_bn:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels),
                activation,
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                activation
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                activation,
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                activation
            )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, activation, use_bn=False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, activation, use_bn=use_bn)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, activation, bilinear=False, use_bn=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, activation, in_channels // 2, use_bn=use_bn)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, activation, use_bn=use_bn)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet2(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, residual=True, activation_type="relu", use_bn=True):
        super(UNet2, self).__init__()

        if activation_type == "leaky_relu":
            activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation_type == "relu":
            activation = nn.ReLU(inplace=True)
        else:
            raise TypeError

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 96, activation, use_bn=use_bn)
        self.down1 = Down(96, 96*2, activation, use_bn=use_bn)
        self.down2 = Down(96*2, 96*4, activation, use_bn=use_bn)

        self.up1 = Up(96*4, 96*2, activation, use_bn=use_bn)
        self.up2 = Up(96*2, 96*1, activation, use_bn=use_bn)
        self.outc = OutConv(96, n_classes)
        self.residual = residual

    def forward(self, input):
        x1 = self.inc(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.outc(x)
        if self.residual:
            x = input + x
        return x

def to_image( x):
    x = x[0].permute(1, 2, 0)
    x = x.cpu().squeeze().numpy()
    x = np.clip(x, 0, 1) * 255
    return x

class UNetLightning(pl.LightningModule):
    def __init__(self, mode='n2c', lr=1e-4, epochs=10):
        super().__init__()
        self.mode = mode
        self.lr = lr
        self.epochs = epochs
        self.model = UNet2(1, 1)
        self.criterion = nn.L1Loss()

        metrics = MetricCollection([StructuralSimilarityIndexMeasure(), PeakSignalNoiseRatio()])
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')   

    def training_step(self, batch, batch_idx):
        x, y, img_ignore, gt = batch
        y_hat = self.model(x)
        if self.mode == 'n2c':
            loss = self.criterion(y_hat, gt) * 10
        elif self.mode == 'n2s':
            mask = 1 - img_ignore
            y_hat = self.model(x)
            y = y * mask
            y_hat = y_hat * mask
            loss = self.criterion(y_hat, y) * 10
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, _, gt = batch
        y_hat = self.model(x)
        if self.mode == 'n2c':
            loss = self.criterion(y_hat, gt) * 10
        elif self.mode == 'n2s':
            loss = self.criterion(y_hat, y) * 10
        self.log('val_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.val_metrics(y_hat, gt, on_epoch=True)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, _, _, gt = batch
        y_hat = self.model(x)
        self.test_metrics(y_hat, gt, on_step=False, on_epoch=True)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, logger=True)
        #log images to wandb
        if batch_idx in [10, 75, 150, 200]:
            self.logger.log_image(key='noisy image', images=[to_image(x)])
            self.logger.log_image(key='denoised image', images=[to_image(y_hat)])
            self.logger.log_image(key='ground truth', images=[to_image(gt)])
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.1, 0.99))
        scheduler = RampedLR(optimizer, self.epochs, 0.1, 0.3)
        return [optimizer], [scheduler]