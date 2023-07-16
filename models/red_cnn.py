import numpy as np
import pytorch_lightning as pl
from torch import nn, optim
from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio, MetricCollection
from models.scheduler import RampedLR

# REDCNN model from https://github.com/SSinyu/RED-CNN
class REDCNN(nn.Module):
    def __init__(self, out_ch=96):
        super(REDCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv4 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv5 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)

        self.tconv1 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv2 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv3 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv4 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv5 = nn.ConvTranspose2d(out_ch, 1, kernel_size=5, stride=1, padding=0)

        self.relu = nn.ReLU()

    def forward(self, x):
        # encoder
        residual_1 = x
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        residual_2 = out
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        residual_3 = out
        out = self.relu(self.conv5(out))
        # decoder
        out = self.tconv1(out)
        out += residual_3
        out = self.tconv2(self.relu(out))
        out = self.tconv3(self.relu(out))
        out += residual_2
        out = self.tconv4(self.relu(out))
        out = self.tconv5(self.relu(out))
        out += residual_1
        out = self.relu(out)
        return out
    
# get image from tensor
def to_image( x):
    x = x[0].permute(1, 2, 0)
    x = x.cpu().squeeze().numpy()
    x = np.clip(x, 0, 1) * 255
    return x

# Pytorch Lightning module
class RedCNNLightning(pl.LightningModule):
    def __init__(self, mode='n2c', lr=1e-4, epochs=10):
        super().__init__()
        self.mode = mode
        self.lr = lr
        self.epochs = epochs
        self.model = REDCNN()
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
        #log only one image from the test set
        if batch_idx in [10, 75, 150, 200]:
            self.logger.log_image(key='noisy image', images=[to_image(x)])
            self.logger.log_image(key='denoised image', images=[to_image(y_hat)])
            self.logger.log_image(key='ground truth', images=[to_image(gt)])

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.1, 0.99))
        scheduler = RampedLR(optimizer, self.epochs, 0.1, 0.3)
        return [optimizer], [scheduler]

