import torch
import numpy as np
import pytorch_lightning as pl
from torch import nn, optim
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score, AUROC

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding='same')
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 25 * 25, 8)
        self.fc2 = nn.Linear(8, 2)
        self.drop = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.bn1(self.relu(self.conv1(x))))
        x = x.view(-1, 16 * 25 * 25)
        x = self.drop(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class CNNLightning(pl.LightningModule):
    def __init__(self, lr, class_weights):
        super().__init__()
        self.lr = lr
        self.model = CNN()
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.loss = nn.CrossEntropyLoss(weight=self.class_weights)
        

        metrics = MetricCollection([Accuracy(task='binary'), Precision(task='binary'),
                                    Recall(task='binary'), F1Score(task='binary'), AUROC(task='binary')])
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_') 

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        class_pred = torch.argmax(torch.sigmoid(y_hat), dim=1)
        self.train_metrics(class_pred, y, on_epoch=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        class_pred = torch.argmax(torch.sigmoid(y_hat), dim=1)
        self.val_metrics(class_pred, y, on_epoch=True)
        self.log('val_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        class_pred = torch.argmax(torch.sigmoid(y_hat), dim=1)
        self.test_metrics(class_pred, y, on_epoch=True)
        self.log('test_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, logger=True)
        return loss