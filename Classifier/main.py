import time
import torch
import numpy as np
import pytorch_lightning as pl
from models.cnn import CNNLightning
from utils.data import LIDC
from torchvision import transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from utils.parser import parser


def main():
    args = parser()

    # set seed for reproducibility
    pl.seed_everything(42)
    torch.set_float32_matmul_precision('medium')

    patches = np.load('data/patches.npy')
    labels = np.load('data/labels.npy')

    # compute class weights for imbalanced dataset
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)

    # split data into train, val, test
    train_patches, test_patches, train_labels, test_labels = train_test_split(patches, labels, test_size=0.2, random_state=42, stratify=labels)
    train_patches, val_patches, train_labels, val_labels = train_test_split(train_patches, train_labels, test_size=0.2, random_state=42, stratify=train_labels)

    transform = transforms.Compose([transforms.RandomRotation(10),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomVerticalFlip()])

    train_dataset = LIDC(train_patches, train_labels, transforms=transform, denoised=args.denoised)
    val_dataset = LIDC(val_patches, val_labels, denoised=args.denoised)
    test_dataset = LIDC(test_patches, test_labels, denoised=args.denoised)

    sampler = WeightedRandomSampler(torch.from_numpy(class_weights).float(), len(train_dataset), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    callbacks = [EarlyStopping(monitor='val_loss', patience=args.patience, mode='min'),
                 ModelCheckpoint(monitor='val_loss', mode='min')]

    if args.denoised:
        run_name = f'{time.strftime("%Y%m%d-%H%M%S")}-denoised'
    else:
        run_name = f'{time.strftime("%Y%m%d-%H%M%S")}'

    logger = WandbLogger(name=run_name, project='Luna16', log_model='all')
    
    model = CNNLightning(lr=args.lr, class_weights=class_weights)

    trainer = Trainer(accelerator='gpu', max_epochs=args.epochs, logger=logger, callbacks=callbacks, reload_dataloaders_every_n_epochs=1)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, dataloaders=(test_loader))


if __name__ == "__main__":
    main()