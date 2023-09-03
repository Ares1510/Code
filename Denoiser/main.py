import time
import torch
import pytorch_lightning as pl
from utils.data import MayoLoader
from models.red_cnn import RedCNNLightning
from models.swinir import SwinIRLightning
from models.unet import UNetLightning
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from utils.parser import parser

def main():
    args = parser()
    print(args)

    # set seed for reproducibility
    pl.seed_everything(42)
    torch.set_float32_matmul_precision('medium')

    data_loader = MayoLoader(args.batch_size)

    train_loader = data_loader.train()
    val_loader = data_loader.val()
    test_loader = data_loader.test()

    callbacks = [EarlyStopping(monitor='val_loss', patience=args.patience, mode='min'),
                 ModelCheckpoint(monitor='val_loss', mode='min')]

    run_name = f'{args.model}_{time.strftime("%Y%m%d-%H%M%S")}'

    logger = WandbLogger(name=run_name, project='MayoDenoiser', log_model='all')
    logger.experiment.config.update(args)
    
    if args.model == 'redcnn':
        model = RedCNNLightning(args.mode, args.lr, args.epochs)
    elif args.model == 'unet':
        model = UNetLightning(args.mode, args.lr, args.epochs)
    elif args.model == 'swinir':
        model = SwinIRLightning(args.mode, args.lr, args.epochs)

    trainer = Trainer(accelerator='gpu', max_epochs=args.epochs, logger=logger, callbacks=callbacks, precision='32')
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, dataloaders=(test_loader))


if __name__ == "__main__":
    main()
