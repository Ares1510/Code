import argparse

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--denoised', type=bool, default=False, help='number of epochs to train')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--patience', type=int, default=5, help='patience for early stopping')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    return parser.parse_args()