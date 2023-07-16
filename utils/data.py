import random
import torch
import cv2
import numpy as np
import torch.nn.functional as F
from pydicom import dcmread
from torch.utils.data import Dataset, DataLoader


def read_file(path):
    data = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split()
            data.append(line)
    return data

def read_dicom(path):
    with open(path, 'rb') as f:
        ds = dcmread(f)
    image = ds.pixel_array
    return image

# from noise2sim repository
def random_rotate_mirror(img_0, random_mode):
    if random_mode == 0:
        img = img_0
    elif random_mode == 1:
        img = img_0[::-1, ...]
    elif random_mode == 2:
        img = cv2.rotate(img_0, cv2.ROTATE_90_CLOCKWISE)
    elif random_mode == 3:
        img_90 = cv2.rotate(img_0, cv2.ROTATE_90_CLOCKWISE)
        img = img_90[:, ::-1, ...]
    elif random_mode == 4:
        img = cv2.rotate(img_0, cv2.ROTATE_180)
    elif random_mode == 5:
        img_180 = cv2.rotate(img_0, cv2.ROTATE_180)
        img = img_180[::-1, ...]
    elif random_mode == 6:
        img = cv2.rotate(img_0, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif random_mode == 7:
        img_270 = cv2.rotate(img_0, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img = img_270[:, ::-1, ...]
    else:
        raise TypeError
    return img

# adapted from noise2sim repository
class MayoDataset(Dataset):
    def __init__(self, split, neighbor=2, ks=7, th=30):
        self.data = read_file(f'datasets/mayo/mayo_{split}.txt')
        self.hu_range = [-160, 240]
        self.split = split
        self.neighbor = neighbor
        self.ks = ks
        self.th = th
        self.kernel = torch.ones(1, 1, ks, ks).to(torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ldct_path, ndct_path, label, idx_min, idx_max = self.data[idx]
        idx_min = int(idx_min)
        idx_max = int(idx_max)

        if label == '0':
            lr = 1
        elif label == '-1':
            lr = -1
        else:
            if np.random.rand() > 0.5:
                lr = 1
            else:
                lr = -1
        
        if lr == 1:
            neightbor_max = min(max(1, idx_max - idx), self.neighbor)
            ln = random.randint(1, neightbor_max)
        else:
            neightbor_max = min(max(1, idx - idx_min), self.neighbor)
            ln = -random.randint(1, neightbor_max)
        
        sim_path, _, _, _, _ = self.data[idx + ln]

        ldct_image = read_dicom(ldct_path)
        ndct_image = read_dicom(ndct_path)
        sim_image = read_dicom(sim_path)

        ldct_image = ldct_image.astype(np.float32)
        ndct_image = ndct_image.astype(np.float32)
        sim_image = sim_image.astype(np.float32)

        ldct_image = np.clip(ldct_image - 1024, self.hu_range[0], self.hu_range[1])
        ndct_image = np.clip(ndct_image - 1024, self.hu_range[0], self.hu_range[1])
        sim_image = np.clip(sim_image - 1024, self.hu_range[0], self.hu_range[1])

        if self.th is not None:
            data1 = torch.from_numpy(ldct_image).to(torch.float32)
            data1 = F.conv2d(data1.unsqueeze(0).unsqueeze(0), self.kernel, None, 1, (self.ks-1)//2, 1, 1) / (self.ks * self.ks)
            data1 = data1.squeeze()

            data2 = torch.from_numpy(sim_image).to(torch.float32)
            data2 = F.conv2d(data2.unsqueeze(0).unsqueeze(0), self.kernel, None, 1, (self.ks-1)//2, 1, 1) / (self.ks * self.ks)
            data2 = data2.squeeze()

            img_ignore = torch.abs(data1 - data2) > self.th
            img_ignore = img_ignore.to(torch.float32).numpy()
        else:
            img_ignore = None

        ldct_image = (ldct_image - self.hu_range[0]) / (self.hu_range[1] - self.hu_range[0])
        sim_image = (sim_image - self.hu_range[0]) / (self.hu_range[1] - self.hu_range[0])
        ndct_image = (ndct_image - self.hu_range[0]) / (self.hu_range[1] - self.hu_range[0])
        img_ignore = (img_ignore - self.hu_range[0]) / (self.hu_range[1] - self.hu_range[0])

        if self.split == 'train':
            random_mode = np.random.randint(0, 8)
            ldct_image = random_rotate_mirror(ldct_image.squeeze(), random_mode)
            sim_image = random_rotate_mirror(sim_image.squeeze(), random_mode)
            img_ignore = random_rotate_mirror(img_ignore.squeeze(), random_mode)
            ndct_image = random_rotate_mirror(ndct_image.squeeze(), random_mode)

        if len(ldct_image.shape) < 3:
                ldct_image = ldct_image.reshape([ldct_image.shape[0], ldct_image.shape[1], 1])

        if len(sim_image.shape) < 3:
            sim_image = sim_image.reshape([sim_image.shape[0], sim_image.shape[1], 1])
        
        if len(ndct_image.shape) < 3:
            ndct_image = ndct_image.reshape([ndct_image.shape[0], ndct_image.shape[1], 1])

        if self.th is not None:
            if len(img_ignore.shape) < 3:
                img_ignore = img_ignore.reshape([sim_image.shape[0], sim_image.shape[1], 1])

        ldct_image = torch.from_numpy(ldct_image.transpose([2, 0, 1]).copy()).to(torch.float32)
        sim_image = torch.from_numpy(sim_image.transpose([2, 0, 1]).copy()).to(torch.float32)
        ndct_image = torch.from_numpy(ndct_image.transpose([2, 0, 1]).copy()).to(torch.float32)
        if self.th is not None:
            img_ignore = torch.from_numpy(img_ignore.transpose([2, 0, 1]).copy()).to(torch.float32)
        else:
            img_ignore = -1

        return ldct_image, sim_image, img_ignore, ndct_image


class MayoLoader():
    def __init__(self, batch_size):
        self.batch_size = batch_size

        self.train_set = MayoDataset('train')
        self.val_set = MayoDataset('val')
        self.test_set = MayoDataset('test')

    def train(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    
    def val(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    def test(self):
        return DataLoader(self.test_set, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
