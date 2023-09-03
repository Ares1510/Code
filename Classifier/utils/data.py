import torch
import numpy as np
from models.swinir_net import SwinIR
from torch.utils.data import Dataset

class LIDC(Dataset):
    def __init__(self, patches, labels, transforms=None, denoised=False):
        self.patches = patches
        self.labels = labels
        self.transforms = transforms
        self.denoised = denoised
        self.model = SwinIR(img_size=512, patch_size=64, window_size=64, in_chans=1, embed_dim=64,
                             depths=[4], num_heads=[4], mlp_ratio=2, qkv_bias=False, upsampler=None, upscale=1)
        checkpoint = torch.load("models/model.ckpt")
        model_weights = checkpoint["state_dict"]

        for key in list(model_weights):
            model_weights[key.replace("model.", "")] = model_weights.pop(key)

        self.model.load_state_dict(model_weights)
        self.model.eval()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        patch = torch.from_numpy(self.patches[idx]).float()
        label = torch.from_numpy(np.asarray(self.labels[idx])).long()
        patch = patch.unsqueeze(0)
        
        
        # if denoise is True load swin model and denoise patch
        if self.denoised == True:
            with torch.no_grad():
                patch = patch.unsqueeze(0)
                patch = patch.cuda()
                model = self.model.cuda()
                patch = self.model(patch)
            patch = patch.squeeze(0).cpu()

        if self.transforms and label == 1:
            patch = self.transforms(patch)
        return patch, label
