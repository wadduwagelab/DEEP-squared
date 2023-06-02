import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset, WeightedRandomSampler
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import numpy as np
import h5py
import torch
import math

def read_h5py(path, isVal):
    with h5py.File(path, "r") as f:
        print(f"Reading from {path} ====================================================")
        # print("Keys in the h5py file : %s" % f.keys())
        a_group_key = list(f.keys())[0]
        b_group_key = list(f.keys())[1]
        # Get the data
        _target = np.array((f[a_group_key]))
        length = _target.shape[0]
        if isVal:
            _target = np.array((f[a_group_key]))[:math.ceil(length/5)]
            _img = np.array((f[b_group_key]))[:math.ceil(length/5)]
            print(f"Number of samples (Validation) : {len(_img)},{len(_target)}")
            print(f"Shape of each data (Validation) : {_img.shape}, {_target.shape}")
        else:
            _target = np.array((f[a_group_key]))[math.floor(length/5):]
            _img = np.array((f[b_group_key]))[math.floor(length/5):]
            print(f"Number of samples (Training) : {len(_img)},{len(_target)}")
            print(f"Shape of each data (Training) : {_img.shape}, {_target.shape}")

        return _target, _img

class HDF5Dataset(Dataset):
    def __init__(self,img_dir, PSF_path, max_im, max_gt, isVal= False, transform=None):
        self.img_dir = img_dir
        self.max_im = max_im
        self.max_gt = max_gt
        self.ground_truth, self.measurement = read_h5py(self.img_dir,isVal)
        self.transform = transform 
    def __len__(self):
        return len(self.ground_truth)

    def __getitem__(self, index):

    
        img = self.measurement[index]
        target = self.ground_truth[index]
        _img = torch.from_numpy(np.divide(img,self.max_im)).float()
        _target = torch.from_numpy(np.divide(target,self.max_gt)).float()

        return _img,_target
    
    
class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))
    
    
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0