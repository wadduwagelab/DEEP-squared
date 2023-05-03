import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset, WeightedRandomSampler
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import numpy as np
import h5py
import torch

class HDF5Dataset(Dataset):
    def __init__(self,img_dir, max_im, max_gt, isTrain=True):
        self.isTrain = isTrain
        #self.data_dict = pd.read_csv(data_dir) 
        if isTrain:
            fold_dir = "txt_files/train.txt"
        else:
            fold_dir = "txt_files/test.txt"

        ids = open(fold_dir, 'r')

        self.index_list = []

        for line in ids:
            self.index_list.append(line[0:-1])
        self.img_dir = img_dir
        self.max_im = max_im
        self.max_gt = max_gt
        
    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, index):
        _img = np.dtype('>u2')
        _target = np.dtype('>u2')
        id_ = int(self.index_list[index])
        with h5py.File(self.img_dir, 'r') as db:
             #print(db['input'].shape)
             _img = db['input'][id_]
             _target = db['gt'][id_]
        if np.max(_target) == 0:
             with h5py.File(self.img_dir, 'r') as db:
                 _img = db['input'][id_+1]
                 _target = db['gt'][id_+1]
        _img = torch.from_numpy(np.divide(_img,self.max_im)).float()
        _target = torch.from_numpy(np.divide(_target,self.max_gt)).float()

        return _img, _target
    
    


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