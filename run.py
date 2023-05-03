from __future__ import print_function
import os
import random
import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys
import torch 
from support_train import HDF5Dataset,RMSLELoss,AverageMeter
import model
import argparse
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset, WeightedRandomSampler
from torch.optim.lr_scheduler import StepLR
from model import UNet
import warnings


All_Accuracy = []
All_Epoch = []

case = sys.argv[1]
img_dir = sys.argv[2]
lossfunc = sys.argv[3]
experiment_name = sys.argv[4]

print(f'The case is {case}')
print(f'The image directory path is {img_dir}')
print(f'The loss function is {lossfunc}')
print(f'The experiment name is {experiment_name}')

if case == 'Beads4SLS':
    max_im = 22.073517
    max_gt = 1
elif case == 'BV2SLS':
    max_im = 114.66137
    max_gt = 56.031727
elif case == 'BV4SLS':
    max_im = 108.59321
    max_gt = 46.701
elif case == 'BV6SLS':
    max_im = 39.146564
    max_gt = 22.953957
elif case == 'Neuronal2SLS':
    max_im = 117.02816
    max_gt = 56.031727 
elif case == 'Neuronal6SLS':
    max_im = 33.970192
    max_gt = 22.953955


os.mkdir("/n/home12/mithunjha/common_python/models/"+case+"/"+experiment_name)
model_path = f"/n/home12/mithunjha/common_python/models/{case}/{experiment_name}"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"



def train(args, model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output,target)
        train_loss += loss
        loss.backward()
        optimizer.step()
    train_loss /= len(train_loader.dataset)
    return train_loss

def test(args, model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_l = criterion(output, target)
            test_loss += test_l

    test_loss /= len(test_loader.dataset)
   
    return test_loss

def main(img_dir):
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 10)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--gamma', type=float, default=0.3, metavar='M',
                        help='Learning rate step gamma (default: 0.3)')
    parser.add_argument('--seed', type=int, default=100, metavar='S',
                        help='random seed (default: 100)')
    parser.add_argument('--output_nc', type=int, default=1, metavar='N',
                        help='output channels')
    args = parser.parse_args(args=[])
    
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

    
    ### Data loader ###################
    train_dataset = HDF5Dataset(img_dir=img_dir, max_im = max_im, max_gt = max_gt, isTrain=True)
    test_dataset = HDF5Dataset(img_dir=img_dir, max_im = max_im, max_gt = max_gt, isTrain=False)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    ### Model initialization ##########
    model = UNet(n_classes=args.output_nc).cuda()
    model = torch.nn.parallel.DataParallel(model)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=20, gamma=args.gamma)
      
    ## Loss Functions ############
    if lossfunc == 'KLDiv':
        criterion = torch.nn.KLDivLoss()
    elif lossfunc == 'RMSLE':
        criterion = RMSLELoss()
    elif lossfunc == 'MSE':
        criterion = torch.nn.MSELoss()
    elif lossfunc == 'L1':
        criterion = torch.nn.SmoothL1Loss()
        
        
    
    val_losses = AverageMeter()
    train_losses = AverageMeter()
    loss_values_train = []
    loss_values_valid = []
    val_loss = 10000000
    
    
    for epoch in range(1, args.epochs + 1):
        tloss= train(args, model, device, train_loader, optimizer, epoch,criterion)
        train_losses.update(tloss.data.item())
        vloss= test(args, model, device, test_loader,criterion)
        val_losses.update(vloss.data.item())
        loss_values_train.append(tloss)
        loss_values_valid.append(vloss)
        print("epoch:%.1f" %epoch, "Train_loss:%.4f" % tloss, "Val_loss:%.4f" % vloss)
        scheduler.step()
        try:
            os.makedirs(model_path)
        except OSError:
            pass
        if val_loss>=vloss : 
            torch.save(model.state_dict(),  model_path +"/fcn_deep_best.pth") 
            val_loss = vloss
        if epoch == 60:
            torch.save(model.state_dict(),  model_path +"/fcn_deep_fix_60.pth") 

    return loss_values_train, loss_values_valid



warnings.filterwarnings('ignore')
if __name__ == '__main__':
    loss_t, loss_v = main(img_dir)
    plt.plot(loss_t)
    plt.plot(loss_v)
    plt.savefig(model_path + f'loss_curve_{lossfunc}_{case}_norm_Unet.png')
    
