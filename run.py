import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys
import torch 
from Modules.support_train import HDF5Dataset,RMSLELoss,AverageMeter
import Modules.model
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
from Modules.model import UNet
import warnings


All_Accuracy = []
All_Epoch = []


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

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch DEEP-sqaured')
    parser.add_argument('--case', type=str, default= 'Beads4SLS', help='Beads4SLS/Neuronal2SLS/Neuronal6SLS/BV2SLS/BV4SLS/BV6SLS/OWNDATA')
    parser.add_argument('--lossfunc', type=str, default= 'KLDiv', help='KLDiv/RMSLE/MSE/L1')
    parser.add_argument('--experiment_name', type=str, default= 'sample', help='experiment name will be the name of the folder')
    parser.add_argument('--save_model_path', type=str, default= None, help='path to the folder to store the model checkpoints')
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
    parser.add_argument('--use_gpu', action='store_true', default=True,
                        help='enables CUDA training')
    parser.add_argument('--n_patterns', type=str, default= 32, help='1/2/4/8/16/32') 
    parser.add_argument('--max_im', type=int, default= None, help='Give the maximum of input image to normalize them') 
    parser.add_argument('--max_gt', type=int, default= None, help='Give the maximum of ground truth image to normalize them')
    parser.add_argument('--data_path', type=str, default= None, help='path to the folder where the test data is stored')
    
    args = parser.parse_args(args=[])
    
    
    use_cuda = args.use_gpu and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}
    
    if args.case == 'Beads4SLS':
        max_im = 22.073517
        max_gt = 1
        img_dir = '/n/holylabs/LABS/wadduwage_lab/Lab/dataset_mithu/dmd_exp_tfm_beads_4sls_maxcount_5/beads_data_4sls_5mc_tr.h5'

    elif args.case == 'BV2SLS':
        max_im = 114.66137
        max_gt = 56.031727
        img_dir = '/n/holylabs/LABS/wadduwage_lab/Lab/temp/_results/_cnn_synthTrData/18-Oct-2021/dmd_exp_tfm_mouse_20201224_100um/mouse_bv_100um_data_2sls_5.603172e+01mc_tr.h5'

    elif args.case == 'BV4SLS':
        max_im = 108.59321
        max_gt = 46.701
        img_dir = '/n/holylabs/LABS/wadduwage_lab/Lab/temp/_results/_cnn_synthTrData/21-Oct-2021/dmd_exp_tfm_mouse_20201224_200um/mouse_bv_200um_data_4sls_4.670100e+01mc_tr.h5'

    elif args.case == 'BV6SLS':
        max_im = 39.146564
        max_gt = 22.953957
        img_dir = '/n/holylabs/LABS/wadduwage_lab/Lab/temp/_results/_cnn_synthTrData/21-Oct-2021/dmd_exp_tfm_mouse_20201224_300um/mouse_bv_300um_data_6sls_2.295395e+01mc_tr.h5'


    elif args.case == 'Neuronal2SLS':
        max_im = 117.02816
        max_gt = 56.031727 
        img_dir = '/n/holylabs/LABS/wadduwage_lab/Lab/dataset_mithu/11-Aug-2022/dmd_exp_tfm_mouse_20201224_100um/mouse_neuronal_100um_data_2sls_5.603172e+01mc_tr.h5'

    elif args.case == 'Neuronal6SLS':
        max_im = 33.970192
        max_gt = 22.953955
        img_dir = '/n/holylabs/LABS/wadduwage_lab/Lab/dataset_mithu/20-Aug-2022/dmd_exp_tfm_mouse_20201224_300um/mouse_neuronal_300um_data_6sls_2.295395e+01mc_tr.h5'

    elif args.case == "own_dataset":
        max_im = args.max_im
        max_gt = args.max_gt
        img_dir = args.data_path


    model_path = f"{args.save_model_path}/{args.case}/{args.experiment_name}"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    print(f'The case is {args.case}')
    print(f'The image directory path is {img_dir}')
    print(f'The loss function is {args.lossfunc}')
    print(f'The experiment name is {args.experiment_name}')
    

    
    ### Data loader ###################
    train_dataset = HDF5Dataset(img_dir=img_dir, max_im = max_im, max_gt = max_gt, n_patterns=args.n_patterns, isVal=False)
    test_dataset = HDF5Dataset(img_dir=img_dir, max_im = max_im, max_gt = max_gt, n_patterns=args.n_patterns ,isVal=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    ### Model initialization ##########
    model = UNet(n_classes=args.output_nc,n_patterns = args.n_patterns).cuda()
    model = torch.nn.parallel.DataParallel(model)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=20, gamma=args.gamma)
      
    ## Loss Functions ############
    if args.lossfunc == 'KLDiv':
        criterion = torch.nn.KLDivLoss()
    elif args.lossfunc == 'RMSLE':
        criterion = RMSLELoss()
    elif args.lossfunc == 'MSE':
        criterion = torch.nn.MSELoss()
    elif args.lossfunc == 'L1':
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
            if not os.path.exists(model_path):
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
    loss_t, loss_v = main()
    plt.plot(loss_t)
    plt.plot(loss_v)
    plt.savefig(model_path + f'loss_curve_{args.lossfunc}_{args.case}_norm_Unet.png')
    
