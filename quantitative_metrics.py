import torch
import numpy as np
from eval_metrics import *
from skimage.metrics import peak_signal_noise_ratio as psnr 
import h5py
import torch.nn as nn
import statistics
from model import UNet
import warnings
warnings.filterwarnings('ignore')

def quantitative_metrics(args,directory,test_file,max_gt,max_im):
    list_new_psnr = []
    ssim_pytorch_list = []
    mse_norm_list = []

    my_model = args.model_path 

    netG = UNet(n_classes=1,n_patterns=args.n_patterns).cuda()
    netG = torch.nn.parallel.DataParallel(netG)
    netG.load_state_dict(torch.load(my_model))
    netG.eval()

    p = 0


    for i in range (0,128):
        with h5py.File(directory+test_file, 'r') as db:
            # print(db['input'].shape,db['gt'].shape)
            # print(i)
            modalities = db['input'][i]
            GT_ = db['gt'][i]
    
        mod_sum = np.sum(modalities,axis = 0)
        GT = torch.from_numpy(np.divide(GT_,max_gt))
        img = torch.from_numpy(np.divide(modalities,max_im)[None, :, :]).float()
        img = img[:,0:args.n_patterns,:,:]
        _img_avg = np.sum(modalities, axis=0)/32
        
        netG = netG.cuda()
        out = netG(img.cuda())
        
        ## check followings ###
        out = out/out.max()       
        out_img_ = out.detach().cpu()
        
        # print(f'out shape {out.shape}')
        # out_img = out.data[0] 
        # print(f'1. out_img shape {out_img.shape}')
        # out_img = np.squeeze(out_img)
        # print(f'2. out_img shape {out_img.shape}')
        # print(f'2. GT shape {GT.shape}')
        GT_ = GT.unsqueeze(dim=0)
        # print(f'2. GT shape {GT.shape}')
        
        

        # out_img_ = torch.from_numpy(np.expand_dims(out_img, [0, 1]))
        # print(f'3. out_img_ shape {out_img_.shape}')
        # GT_ = torch.from_numpy(np.expand_dims(GT, [0, 1]))
        # print(f'3. GT_ shape {GT_.shape}')

        out_img_norm = (out_img_ - out_img_.min())/(out_img_.max() - out_img_.min())
        GT_norm = (GT_ - GT_.min())/(GT_.max() - GT_.min())


        ################## MSE error ##################
        loss_mse = nn.MSELoss()
        mse_norm = loss_mse(out_img_norm,GT_norm)
        mse_norm_list.append(mse_norm.item())

        ################ SSIM ###################
        ssim_pytorch_val= ssim_pytorch(out_img_norm.squeeze(dim=0), GT_norm.squeeze(dim=0), 
                                       k = 11, data_range = np.array(out_img_norm).max() - np.array(out_img_norm).min(),
                                       range_independent = False, double_check = True)
        ssim_pytorch_list.append(ssim_pytorch_val)

        ################ PSNR #####################
        new_psnr = psnr(np.array(GT_norm), np.array(out_img_norm.detach().numpy()),data_range=out_img_norm.max() - out_img_norm.min())
        list_new_psnr.append(new_psnr.item())

    print('MSE error', statistics.mean(mse_norm_list),'std',statistics.stdev(mse_norm_list))
    print(f'SSIM {statistics.mean(ssim_pytorch_list)},std {statistics.stdev(ssim_pytorch_list)}')
    print(f'PSNR {statistics.mean(list_new_psnr)},std {statistics.stdev(list_new_psnr)}')
    