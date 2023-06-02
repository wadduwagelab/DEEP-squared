import torch.nn.functional as F
from ignite.metrics import SSIM as SSIM_IGNITE
import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity as ski_ssim
from pytorch_msssim import ssim as pytorch_ssim
import warnings
warnings.simplefilter('default', UserWarning)

def mse_distance(X_hat, X):
    '''
        Function to calculate MSE distance between predicted and ground truth.
        
        Args:
            X_hat : Predicted image ((n_samples, img_size, img_size), dtype= torch.float)  
            X     : Ground Truth image ((n_samples, img_size, img_size), dtype= torch.float)  
            
        Returns:
             MSE distance
    '''
    return F.mse_loss(X_hat.unsqueeze(dim=1), X.unsqueeze(dim=1)).item() # check

def L1_distance(X_hat, X):
    '''
        Function to calculate L1 distance between predicted and ground truth.
        
        Args:
            X_hat : Predicted image ((n_samples, img_size, img_size), dtype= torch.float)  
            X     : Ground Truth image ((n_samples, img_size, img_size), dtype= torch.float)  
            
        Returns:
             L1 distance
    '''
    return F.l1_loss(X_hat.unsqueeze(dim=1), X.unsqueeze(dim=1)).item()
    
def ssim_ignite(X_hat, X, k= 11):
    '''
        Function to calculate SSIM score between predicted and ground truth.
        
        Args:
            X_hat : Predicted image ((n_samples, img_size, img_size), dtype= torch.float) | torch.Tensor
            X     : Ground Truth image ((n_samples, img_size, img_size), dtype= torch.float) | torch.Tensor
            k     : Kernel size for SSIM (defaults to 11) | int
            
        Returns:
             SSIM score
    '''
    metric = SSIM_IGNITE(data_range = 1.0, kernel_size= (k,k))
    metric.update((X_hat.unsqueeze(dim=1), X.unsqueeze(dim=1)))
    return metric.compute().item()
    # raise AssertionError("Inaccurate SSIM called! Anyway Answer is ", metric.compute().item())

def error_quantification(gt, pred, error_x, error_x2, e_n, bin_size, angle_max = 2*np.pi):
    '''
        Acculmunate error quantification metrics for all the bin_sizes, which can be used for plotting.

        Accepts : 
            gt   : Ground truth phase image (n_samples, img_size, img_size) | torch.Tensor
            pred : Predicted phase image    (n_samples, img_size, img_size) | torch.Tensor
            angle_max : Maximum angle value for the phase image (defaults to 2*pi) | float

            => gt   is normalized in range [0,1]  
            => pred is normalized in range [0,1]

            error_x  : Accumulated error in phase   | Dictionary
            error_x2 : Acculumated error^2 in phase | Dictionary
            e_n      : Number of pixels in each bin | Dictionary

        Returns:
            error_x, error_x2, e_n : Updated error_x, error_x2, e_n
    '''

    #check if out of bounds
    if(gt.max() > 1 or gt.min() < 0 or pred.max() > 1 or pred.min() < 0):
        warnings.warn("Error Quantification : gt values not in range [0,1] or pred values not in range [0,1] ")
        # print("gt.max() = ", gt.max())
        # print("gt.min() = ", gt.min())
        # print("pred.max() = ", pred.max())
        # print("pred.min() = ", pred.min())

    #convert [0,1] to [0, angle_max] (unit = rad)
    gt   = gt * angle_max
    pred = pred * angle_max 

    error_img = abs((gt-pred)).detach().cpu().numpy()*180/np.pi  # in degrees
    
    for idx in range(0,gt.shape[0]): # loop over all samples
        
        #Loop through each pixel
        for x in range(0,gt.shape[1]):
            for y in range(0,gt.shape[2]):
                
                gt_phase_degrees = gt[idx][x][y].detach().cpu().numpy()*180/np.pi
                angle = (gt_phase_degrees//bin_size) * bin_size
                
                if angle not in error_x.keys():
                    error_x[angle]  = error_img[idx][x][y]
                    error_x2[angle] = error_img[idx][x][y]**2

                    e_n[angle] = 1
                else: 
                    error_x[angle]  += error_img[idx][x][y]
                    error_x2[angle] += error_img[idx][x][y]**2
                    e_n[angle] += 1
                    
    return error_x, error_x2, e_n

def ssim_cvpr(img1, img2, win_size, data_range, K = [0.01, 0.03]):
        '''
            Author : [CVPR'20] TTSR: Learning Texture Transformer Network for Image Super-Resolution Resources
            https://github.com/researchmm/TTSR/blob/2836600b20fd8f38e0f1550ab0b87c8d2a2bd276/utils.py
        '''

        C1 = (K[0] * data_range)**2
        C2 = (K[1] * data_range)**2 

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(win_size, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()


def ssim_verify(X_hat, X, k, dr, value_to_check):
    '''
        Function to verify SSIM with Skimage and CVPR implementation.
    '''


    X_hat = X_hat.cpu().numpy()
    X = X.cpu().numpy()
    
    ski_ssim_vals  = []
    cvpr_ssim_vals = []

    for i in range(0,X_hat.shape[0]):
        ski_ssim_vals.append(  ski_ssim(X_hat[i], X[i], win_size=k, gaussian_weights=True, data_range = dr))
        cvpr_ssim_vals.append(ssim_cvpr(X_hat[i], X[i], win_size=k, data_range = dr))
    
    v1 = np.mean(ski_ssim_vals)
    v2 = np.mean(cvpr_ssim_vals)

    check1 = np.allclose(v1, v2, atol = 5e-3)
    check2 = np.allclose(v1, value_to_check, atol = 5e-3) #.cpu().numpy()
    
    if((check1 or check2) == False):
        # print(check1)
        # print(check2)
        raise AssertionError("Skimage ssim = ", v1, ",CVPR ", v2, ",Pytorch SSIM = ", value_to_check.item())#.numpy()

def ssim_pytorch(X_hat, X, k = 11, data_range = 1.0, range_independent = True, double_check = False):
    '''
        Function to calculate SSIM score between predicted and ground truth.
        
        Args:
            X_hat : Predicted image ((n_samples, img_size, img_size), dtype= torch.float) | torch.Tensor
            X     : Ground Truth image ((n_samples, img_size, img_size), dtype= torch.float) | torch.Tensor
            k     : Kernel size for SSIM (defaults to 11) | int
            data_range : Data range for SSIM (defaults to 1.0) | float
            range_independent : Whether to use range independent SSIM (defaults to True) | bool
            double_check : Double check the SSIM score with other implementations (defaults to False) | bool
            
        Returns:
             SSIM score
    '''
    # print(f'x_hat {X_hat.shape}')
    X_hat_ = X_hat.unsqueeze(dim=1)
    # print(X_hat.shape)
    X_     = X.unsqueeze(dim=1)
    
    if(range_independent):
        ans =  torch.mean(pytorch_ssim(X_hat_, X_, win_size = k, data_range = data_range, size_average = False, K = (1e-6,1e-6)))
        
        if(double_check):
            ssim_verify(X_hat, X, k, data_range, ans)
            # print("Checks Passed!")
        return ans.item()

    else:
        ### Regular SSIM calculation with default parameters ;
        ans = torch.mean(pytorch_ssim(X_hat_, X_, win_size = k, data_range = data_range , size_average = False))
        
        if(double_check):
            ssim_verify(X_hat, X, k, data_range, ans)
            # print("Checks Passed!")
        return ans.item()
