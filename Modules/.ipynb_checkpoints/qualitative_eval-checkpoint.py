from Modules.model import UNet
import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib.font_manager as fm
import os
import warnings
warnings.filterwarnings('ignore')

def qualitative_eval(args,directory,test_file,max_gt,max_im):
    my_model = args.model_path

    netG = UNet(n_classes=1, n_patterns = args.n_patterns).cuda()
    netG = torch.nn.parallel.DataParallel(netG)
    my_model = args.model_path
    netG.load_state_dict(torch.load(my_model))
    netG.eval()
    

    for id_ in range(args.idx,args.idx+1):
        with h5py.File(directory+test_file, 'r') as db:
            modalities = db['input'][id_]
            GT_ = db['gt'][id_]

        # mod_sum = np.sum(modalities,axis = 0)
        GT = torch.from_numpy(np.divide(GT_,max_gt))
        img = torch.from_numpy(np.divide(modalities,max_im)[None, :, :]).float()
        img = img[:,0:args.n_patterns,:,:]
        
        
        netG = netG.cuda()
        
        out = netG(img.cuda())
        out = out/out.max()
        out = out.detach().cpu()

        out_img = np.squeeze(out)
        GT = np.squeeze(GT)
        
        _img_avg = np.sum(modalities, axis=0)/32
        out_img = out_img - out_img.min()
        _img_avg = _img_avg/_img_avg.max()
        
        
        
        predict_path= args.save_path #'/n/home12/mithunjha/GITHUB_deep2/'
        if not os.path.exists(predict_path):
            os.makedirs(predict_path)
        plt.rcParams["figure.figsize"] = (200,200)
        fontprops = fm.FontProperties(size=150)
        
 
        f, axarr = plt.subplots(1,4)
        
        img_sum_ = axarr[0].imshow(img[:,0,:,:].squeeze(),cmap='inferno')#mod_sum
        axarr[0].set_title('Single Patterned Input Image')
        axarr[0].title.set_size(180)
        img_avg_ = axarr[1].imshow(_img_avg,cmap='inferno')
     
        axarr[1].set_title('Average Patterned Input Image')
        axarr[1].title.set_size(180)
        img_ptn_ = axarr[2].imshow(out_img/out_img.max(),cmap='inferno')
        
        axarr[2].set_title('Prediction')
        axarr[2].title.set_size(180)
        img_tar_ = axarr[3].imshow(GT,cmap='inferno')
        
        axarr[3].set_title('Target')
        axarr[3].title.set_size(180)
        f.savefig(predict_path + str(args.idx) + '_prediction.png')
        plt.show()
        
        

