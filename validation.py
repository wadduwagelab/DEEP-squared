import argparse
import os
from quantitative_metrics import quantitative_metrics
from qualitative_eval import qualitative_eval

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
######################


def parse_option():
    parser = argparse.ArgumentParser('Argument for Validation')
    parser.add_argument('--case', type=str, default= 'Beads4SLS', help='Beads4SLS/Neuronal2SLS/Neuronal6SLS/BV2SLS/BV4SLS/BV6SLS/OWNDATA')
    parser.add_argument('--output_path', type=str, default='/n/home12/mithunjha/common_python', help='output folder name')
    parser.add_argument('--model_path', type=str, default=f'/n/home12/mithunjha/common_python/models/Beads5SLS/DEEP2-45/fcn_deep_best.pth',help='model path')
    parser.add_argument('--n_patterns', type=str, default= 32, help='1/2/4/8/16/32') 
    parser.add_argument('--idx', type=int, default= 20, help='any interger value <= 127') 
    parser.add_argument('--save_path', type = str, default = '/n/home12/mithunjha/GITHUB_deep2/', help = 'save folder name')
    
    opt = parser.parse_args()
    parser.add_argument('--max_im', type=int, default= None, help='Give the maximum of input image to normalize them') 
    parser.add_argument('--max_gt', type=int, default= None, help='Give the maximum of ground truth image to normalize them')
    parser.add_argument('--data_path', type=int, default= None, help='path to the folder where the test data is stored')
    return opt

args = parse_option()

def main():
    if args.case == 'Beads4SLS':
        # print(args.case)
        max_im = 22.073517  
        max_gt = 1
        directory = "/n/holylabs/LABS/wadduwage_lab/Lab/dataset_mithu/dmd_exp_tfm_beads_4sls_maxcount_5/"
        # train_file = 'beads_data_4sls_5mc_tr.h5'
        test_file = 'beads_data_4sls_5mc_test.h5'

    elif args.case == 'Neuronal2SLS':
        max_im = 117.02816 
        max_gt = 56.031727
        directory = '/n/holylabs/LABS/wadduwage_lab/Lab/dataset_mithu/11-Aug-2022/dmd_exp_tfm_mouse_20201224_100um/'
        # train_file = 'mouse_neuronal_100um_data_2sls_5.603172e+01mc_tr.h5'
        test_file = 'mouse_neuronal_100um_data_2sls_5.603172e+01mc_test.h5'
        
    elif args.case == 'Neuronal6SLS':
        max_im = 33.970192 
        max_gt = 22.953955
        directorys = '/n/holylabs/LABS/wadduwage_lab/Lab/dataset_mithu/20-Aug-2022/dmd_exp_tfm_mouse_20201224_300um/'
        # train_file = 'mouse_neuronal_300um_data_6sls_2.295395e+01mc_tr.h5'
        test_file = 'mouse_neuronal_300um_data_6sls_2.295395e+01mc_test.h5'
        
    elif args.case == 'BV2SLS':
        max_im = 114.66137 
        max_gt = 56.031727
        directory = '/n/holylabs/LABS/wadduwage_lab/Lab/temp/_results/_cnn_synthTrData/18-Oct-2021/dmd_exp_tfm_mouse_20201224_100um/'
        test_file = 'mouse_bv_100um_data_2sls_5.603172e+01mc_test.h5'
        # train_file = 'mouse_bv_100um_data_2sls_5.603172e+01mc_tr.h5'
        
    elif args.case == 'BV4SLS':
        max_im = 108.59321 
        max_gt = 46.701
        directory = '/n/holylabs/LABS/wadduwage_lab/Lab/temp/_results/_cnn_synthTrData/21-Oct-2021/dmd_exp_tfm_mouse_20201224_200um/'
        test_file = 'mouse_bv_200um_data_4sls_4.670100e+01mc_test.h5'
        # train_file = 'mouse_bv_200um_data_4sls_4.670100e+01mc_tr.h5'
        
    elif args.case == 'BV6SLS':
        max_im = 39.146564 
        max_gt = 22.953957
        directory = '/n/holylabs/LABS/wadduwage_lab/Lab/temp/_results/_cnn_synthTrData/21-Oct-2021/dmd_exp_tfm_mouse_20201224_300um/'
        test_file = 'mouse_bv_300um_data_6sls_2.295395e+01mc_test.h5'
        # train_file = 'mouse_bv_300um_data_6sls_2.295395e+01mc_tr.h5'
        
        
    elif args.case == 'OWNDATA':
        max_im = args.max_im
        max_gt = args.max_gt
        directory = args.data_path

        


    

    # quantitative_metrics(args,directory, test_file,max_gt,max_im)
    qualitative_eval(args,directory, test_file,max_gt,max_im)
    
if __name__ == '__main__':
    main()