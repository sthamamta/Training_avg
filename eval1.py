'''
checkpoints to load the models
model_name to load correct model and use their function to load weights from checkpoints
factor: determines on which downsampled factor images model is tested
plot-dir: dir where plots is saved---leave as default
output-dir: dir where outputs are saved --- leave as deault
addition: boolean (not implemented) if True add the pred with images to get the final output of the model

THIS FILE IS CREATED TO EVALUATE GAUSSIAN IMAGE TRAINED MODEL WITH KSPACE PADDED, KSPACE PADDED AND GAUSSIAN DOWSAMPLE 50 MICRON IMAGES FROM ALL SUBJECTS

'''

from venv import create
import torch
import torch.nn as nn
import  cv2
import matplotlib.pyplot as plt
import argparse
import statistics
import json
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'


# from models.densenet_new import SRDenseNet

from models.densenet_smchannel import SRDenseNet
from utils.preprocess import tensor2image, image2tensor,create_dictionary
from utils.prepare_test_set_image import crop_pad_kspace, prepare_image_fourier
from utils.image_quality_assessment import PSNR,SSIM
from utils.general import min_max_normalize, NRMSELoss
from utils.train_utils import read_dictionary
from models.srcnn import SRCNN
from models.rrdbnet import RRDBNet
import os
import numpy as np
from utils.preprocess import hfen_error

import warnings
warnings.filterwarnings("ignore")



def load_model(checkpoint_path,device,name='srdense'):
    if name in ['srdense']:
        model = load_srdense(checkpoint_path,device)
    elif name in ['srcnn']:
        model = load_srcnn(checkpoint_path,device)
    elif name in ['rrdbnet']:
        model = load_rrdbnet(checkpoint_path,device)
    return model


def load_rrdbnet(checkpoint_path,device):
    checkpoint = torch.load(checkpoint_path,map_location=torch.device(device))
    num_blocks = checkpoint['num_blocks']
    model = RRDBNet(num_block=num_blocks)
    state_dict = model.state_dict()
    for n, p in checkpoint['model_state_dict'].items():
        new_key = n[7:]
        # new_key = n
        if new_key in state_dict.keys():
            state_dict[new_key].copy_(p)
        elif n in state_dict.keys():
            state_dict[n].copy_(p)           
    return model

def load_srdense(checkpoint_path,device):
    checkpoint = torch.load(checkpoint_path,map_location=torch.device(device))
    growth_rate = checkpoint['growth_rate']
    num_blocks = checkpoint['num_blocks']
    num_layers =checkpoint['num_layers']
    model = SRDenseNet(growth_rate=growth_rate,num_blocks=num_blocks,num_layers=num_layers)
    state_dict = model.state_dict()
    for n, p in checkpoint['model_state_dict'].items():
        new_key = n[7:]
        # new_key = n
        if new_key in state_dict.keys():
            state_dict[new_key].copy_(p)
        elif n in state_dict.keys():
            state_dict[n].copy_(p)
            
    return model

def load_srcnn(checkpoint_path,device):
    checkpoint = torch.load(checkpoint_path,map_location=torch.device(device))
    
    model = SRCNN()
    state_dict = model.state_dict()
    for n, p in checkpoint['model_state_dict'].items():
        new_key = n[7:]
        if new_key in state_dict.keys():
            state_dict[new_key].copy_(p)
        elif n in state_dict.keys():
            state_dict[n].copy_(p)
            
    return model
 


def predict_model(model,image_path,label_path,device,psnr,ssim,mse,nrmse):

    # print("image path",image_path)
    # print("label path",label_path)
    # print("**********************************************************************************************")

    degraded = cv2.imread(image_path)
    degraded = cv2.cvtColor(degraded, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ref = cv2.imread(label_path)
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)

    degraded = degraded/255.
    ref = ref/255.   

    degraded = image2tensor(degraded).unsqueeze(dim=0).to(device)
    ref = image2tensor(ref).unsqueeze(dim=0).to(device)  

    pre = model(degraded)

    pre = pre.clamp(0.,1.)

    init_psnr = psnr(degraded, ref).item()
    init_ssim = ssim(degraded, ref).item()
    init_mse = mse(degraded, ref).item()
    init_nrmse = nrmse(ref,degraded).item()
    

    model_psnr = psnr(pre, ref).item()
    model_ssim = ssim(pre, ref).item()
    model_mse = mse(pre, ref).item()
    model_nrmse = nrmse(ref, pre).item()


    ref_arr = (ref.squeeze().detach().cpu().numpy()) *255.
    degraded_arr = (degraded.squeeze().detach().cpu().numpy())*255.
    pre_arr = (pre.squeeze().detach().cpu().numpy())*255.
    init_hfen = hfen_error(ref_arr,degraded_arr).astype(np.float16).item()
    model_hfen = hfen_error(ref_arr,pre_arr).astype(np.float16).item()

    return  {'init_psnr':init_psnr,
            'init_ssim': init_ssim,
            'init_mse': init_mse,
            'init_nrmse': init_nrmse,
            'init_hfen': init_hfen,
            'model_psnr':model_psnr,
            'model_ssim':model_ssim,
            'model_mse': model_mse,
            'model_nrmse':model_nrmse,
            'model_hfen': model_hfen}

def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

def create_each_plot(initial,model,save_name="ssim",y_min=None,y_max=None, ticks=None):
    # initial_pad = [initial_50,initial_75,initial_100,initial_125,initial_150]
    #....

    title = save_name.upper()
    plt.figure()

    bpl = plt.boxplot(initial, positions=np.array(range(len(initial)))*2.0-0.4, sym='', widths=0.4)
    bpr = plt.boxplot(model, positions=np.array(range(len(model)))*2.0+0.1, sym='', widths=0.4)

    set_box_color(bpl, '#D7191C') # colors are from http://colorbrewer2.org/
    set_box_color(bpr, '#2C7BB6')


    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c='#D7191C', label='Initial')
    plt.plot([], c='#2C7BB6', label='Model')
    plt.legend()

    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    plt.xlim(-2, len(ticks)*2)
    plt.ylim(y_min, y_max)
    plt.ylabel(title)
    plt.xlabel('datasets')
    plt.tight_layout()
    plt.title(title+' PLOT')

    plt.show()
    plt.savefig(save_name+'.png')


def plot_boxplot(initial_list, model_list):

    keys = ['psnr', 'ssim', 'mse','nrmse', 'hfen']
    # names = ['sigma_100', 'hanning', 'hamming', 'bicubic','mean','median'] # dataset names xticks
    names = ['gaussian', 'hanning', 'hamming', 'bicubic','mean','median']
    # initial_pad_list list of dictionary , each dictionary is the metric value for each sigma, each dictionary having keys and values for each metric

    # ylim_min = [24, 0.6, 0.0001, 0.05, 0.0]
    # ylim_max = [37, 0.98, 0.0025, 0.2, 0.2]

    # for testing sigma_25 model on all dataset
    # ylim_min = [10, 0.2, 0.0000, 0.05, 0.0]
    # ylim_max = [38, 0.98, 0.009, 0.8, 2]

    # # for testing sigma_50 model on all dataset
    # ylim_min = [15.45, 0.3, 0.0003, 0.06, 0.01]
    # ylim_max = [36, 0.91, 0.02, 0.7, 0.8]

    # ['psnr', 'ssim', 'mse','nrmse', 'hfen']
    # # # for testing sigma_75 model on all dataset
    # ylim_min = [15.45, 0.3, 0.0003, 0.04, 0.004]
    # ylim_max = [36, 0.98, 0.02, 0.5, 0.8]

    # ['psnr', 'ssim', 'mse','nrmse', 'hfen']
    # # for testing sigma_100 model on all dataset
    # ylim_min = [15.42, 0.41, 0.0002, 0.02, 0.022]
    # ylim_max = [39, 0.98, 0.028, 0.5, 0.7]

    # ['psnr', 'ssim', 'mse','nrmse', 'hfen']
    # for testing sigma_125 model on all dataset
    # ylim_min = [15.42, 0.41, 0.0001, 0.01, 0.018]
    # ylim_max = [41, 0.99, 0.028, 0.5, 0.7]

    # ['psnr', 'ssim', 'mse','nrmse', 'hfen']
    # for testing sigma_150 model on all dataset
    # ylim_min = [15.42, 0.41, 0.0001, 0.015, 0.005]
    # ylim_max = [43, 1, 0.028, 0.5, 0.75]


    # ***********************************************************
    # for testing sigma_100 model on all dataset
    ylim_min = [20.95, 0.4, 0.00019, 0.0333, 0.01]
    # ylim_max = [37, 0.98, 0.028, 0.48, 0.28]
    ylim_max = [37, 0.98, 0.028, 0.48, 0.31]

    # for testing hanning model on all dataset
    # ylim_min = [20, 0.66, 0.00019, 0.055, 0.018]
    # ylim_max = [36, 0.91, 0.028, 0.41, 0.36]
    # ['psnr', 'ssim', 'mse','nrmse', 'hfen']

    # for testing hamming model on all dataset
    # ylim_min = [20.95, 0.63, 0.00019, 0.033, 0.019]
    # ylim_max = [37.17, 0.925, 0.028, 0.5, 0.28]
   

    # for testing bicubic model
    # ylim_min = [20.95, 0.63, 0.00019, 0.058, 0.01]
    # ylim_max = [37.97, 0.925, 0.028, 0.4, 0.303]
    #           ['psnr', 'ssim', 'mse','nrmse', 'hfen']

    # for testing mean model
    # ylim_min = [20.95, 0.53, 0.00019, 0.029, 0.019]
    # ylim_max = [42, 0.972, 0.028, 0.4, 0.35]

     # for testing median model
    # ylim_min = [20.95, 0.63, 0.00019, 0.05, 0.019]
    # ylim_max = [37.175, 0.99, 0.028, 0.4, 0.35]

         # for testing all degradation fix dataset model
    # ylim_min = [21.51, 0.63, 0.00019, 0.045, 0.02]
    # ylim_max = [39, 0.99, 0.028, 0.45, 0.287]
    # ['psnr', 'ssim', 'mse','nrmse', 'hfen']

     # for testing gaussian large dataset model
    # ylim_min = [15.42, 0.41, 0.00019, 0.01, 0.03]
    # ylim_max = [43, 1, 0.028, 0.5, 0.7]
    # # ['psnr', 'ssim', 'mse','nrmse', 'hfen']


    for i,key in enumerate(keys):  # for each metric
        initial = []
        model = []
        for index,element in enumerate(initial_list): # for each sigmas
            # print("Index Value", index)
            initial.append(initial_list[index][key])
            model.append(model_list[index][key])

        # ylim_min = (min(min(initial+model)))
        # ylim_max = (max(max(initial+model)))

        # print("Key ", key)
        # print(" min",ylim_min )
        # print(" max",ylim_max )

        create_each_plot(initial=initial, model=model,save_name=key,y_min=ylim_min[i],y_max=ylim_max[i],ticks=names)
        # create_each_plot(initial=initial, model=model,save_name=key,y_min=ylim_min,y_max=ylim_max,ticks=names)


# new function for evaluating upsample 50 micron as dictionary structure is different
def evaluate_model(opt):
    
    initial ={'psnr':[],'ssim':[],'mse':[],'nrmse':[],'hfen':[]}
    model = {'psnr':[],'ssim':[],'mse':[],'nrmse':[],'hfen':[]}

    for key in opt.dictionary: 
        label_name = opt.dictionary[key]
        image_path = os.path.join(opt.image_path,key)
        label_path = os.path.join(opt.label_path, label_name)

        output = predict_model(model=opt.model,image_path=image_path,label_path=label_path,device=opt.device,psnr=opt.psnr,ssim=opt.ssim,mse = opt.mse,nrmse=opt.nrmse)
            
        # append initial metric
        initial['psnr'].append(output['init_psnr'])
        initial['ssim'].append(output['init_ssim'])
        initial['mse'].append(output['init_mse'])
        initial['nrmse'].append(output['init_nrmse'])
        initial['hfen'].append(output['init_hfen'])
        model['psnr'].append(output['model_psnr'])
        model['ssim'].append(output['model_ssim'])
        model['mse'].append(output['model_mse'])
        model['nrmse'].append(output['model_nrmse'])
        model['hfen'].append(output['model_hfen'])

    #print min, max std and median
    # print('metric for initial')
    # for key in initial.keys():
    #     print('key is',key) 
    #     # print('min : ',min(initial[key]))
    #     # print('max : ',max(initial[key]))  
    #     # if key == 'hfen':
    #     #     pass
    #     # else:
    #     #     print( ' std :', statistics.pstdev(initial[key]))
    #     print( ' mean :', statistics.mean(initial[key]))
    #     # print( ' median :', statistics.median(initial[key]))
    #     # print('************************************************************************************')
    # print('************************************************************************************')
    print('metric for model')
    for key in model.keys():
        # print('key is',key) 
        # print('min : ',min(model[key])) 
        # print('max : ',max(model[key])) 
        # if key == 'hfen':
        #     pass
        # else:
        #     print( ' std :', statistics.pstdev(initial[key]))
        print( ' mean :', statistics.mean(model[key]))
        # print( ' median :', statistics.median(model[key]))
        # print('************************************************************************************')

    # with open(opt.model_name+'.yaml', 'w') as f:
    #     json.dump(model, f, indent=2)

    # with open(opt.initial_name+'.yaml', 'w') as f:
    #     json.dump(initial, f, indent=2)

    return initial, model


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='model demo')
    parser.add_argument('--checkpoint', type=str, metavar='',required=False,help='checkpoint path')
    opt = parser.parse_args()

    opt.model_name = 'srdense'
    print("loading DEnseNet")

    sigma_value = [25,50,75,100,125,150]
    dataset_names = ['sigma_25', 'sigma_50', 'sigma_75', 'sigma_100','sigma_125','sigma_150']
    # dataset_names= ['sigma_100', 'hanning', 'hamming', 'bicubic','mean_blur','median_blur']

    # checkpoint1=[]
    # for sigma in  sigma_value:
    #     checkpoint = 'outputs/gaussian_dataset25_sigma'+str(sigma)+'/srdense/gaussian_mul_wo_circular_mask_sigma'+str(sigma)+'/checkpoints/z_axis/factor_2/epoch_500_f_2.pth'
    #     checkpoint1.append(checkpoint)


    checkpoint1=[
       'outputs/gaussian_dataset25_sigma100/srdense/gaussian_mul_wo_circular_mask_sigma100/checkpoints/z_axis/factor_2/epoch_500_f_2.pth',
       'outputs/dataset_images/hanning_dataset25/srdense/srdense_hanning_blur/checkpoints/z_axis/factor_2/epoch_500_f_2.pth',
       'outputs/dataset_images/hamming_dataset25/srdense/srdense_hamming_blur/checkpoints/z_axis/factor_2/epoch_500_f_2.pth',
       'outputs/dataset_images/bicubic_dataset25/srdense/bicubic_up_and_down_factor_2/checkpoints/z_axis/factor_2/epoch_500_f_2.pth',
       'outputs/dataset_images/mean_blur_dataset25/srdense/srdense_mean_blur/checkpoints/z_axis/factor_2/epoch_500_f_2.pth',
       'outputs/dataset_images/median_blur_dataset25/srdense/srdense_median_blur/checkpoints/z_axis/factor_2/epoch_500_f_2.pth',
       'outputs/dataset_images/bicubic_dataset25/srcnn/srcnn_bicubic_dataset25/checkpoints/z_axis/factor_2/epoch_500_f_2.pth',
       'outputs/dataset_images_rrdbnet/median_blur_dataset25/rrdbnet/rrdbnet_median_blur_dataset25/checkpoints/z_axis/factor_2/epoch_500_f_2.pth',

        'outputs/dataset_images_srdense/VGG_loss/combined_all_fix_dataset25/srdense/srcnn_combine_all_fix_dataset25_vgg/checkpoints/z_axis/factor_2/epoch_500_f_2.pth',

       'outputs/dataset_images_srdense/L1_loss/hamming_dataset25/srdense/srdense_hamming_dataset25_l1/checkpoints/z_axis/factor_2/epoch_500_f_2.pth',

       'outputs/dataset_images_rrdbnet/combined_fixed_dataset25/rrdbnet/rrdbnet_combined_fixed_dataset25/checkpoints/z_axis/factor_2/epoch_500_f_2.pth',
       'outputs/dataset_images_srdense/L1_loss/mean_blur_dataset25/srdense/srdense_mean_blur_dataset25/checkpoints/z_axis/factor_2/epoch_500_f_2.pth'
    ] 
    checkpoint_names = ['gaussian_model', 'hanning_model', 'hamming_model', 'bicubic_model', 'mean_model', 'median_model', "srcnn_bicubic","rrdbnet_median_blur","dense_vgg_comnined", "srdense_ssim_loss_combined","rrdbnet_combine_fix","dense_L1_mean_blur"]
   
    index_number = 9
    checkpoint= checkpoint1[index_number]
    checkpoint_name = checkpoint_names[index_number]

    # checkpoint = 'outputs/dataset_images/combine_all_degradation_large_dataset/srdense/srdense_combine_all_degradation_large_dataset/checkpoints/z_axis/factor_2/epoch_500_f_2.pth'
    # checkpoint = 'outputs/dataset_images/combine_all_degradation_fix_dataset/srdense/srdense_combine_all_degradation_fix_dataset/checkpoints/z_axis/factor_2/epoch_450_f_2.pth'
    # checkpoint_name = 'combine fix model'
    # checkpoint ='outputs/dataset_images/combine_gaussian_large_dataset/srdense/srdense_combine_gaussian_large_dataset/checkpoints/z_axis/factor_2/epoch_500_f_2.pth'
    # checkpoint = 'outputs/dataset_images/combine_gaussian_fix_dataset/srdense/srdense_combine_gaussian_fix_dataset/checkpoints/z_axis/factor_2/epoch_500_f_2.pth'

    # names = ['sigma_25', 'sigma_50', 'sigma_75', 'sigma_100','sigma_125','sigma_150']
    names = ['sigma_100', 'hanning_model', 'hamming_model', 'bicubic','mean','median']

    # annotations = 'upsample/combine/annotation_hr1_dict.pkl'

    annotations = [
        # 'dataset_images/gaussian_dataset25_sigma25/test_annotation.pkl',
        # 'dataset_images/gaussian_dataset25_sigma50/test_annotation.pkl',
        # 'dataset_images/gaussian_dataset25_sigma75/test_annotation.pkl',
        # 'dataset_images/gaussian_dataset25_sigma100/test_annotation.pkl',
        # 'dataset_images/gaussian_dataset25_sigma125/test_annotation.pkl',
        # 'dataset_images/gaussian_dataset25_sigma150/test_annotation.pkl',


        'dataset_images/gaussian_dataset25_sigma100F/test_annotation.pkl',
        'dataset_images/hanning_dataset25/annotation_test_dict.pkl',
        'dataset_images/hamming_dataset25/annotation_test_dict.pkl',
        'dataset_images/bicubic_dataset25/z_axis/factor_2/annotation_test_dict.pkl',
        'dataset_images/mean_blur_dataset25/annotation_test_dict.pkl',
        'dataset_images/median_blur_dataset25/annotation_test_dict.pkl'
        
    ] 

    image_path_sigma = [
    
        # 'dataset_images/gaussian_dataset25_sigma25/z_axis/factor_2/test',
        # 'dataset_images/gaussian_dataset25_sigma50/z_axis/factor_2/test',
        # 'dataset_images/gaussian_dataset25_sigma75/z_axis/factor_2/test',
        # 'dataset_images/gaussian_dataset25_sigma100/z_axis/factor_2/test',
        # 'dataset_images/gaussian_dataset25_sigma125/z_axis/factor_2/test',
        # 'dataset_images/gaussian_dataset25_sigma150/z_axis/factor_2/test',

        'dataset_images/gaussian_dataset25_sigma100F/z_axis/factor_2/test',
        'dataset_images/hanning_dataset25/z_axis/factor_2/test',
        'dataset_images/hamming_dataset25/z_axis/factor_2/test',
        'dataset_images/bicubic_dataset25/z_axis/factor_2/test',
        'dataset_images/mean_blur_dataset25/z_axis/factor_2/test',
        'dataset_images/median_blur_dataset25/z_axis/factor_2/test',
    ]

    label_path ='dataset_images/combined_kspace_gaussian_bicubic/z_axis/label/test'

    '''set device'''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.device = device

    metrics = ['psnr', 'ssim', 'mse', 'nrmse','hfen']

    psnr = PSNR()
    ssim = SSIM()

    psnr = psnr.to(device=opt.device, memory_format=torch.channels_last, non_blocking=True)
    ssim = ssim.to(device=opt.device, memory_format=torch.channels_last, non_blocking=True)

    opt.psnr= psnr
    opt.ssim=ssim
    opt.mse = nn.MSELoss().to(device=opt.device)
    opt.nrmse = NRMSELoss().to(device=opt.device)


    initial_list = []  # list of dictionary for each dataset
    model_list = []

    model = load_model(checkpoint,device,opt.model_name)
    if device != 'cpu':
        num_of_gpus = torch.cuda.device_count()
        model = nn.DataParallel(model,device_ids=[*range(num_of_gpus)])
    model.to(device)
    model.eval()
    opt.model=model

    for index,dataset in enumerate(dataset_names):
        opt.dictionary = read_dictionary(annotations[index])
        opt.image_path = image_path_sigma[index]
        if dataset_names[index] == '50_micron':
            opt.label_path = 'dataset_images/upscaled_50_micron_datset/z_axis/label/test'
        else:
            opt.label_path = label_path
        
        print("Image path", opt.image_path)
        # print("Label path",opt.label_path)
        print("checkpoint", checkpoint)
        print("Evaluating for model", checkpoint_name)

        initial, model = evaluate_model(opt)  # each dictionary of list with keys psnr,ssim,mse,nrmse,hfen
        model_list.append(model)
        initial_list.append(initial)
        
        # print("MODEL METRIC")
        # for metric in metrics:
        #     print("Average Values for {} is {} ".format(metric,statistics.mean(model[metric])))
        # print("INITIAL METRIC")
        # for metric in metrics:
        #     print("Average Values for {} is {} ".format(metric,statistics.mean(initial[metric])))

        # print("**************************************************************************************************")

    plot_boxplot(initial_list=initial_list,model_list=model_list)    



