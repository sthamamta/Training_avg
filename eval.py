# load array
#load model
# in a loop apply model and measure psnr, ssim, mse, nrmse

import torch
import torch.nn as nn
import nibabel as nib
import  cv2
import matplotlib.pyplot as plt
import argparse
import statistics
import json
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'

from utils.preprocess import tensor2image, image2tensor,create_dictionary
from utils.prepare_test_set_image import crop_pad_kspace, prepare_image_fourier
from utils.image_quality_assessment import PSNR,SSIM
from utils.general import min_max_normalize, NRMSELoss
from model.densenet import SRDenseNet,SRDenseNetWOAVG
from model.rrdbnet import RRDBNet,RRDBNetWOAVG
import os
import numpy as np
from utils.preprocess import hfen_error
from utils.load_checkpoint import load_checkpoint

import warnings
warnings.filterwarnings("ignore")

def load_data_nii(fname):
    img = nib.load(fname)
    data = img.get_fdata()
    data_norm = torch.from_numpy(data)
    return data_norm 

def min_max_normalize(image):
    max_img = image.max()
    min_img = image.min()
    denom = (max_img-min_img) + 0.00000000001
    norm_image = (image-min_img)/denom
    return norm_image 

def prepare_tensor(image_array):
    # image_tensor = torch.from_numpy(image_array).float()
    image_tensor= torch.unsqueeze(image_array,0)
    image_tensor= torch.unsqueeze(image_tensor,0)
    return image_tensor

parser = argparse.ArgumentParser(description='model demo')
parser.add_argument('-checkpoint', type=str, metavar='',required=False,help='checkpoint path', default= 'outputs/checkpoints/srdense_train_full_axis2_mse_ssim_derivative_loss/factor_2/epoch_400_f_2.pth')
parser.add_argument('-hr-array-path', type=str, metavar='',required=False,help='test hr array path',default= '../3D_Training/data_array/25/f1_25.nii')
parser.add_argument('-lr-array-path', type=str, metavar='',required=False,help='test lr array path',default= '../3D_Training/data_array/50/f1_50.nii')
parser.add_argument('-save-path', type=str, metavar='',required=False,help='plots save path',default= 'plots')
parser.add_argument('-model-name', type=str, metavar='',required=False,help='model name',default= 'srdense')
args = parser.parse_args()

'''create output directory'''
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
    print("Folder created")

'''load lr array'''
lr_array = load_data_nii(args.lr_array_path)
print(lr_array.shape)


'''load hr array'''
hr_array = load_data_nii(args.hr_array_path)
print(hr_array.shape)



'''set device'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.device = device


"""load model"""
model =  load_checkpoint(args.checkpoint, device, args.model_name)
if device != 'cpu':
        num_of_gpus = torch.cuda.device_count()
        model = nn.DataParallel(model,device_ids=[*range(num_of_gpus)])
model.to(device)
model.eval()



'''load image quality metric'''
psnr = PSNR()
ssim = SSIM()
psnr = psnr.to(device= device, memory_format=torch.channels_last, non_blocking=True)
ssim = ssim.to(device= device, memory_format=torch.channels_last, non_blocking=True)
mse = nn.MSELoss().to(device=device)
nrmse = NRMSELoss().to(device=device)

''' set  corresponding index for lr and hr according to axis'''
axis = 0
lr_index_list = [x for x in range(6,lr_array.shape[axis]-35)]
hr_index_list = [2*x-1 for x in lr_index_list]


start_index = 20
end_index= lr_array.shape[axis]-50
step =1
plot_image =True
plot_metric = True

'''loop through lr array and apply model and measure metric'''
initial_metric ={'psnr':[],'ssim':[],'mse':[],'nrmse':[],'hfen':[]}
model_metric = {'psnr':[],'ssim':[],'mse':[],'nrmse':[],'hfen':[]}

for i in range(start_index,end_index,step):
    lr_index = lr_index_list[i]
    hr_index =  hr_index_list[i]
    if axis == 0:
        lr_image = lr_array[lr_index,:,:]
        hr_image = hr_array[hr_index,:,:]

    elif axis==2:
        lr_image = lr_array[:,:,lr_index]
        hr_image= hr_array[:,:,hr_index]
    else:
        print("axis not implemented")

    lr_image= min_max_normalize(lr_image)
    hr_image= min_max_normalize(hr_image)

    lr_tensor= prepare_tensor(lr_image).float().to(device)
    hr_tensor = prepare_tensor(hr_image).float().to(device)

    output_tensor =  model(lr_tensor)
    output_image = output_tensor.squeeze().detach().cpu().numpy()

    '''calculate image metric'''
    model_psnr = psnr(output_tensor, hr_tensor).item()
    model_ssim = ssim(output_tensor, hr_tensor).item()
    model_mse = mse(output_tensor, hr_tensor).item()
    model_nrmse = nrmse(output_tensor, hr_tensor).item()

    model_metric['psnr'].append(model_psnr)
    model_metric['ssim'].append(model_ssim)
    model_metric['mse'].append(model_mse)
    model_metric['nrmse'].append(model_nrmse)

    if plot_image:
        fnsize = 17
        fig = plt.figure(figsize=(25,15))

        fig.add_subplot(1, 3, 1)
        plt.title('input'+str(lr_index),fontsize=fnsize)
        plt.imshow(lr_image,cmap='gray')

        fig.add_subplot(1, 3, 2)
        plt.title('output',fontsize=fnsize)
        plt.imshow(output_image,cmap='gray')
        plt.xlabel('PSNR=%.2f\nSSIM=%.4f \n (2)' % (model_psnr, model_ssim),fontsize=fnsize)


        fig.add_subplot(1, 3, 3)
        plt.title('label'+str(hr_index),fontsize=fnsize)
        plt.imshow(hr_image,cmap='gray')

        plt.tight_layout()
        save_name = 'output_'+ str(i)+'.png'

        path = os.path.join(args.save_path,save_name)

        print(plt.savefig(path))
        print('figure saved')

keys = ['psnr', 'ssim', 'mse','nrmse']
# if plot_metric:
#     for i,key in enumerate(keys):  # for each metric
#         ylim_min = (min(min(model)))
#         ylim_max = (max(max(model)))

#         # print("Key ", key)
#         # print(" min",ylim_min )
#         # print(" max",ylim_max )

#         title = save_name.upper()
#         plt.figure()

#         bpl = plt.boxplot(initial, positions=np.array(range(len(initial)))*2.0-0.4, sym='', widths=0.4)
#         bpr = plt.boxplot(model, positions=np.array(range(len(model)))*2.0+0.1, sym='', widths=0.4)

#         set_box_color(bpl, '#D7191C') # colors are from http://colorbrewer2.org/
#         set_box_color(bpr, '#2C7BB6')


#         # draw temporary red and blue lines and use them to create a legend
#         plt.plot([], c='#D7191C', label='Initial')
#         plt.plot([], c='#2C7BB6', label='Model')
#         plt.legend()

#         plt.xticks(range(0, len(ticks) * 2, 2), ticks)
#         plt.xlim(-2, len(ticks)*2)
#         plt.ylim(y_min, y_max)
#         plt.ylabel(title)
#         plt.xlabel('datasets')
#         plt.tight_layout()
#         plt.title(title+' PLOT')

#         plt.show()
#         plt.savefig(save_name+'.png')


print("complete appply model")

