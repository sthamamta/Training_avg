# load model
# load array
# apply model
# store all output to list
# average the list
# plot the lr and hr image 

import argparse
import torch
import nibabel as nib
from model.densenet import SRDenseNet
from utils.image_quality_assessment import PSNR,SSIM
from utils.general import min_max_normalize, NRMSELoss
from utils.load_checkpoint import load_checkpoint
import torch.nn as nn
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0,1'

def total_variation_loss(img, weight):
     bs_img, c_img, h_img, w_img = img.size()
     tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
     tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()
     return weight*(tv_h+tv_w)/(bs_img*c_img*h_img*w_img)


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

def prepare_tensor(lr_Image):
    lr_tensor= min_max_normalize(lr_image).float()
    lr_tensor= torch.unsqueeze(lr_tensor,0)
    lr_tensor= torch.unsqueeze(lr_tensor,0)
    return lr_tensor

parser = argparse.ArgumentParser(description='model demo')
parser.add_argument('-checkpoint', type=str, metavar='',required=False,help='checkpoint path', default= 'outputs/checkpoints/rrdbnet_avg_image_train_mse_ssim_derivative_loss/factor_2/epoch_1250_f_2.pth')
parser.add_argument('-hr-array-path', type=str, metavar='',required=False,help='test hr array path',default= '../array_data/25/f1_25.nii')
parser.add_argument('-lr-array-path', type=str, metavar='',required=False,help='test lr array path',default= '../array_data/50/f1_50.nii')
parser.add_argument('-save-path', type=str, metavar='',required=False,help='plots save path',default= 'plots')
parser.add_argument('-model-name', type=str, metavar='',required=False,help='model name',default= 'rrdbnet')
args = parser.parse_args()

'''create output directory'''
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

'''load array'''
lr_array = load_data_nii(args.lr_array_path)
print(lr_array.shape)

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

axis = 2
output_array = {}
start_index = 50
'''loop through lr array and apply model and append to otuput_array'''
for i in range(start_index,100):

    lr_image = lr_array[i,:,:]
    lr_tensor= prepare_tensor(lr_image)
    first_out,second_out,third_out =  model(lr_tensor)

    if i==start_index:
        first_pred = first_out
    else:
        first_pred = (first_out + addition)/2
    second_pred = second_out

    output_array[2*i-1] = first_pred
    output_array[2*i]= second_pred
    addition = third_out

    print("output shape",first_out.shape)
    print(i)

print("complete appply model")

hr_array = load_data_nii(args.hr_array_path)


for i in range (55,95):
    hr_tensor = hr_array[2*i,:,:].unsqueeze(0).unsqueeze(0).float()
    lr_tensor = lr_array[i,:,:].unsqueeze(0).unsqueeze(0).float()
    pred_tensor = output_array[2*i].float()

    diff_x_hr = hr_tensor[:,:,:,1:]-hr_tensor[:,:,:,:-1]
    diff_x_hr = diff_x_hr.squeeze().numpy()

    diff_y_hr = hr_tensor[:,:,1:,:]-hr_tensor[:,:,:-1,:]
    diff_y_hr = diff_y_hr.squeeze().numpy()

    
    diff_x_lr = lr_tensor[:,:,:,1:]-lr_tensor[:,:,:,:-1]
    diff_x_lr = diff_x_lr.squeeze().numpy()

    diff_y_lr = lr_tensor[:,:,1:,:]-lr_tensor[:,:,:-1,:]
    diff_y_lr = diff_y_lr.squeeze().numpy()


    diff_x_pred = pred_tensor[:,:,:,1:]-pred_tensor[:,:,:,:-1]
    diff_x_pred = diff_x_pred.squeeze().detach().cpu().numpy()

    diff_y_pred = pred_tensor[:,:,1:,:]-pred_tensor[:,:,:-1,:]
    diff_y_pred = diff_y_pred.squeeze().detach().cpu().numpy()

    fnsize = 17
    fig = plt.figure(figsize=(25,15))


    fig.add_subplot(1, 3, 1)
    plt.title('lr_tensor',fontsize=fnsize)
    plt.imshow(diff_x_hr,cmap='gray')

    fig.add_subplot(1, 3, 2)
    plt.title('hr_tensor',fontsize=fnsize)
    plt.imshow(diff_x_lr,cmap='gray')

    fig.add_subplot(1, 3, 3)
    plt.title('pred_tensor',fontsize=fnsize)
    plt.imshow(diff_x_pred,cmap='gray')

    plt.tight_layout()
    save_name = 'output_'+ str(i)+'.png'

    plt.savefig('test_figure'+'/'+save_name)
    print('figure saved')

# quit();

'''plot and measure metric'''
for i in range(55,95):
    lr_index = i//2
    pred = output_array[i]
    label = hr_array[i,:,:]
    input = lr_array[lr_index,:,:]
    # label = prepare_tensor(label)

    pred = pred.clamp(0.,1.).squeeze().detach().cpu().numpy()
    # label = label.squeeze(0).numpy()

    label =  min_max_normalize(label)
    input = min_max_normalize(input)
    pred = min_max_normalize(pred)
    error =  label - pred



    fnsize = 17
    fig = plt.figure(figsize=(25,15))


    fig.add_subplot(1, 3, 1)
    plt.title('Pred',fontsize=fnsize)
    plt.imshow(pred,cmap='gray',vmin=0.,vmax=0.9)
    # psnr = psnr(labels_0,labels_0,data_range=labels_0.max() - labels_0.min())
    # ssim = ssim(labels_0,labels_0,multichannel=False,gaussian_weights=True, sigma=1.5, 
    #                     use_sample_covariance=False, data_range=labels_0.max() - labels_0.min())
    # plt.xlabel('PSNR=%.2f\nSSIM=%.4f \n (1)' % (psnr, ssim,),fontsize=fnsize)

    fig.add_subplot(1, 3, 2)
    plt.title('Label',fontsize=fnsize)
    plt.imshow(label,cmap='gray')
    # psnr = psnr(labels_0,images_0,data_range=labels_0.max() - labels_0.min())
    # ssim = ssim(labels_0,images_0,multichannel=False,gaussian_weights=True, sigma=1.5, 
    #                     use_sample_covariance=False, data_range=labels_0.max() - labels_0.min())
    # plt.xlabel('PSNR=%.2f\nSSIM=%.4f \n (2)' % (psnr, ssim),fontsize=fnsize)

    fig.add_subplot(1, 3, 3)
    plt.title('Input',fontsize=fnsize)
    plt.imshow(input,cmap='gray')

    plt.tight_layout()
    save_name = 'output_'+ str(i)+'.png'

    plt.savefig(args.save_path+'/'+save_name)
    print('figure saved')
    # plt.show()

