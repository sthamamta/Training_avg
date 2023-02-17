import argparse
import torch
import nibabel as nib
from model.densenet import SRDenseNet
from utils.image_quality_assessment import PSNR,SSIM
from utils.general import min_max_normalize, NRMSELoss
from utils.load_checkpoint import load_checkpoint
import torch.nn as nn
import matplotlib.pyplot as plt
from dataset.downsample_utils import *
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0,1'

hr_array_path = '../3D_Training/data_array/25/f1_25.nii'

save_path = 'test_figure'
'''create output directory'''
if not os.path.exists(save_path):
    os.makedirs(save_path)


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

'''load array'''
hr_array = load_data_nii(hr_array_path)
print(hr_array.shape)
axis = 2

save_paths = ['test_figure/hr','test_figure/bicubic', 'test_figure/crop','test_figure/bic_gauss50','test_figure/bic_gauss75','test_figure/crop_gauss50','test_figure/crop_gauss75']

for path in save_paths:
    if not os.path.exists(path):
        os.makedirs(path)

for i in range(10, hr_array.shape[2]-10, 10):
    hr_image = hr_array[:,:,i]

    lr_image_bicubic = downsample_bicubic(hr_image,2)
    lr_image_crop = downsample_kspace (hr_image=hr_image,gaussian=False,factor=2)
    lr_image_bicubic_gaussian50 = downsample_bicubic_gaussian(hr_image=hr_image,sigma=50)
    lr_image_bicubic_gaussian75 = downsample_bicubic_gaussian(hr_image=hr_image,sigma=75)
    lr_image_crop_gaussian50 = downsample_kspace(hr_image=hr_image, gaussian=True,sigma=50,factor=2)
    lr_image_crop_gaussian75 = downsample_kspace(hr_image=hr_image, gaussian=True,sigma=75,factor=2)

    images = [hr_image,lr_image_bicubic,lr_image_crop,lr_image_bicubic_gaussian50,lr_image_bicubic_gaussian75,lr_image_crop_gaussian50,lr_image_crop_gaussian75]
    for idx, image in enumerate(images):
        fnsize = 17
        fig = plt.figure(figsize=(25,15))
        plt.imshow(image,cmap='gray')
        plt.tight_layout()
        save_name = 'image_'+ str(i)+'.png'
        path = os.path.join(save_paths[idx],save_name)
        plt.savefig(path)
        print('figure saved')




