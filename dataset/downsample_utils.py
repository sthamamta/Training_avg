
'''With the use of this dataset, we downsample the hr image by factor 2 to get lr using various methods and train model to increase the size of the image'''

import torch
import  cv2, os
from torch.utils.data import Dataset
import numpy as np
import utils as ut

import numpy as np
import nibabel as nib
import torch
from PIL import Image
import math

import warnings
warnings.filterwarnings("ignore")


def min_max_normalize(image):
    max_img = image.max()
    min_img = image.min()
    denom = (max_img-min_img) + 0.00000000001
    norm_image = (image-min_img)/denom
    return norm_image 


def load_data_nii(fname):
    img = nib.load(fname)
    data = img.get_fdata()
    data_norm = torch.from_numpy(data)
    return data_norm 


def prepare_lr_array(hr_array,factor=2,pad=True):
    # 3d fourier transform
    spectrum_3d = np.fft.fftn(hr_array)  

    # Apply frequency shift along spatial dimentions 
    spectrum_3d_sh = np.fft.fftshift(spectrum_3d, axes=(0,1,2))  

    x,y,z = spectrum_3d_sh.shape
    data_pad = np.zeros((x,y,z),dtype=np.complex_)
    center_y = y//2 #defining the center of image in x, y and z direction
    center_x = x//2
    center_z = z//2
    startx = center_x-(x//(factor*2))  
    starty = center_y-(y//(factor*2))
    startz = center_z-(z//(factor*2))
    arr = spectrum_3d_sh[startx:startx+(x//factor),starty:starty+(y//factor),startz:startz+(z//factor)]
      
    if pad:
        data_pad[startx:startx+(x//factor),starty:starty+(y//factor),startz:startz+(z//factor)] = arr
        img_reco_cropped = np.fft.ifftn(np.fft.ifftshift(data_pad))
    else:
        img_reco_cropped = np.fft.ifftn(np.fft.ifftshift(arr)) 
        
    return np.abs(img_reco_cropped)



def prepare_lr_image(hr_image):
    pass



def load_array_from_list(hr_path_main, hr_files_list,axis,degradation_methods):
    
    hr_path = os.path.join(hr_path_main,hr_files_list[0])
    hr_array = load_data_nii(hr_path)

    del hr_files_list[0]

    if axis == 2:
        #index for f1
        hr_index_list_f1 = [x for x in range(6,hr_array.shape[axis]-6)]

        #index for f2
        hr_index_list_f2 = [x for x in range(6,hr_array.shape[axis]-6)]
        hr_index_list_f2 = [x+304 for x in hr_index_list_f2]

        #index for f5 
        hr_index_list_f5 = [x for x in range(6,hr_array.shape[axis]-6)]
        hr_index_list_f5 = [x+608 for x in hr_index_list_f5]

    elif axis == 0: #images along x-axis 

        #index for f1
        hr_index_list_f1 = [x for x in range(6,hr_array.shape[axis]-35)]

        #index for f2
        hr_index_list_f2 = [x for x in range(6,hr_array.shape[axis]-35)]
        hr_index_list_f2 = [x+720 for x in hr_index_list_f2]

        #index for f5 
        hr_index_list_f5 = [x for x in range(6,hr_array.shape[axis]-35)]
        hr_index_list_f5 = [x+1440 for x in hr_index_list_f5]
    else:
        print("image extraction axis not implemeted in dataset")


    hr_index_list = hr_index_list_f1+hr_index_list_f2+hr_index_list_f5
    list_length_for_single_degradation_type = len(hr_index_list)
    degradation_lists = [method]* list_length_for_single_degradation_type
    
    del degradation_methods[0]
    for method in degradation_methods:
        degradation_list = [method]* list_length_for_single_degradation_type
        degradation_lists += degradation_list
        hr_index_list += hr_index_list


    #appending hr array
    for path in hr_files_list:
        path = os.path.join(hr_path_main,path)
        array = load_data_nii(path)
        hr_array = np.append(hr_array,array,axis=axis)

    
   
    return hr_array,hr_index_list,degradation_lists
              

def downsample_bicubic(hr_image,factor=2):
    hr_image = min_max_normalize(hr_image)
    # hr_image = (hr_image-hr_image.min())/(hr_image.max()-hr_image.min())
    hr_image = hr_image.numpy()
    array = hr_image * 255.
    array = array.astype(np.uint8)
    image = Image.fromarray(array)

    x,y = image.size
    
    shape = (x//(2*factor), y//(2*factor))  #downsample by factor 4

    image = image.resize(shape,Image.BICUBIC)

    shape = (x//factor, y//factor)  #upsample by factor 2

    image = image.resize(shape,Image.BICUBIC)

    image = np.array(image).astype(np.float32)
    image = min_max_normalize(image)

    return image
    
def downsample_kspace(hr_image, gaussian = False, sigma = 75,factor=2):
    F = np.fft.fft2(hr_image)
    fshift = np.fft.fftshift(F)
    
    y,x = fshift.shape
    data_pad = np.zeros((y,x),dtype=np.complex_)
    center_y = y//2 #defining the center of image in x and y direction
    center_x = x//2
    startx = center_x-(x//(factor*2))  
    starty = center_y-(y//(factor*2))
    
    arr = fshift[starty:starty+(y//factor),startx:startx+(x//factor)]
    
    img_reco_cropped = np.fft.ifft2(np.fft.ifftshift(arr)) 
    image = np.abs(img_reco_cropped )
    
    if gaussian:
        image = downsample_gaussian_with_sigma(image,sigma)

    return np.abs(image)

def get_gaussian_low_pass_filter(shape, sigma = None):
        """Computes a gaussian low pass mask
        shape: the shape of the mask to be generated
        sigma: the cutoff frequency of the gaussian filter 
        factor: downsampling factor for given image
        returns a gaussian low pass mask"""
        rows, columns = shape
        d0 = sigma
        mask = np.zeros((rows, columns), dtype=np.complex_)
        mid_R, mid_C = int(rows / 2), int(columns / 2)
        for i in range(rows):
            for j in range(columns):
                d = math.sqrt((i - mid_R) ** 2 + (j - mid_C) ** 2)
                mask[i, j] = np.exp(-(d * d) / (2 * d0 * d0)) #dont divide by 2pi(sigma)^2 as it reduces the value of mask to the order of e-6
                
        final_mask = mask
        return final_mask

def apply_filter(image_arr,filter_apply):
    F = np.fft.fft2(image_arr)  #fourier transform of image
    fshift = np.fft.fftshift(F)  #shifting 

    FFL = filter_apply* fshift  #multiplying with gaussian filter

    img_recon = np.abs(np.fft.ifft2(np.fft.ifftshift(FFL))) #inverse shift and inverse fourier transform
    
    return img_recon
    
def downsample_gaussian_with_sigma(image_arr,sigma=150):
    low_filter = get_gaussian_low_pass_filter(image_arr.shape,sigma=sigma)
    image_downsampled = apply_filter(image_arr,low_filter)

    return image_downsampled


def downsample_bicubic_gaussian(hr_image,sigma=50):
    lr_image = downsample_bicubic(hr_image)
    lr_image = downsample_gaussian_with_sigma(lr_image,sigma)
    return lr_image



def prepare_lr_image(hr_image, degradation_method):
    if degradation_method == 'bicubic':
        lr_image = downsample_bicubic(hr_image)
    elif degradation_method in ['crop_kspace','kspace']:
        lr_image = downsample_kspace(hr_image)
    elif degradation_method == 'kspace_gaussian50':
        lr_image = downsample_kspace(hr_image=hr_image, gaussian =True, sigma=50)
    elif degradation_method == 'bicubic_gaussian_50':
        lr_image = downsample_bicubic_gaussian(hr_image=hr_image,sigma=50)
    elif degradation_method == 'kspace_gaussian75':
        lr_image = downsample_kspace(hr_image=hr_image, gaussian =True, sigma=75)
    elif degradation_method == 'bicubic_gaussian75':
        lr_image = downsample_bicubic_gaussian(hr_image=hr_image,sigma=50)
    else:
        lr_image = downsample_bicubic(hr_image)
    return lr_image
