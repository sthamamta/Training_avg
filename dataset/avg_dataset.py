"this file is created to write a dataset class that takes an lr and hr array path and pass this input lr and multiple hr as label image, this dataset class is for 2d images for averaging the output image"

import torch
import  cv2, os
from torch.utils.data import Dataset
import numpy as np
import utils as ut

import numpy as np
import nibabel as nib
import torch
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





class MRIDataset(Dataset):
    def __init__(self,hr_array_path,lr_array_path, factor=2,eval=False,axis = 2):
        self.factor = factor
        self.eval = eval
        self.hr_array_path = hr_array_path
        self.lr_array_path = lr_array_path
        self.hr_array = load_data_nii(self.hr_array_path)
        self.lr_array =  load_data_nii(self.lr_array_path)
        self.axis = axis  # axis 0 means along x-direction, only works for z-axis, need to modify getitem function
        self.indexes = [*range(1,self.lr_array.shape[self.axis]-2)]
        # print("INdex values",self.indexes)

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index):
        lr_index = self.indexes[index]
        lr_image = self.lr_array[:,:,lr_index]
        hr_index =  lr_index * 2
        first_hr = self.hr_array[:,:,hr_index-1]
        second_hr = self.hr_array[:,:,hr_index]
        third_hr = self.hr_array[:,:,hr_index+1]
       
        # print("lr index", lr_index)
        # print("Hr index", hr_index)
        # print("*********************************************************************************************************")
        lr_tensor= min_max_normalize(lr_image).float()
        first_hr_tensor = min_max_normalize(first_hr).float()
        second_hr_tensor = min_max_normalize(second_hr).float()
        third_hr_tensor =  min_max_normalize(third_hr).float()

        # lr_tensor = torch.from_numpy(lr_image).float()
        # first_hr_tensor = torch.from_numpy(first_hr).float()
        # second_hr_tensor = torch.from_numpy(second_hr).float()
        # third_hr_tensor = torch.from_numpy(third_hr).float()

        lr_tensor= torch.unsqueeze(lr_tensor,0)
        first_hr_tensor = torch.unsqueeze(first_hr_tensor,0)
        second_hr_tensor = torch.unsqueeze(second_hr_tensor,0)
        third_hr_tensor = torch.unsqueeze(third_hr_tensor,0)


        return {
                "lr": lr_tensor, 
                "first_hr": first_hr_tensor,
                "second_hr":second_hr_tensor,
                "third_hr":third_hr_tensor
        }