
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
from downsample_utils import *

import warnings
warnings.filterwarnings("ignore")




class MRIDatasetWOAVGAug(Dataset):
    def __init__(self,hr_path_main, hr_files_list, factor=2,eval=False,axis = 2, degradation_methods = ['bicubic', 'crop_kspace', 'bicubic_gaussian50','kspace_gaussian50','bicubic_gaussian75','kspace_gaussian75']):
        self.factor = factor
        self.eval = eval
        self.hr_array,self.index_list,self.degradation_type_list = load_array_from_list(hr_path_main=hr_path_main, hr_files_list=hr_files_list,axis=axis, degradation_methods=degradation_methods)

        self.axis = axis  # axis 0 means along x-direction, only works for z-axis, need to modify getitem function
        print("Without avg dataset")

        # print("Length of dataset is ", len(self.lr_index_list))
        # print("Hr array shape", self.hr_array.shape)
        # print("Lr array shape", self.lr_array.shape)


    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, index):
        hr_index =  self.index_list[index]
        degradation = self.degradation_type_list[index]

        if self.axis == 0:
            hr_image = self.hr_array[hr_index,:,:]
        elif self.axis == 2:
            hr_image = self.hr_array[:,:,hr_index]
        else:
           print("Axis not implemented in dataset class")

        # print("lr index", lr_index)
        # print("Hr index", hr_index)
        # print("lr shape", lr_image.shape)
        # print("Hr shape", first_hr.shape)
        # print("*********************************************************************************************************")
        hr_image = min_max_normalize(hr_image)
        lr_image =  prepare_lr_image(hr_image=hr_image, degradation_method=degradation)

        
        lr_tensor = torch.from_numpy(lr_image).float()
        hr_tensor = torch.from_numpy(hr_image).float()
    

        lr_tensor= torch.unsqueeze(lr_tensor,0)
        hr_tensor = torch.unsqueeze(hr_tensor,0)
        

        return {
                "lr": lr_tensor, 
                "hr": hr_tensor
        }