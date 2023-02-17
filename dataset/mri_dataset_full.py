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


def load_array_from_list(hr_path_main, lr_path_main, hr_files_list, lr_files_list,axis):
    
    hr_path = os.path.join(hr_path_main,hr_files_list[0])
    lr_path = os.path.join(lr_path_main,lr_files_list[0])

    hr_array = load_data_nii(hr_path)
    lr_array = load_data_nii(lr_path)

    del hr_files_list[0]
    del lr_files_list[0]

    if axis == 2:
        #index for f1
        lr_index_list_f1 = [x for x in range(6,lr_array.shape[axis]-6)]
        hr_index_list_f1 = [2*x for x in lr_index_list_f1]

        #index for f2
        lr_index_list_f2 = [x for x in range(6,lr_array.shape[axis]-6)]
        hr_index_list_f2 = [(2*x)-6 for x in lr_index_list_f2]
        lr_index_list_f2 = [x+152 for x in lr_index_list_f2]
        hr_index_list_f2 = [x+304 for x in hr_index_list_f2]


        #index for f5 
        lr_index_list_f5 = [x for x in range(6,lr_array.shape[axis]-6)]
        hr_index_list_f5 = [(2*x)-6 for x in lr_index_list_f5]
        lr_index_list_f5 = [x+304 for x in lr_index_list_f5]
        hr_index_list_f5 = [x+608 for x in hr_index_list_f5]

    elif axis == 0: #images along x-axis 

        #index for f1
        lr_index_list_f1 = [x for x in range(6,lr_array.shape[axis]-35)]
        hr_index_list_f1 = [2*x-1 for x in lr_index_list_f1]

        #index for f2
        lr_index_list_f2 = [x for x in range(6,lr_array.shape[axis]-35)]
        hr_index_list_f2 = [(2*x)+7 for x in lr_index_list_f2]
        lr_index_list_f2 = [x+360 for x in lr_index_list_f2]
        hr_index_list_f2 = [x+720 for x in hr_index_list_f2]


        #index for f5 
        lr_index_list_f5 = [x for x in range(6,lr_array.shape[axis]-35)]
        hr_index_list_f5 = [(2*x)+2 for x in lr_index_list_f5]
        lr_index_list_f5 = [x+720 for x in lr_index_list_f5]
        hr_index_list_f5 = [x+1440 for x in hr_index_list_f5]
    else:
        print("image extraction axis not implemeted in dataset")


    hr_index_list = hr_index_list_f1+hr_index_list_f2+hr_index_list_f5
    lr_index_list =lr_index_list_f1+lr_index_list_f2+lr_index_list_f5

    #appending hr array
    for path in hr_files_list:
        path = os.path.join(hr_path_main,path)
        array = load_data_nii(path)
        hr_array = np.append(hr_array,array,axis=axis)
   
    #appending lr array
    for path in lr_files_list:
        path = os.path.join(lr_path_main,path)
        array = load_data_nii(path)
        lr_array = np.append(lr_array,array,axis=axis)

    return hr_array,lr_array,lr_index_list,hr_index_list
              





class MRIDatasetWOAVGFull(Dataset):
    def __init__(self,hr_path_main,lr_path_main, hr_files_list, lr_files_list, factor=2,eval=False,axis = 2):
        self.factor = factor
        self.eval = eval
        self.hr_array,self.lr_array,self.lr_index_list,self.hr_index_list = load_array_from_list(hr_path_main=hr_path_main, lr_path_main=lr_path_main, hr_files_list=hr_files_list, lr_files_list=lr_files_list,axis=axis)

        self.axis = axis  # axis 0 means along x-direction, only works for z-axis, need to modify getitem function
        print("Without avg dataset")

        # print("Length of dataset is ", len(self.lr_index_list))
        # print("Hr array shape", self.hr_array.shape)
        # print("Lr array shape", self.lr_array.shape)


    def __len__(self):
        return len(self.lr_index_list)

    def __getitem__(self, index):
        lr_index = self.lr_index_list[index]
        hr_index =  self.hr_index_list[index]


        if self.axis == 0:
            lr_image = self.lr_array[lr_index,:,:]
            hr_image = self.hr_array[hr_index,:,:]
        elif self.axis == 2:
            lr_image = self.lr_array[:,:,lr_index]
            hr_image = self.hr_array[:,:,hr_index]
        else:
           print("Axis not implemented in dataset class")

        # print("lr index", lr_index)
        # print("Hr index", hr_index)
        # print("lr shape", lr_image.shape)
        # print("Hr shape", first_hr.shape)
        # print("*********************************************************************************************************")
        lr_tensor= min_max_normalize(lr_image)
        hr_tensor = min_max_normalize(hr_image)

        

        lr_tensor = torch.from_numpy(lr_tensor).float()
        hr_tensor = torch.from_numpy(hr_tensor).float()
    

        lr_tensor= torch.unsqueeze(lr_tensor,0)
        hr_tensor = torch.unsqueeze(hr_tensor,0)
        

        return {
                "lr": lr_tensor, 
                "hr": hr_tensor
        }