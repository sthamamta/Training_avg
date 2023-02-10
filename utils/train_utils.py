

import torch
# from utils.preprocess import 
import torch.nn as nn
from model.densenet import SRDenseNet
from model.rrdbnet import RRDBNet
from dataset.avg_dataset import MRIDataset


import torch.optim as optim
import pickle
from loss.ssim_loss import SSIM
from loss.laplacian_pyramid_loss import LaplacianPyramidLoss
from loss.derivative_loss import DerivativeLoss

def read_dictionary(dir_dict):
    '''Read annotation dictionary pickle'''
    a_file = open(dir_dict, "rb")
    output = pickle.load(a_file)
    # print(output)
    a_file.close()
    return output


''' set the dataset path based on opt.dataset,opt.factor values and load & return the same dataset/dataloader'''
def load_dataset(opt, load_eval=True):
    train_dataloader,train_datasets =load_train_dataset(opt)
    if load_eval:
        eval_dataloader,val_datasets = load_val_dataset(opt)
        return train_dataloader,eval_dataloader,train_datasets,val_datasets
    else: 
        return train_dataloader,train_datasets

def load_train_dataset(opt):
    train_datasets = MRIDataset(hr_array_path = opt.train_hr_array_path, lr_array_path=opt.train_lr_array_path, factor = opt.factor,eval=False,axis=opt.axis)
    train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size = opt.train_batch_size,shuffle=True,
        num_workers=8,pin_memory=False,drop_last=False)
    return train_dataloader,train_datasets


def load_val_dataset(opt):
    val_datasets = MRIDataset(hr_array_path = opt.eval_hr_array_path, lr_array_path=opt.eval_lr_array_path, factor = opt.factor,eval=True,axis=opt.axis)
    eval_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size = opt.val_batch_size, shuffle=True,
        num_workers=8,pin_memory=False,drop_last=False)
    return eval_dataloader,val_datasets




'''reduce learning rate of optimizer by half on every  150 and 225 epochs'''
def adjust_learning_rate(optimizer, epoch,lr,lr_factor=0.5):
    if lr <= 0.00001:
        return lr
    else:
        if epoch % 150 == 0 or epoch % 250 == 0:
            lr = lr * lr_factor
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr



'''load the model instance based on opt.model_name value'''
def load_model(opt):
    if opt.model_name in ['srdense','dense']:
        model =  SRDenseNet(num_channels=1, growth_rate = opt.growth_rate, num_blocks = opt.num_blocks, num_layers=opt.num_layers, upscale_factor=opt.factor).to(opt.device)
    if opt.model_name in ['rrdbnet','RRDBNet','RRDBNET']:
        model =  RRDBNet(in_channels=1, out_channels = opt.out_channels,channels=opt.channels, growth_channels = opt.growth_channels,num_blocks = opt.num_blocks, upscale_factor = opt.factor, mode=opt.mode).to(opt.device)
    else:
        print(f'Model {opt.model_name} not implemented')
    return model

'''get the optimizer based on opt.criterion value'''
def get_criterion(name):
    if name in ['mse']:
        print("Using MSE Loss")
        criterion = nn.MSELoss()
    elif name in ['l1']:
        criterion = nn.L1Loss()
        print("Using L1 lOSS")
    elif name in ['SSIM','ssim']:
        print("Using SSIM loss")
        criterion  = SSIM()
    elif name in ['pyramid loss','laplacian loss']:
        print("Using Pyramid loss")
        criterion  = LaplacianPyramidLoss()
    elif name in ['derivative loss']:
        print("Using Derivative loss")
        criterion  = DerivativeLoss()
    else:
        print("Criterion not implemented")
        criterion = None
    return criterion



'''get the optimizer based on opt.optimizer value'''
def get_optimizer(opt,model):
    if opt.optimizer in ['adam']:
        optimizer = optim.Adam(model.parameters(), lr=opt.lr)
        print('Using ADAm Optimizer')
        return optimizer
    elif opt.optimizer in ['sgd']:
        print('Using SGD Optimizer')
        optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
        return optimizer
    else:
        print(f'optimizer type {opt.optimizer} not found')
        return None



 