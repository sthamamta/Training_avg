# functions for loading the checkpoints

import torch
from model.densenet import SRDenseNet, SRDenseNetWOAVG
from model.rrdbnet import RRDBNet, RRDBNetWOAVG

def load_checkpoint(checkpoint_path,device, model_name):
    if model_name in ['srdense','densenet', 'srdensenet']:
        model = load_dense_net(checkpoint_path=checkpoint_path,device=device)
    elif model_name in ['rrdbnet', 'RRDBNet','RRDBNET']:
        model =load_rrdbnet(checkpoint_path,device)

    return model
    

def load_dense_net(checkpoint_path,device):
    checkpoint = torch.load(checkpoint_path,map_location=torch.device(device))

    num_channels = checkpoint['num_channels']
    growth_rate = checkpoint['growth_rate']
    num_blocks = checkpoint['num_blocks']
    num_layers =checkpoint['num_layers']
    upscale_factor = checkpoint["upscale_factor"]
    mode = checkpoint["mode"]
    model = SRDenseNetWOAVG(num_channels=num_channels, growth_rate=growth_rate, num_blocks=num_blocks, num_layers=num_layers, upscale_factor=upscale_factor, mode = mode)
    state_dict = model.state_dict()
    for n, p in checkpoint['model_state_dict'].items():
        new_key = n[7:]
        # new_key = n
        if new_key in state_dict.keys():
            state_dict[new_key].copy_(p)
        elif n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)
    return model

def load_rrdbnet(checkpoint_path,device):
    checkpoint = torch.load(checkpoint_path,map_location=torch.device(device))
    in_channels = checkpoint['in_channels']
    out_channels = checkpoint['out_channels']
    channels = checkpoint['channels']
    growth_channels =checkpoint['growth_channels']
    num_blocks = checkpoint["num_blocks"]
    mode = checkpoint["mode"]
    upscale_factor = checkpoint["upscale_factor"]
    model = RRDBNetWOAVG(in_channels=in_channels, out_channels=out_channels, channels=channels, growth_channels=growth_channels,num_blocks=num_blocks, upscale_factor=upscale_factor, mode = mode)
    state_dict = model.state_dict()
    for n, p in checkpoint['model_state_dict'].items():
        new_key = n[7:]
        if new_key in state_dict.keys():
            state_dict[new_key].copy_(p)
        elif n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)
    return model

