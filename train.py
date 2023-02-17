# process_yaml.py file
#imports
import yaml
import argparse
import sys
from utils.train_utils import load_dataset_wo_avg,load_model_wo_avg,get_criterion,get_optimizer, load_dataset_wo_avg_full
import torch
import torch.nn as nn
from utils.image_quality_assessment import PSNR,SSIM
import copy
from utils.logging_metric import LogMetric,create_loss_meters_srdense
from utils.train_utils import adjust_learning_rate
from utils.train_epoch import train_epoch_srdense_wo_avg
from utils.general import save_configuration_yaml,LogOutputs
from utils.config import set_outputs_dir,set_training_metric_dir
import os
import wandb
os.environ["CUDA_VISIBLE_DEVICES"]='6,7'


def train(opt,model,criterions,optimizer,train_datasets,train_dataloader,eval_dataloader=None,wandb=None):

    if opt.wandb:
        log_table_output = LogOutputs()

    # best_weights = copy.deepcopy(model.state_dict())
    # best_epoch = 0
    # best_psnr = 0.0
    
    for epoch in range(opt.num_epochs):
        '''reduce learning rate by factor 0.5 on every 150 or 225 epoch'''
        opt.lr = adjust_learning_rate(optimizer, epoch,opt.lr,opt.lr_decay_factor)

        '''setting model in train mode'''
        model.train()

        '''train one epoch and evaluate the model'''
    
        epoch_losses = create_loss_meters_srdense()  #create a dictionary
        images,labels,preds = train_epoch_srdense_wo_avg(opt,model,criterions,optimizer,train_datasets,train_dataloader,epoch,epoch_losses)
        # eval_loss, eval_l1,eval_psnr, eval_ssim,eval_hfen = validate_srdense(opt,model, eval_dataloader,criterion,addition=opt.addition)
        
        # apply_model_using_cv(model,epoch,opt,addition= opt.addition)

        if opt.wandb:
            wandb.log({
            # "val/val_loss" : eval_loss,
            # "val/val_l1_error":eval_l1,
            # "val/val_psnr": eval_psnr,
            # "val/val_ssim":eval_ssim,
            # "val/val_hfen":eval_hfen,
            "epoch": epoch,
            })
            for key in epoch_losses.keys():
                wandb.log({"train/{}".format(key) : epoch_losses[key].avg,
                })
            wandb.log({"other/learning_rate": opt.lr})
            
            if epoch % opt.n_freq == 0:
                print('images shape',images.shape)
                log_table_output.append_list(epoch,images,labels,preds)  #create a class with list and function to loop through list and add to log table

        # print('eval psnr: {:.4f}'.format(eval_psnr))

        # if eval_psnr > best_psnr:
        #     best_epoch = epoch
        #     best_psnr = eval_psnr
        #     best_weights = copy.deepcopy(model.state_dict())

        # '''adding to the dictionary'''
        # metric_dict.update_dict([eval_loss,eval_l1,eval_psnr,eval_ssim,eval_hfen],training=False)

        
        metric_dict.update_dict([epoch_losses['train_loss'].avg])  

    if opt.wandb:
        log_table_output.log_images(columns = ["epoch","image", "pred", "label"],wandb=wandb)

    path = metric_dict.save_dict(opt)
    _ = save_configuration_yaml(opt)

 
    # path="best_weights_factor_{}_epoch_{}.pth".format(opt.factor,best_epoch)
    # path = os.path.join(opt.checkpoints_dir, path)
    # if opt.data_parallel:
    #     model.module.save(best_weights,opt,path,optimizer.state_dict(),best_epoch)
    # else:
    #     model.save(best_weights,opt,path,optimizer.state_dict(),best_epoch)


    print('model saved')


if __name__ == "__main__":
    '''get the configuration file'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="configuration file *.yml", type=str, required=False, 
    default='configuration/srdense_axis_x/srdensenet_full1.yaml')
    sys.argv = ['-f']
    opt   = parser.parse_known_args()[0]

    '''load the configuration file and append to current parser arguments'''
    ydict = yaml.load(open(opt.config), Loader=yaml.FullLoader)
    for k,v in ydict.items():
        if k=='config':
            pass
        else:
            parser.add_argument('--'+str(k), required=False, default=v)
    opt  = parser.parse_args()

    '''adding seed for reproducibility'''
    # torch.manual_seed(opt.seed)

    '''set device'''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.device = device

    '''load dataset (loading dataset based on dataset name and factor on arguments)'''
    if opt.train_full:
        print("LOading full dataset")
        train_dataloader,train_datasets = load_dataset_wo_avg_full(opt=opt,load_eval=False)
    else:
        train_dataloader,train_datasets = load_dataset_wo_avg(opt=opt,load_eval=False) 

    '''load model'''
    model = load_model_wo_avg(opt)

    '''print model'''
    print(model)

    '''setup the outputs and logging metric dirs on '''
    set_outputs_dir(opt) 
    set_training_metric_dir(opt) 



    '''wrap model for data parallelism'''
    num_of_gpus = torch.cuda.device_count()
    print("Number of GPU available", num_of_gpus)
    if num_of_gpus>1:
        model = nn.DataParallel(model,device_ids=[*range(num_of_gpus)])
        opt.data_parallel = True
        print("Multiple GPU Training")
    else:
        opt.data_parallel=False

    '''setup loss and optimizer '''
   
    optimizer = get_optimizer(opt,model)
    criterions = []
    for criteria in opt.criterions:
        criteria  = get_criterion(criteria)
        criterions.append(criteria)

    print('training for factor ',opt.factor)
    # print(model)

    '''initialize the logging dictionary'''
    metric_dict = LogMetric( { 'train_loss' : [],'epoch':[]})

    #setting metric for evaluation
    psnr = PSNR()
    ssim = SSIM()
    opt.psnr = psnr.to(device=opt.device, memory_format=torch.channels_last, non_blocking=True)
    opt.ssim = ssim.to(device=opt.device, memory_format=torch.channels_last, non_blocking=True)

    # quit();

    if opt.wandb:
        wandb.init(
        project=opt.project_name,
                name = opt.exp_name,
                config = opt )

        wandb.watch(model,log="all",log_freq=1)

    else:
        wandb=None

    # print(opt)
    # quit();

    '''training the model'''
    train(opt=opt,model=model,criterions=criterions,optimizer=optimizer,train_datasets=train_datasets,train_dataloader=train_dataloader,wandb = wandb)

    if opt.wandb:
        wandb.unwatch(model)
        wandb.finish()