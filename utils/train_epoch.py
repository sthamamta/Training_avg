
from tqdm import tqdm
import torch
import os

def update_epoch_losses(epoch_losses, count,values=[]):
    for (loss_meter,val) in zip(epoch_losses.keys(),values):
        epoch_losses[loss_meter].update(val,n=count)


def train_epoch_srdense(opt,model,criterions,optimizer,train_dataset,train_dataloader,epoch,epoch_losses): 
    with tqdm(total=(len(train_dataset) - len(train_dataset) % opt.train_batch_size), ncols=80) as t:
        t.set_description('epoch: {}/{}'.format(epoch, opt.num_epochs - 1))

        for idx, data in enumerate(train_dataloader):
            # "lr": lr_tensor, 
            #     "first_hr": first_hr_tensor,
            #     "second_hr":second_hr_tensor,
            #     "third_hr":third_hr_tensor
            images = data['lr'].to(opt.device)
            label_first = data['first_hr'].to(opt.device)
            label_second = data['second_hr'].to(opt.device)
            label_third = data['third_hr'].to(opt.device)
            first_out,second_out,third_out = model(images)
            labels = [label_first,label_second,label_third]
            outputs = [first_out,second_out,third_out]

            loss = 0
            for _,criterion in enumerate(criterions):
                for idx ,label in enumerate(labels):
                    # print("prediction shape",outputs[idx].shape)
                    loss_val = criterion(outputs[idx], label)
                    loss += loss_val
            
            update_epoch_losses(epoch_losses, count=len(images),values=[loss.item()])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        t.set_postfix(loss='{:.6f}'.format(epoch_losses['train_loss'].avg))
        t.update(len(images))

        if epoch % opt.n_freq==0:
            if not os.path.exists(opt.checkpoints_dir):
                os.makedirs(opt.checkpoints_dir)
            path = os.path.join(opt.checkpoints_dir, 'epoch_{}_f_{}.pth'.format(epoch,opt.factor))
            if opt.data_parallel:
                model.module.save(model.state_dict(),path,optimizer.state_dict(),epoch)
            else:
                model.save(model.state_dict(),path,optimizer.state_dict(),epoch)

    return images,label_second,second_out


def train_epoch_srdense_wo_avg(opt,model,criterions,optimizer,train_dataset,train_dataloader,epoch,epoch_losses): 
    print("Without avg training")
    with tqdm(total=(len(train_dataset) - len(train_dataset) % opt.train_batch_size), ncols=80) as t:
        t.set_description('epoch: {}/{}'.format(epoch, opt.num_epochs - 1))

        for idx, data in enumerate(train_dataloader):
            input = data['lr'].to(opt.device)
            label= data['hr'].to(opt.device)
            out = model(input)
        
            loss = 0
            for _,criterion in enumerate(criterions):
                loss_val = criterion(out, label)
                loss += loss_val
            
            update_epoch_losses(epoch_losses, count=len(input),values=[loss.item()])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        t.set_postfix(loss='{:.6f}'.format(epoch_losses['train_loss'].avg))
        t.update(len(input))

        if epoch % opt.n_freq==0:
            if not os.path.exists(opt.checkpoints_dir):
                os.makedirs(opt.checkpoints_dir)
            path = os.path.join(opt.checkpoints_dir, 'epoch_{}_f_{}.pth'.format(epoch,opt.factor))
            if opt.data_parallel:
                model.module.save(model.state_dict(),path,optimizer.state_dict(),epoch)
            else:
                model.save(model.state_dict(),path,optimizer.state_dict(),epoch)

    return input,label,out

