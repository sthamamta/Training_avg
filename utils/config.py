

def set_outputs_dir(opt):
    opt.checkpoints_dir = 'outputs/checkpoints/{}/factor_{}/'.format(opt.exp_name,opt.factor)
    opt.epoch_images_dir ='outputs/epoch_images/{}/factor_{}/'.format(opt.exp_name,opt.factor)
    opt.input_images_dir ='outputs/input_images/{}/factor_{}/'.format(opt.exp_name,opt.factor)
    opt.output_images_dir ='outputs/output_images/{}/factor_{}/'.format(opt.exp_name, opt.factor)


# training metric paths
def set_training_metric_dir(opt):
    opt.loss_dir = 'outputs/losses/{}/factor_{}/'.format(opt.exp_name,opt.factor)
    opt.grad_norm_dir ='outputs/grad_norm/{}/factor_{}/'.format(opt.exp_name,opt.factor)


#plots path
def set_plots_dir(opt):    
    opt.plot_dir = 'outputs/plots/{}/factor_{}/'.format(opt.exp_name, opt.factor)

