import torch

class DerivativeLoss(torch.nn.Module):
    def __init__(self, weight = 1.0):
        super(DerivativeLoss, self).__init__()
        self.weight = weight
    
    def forward(self, lr_tensor, hr_tensor):
        bs_img, c_img, h_img, w_img = lr_tensor.shape

        diff_x_hr = hr_tensor[:,:,:,1:]-hr_tensor[:,:,:,:-1]
        diff_x_lr = lr_tensor[:,:,:,1:]-lr_tensor[:,:,:,:-1]

        diff_y_hr = hr_tensor[:,:,1:,:]-hr_tensor[:,:,:-1,:]
        diff_y_lr = lr_tensor[:,:,1:,:]-lr_tensor[:,:,:-1,:]

        diff_x = torch.pow(diff_x_hr-diff_x_lr, 2).sum()
        diff_y = torch.pow(diff_y_hr-diff_y_lr, 2).sum()

        return self.weight*(diff_x+diff_y)/(bs_img*c_img*h_img*w_img)
     
