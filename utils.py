import os

import numpy as np
import torch
from torch import nn


def set_seed(seed_value, pytorch=True):
    """
    Set seed for deterministic behavior

    Parameters
    ----------
    seed_value : int
        Seed value.
    pytorch : bool
        Whether the torch seed should also be set. The default is True.

    Returns
    -------
    None.
    """
    import random
    random.seed(seed_value)
    np.random.seed(seed_value)
    if pytorch:
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True

class TrainableTensor(torch.nn.Module):
    def __init__(self, tensor: torch.Tensor):
        super().__init__()
        self.tensor = torch.nn.Parameter(tensor).to(tensor.device)

    def forward(self) -> torch.Tensor:
        return self.tensor


class NCC(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    Obtained from: https://github.com/cwmok/Conditional_LapIRN.git
    """
    def __init__(self, win=5, eps=1e-5):
        super(NCC, self).__init__()
        self.win = win
        self.eps = eps
        self.w_temp = win

    def forward(self, I, J):
        ndims = 3
        win_size = self.w_temp

        # set window size
        if self.win is None:
            self.win = [5] * ndims
        else:
            self.win = [self.w_temp] * ndims

        weight_win_size = self.w_temp
        weight = torch.ones((1, 1, weight_win_size, weight_win_size, weight_win_size), device=I.device, requires_grad=False)
        conv_fn = nn.functional.conv3d

        # compute CC squares
        I2 = I*I
        J2 = J*J
        IJ = I*J

        # compute filters
        # compute local sums via convolution
        I_sum = conv_fn(I, weight, padding=int(win_size/2))
        J_sum = conv_fn(J, weight, padding=int(win_size/2))
        I2_sum = conv_fn(I2, weight, padding=int(win_size/2))
        J2_sum = conv_fn(J2, weight, padding=int(win_size/2))
        IJ_sum = conv_fn(IJ, weight, padding=int(win_size/2))

        # compute cross correlation
        win_size = np.prod(self.win)
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross * cross / (I_var * J_var + self.eps)

        # return negative cc.
        return -1.0 * torch.mean(cc)

class multi_resolution_NCC(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    Obtained from: https://github.com/cwmok/Conditional_LapIRN.git
    """
    def  __init__(self, win=None, eps=1e-5, scale=3):
        super(multi_resolution_NCC, self).__init__()
        self.num_scale = scale
        # self.similarity_metric = NCC(win=win)

        self.similarity_metric = []

        for i in range(scale):
            self.similarity_metric.append(NCC(win=win - (i*2)))
            # self.similarity_metric.append(Normalized_Gradient_Field(eps=0.01))

    def forward(self, I, J):
        total_NCC = []
        # scale_I = I
        # scale_J = J
        #
        # for i in range(self.num_scale):
        #     current_NCC = similarity_metric(scale_I,scale_J)
        #     # print("Scale ", i, ": ", current_NCC, (2**i))
        #     total_NCC += current_NCC/(2**i)
        #     # print(scale_I.size(), scale_J.size())
        #     # print(current_NCC)
        #     scale_I = nn.functional.interpolate(I, scale_factor=(1.0/(2**(i+1))))
        #     scale_J = nn.functional.interpolate(J, scale_factor=(1.0/(2**(i+1))))

        for i in range(self.num_scale):
            current_NCC = self.similarity_metric[i](I, J)
            total_NCC.append(current_NCC/(2**i))
            # print(scale_I.size(), scale_J.size())

            I = nn.functional.avg_pool3d(I, kernel_size=3, stride=2, padding=1, count_include_pad=False)
            J = nn.functional.avg_pool3d(J, kernel_size=3, stride=2, padding=1, count_include_pad=False)

        return sum(total_NCC)


def save_data(savedir, batch_idx, moving, fixed, warped, dvf, category):
    np.save(os.path.join(savedir, 'batch_' + str(batch_idx) + '__' + category + '_mov.npy'),
            moving.detach().squeeze().cpu().numpy())
    np.save(os.path.join(savedir, 'batch_' + str(batch_idx) + '__' + category + '_fix.npy'),
            fixed.detach().squeeze().cpu().numpy())
    np.save(os.path.join(savedir, 'batch_' + str(batch_idx) + '__' + category + '_warped.npy'),
            warped.detach().squeeze().cpu().numpy())
    np.save(os.path.join(savedir, 'batch_' + str(batch_idx) + '__' + category + '_dvf.npy'),
            dvf.detach().squeeze().cpu().numpy())

def read_data(savedir, batch_idx, category):
    moving = torch.from_numpy(
        np.load(os.path.join(savedir, 'batch_' + str(batch_idx) + '__' + category + '_mov.npy'))).unsqueeze(
        0).unsqueeze(0)
    fixed = torch.from_numpy(
        np.load(os.path.join(savedir, 'batch_' + str(batch_idx) + '__' + category + '_fix.npy'))).unsqueeze(
        0).unsqueeze(0)
    warped = torch.from_numpy(
        np.load(os.path.join(savedir, 'batch_' + str(batch_idx) + '__' + category + '_warped.npy'))).unsqueeze(
        0).unsqueeze(0)
    dvf = torch.from_numpy(
        np.load(os.path.join(savedir, 'batch_' + str(batch_idx) + '__' + category + '_dvf.npy'))).unsqueeze(
        0).unsqueeze(0)
    return moving, fixed, warped, dvf


def acs_slice(img, img_slice, ax_cor_sag):
    """
    :param img:         A 3D image (Numpy array)
    :param img_slice:   The image slice at which to take the 2D slice from
    :param ax_cor_sag:  Axial, coronal or sagittal slice

    :return:            The 2D image slice
    """
    if img == []:
        img_2d = []
    else:
        assert ax_cor_sag in ['ax', 'cor', 'sag'], \
            "ax_cor_sag should be either one of 'ax', 'cor' or 'sag' to get the axial, coronal or sagitall view"

        img_2d = None
        if ax_cor_sag == 'sag':
            img_2d = img[img_slice, :, :]
        if ax_cor_sag == 'cor':
            img_2d = img[:, img_slice, :]
        if ax_cor_sag == 'ax':
            img_2d = img[:, :, img_slice]
    return img_2d

def to_numpy(im):
    return im.squeeze().cpu().numpy()
