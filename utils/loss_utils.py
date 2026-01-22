#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
try:
    from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward
except:
    pass

C1 = 0.01 ** 2
C2 = 0.03 ** 2

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2):
        ssim_map = fusedssim(C1, C2, img1, img2)
        ctx.save_for_backward(img1.detach(), img2)
        ctx.C1 = C1
        ctx.C2 = C2
        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2 = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        return None, None, grad, None

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l1_loss_mask(network_output, gt, mask = 1):
    return torch.abs((network_output - gt) * mask).mean()

# def l2_loss(network_output, gt):
#     return ((network_output - gt) ** 2).mean()

# added by mwx #################################
def l2_loss(network_output, gt,mask=1):
    return (((network_output - gt)*mask) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

# def ssim(img1, img2, window_size=11, size_average=True):
#     channel = img1.size(-3)
#     window = create_window(window_size, channel)

#     if img1.is_cuda:
#         window = window.cuda(img1.get_device())
#     window = window.type_as(img1)

#     return _ssim(img1, img2, window, window_size, channel, size_average)
# added by mwx #################################

def ssim(img1, img2, mask=1, window_size=11,size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average,mask)
	
# def _ssim(img1, img2, window, window_size, channel, size_average=True):
#     mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
#     mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

#     mu1_sq = mu1.pow(2)
#     mu2_sq = mu2.pow(2)
#     mu1_mu2 = mu1 * mu2

#     sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
#     sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
#     sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

#     C1 = 0.01 ** 2
#     C2 = 0.03 ** 2

#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

#     if size_average:
#         return ssim_map.mean()
#     else:
#         return ssim_map.mean(1).mean(1).mean(1)
def _ssim(img1, img2, window, window_size, channel, size_average=True, mask=1):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim_map = ssim_map*mask
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
def depth_loss_mse(pre_depth,gtdepth):
    pre_depth = torch.squeeze(pre_depth,0)
    mask1 = gtdepth>0
    #在大高斯处增加loss，弱纹理处也会产生稠密点云，使用稠密点云生成的深度图一定程度将原始分辨率的点云带入，降低低分辨率点云的影响
    return ((pre_depth-gtdepth)[mask1]**2).mean()

def fast_ssim(img1, img2):
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    return ssim_map.mean()
# added by mwx #################################
def depth_loss_l1(pre_depth,gtdepth):
    pre_depth = torch.squeeze(pre_depth,0)
    mask1 = gtdepth>0
    #在大高斯处增加loss，弱纹理处也会产生稠密点云，使用稠密点云生成的深度图一定程度将原始分辨率的点云带入，降低低分辨率点云的影响
    return (torch.abs((pre_depth-gtdepth)[mask1])).mean()

# added by mwx #################################
# ssim 加权金字塔
import torch.nn.functional as F

def pyramid_ssim_loss(img, gt, mask, levels=3, weight_list=None):
    """
    计算金字塔 SSIM Loss
    Args:
        img: 渲染图像 (B, C, H, W)
        gt: 真值图像 (B, C, H, W)
        mask: 掩膜 (B, 1, H, W)
        levels: 金字塔层数 (默认3层: 原图, 1/2, 1/4)
        weight_list: 每层的权重列表
    Returns:
        total_loss: 加权后的 SSIM Loss
    """
    if weight_list is None:
        # 默认权重：越底层的分辨率（大结构）权重稍微大一点，或者平均
        # 也可以参考 MS-SSIM 的经典权重，这里简单起见设为平均或递减
        # 例如: [1.0, 1.0, 1.0] 表示每层都同等重要
        weight_list = [1.0] * levels
    
    total_loss = 0.0
    total_weight = 0.0
    
    current_img = img
    current_gt = gt
    current_mask = mask
    
    for i in range(levels):
        # 1. 计算当前层的 SSIM Loss (1 - SSIM)
        ssim_val = ssim(current_img, current_gt, current_mask)
        loss_level = 1.0 - ssim_val
        
        total_loss += loss_level * weight_list[i]
        total_weight += weight_list[i]
        
        # 2. 如果不是最后一层，进行下采样
        if i < levels - 1:
            # 使用平均池化进行下采样 (抗锯齿效果比最近邻好)
            current_img = F.avg_pool2d(current_img, kernel_size=2, stride=2)
            current_gt = F.avg_pool2d(current_gt, kernel_size=2, stride=2)
            
            # Mask 下采样后需要重新二值化 (保持 Hard Mask)
            # 或者将其视为 Soft Mask (权重)，推荐二值化以保持逻辑一致
            current_mask = F.avg_pool2d(current_mask.float(), kernel_size=2, stride=2)
            current_mask = (current_mask > 0.5).float() # 阈值化
            
    return total_loss / total_weight