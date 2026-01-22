from typing import NamedTuple
import torch.nn as nn
import torch
from fused_ssim_cuda import fusedssim, fusedssim_backward

allowed_padding = ["same", "valid"]

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2, padding="same", train=True):
        ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = fusedssim(C1, C2, img1, img2, train)

        if padding == "valid":
            ssim_map = ssim_map[:, :, 5:-5, 5:-5]

        ctx.save_for_backward(img1.detach(), img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
        ctx.C1 = C1
        ctx.C2 = C2
        ctx.padding = padding

        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = ctx.saved_tensors
        C1, C2, padding = ctx.C1, ctx.C2, ctx.padding
        dL_dmap = opt_grad
        if padding == "valid":
            dL_dmap = torch.zeros_like(img1)
            dL_dmap[:, :, 5:-5, 5:-5] = opt_grad
        grad = fusedssim_backward(C1, C2, img1, img2, dL_dmap, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
        return None, None, grad, None, None, None

# def fused_ssim(img1, img2, padding="same", train=True):
#     C1 = 0.01 ** 2
#     C2 = 0.03 ** 2

#     assert padding in allowed_padding

#     map = FusedSSIMMap.apply(C1, C2, img1, img2, padding, train)
#     return map.mean()
def fused_ssim(img1, img2, padding="same", train=True, mask=None):
    """
    Args:
        img1: (B, C, H, W)
        img2: (B, C, H, W)
        padding: "same" or "valid"
        train: bool
        mask: (Optional) (B, 1, H, W) or matching broadcastable shape. 
              Values should be 0 or 1 (float or bool).
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    assert padding in allowed_padding

    # 1. 计算完整的 SSIM Map
    # map shape: [B, C, H, W]
    map = FusedSSIMMap.apply(C1, C2, img1, img2, padding, train)
    
    # 2. 如果没有 mask，直接返回均值 (原逻辑)
    if mask is None:
        return map.mean()
    
    # 3. 如果有 mask，进行处理
    else:
        # 确保 mask 是 float 类型
        if mask.dtype != map.dtype:
            mask = mask.to(map.dtype)

        # 处理 padding="valid" 的情况
        # 因为 map 在 forward 里被切除了边缘 (5:-5)，Mask 也必须做同样的切除才能对齐
        if padding == "valid":
            # 假设 mask 是 [B, 1, H, W] 或 [B, C, H, W]
            if mask.shape[-1] == img1.shape[-1]: # 只有当 mask 还是原始大小时才切
                mask = mask[:, :, 5:-5, 5:-5]

        # 4. 计算带 Mask 的均值
        # SSIM = (Map * Mask) 的总和 / (Mask 的有效像素数 * 通道数)
        # 假设 mask 在 batch 和 channel 维度可以广播 (例如 mask 是 [1, 1, H, W])
        
        masked_map = map * mask
        
        # 计算分母：有效元素的个数
        # map.shape[1] 是通道数 (C)，因为 mask 通常是单通道的 [1, 1, H, W]，
        # 而 SSIM 是对每个通道分别计算的，所以有效像素总数需要乘以通道数
        normalization = mask.sum() * map.shape[1] 
        
        # 防止除以 0 (虽然训练中不太可能全 0)
        if normalization == 0:
            return torch.tensor(0.0, device=map.device, requires_grad=True)
        
        return masked_map.sum() / normalization