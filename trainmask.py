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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l1_loss_mask,l2_loss,depth_loss_l1
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import numpy as np
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)

    
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

   # added by mwx 
    mem_log_path = os.path.join(scene.model_path, "memory_log.txt")
    # 初始化文件，写入表头 (使用 "w" 模式，仅在开始训练前清空一次)
    with open(mem_log_path, "w") as f:
        f.write("Iteration | Stage      | Allocated (MB) | Peak (MB)\n")
        f.write("-" * 60 + "\n")
    # ====================================================================

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()
        # added by mwx 
        check_mem = (iteration % 1000 == 0)
        if check_mem:
            torch.cuda.synchronize() # 确保之前的操作都已完成，读数更准
            torch.cuda.empty_cache() # 清理碎片
            torch.cuda.reset_peak_memory_stats() # 重置峰值统计
            
            # 记录初始状态
            alloc = torch.cuda.memory_allocated() / 1024**2
            with open(mem_log_path, "a") as f:
                f.write(f"\n[Iter {iteration}] Start Cycle\n")
                f.write(f"{iteration:<9} | {'Start':<10} | {alloc:.2f}\n")
        ################################################
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        ## added by mwx
        if check_mem:
            alloc = torch.cuda.memory_allocated() / 1024**2
            peak = torch.cuda.max_memory_allocated() / 1024**2
            with open(mem_log_path, "a") as f:
                f.write(f"{iteration:<9} | {'Render':<10} | {alloc:.2f} | {peak:.2f}\n")
        ################################################
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        # added by mwx 2026/01/16
        # ==================== Mask ====================
        # 1. 创建全 1 的 Mask
        hard_mask = torch.ones_like(image)

        # 2. 根据 Colmap ID 填 0 (你的自定义区域)
        if viewpoint_cam.colmap_id==1:
            hard_mask[:, 780:, 510:1150] = 0
        elif viewpoint_cam.colmap_id==3:
            hard_mask[:, 940:, 720:905] = 0

        # ===============================================

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        
        # added by mwx 2026/01/16
        # ===============================================

        # Ll1 = l1_loss_mask(image, gt_image,hard_mask)
        Ll1 = l2_loss(image,gt_image,hard_mask)
        # if FUSED_SSIM_AVAILABLE:
        #     ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        # else:
        #     ssim_value = ssim(image, gt_image)
        ssim_value = ssim(image, gt_image,hard_mask)
        # loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # ===============================================

        # Depth regularization
        # Ll1depth_pure = 0.0
        # if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
        #     invDepth = render_pkg["depth"]
        #     mono_invdepth = viewpoint_cam.invdepthmap.cuda()
        #     depth_mask = viewpoint_cam.depth_mask.cuda(
        #     Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
        #     Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
        #     loss += Ll1depth
        #     Ll1depth = Ll1depth.item()
        # else:
        #     Ll1depth = 0

        # added by mwx 2026/01/16
        # ====================  NPZ 读取 + Inverse 空间 Loss ====================
        # 1. 获取渲染出来的“逆深度” (from forward.cu)
        surf_inv_depth = render_pkg['depth']
        # 2. 构造 npz 路径
        base_name = os.path.splitext(viewpoint_cam.image_name)[0] 
        depth_path = os.path.join(dataset.source_path, "npz_depths", base_name + ".npz")
        # 3. 加载 GT (Metric 米制深度)
        # 使用 ["arr_0"] 读取 npz
        # gt_metric_depth = torch.from_numpy(np.load(depth_path)["arr_0"]).cuda()
        gt_depth_data = np.load(depth_path)
        gt_metric_depth = torch.from_numpy(gt_depth_data[gt_depth_data.files[0]].astype(np.float32)).cuda()
        # 维度检查：确保 GT 是 [1, H, W]
        if len(gt_metric_depth.shape) == 2:
            cc = gt_metric_depth.unsqueeze(0)
        
        # 4. 计算 L1 Loss 
        # 现在 Render 和 GT 都是正深度，直接求 Loss
        depth_loss = depth_loss_l1(pre_depth=surf_inv_depth,gtdepth=gt_metric_depth)
        # depth_loss_l1(surf_inv_depth, gt_metric_safe,final_depth_mask)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value) + depth_loss
        # depth_loss = depth_loss.item()
        # added by mwx
        if check_mem:
            alloc = torch.cuda.memory_allocated() / 1024**2
            with open(mem_log_path, "a") as f:
                f.write(f"{iteration:<9} | {'Loss Calc':<10} | {alloc:.2f}\n")
        ################################################
        
        loss.backward()
        # added by mwx
        if check_mem:
            alloc = torch.cuda.memory_allocated() / 1024**2
            peak = torch.cuda.max_memory_allocated() / 1024**2
            with open(mem_log_path, "a") as f:
                f.write(f"{iteration:<9} | {'Backward':<10} | {alloc:.2f} | {peak:.2f}\n")
        ################################################
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            # ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            # added by mwx 2026/01/16
            ema_depth_for_log = depth_loss
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
           
            # # Log and save
            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)

            # --- 修改 training 函数中的 Log 部分 ---
            # Log and save
            # if tb_writer:
            #     # 记录我们手动添加的那个 depth_loss
            #     tb_writer.add_scalar('train_loss_patches/explicit_depth_loss', depth_loss, iteration)
            #     # 记录原有的 Ll1depth (如果有)
            #     tb_writer.add_scalar('train_loss_patches/ema_depth_l1', ema_Ll1depth_for_log, iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss,depth_loss,iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)

                # modified by mwx 2026/01/16
                # if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                #     gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                if check_mem:
                    alloc = torch.cuda.memory_allocated() / 1024**2
                    with open(mem_log_path, "a") as f:
                        f.write(f"{iteration:<9} | {'Optimizer':<10} | {alloc:.2f}\n")
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

# added by mwx 2026/01/16
# --- 修改 training_report 函数 ---
def training_report(tb_writer, iteration, Ll1, loss, l1_loss ,depth_loss,elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        # 基础指标
        tb_writer.add_scalar('train_loss_patches/rgb_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/depth_loss',depth_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        # 记录当前学习率 (假设 gaussians 是在 renderArgs 之前或可以通过 scene 获取)
        for i, group in enumerate(scene.gaussians.optimizer.param_groups):
            tb_writer.add_scalar(f'learning_rate/{group["name"]}', group['lr'], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    # 获取渲染包
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    
                    if tb_writer and (idx < 5):
                        # 1. 记录 RGB 渲染图和真值
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                        
                        # 2. 记录深度图 (可视化)
                        if "depth" in render_pkg:
                            depth = render_pkg["depth"]
                            # 归一化深度图以便显示
                            depth_vis = depth / (depth.max() + 1e-5)
                            tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth_vis[None], global_step=iteration)

                        # 3. 记录法线图 (如果 2DGS 输出了 surf_normal)
                        if "surf_normal" in render_pkg:
                            norm = render_pkg["surf_normal"] * 0.5 + 0.5 # 映射到 [0, 1]
                            tb_writer.add_images(config['name'] + "_view_{}/normal".format(viewpoint.image_name), norm[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            # 属性分布直方图
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_histogram("scene/scaling_histogram", scene.gaussians.get_scaling, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
