# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
import os
import torch
from random import randint
# [Modified] 引入必要的损失函数
from utils.loss_utils import  depth_loss_l1, pyramid_ssim_loss,l1_loss,ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import numpy as np
import random
# modified by mwx - 2026-03-09: 引入 SPA 优化器
from optimizing_spa import OptimizingSpa
##############################################
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



def preload_all_depths(scene, depth_root):
    depth_dict = {}
    train_cams = scene.getTrainCameras()
    for cam in tqdm(train_cams, desc="Loading Depth to RAM"):
        pure_name = os.path.splitext(cam.image_name)[0]
        path = os.path.join(depth_root, pure_name + ".npz")
        
        if os.path.exists(path):
            try:
                # 尝试不同的键名
                data = np.load(path)
                if 'depth' in data.files:
                    d = data['depth']
                elif 'arr_0' in data.files:
                    d = data['arr_0']
                elif len(data.files) > 0:
                    # 如果有其他键名，使用第一个
                    key = data.files[0]
                    d = data[key]
                else:
                    depth_dict[cam.image_name] = None
                    continue
                
                depth_dict[cam.image_name] = torch.from_numpy(d).half()
            except Exception as e:
                depth_dict[cam.image_name] = None
        else:
            depth_dict[cam.image_name] = None
    return depth_dict

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"试图使用 sparse adam 但未检测到安装。")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    
    # [修改点 3] 预备全局 LiDAR 几何基准
    global_lidar_xyz = gaussians.get_xyz.detach().clone().float().cuda()

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    # modified by mwx - 2026-03-09: 初始化优化器
    optimizing_spa = None
    ##########################################
    
    # bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    viewpoint_stack = scene.getTrainCameras().copy()
    ema_loss_for_log = 0.0
    ema_depth_loss_for_log = 0.0

    depth_gt_root = os.path.join(dataset.source_path, "depth/npz_depths") 
    depth_memory_cache = preload_all_depths(scene, depth_gt_root)
    
    # 检查有多少深度文件成功加载
    loaded_count = sum(1 for v in depth_memory_cache.values() if v is not None)
    
    lambda_depth = 0.5

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    if opt.pyramid:
        print("[INFO] 已启用 pyramid SSIM loss")
    else:
        print("[INFO] 未启用 pyramid，当前使用普通 SSIM loss")
        
    for iteration in range(first_iter, opt.iterations + 1):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
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
        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(random.randint(0, len(viewpoint_stack) - 1))

        bg = background

        # 1. 前向渲染
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        rendered_depth = render_pkg["depth"]
        pixels = render_pkg["pixels"].reshape(-1)

 
        # 2. RGB Loss
        gt_image = viewpoint_cam.original_image.cuda()

        Ll1 = l1_loss(image, gt_image)
        # modified by mwx - 2026-03-26
        # pyramid ssim loss - 2026-03-26: 可选的金字塔 SSIM 损失
        if opt.pyramid:
            pyramid_weights = [1.0, 1.0, 1.0]
            ssim_term = pyramid_ssim_loss(
                image,
                gt_image,
                levels=3,
                weight_list=pyramid_weights
            )
        else:
            ssim_term = 1.0 - ssim(image, gt_image)

        loss_rgb = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_term
        #########################################################

        # 3. 2D 深度损失 (辅助) - 只使用图像下半部1/3的深度
        loss_depth = torch.tensor(0.0).cuda()
        gt_depth_raw = depth_memory_cache.get(viewpoint_cam.image_name)

        if gt_depth_raw is not None:
            gt_depth = gt_depth_raw.cuda().float()
            
            # 统一深度图分辨率
            if rendered_depth.shape[1:] != gt_depth.shape[-2:]:
                if gt_depth.dim() == 2:
                    gt_depth_resized = gt_depth.unsqueeze(0)
                else:
                    gt_depth_resized = gt_depth
                
                gt_depth_resized = torch.nn.functional.interpolate(
                    gt_depth_resized.unsqueeze(0),
                    size=rendered_depth.shape[1:], 
                    mode='nearest'
                ).squeeze(0)
            else:
                if gt_depth.dim() == 2:
                    gt_depth_resized = gt_depth.unsqueeze(0)
                else:
                    gt_depth_resized = gt_depth
            
            # 创建图像下半部1/3的掩码 - 将上面2/3设为0，保留下面1/3
            height = gt_depth_resized.shape[1]
            bottom_third_start = int(height * 2 / 3)  # 上面2/3的结束位置，即下面1/3的开始位置
            
            # 创建掩码：下面1/3为True，上面2/3为False
            mask = torch.zeros_like(gt_depth_resized, dtype=torch.bool)
            mask[:, bottom_third_start:, :] = True  # 下面1/3区域设为True
            
            # 应用掩码，将上面2/3的深度值设为0
            processed_gt_depth = gt_depth_resized * mask.float()
            
            # 只在有有效深度值的位置计算深度损失
            valid_mask = (processed_gt_depth > 0.1) & mask  # 结合深度有效性检查和区域掩码
            
            if valid_mask.sum() > 0:
                # 获取有效区域的渲染深度和真实深度
                valid_rendered_depth = rendered_depth * valid_mask.float()
                valid_gt_depth = processed_gt_depth * valid_mask.float()
                
                # 计算深度损失，只在有深度值的位置计算
                loss_depth = depth_loss_l1(valid_rendered_depth, valid_gt_depth)

        # 5. 总 Loss 合并
        depths_loss = lambda_depth * loss_depth
        # total_loss = loss_rgb + depths_loss 

        # ===== GaussianSpa: 后半程开始 SPA loss ===== modified by mwx - 2026-03-09
        if getattr(opt, "optimizing_spa", False) and iteration == opt.optimizing_spa_start_iter:
            optimizing_spa = OptimizingSpa(gaussians, opt, device="cuda")

        total_loss = loss_rgb + depths_loss

        if (
            optimizing_spa is not None
            and opt.optimizing_spa_start_iter <= iteration < opt.optimizing_spa_stop_iter
        ):
            optimizing_spa.adjust_rho(iteration, opt.iterations)
            total_loss = optimizing_spa.append_spa_loss(total_loss)
        ###############################################################################

        total_loss.backward()
        if iteration == first_iter:
            g = viewspace_point_tensor.grad
            print("viewspace_points:", tuple(viewspace_point_tensor.shape),
                "grad:", None if g is None else tuple(g.shape),
                "absgrad_sum:", 0.0 if g is None else float(g[:, 2:].abs().sum().item()))                                              
        iter_end.record()

        # 6. 统计与稠密化 (完整补全)
        with torch.no_grad():
            ema_loss_for_log = 0.4 * total_loss.item() + 0.6 * ema_loss_for_log
            ema_depth_loss_for_log = 0.4 * depths_loss.item() + 0.6 * ema_depth_loss_for_log
            
            if iteration % 10 == 0:
                progress_bar.set_postfix({"L": f"{ema_loss_for_log:.4f}", "Depth": f"{ema_depth_loss_for_log:.4f}", "Pts": gaussians.get_xyz.shape[0]})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            training_report(tb_writer, iteration, Ll1, total_loss, l1_loss, depths_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)

            if (iteration in saving_iterations):
                scene.save(iteration)

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
                # modified by mwx - 2026-03-09: SPA 优化器步进
                # ===== GaussianSpa: 周期性更新 z / u =====
                if (
                    optimizing_spa is not None
                    and opt.optimizing_spa_start_iter <= iteration < opt.optimizing_spa_stop_iter
                    and iteration % opt.optimizing_spa_interval == 0
                ):
                    optimizing_spa.update()

                # ===== GaussianSpa: 在 stop_iter 做一次硬 prune =====
                if optimizing_spa is not None and iteration == opt.optimizing_spa_stop_iter:
                    prune_mask = optimizing_spa.build_prune_mask(
                        ratio=opt.prune_ratio2,
                        min_opacity=None,
                        use_z=True
                    )
                    gaussians.prune_points(prune_mask)
                    print(f"[GaussianSpa] hard prune at iter {iteration}, remain points: {gaussians.get_xyz.shape[0]}")
                ###################################################
                
            if (iteration in checkpoint_iterations):
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            # 稠密化与修建 (3DGS 核心逻辑)
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, pixels)#pixel修改为之

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.densify_grad_abs_threshold,0.005, scene.cameras_extent, size_threshold, radii)
                    print("N xyz:", gaussians.get_xyz.shape[0],
                            "acc:", gaussians.xyz_gradient_accum.shape[0],
                            "acc_abs:", gaussians.xyz_gradient_accum_abs.shape[0],
                            "denom:", gaussians.denom.shape[0],
                            "max_r2d:", gaussians.max_radii2D.shape[0])

                # if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                #     gaussians.reset_opacity()

            # 优化器步进



def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    print("输出文件夹 (Output folder): {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("未找到 Tensorboard: 将不记录训练曲线")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss_fn, depths_loss,  elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/depth_loss', depths_loss.item() if isinstance(depths_loss, torch.Tensor) else depths_loss, iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            bytes_to_mb = 1024.0 ** 2
            tb_writer.add_scalar('gpu_memory/max_allocated_mb', torch.cuda.max_memory_allocated(device) / bytes_to_mb, iteration)
        
        for i, group in enumerate(scene.gaussians.optimizer.param_groups):
            tb_writer.add_scalar(f'learning_rate/{group["name"]}', group['lr'], iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                        
                        if "depth" in render_pkg:
                            pred_depth = render_pkg["depth"]
                            depth_vis = pred_depth / (pred_depth.max() + 1e-5)
                            tb_writer.add_images(config['name'] + "_view_{}/depth_render".format(viewpoint.image_name), depth_vis[None], global_step=iteration)
                        
                    l1_test += l1_loss_fn(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] 评估 {}: L1 {:.4f} PSNR {:.4f}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[2_000, 7_000, 15_000,20_000,30_000,40_000,50_000,60_000,70_000,75_000,90_000,100_000,120_000,140_000,150_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 100_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("正在优化 (Optimizing): " + args.model_path)

    safe_state(args.quiet)

    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    print("\n训练完成 (Training complete).")
