from typing import Tuple, Callable, Optional
from time import time
from numbers import Number

import numpy as np
from torch import optim, Tensor
import mrphy_rf
from mrphy_rf.mobjs import SpinCube, Pulse
import torch
import torch.nn.functional as F
import numpy as np
import random

from adpulses_rf.penalties import pen_l2_mat, pen_inf_mat, rf_smooth, crush_stop, pen_traj
from adpulses_rf.metrics import err_stdmxy

from scipy.io import savemat
import os

import math

def arctanLBFGS(
        target: dict, cube: SpinCube, pulse: Pulse,
        fn_err: Callable[[Tensor, Tensor, Optional[Tensor]], Tensor],
        fn_pen: Callable[[Tensor], Tensor],
        niter: int = 1, niter_gr: int = 1, niter_rf: int = 1,
        eta: Number = 4., b1Map_: Optional[Tensor] = None, doRelax: bool = True,
        train_rf: bool = True, train_nr: bool = True,
        lr_rf: float = 0.001, lr_nr: float = 0.001,
        save_mag: bool = True, save_path: str = '',
        shim_path: str = '', if_so_spatial: bool = True,
        if_so_spectral: bool = False, op_fat: bool = False,
        res: float = 0.1, lambda_loss: float = 1.0,
        lambda_pengr: float = 0.0, lambda_edge: float = 0.0,
        adaptive_weights: bool = False, weight_freq: int = 1, 
        weights_blur: bool = False, slice_weight: bool = False, 
        traj_con: bool = False, traj_type: str = 'z', 
        lambda_traj: float = 0.1, save_hist: str = '') -> Tuple[Pulse, dict]:
    r"""Joint RF/GR optimization via direct arctan trick

    Usage:
        ``arctanLBFGS(target, cube, pulse, fn_err, fn_pen; eta=eta)``

    Inputs:
        - ``target``: dict, with fields:
            ``d_``: `(1, nM, xy)`, desired excitation;
            ``weight_``: `(1, nM)`.
        - ``cube``: mrphy_rf.mobjs.SpinCube.
        - ``pulse``: mrphy_rf.mobjs.Pulse.
        - ``fn_err``: error metric function. See :mod:`~adpulses.metrics`.
        - ``fn_pen``: penalty function. See :mod:`~adpulses.penalties`.
    Optionals:
        - ``niter``: int, number of iterations.
        - ``niter_gr``: int, number of LBFGS iters for updating *gradients*.
        - ``niter_rf``: int, number of LBFGS iters for updating *RF*.
        - ``eta``: `(1,)`, penalization term weighting coefficient.
        - ``b1Map_``: `(1, nM, xy,(nCoils))`, a.u., transmit sensitivity.
        - ``doRelax``: [T/f], whether accounting relaxation effects in simu.
        - ``train_rf``: optimize rf or not
        - ``train_nr``: optimize shim current ot not
        - ``save_mag``: save magentization after applying optimized rf and shim
        - ``save_path``: save path
        - ``shim_path``: path of shim fields
        - ``if_so_spatial``: whether use stochastic offset strategy (spatial)
        - ``op_fat``: whether including fat into optimization. (uniform excitation)
    Outputs:
        - ``pulse``: mrphy_rf.mojbs.Pulse, optimized pulse.
        - ``optInfos``: dict, optimization informations.
    """
    
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Set up: Interior mapping
    tρ, θ = mrphy_rf.utils.rf2tρθ(pulse.rf, pulse.rfmax)
    tsl_nr = mrphy_rf.utils.s2ts(mrphy_rf.utils.g2s(pulse.nr, pulse.dt), pulse.smax)  #This is used for lineara gradient. g2s computes the slew rate. s2ts computes tan() to avoid contraints
    tsl_lr = mrphy_rf.utils.s2ts(mrphy_rf.utils.g2s(pulse.lr, pulse.dt), pulse.smax)

    # Set up optimizers for RF and shim currents (after interior mapping).

    if train_rf:
        opt_rf = optim.LBFGS([tρ, θ], lr=lr_rf, max_iter=10, history_size=100,
                         tolerance_change=1e-4,
                         line_search_fn='strong_wolfe') 

        scheduler_rf = optim.lr_scheduler.MultiStepLR(opt_rf, milestones=[150, 170], gamma=0.1)
        tρ.requires_grad = θ.requires_grad = True
    if train_nr:
        opt_sl = optim.LBFGS([tsl_nr], lr=lr_nr, max_iter=20, history_size=100, # 1
                         tolerance_change=1e-6,
                         line_search_fn='strong_wolfe')
        scheduler_gr = optim.lr_scheduler.MultiStepLR(opt_sl, milestones=[150, 170], gamma=0.1)               
        tsl_nr.requires_grad = True

    # Weights for loss
    loss_hist = np.full((niter*(niter_gr+niter_rf),), np.nan)
    loss_mag = np.full((niter*(niter_gr+niter_rf),), np.nan)
    Md_, w_ = target['d_'], target['weight_'].sqrt()  # (1, nM, 3), (1, nM)
    
    # if enable adaptive weighting, w_ would change. 
    w_ori_ = w_
    diff = []
    # blur kernel (use _gaussian blur to create the kernel)
    # if weights_blur:
    #     blur_kernel = gaussian_kernel(5, sigma=2., dim=3, channels=1)
        # dims = 5
    
    rfmax, smax = pulse.rfmax, pulse.smax

    gymo_ = cube.γ_

    def fn_loss(cube, pulse, w_ = None, offset_loc_ = None, baseband = 'water'):
        if if_so_spectral:
            # Note that change cube.γ_ will also change cube.γ
            offsets_spectral = (0.5 - torch.rand(gymo_.shape))
        else:
            offsets_spectral = torch.zeros(gymo_.shape)

        if baseband == 'water':
            gymo_copy = gymo_
        elif baseband == 'fat':
            gymo_copy = gymo_ - 0.014 * 3
            
        # add offset in spectral domain.
        cube.γ_ = gymo_copy + offsets_spectral.to(device = gymo_.device)

        Mr_, traj_hist = cube.applypulse(pulse, 
                                         b1Map_=b1Map_, 
                                         doRelax=doRelax, 
                                         offset_loc_ = offset_loc_, 
                                         resolution = res, 
                                         shim_path = shim_path,
                                         traj_con = traj_con,
                                         save_hist = save_hist)
        
        loss_err, loss_pen, loss_pengr, loss_edge = fn_err(Mr_, Md_, w_ = w_), \
                                                                fn_pen(pulse.rf), \
                                                                pen_inf_mat(pulse.gr), \
                                                                pen_l2_mat(pulse.gr)
        if traj_con: 
            loss_traj = pen_traj(traj_hist, traj_type, Md_, w_) 
        else:
            loss_traj = torch.tensor(0)
            
        # cube.γ_ = gymo_
        return loss_err, loss_pen, loss_pengr, loss_edge, loss_traj, Mr_
    
    # used for refocusing pulse design
    def crush_mod_loss(Mx, My, w):
        return crush_stop(Mx, My, w)
    

    log_col = '\n#iter\t ‖ elapsed time\t ‖ error\t ‖ penalty\t ‖ total loss'

    def logger(i, t0, loss_err1, loss_pen, loss_pengr, loss_edge, loss_traj):
        loss = lambda_loss*loss_err1 + eta*loss_pen + lambda_pengr*loss_pengr + lambda_edge*loss_edge + lambda_traj*loss_traj
        # loss = 1 * loss_err1 + eta*loss_pen + 0.0*loss_pengr + 0.0*loss_edge + 0.0*loss_traj
        print("%i\t | %.1f  \t | %.3f\t | %.3f\t | %.3f\t | %.3f\t | %.3f\t | %.3f" \
              %(i, time()-t0, loss_err1.item(),loss_pen.item(), loss_pengr.item(), loss_edge.item(), loss_traj.item(), loss.item()))
        return loss

    # Optimization
    cube_num = cube.size()[1:]
    
    t0 = time()
    for i in range(niter):
        if not (i % 5):
            print(log_col)

        log_ind = 0
        
        if if_so_spatial:
            offset_loc_f1 = torch.rand((1, *cube_num, 3))
        else:
            offset_loc_f1 = torch.ones((1, *cube_num, 3)) * 0.5

        def closure():
            if train_rf: opt_rf.zero_grad()
            if train_nr: opt_sl.zero_grad()
            
            # convert rf back.
            pulse.rf = mrphy_rf.utils.tρθ2rf(tρ, θ, rfmax)
            
            # convert b0 back.
            pulse.lr = mrphy_rf.utils.s2g(mrphy_rf.utils.ts2s(tsl_lr, smax), pulse.dt)

            pulse.nr = mrphy_rf.utils.s2g(mrphy_rf.utils.ts2s(tsl_nr, smax), pulse.dt)
            pulse.gr = torch.cat((pulse.lr,pulse.nr),1)
            
            loss_err1, loss_pen, loss_pengr, loss_edge, loss_traj, _ = fn_loss(cube, pulse, w_, offset_loc_f1, baseband = 'water')
            loss = lambda_loss*loss_err1 + eta*loss_pen + lambda_pengr*loss_pengr + lambda_edge*loss_edge + lambda_traj*loss_traj
            if op_fat:
                loss_err1, loss_pen, loss_pengr, loss_edge, loss_traj, _ = fn_loss(cube, pulse, w_, offset_loc_f1, baseband = 'fat')
                loss = lambda_loss*loss_err1 + eta*loss_pen + lambda_pengr*loss_pengr + lambda_edge*loss_edge + lambda_traj*loss_traj + loss     
                  
            loss.backward()
            
            return loss
        
        
        print('rf-loop: ', niter_rf)
        # print(torch.max(w_[w_ori_ < 2]), torch.min(w_[w_ori_ < 2]), torch.max(w_[w_ori_ > 2]), torch.min(w_[w_ori_ > 2]))
        ################################################################################################
        # adaptive weights is perfromed here..
        # At first, it's voxel wise weights adaption + exponential average + gaussian blue. Yet the spikes and peaks are still there.
        # To improve that, implement a slice-wise in the slice direction adaption.
        if adaptive_weights and ((i+1) % weight_freq == 0):
            if len(diff):
                if slice_weight:
                    diff = sum(diff) / niter_rf
                    container = torch.zeros((1, *cube_num)).to(w_ori_.device) # 1, 88, 88, 60
                    container[cube.mask > 0] = diff
                    # calculate among slice dimension (-1)
                    container = torch.sum(container, dim = (1, 2), keepdim = True)
                    container = container / torch.sum(container)
                    container = container / torch.max(container)
                    container = container.repeat(1, cube_num[0], cube_num[1], 1)
                    diff = container[cube.mask > 0][None]
                    # w_ = 0.99 * w_ + 0.01 * w_ori_ * diff * 10
                    w_ = 0.9 * w_ + 0.1 * diff * 2
                    w_[w_ori_ > 2] = w_ori_[w_ori_ > 2]
                    
                else:     
                    diff = sum(diff) / niter_rf
                    diff = diff / torch.sum(diff)
                    diff = diff / torch.max(diff)           
                    # exponential moving average
                    w_ = 0.9 * w_ + 0.1 * w_ori_ * diff * 10
                    w_[w_ori_ > 2] = w_ori_[w_ori_ > 2]
                    if weights_blur:
                        container = torch.zeros((1, *cube_num)).to(w_ori_.device)
                        container[cube.mask > 0] = w_
                        container = _gaussian_blur(container, 7)[None]
                        w_ = container[cube.mask > 0]
                        w_ = w_[None]
                        
                print(torch.max(w_[w_ori_ < 2]), torch.min(w_[w_ori_ < 2]), torch.max(w_[w_ori_ > 2]), torch.min(w_[w_ori_ > 2]))
                w_tosave = w_.cpu().detach().numpy()
                savemat(os.path.join(save_path, 'iter_' + str(i)+'.mat'), {'w_tosave': w_tosave})
                diff = []
        ################################################################################################
        
        for _ in range(niter_rf):
            if train_rf:
                opt_rf.step(closure)
            
            off_coef = (1/niter_rf) * _
            offset_loc_zero = torch.ones((1, *cube_num, 3)) * off_coef
            
            for idx, rf_scale in enumerate(range(1)): # change the scale of rf. disabled. 
                if idx > 0: 
                    pulse.rf = pulse.rf * rf_scale / (rf_scale - 1)
                else:
                    pulse.rf = pulse.rf * 1

                loss_err1, loss_pen, loss_pengr, loss_edge, loss_traj, M1_ = fn_loss(cube, pulse, w_ori_, offset_loc_zero, baseband = 'water')
              
                if adaptive_weights:
                    #diff.append(abs((M1_[..., -1].detach()**2 - Md_[..., -1]**2))**0.5)
                    diff = abs((torch.sqrt(1-M1_[..., -1].detach()**2 - Md_[..., -1]**2)))
                loss = logger(i, t0, loss_err1, loss_pen, loss_pengr, loss_edge, loss_traj)
              
                if save_mag:
                    MM1 = M1_.cpu().detach().numpy()
                    ww_ = w_.cpu().detach().numpy()
                    savemat(os.path.join(save_path, str(off_coef)+'ex.mat'), {'Mx': MM1, 'mask_in':ww_})
                
            loss_hist[i*(niter_gr+niter_rf)+log_ind] = loss.item()
            loss_mag[i*(niter_gr+niter_rf)+log_ind] = loss_err1.item()
            log_ind += 1
        
        if train_rf:
            scheduler_rf.step()
        
        print('gr-loop: ', niter_gr)
        for _ in range(niter_gr):
            if train_nr:
                opt_sl.step(closure)
                
            off_coef = (1/niter_gr) * _
            offset_loc_zero = torch.ones((1, *cube_num, 3)) * off_coef

            loss_err1, loss_pen, loss_pengr, loss_edge, loss_traj, M1 = fn_loss(cube, pulse, w_ori_, offset_loc_zero, baseband = 'water')
            loss = logger(i, t0, loss_err1, loss_pen, loss_pengr, loss_edge, loss_traj)

            loss_hist[i*(niter_gr+niter_rf) + log_ind] = loss.item()
            loss_mag[i*(niter_gr+niter_rf) + log_ind] = loss_err1.item()
            log_ind += 1
        if train_nr:
            scheduler_gr.step()
        
    print('\n== Results: ==')
    print(log_col)

    optInfos = {'loss_hist': loss_hist}
    savemat(os.path.join(save_path, 'loss.mat'), {'loss_hist': loss_hist, 'loss_mag': loss_mag})
    return pulse, optInfos


def gaussian_kernel(size, sigma=2., dim=3, channels=1):
    # The gaussian kernel is the product of the gaussian function of each dimension.
    # kernel_size should be an odd number.
    
    kernel_size = 2*size + 1

    kernel_size = [kernel_size] * dim
    sigma = [sigma] * dim
    kernel = 1
    meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
    
    for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
        mean = (size - 1) / 2
        kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

    # Make sure sum of values in gaussian kernel equals 1.
    kernel = kernel / torch.sum(kernel)

    # Reshape to depthwise convolutional weight
    kernel = kernel.view(1, 1, *kernel.size())
    kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

    return kernel

def _gaussian_blur(x, size):

    kernel = gaussian_kernel(size=size).to(x.device)
    kernel_size = 2*size + 1

    x = x[None,...]
    padding = int((kernel_size - 1) / 2)

    if x.dim() == 5:
        paddings = (padding, padding, padding, padding, padding, padding)
    elif x.dim() == 4:
        paddings = (padding, padding, padding, padding)   
    x = F.pad(x, paddings, mode='constant')
    
    x = torch.squeeze(F.conv3d(x, kernel))

    return x