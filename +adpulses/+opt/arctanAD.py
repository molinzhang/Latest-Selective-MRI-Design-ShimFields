# arctan.py
import sys
sys.path.insert(0, '/home/molin/github_code_upload/MR_excitation_op_update')

import torch
import os
from adpulses_rf import io, optimizers, metrics, penalties
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
if __name__ == "__main__":
    import sys

    if len(sys.argv) <= 1:  # mode DEBUG
        import os
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

    m2pName = ('m2p.mat' if len(sys.argv) <= 1 else sys.argv[1])
    p2mName = ('p2m.mat' if len(sys.argv) <= 2 else sys.argv[2])
    gpuID = ('0' if len(sys.argv) <= 3 else sys.argv[3])
    
    print('gpuid!!!\t',gpuID,'gpuid!!!\t')
    
    # %% load
    if gpuID == '-1':
        dkw = {'device': torch.device('cpu'), 'dtype': torch.float32}
    else:
        dkw = {'device': torch.device('cuda:'+gpuID), 'dtype': torch.float32}

    target, cube, pulse, arg = io.m2p(m2pName, **dkw)

    def dflt_arg(k, v, fn):
        return (fn(k) if ((k in arg.keys()) and (arg[k].size > 0)) else v)

    arg['doRelax'] = dflt_arg('doRelax', True, lambda k: bool(arg[k].item()))

    arg['b1Map_'] = dflt_arg('b1Map_', None,
                             lambda k: f_tensor(f_c2r_np(arg[k], -2)))

    arg['niter'] = dflt_arg('niter', 1, lambda k: arg[k].item())
    arg['niter_gr'] = dflt_arg('niter_gr', 0, lambda k: arg[k].item())
    arg['niter_rf'] = dflt_arg('niter_rf', 0, lambda k: arg[k].item())
    arg['train_rf'] = dflt_arg('train_rf', True, lambda k: arg[k].item())
    arg['train_nr'] = dflt_arg('train_nr', True, lambda k: arg[k].item())
    arg['lr_rf'] = dflt_arg('lr_rf', 0.001, lambda k: arg[k].item())
    arg['lr_nr'] = dflt_arg('lr_nr', 0.001, lambda k: arg[k].item())
    arg['save_mag'] = dflt_arg('save_mag', False, lambda k: arg[k].item())
    arg['save_path'] = dflt_arg('save_path', './results', lambda k: arg[k].item())
    arg['shim_path'] = dflt_arg('shim_path', '', lambda k: arg[k].item())
    arg['if_so_spatial'] = dflt_arg('if_so_spatial', True, lambda k: arg[k].item())
    arg['op_fat'] = dflt_arg('op_fat', False, lambda k: arg[k].item())
    arg['res'] = dflt_arg('res', 0.1, lambda k: arg[k].item())
    arg['adaptive_weights'] = dflt_arg('adaptive_weights', False, lambda k: arg[k].item())
    arg['weight_freq'] = dflt_arg('weight_freq', 1, lambda k: arg[k].item())
    arg['weights_blur'] = dflt_arg('weights_blur', False, lambda k: arg[k].item())
    arg['slice_weight'] = dflt_arg('slice_weight', False, lambda k: arg[k].item())
    arg['traj_con'] = dflt_arg('traj_con', False, lambda k: arg[k].item())
    arg['traj_type'] = dflt_arg('traj_type', 'z', lambda k: arg[k].item())
    arg['lambda_traj'] = dflt_arg('lambda_traj', 0.10, lambda k: arg[k].item())
    arg['lambda_loss'] = dflt_arg('lambda_loss', 1.0, lambda k: arg[k].item())
    arg['lambda_pengr'] = dflt_arg('lambda_pengr', 0.0, lambda k: arg[k].item())
    arg['lambda_edge'] = dflt_arg('lambda_edge', 0.0, lambda k: arg[k].item())
    arg['save_hist'] = dflt_arg('save_hist', '', lambda k: arg[k].item())   
     
    eta = dflt_arg('eta', 4, lambda k: float(arg[k].item()))
    print('eta: ', eta)

    err_meth = dflt_arg('err_meth', 'l2xy', lambda k: arg[k].item())
    pen_meth = dflt_arg('pen_meth', 'l2', lambda k: arg[k].item())

    err_hash = {'null': metrics.err_null,
                'l2xy': metrics.err_l2xy, 'ml2xy': metrics.err_ml2xy,
                'l2z': metrics.err_l2z}
    pen_hash = {'null': penalties.pen_null, 'l2': penalties.pen_l2}

    fn_err, fn_pen = err_hash[err_meth], pen_hash[pen_meth]

    # %% pulse design
    kw = {k: arg[k] for k in ('b1Map_', 'niter', 'niter_gr', 'niter_rf',
                              'doRelax', 'train_rf', 'train_nr', 'lr_rf', 
                              'lr_nr', 'save_mag', 'save_path', 'shim_path', 
                              'if_so_spatial', 'op_fat', 'res', 'adaptive_weights',
                              'weight_freq', 'weights_blur', 'slice_weight',
                              'traj_con', 'traj_type', 'lambda_traj', 
                              'lambda_edge', 'lambda_loss', 'lambda_pengr', 'save_hist')}

    pulse, optInfos = optimizers.arctanLBFGS(target, cube, pulse,
                                             fn_err, fn_pen, eta=eta, **kw)

    # %% saving
    #print('python saved pulse',pulse.gr.shape)
    io.p2m(p2mName, pulse, optInfos)
