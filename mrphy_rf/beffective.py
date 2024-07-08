r""" B-effective related functions
"""

import torch
import torch.nn.functional as F
from torch import tensor, Tensor
from typing import Optional

from mrphy_rf import γH, dt0, π
from mrphy_rf import utils

import scipy.io as sio
import numpy as np
# TODO:
# - Faster init of AB in `beff2ab`


__all__ = ['beff2ab', 'beff2uφ', 'rfgr2beff']


def beff2uϕ(beff: Tensor, γ2πdt: Tensor, dim=-1):
    r"""Compute rotation axes and angles from B-effectives

    Usage:
        ``U, Φ = beff2uϕ(beff, γ2πdt)``
    Inputs:
        - ``beff``: `(N, *Nd, xyz)`, "Gauss", B-effective, magnetic field \
          applied on `M`.
        - ``γ2πdt``: `(N, 1,)`, "Rad/Gauss", gyro ratio in radians, global.
    Optionals:
        - ``dim``: int. Indicate the `xyz`-dim, allow \
          `beff.shape != (N, *Nd, xyz)`
    Outputs:
        - ``U``: `(N, *Nd, xyz)`, rotation axis
        - ``Φ``: `(N, *Nd)`, rotation angle
    """
    U = F.normalize(beff, dim=dim)
    Φ = -torch.norm(beff, dim=dim) * γ2πdt  # negate: BxM -> MxB
    return U, Φ


def beff2ab(
        beff: Tensor,
        E1: Optional[Tensor] = None, E2: Optional[Tensor] = None,
        γ: Optional[Tensor] = None, dt: Optional[Tensor] = None):
    r"""Compute Hargreave's 𝐴/𝐵, mat/vec, from B-effectives

    See: `doi:10.1002/mrm.1170 <https://doi.org/10.1002/mrm.1170>`_.

    Usage:
        ``A, B = beff2ab(beff, T1=(Inf), T2=(Inf), γ=γ¹H, dt=(dt0))``

    Inputs:
        - ``beff``: `(N,*Nd,xyz,nT)`, B-effective.
    Optionals:
        - ``T1``: `(N, *Nd,)`, "Sec", T1 relaxation.
        - ``T2``: `(N, *Nd,)`, "Sec", T2 relaxation.
        - ``γ``:  `(N, *Nd,)`, "Hz/Gauss", gyro ratio in Hertz.
        - ``dt``: `(N, 1, )`, "Sec", dwell time.
    Outputs:
        - ``A``: `(N, *Nd, xyz, 3)`, `A[:,iM,:,:]`, is the `iM`-th 𝐴.
        - ``B``: `(N, *Nd, xyz)`, `B[:,iM,:]`, is the `iM`-th 𝐵.
    """
    shape = beff.shape
    device, dtype, d = beff.device, beff.dtype, beff.dim()-2

    # defaults
    dkw = {'device': device, 'dtype': dtype}
    dt = tensor(dt0, **dkw) if (dt0 is None) else dt.to(device)
    γ = tensor(γH, **dkw) if (γ is None) else γ.to(device)
    E1 = tensor(0, **dkw) if (E1 is None) else E1.to(device)
    E2 = tensor(0, **dkw) if (E2 is None) else E2.to(device)

    # reshaping
    E1, E2, γ, dt = map(lambda x: x.reshape(x.shape+(d-x.dim())*(1,)),
                        (E1, E2, γ, dt))  # broadcastable w/ (N, *Nd)

    E1, E2, γ2πdt = E1[..., None], E2[..., None, None], 2*π*γ*dt
    E1_1 = E1.squeeze(dim=-1) - 1

    # C/Python `reshape/view` is different from Fortran/MatLab/Julia `reshape`
    NNd, nT = shape[0:-2], shape[-1]
    s1, s0 = NNd+(1, 1), NNd+(1, 4)

    AB = torch.cat([torch.ones(s1, **dkw), torch.zeros(s0, **dkw),
                    torch.ones(s1, **dkw), torch.zeros(s0, **dkw),
                    torch.ones(s1, **dkw), torch.zeros(s1, **dkw)],
                   dim=-1).view(NNd+(3, 4))  # -> (N, *Nd, xyz, 3+1)

    # simulation
    for t in range(nT):
        u, ϕ = beff2uϕ(beff[..., t], γ2πdt)

        if torch.any(ϕ != 0):
            AB1 = utils.uϕrot(u, ϕ, AB)
        else:
            AB1 = AB

        # Relaxation
        AB1[..., 0:2, :] *= E2
        AB1[..., 2, :] *= E1
        AB1[..., 2, 3] -= E1_1
        AB, AB1 = AB1, AB

    A, B = AB[..., 0:3], AB[..., 3]

    return A, B


def rfgr2beff(
        mask : Tensor, rf: Tensor, gr: Tensor, loc: Tensor,
        Δf: Optional[Tensor] = None, b1Map: Optional[Tensor] = None,
        γ: Tensor = γH, offset_loc_: Optional[Tensor] = None,
        resolution = None, shim_path = ''):
    r"""Compute B-effectives from rf and gradients

    Usage:
        ``beff = rfgr2beff(rf, gr, loc, Δf, b1Map, γ)``
    Inputs:
        - ``rf``: `(N,xy,nT,(nCoils))`, "Gauss", `xy` for separating real and \
          imag part.
        - ``gr``: `(N,xyz,nT)`, "Gauss/cm".
    Optionals:
        - ``loc``: `(N,*Nd,xyz)`, "cm", locations.
        - ``Δf``: `(N,*Nd,)`, "Hz", off-resonance.
        - ``b1Map``: `(N,*Nd,xy,nCoils)`, a.u., transmit sensitivity.
        - ``γ``: `(N,1)`, "Hz/Gauss", gyro-ratio
    Outputs:
        - ``beff``: `(N,*Nd,xyz,nT)`, "Gauss"
    """   
    assert(rf.device == gr.device == loc.device)

    # loc shape = [1, nM, 3]

    device = rf.device

    shape = loc.shape #(1, points in mask, 3) #rf (1, 2, time point) #gr (1, 3, time point)
    N, Nd, d = shape[0], shape[1:-1], loc.dim()-2
    q = gr.clone().detach().cpu().numpy().shape[1]

    # Shape of mat_content should be Nx, Ny, Nz, num_coils. Nx Ny and Nz should be the same as those at Cube.
    if len(shim_path):
        coil_loop = sio.loadmat(shim_path)
        mat_content = coil_loop['fields']
        mat_content = np.reshape(mat_content, (*mask.shape[1:],mat_content.shape[-1]), order="F").astype('float64')
        fc = torch.unsqueeze(torch.from_numpy(mat_content), 0).type(torch.float32).to(device = device) 
        # (1, points in mask, coil channel -> 6)

        if type(offset_loc_) == type(None):
            fc = fc[mask == True]
        else:
            # Self implemented interplation.
            # Alternatively, could use meshgrid
            offset_loc_ = (-1/2 + offset_loc_) * 1.0
            indx = offset_loc_.clone().to(device = device)
            indx[indx < 0] = -1
            indx[indx >= 0] = 1 # 1, 88, 88, 60, 3
            
            rx1 = torch.zeros(fc.shape).to(device = device)
            rx1[:, 1:, :, :, :] = fc[:, :-1, :, :, :] #negative
            rx2 = torch.zeros(fc.shape).to(device = device)
            rx2[:, :-1, :, :, :] = fc[:, 1:, :, :, :] # positive
            #print(indx.shape, rx2.shape, rx1.shape)
            
            rx = 0.5*((indx[:, :, :, :, 0:1]+1)*rx2 - (indx[:, :, :, :, 0:1]-1)*rx1)
        
            ry1 = torch.zeros(fc.shape).to(device = device)
            ry1[:, :, 1:, :, :] = fc[:, :, :-1, :, :]
            ry2 = torch.zeros(fc.shape).to(device = device)
            ry2[:, :, :-1, :, :] = fc[:, :, 1:, :, :]
            ry = 0.5*((indx[:, :, :, :, 1:2]+1)*ry2 - (indx[:, :, :, :, 1:2]-1)*ry1)
        
        
            rz1 = torch.zeros(fc.shape).to(device = device)
            rz1[:, :, :, 1:, :] = fc[:, :, :, :-1, :]
            rz2 = torch.zeros(fc.shape).to(device = device)
            rz2[:, :, :, :-1, :] = fc[:, :, :, 1:, :]
            rz = 0.5*((indx[:, :, :, :, 2:3]+1)*rz2 - (indx[:, :, :, :, 2:3]-1)*rz1)
        
            coeff = torch.abs(offset_loc_).to(device = device) # 0 - 0.5
            
            merged_x = coeff[:, :, :, :, 0:1] *(rx - fc) + fc
            merged_y = coeff[:, :, :, :, 1:2] *(ry - merged_x) + merged_x
            intp = coeff[:, :, :, :, 2:3] *(rz - merged_y) + merged_y 
        
            fc = intp[mask == True]
            offset_loc_ = offset_loc_[mask == True] * resolution ## only apply to the gradient mag
            loc = loc + offset_loc_.to(loc.device)

    else:
        offset_loc_ = (-1/2 + offset_loc_) * 1.0
        offset_loc_ = offset_loc_[mask == True] * resolution ## only apply to the gradient mag
        loc = loc + offset_loc_.to(loc.device)

        fc = torch.zeros((*loc.shape[:-1],(q-3))).to(loc.device)
        
    fc = torch.unsqueeze(fc, 0)
    #print(fc.shape)
    
    
    if q == 3:
        Bz = (loc.reshape(N, -1, 3) @ gr).reshape((N, *Nd, 1, -1)) #(1, points in mask, 1, time point) # requires_grad = true
    else:
        Bz = (loc.reshape(N, -1, 3) @ gr[:,:3,:]).reshape((N, *Nd, 1, -1))
        Bz_coil = (fc @ gr[:,3:,:]).reshape((N, *Nd, 1, -1))

        # linear gradient is correct. 
        Bz = Bz + Bz_coil

    if Δf is not None:  # Δf: -> (N, *Nd, 1, 1); 3 from 1(dim-N) + 2(dim-xtra)
        γ = γ.to(device=device)
        Δf, γ = map(lambda x: x.reshape(x.shape+(d+3-x.dim())*(1,)), (Δf, γ))
        Bz += Δf/γ

    # rf -> (N, *len(Nd)*(1,), xy, nT, (nCoils))
    rf = rf.reshape((-1, *d*(1,))+rf.shape[1:])
    # Real as `Bx`, Imag as `By`.
    if b1Map is None:
        if rf.dim() == Bz.dim()+1:  # (N, *len(Nd)*(1,), xy, nT, nCoils)
            rf = torch.sum(rf, dim=-1)  # -> (N, *len(Nd)*(1,), xy, nT)
        Bx, By = rf[..., 0:1, :].expand_as(Bz), rf[..., 1:2, :].expand_as(Bz)
        
    else:
        b1Map = b1Map.to(device)
        b1Map = b1Map[..., None, :]  # -> (N, *Nd, xy, 1, nCoils)
        Bx = torch.sum((b1Map[..., 0:1, :, :]*rf[..., 0:1, :, :]
                        - b1Map[..., 1:2, :, :]*rf[..., 1:2, :, :]),
                       dim=-1).expand_as(Bz)  # -> (N, *Nd, x, nT)
        By = torch.sum((b1Map[..., 0:1, :, :]*rf[:, :, 1:2, ...]
                        + b1Map[..., 1:2, :, :]*rf[:, :, 0:1, ...]),
                       dim=-1).expand_as(Bz)  # -> (N, *Nd, y, nT)

    beff = torch.cat([Bx, By, Bz], dim=-2)  # -> (N, *Nd, xyz, nT)
    return beff

