from typing import Optional
from torch import Tensor
from scipy.io import savemat
import torch

def err_null(Mr_: Tensor, Md_: Tensor, w_: Optional[Tensor] = None) -> Tensor:
    """
    *INPUTS*
    - `Mr_` (1, nM, xyz)
    - `Md_` (1, nM, xyz)
    *OPTIONALS*
    - `w_`  (1, nM)
    *OUTPUTS*
    - `err` (1,)
    """
    return Mr_.new_zeros([])


def err_l2z(Mr_: Tensor, Md_: Tensor, w_: Optional[Tensor] = None) -> Tensor:
    """
    *INPUTS*
    - `Mr_` (1, nM, xyz)
    - `Md_` (1, nM, xyz)
    *OPTIONALS*
    - `w_`  (1, nM)
    *OUTPUTS*
    - `err` (1,)
    """
    Me_ = (Mr_[..., 2] - Md_[..., 2])  # (1, nM)   
    err = (Me_ if w_ is None else Me_*w_).norm()**2
    return err


def err_l2xy(Mr_: Tensor, Md_: Tensor, w_: Optional[Tensor] = None) -> Tensor:
    """
    *INPUTS*
    - `Mr_` (1, nM, xyz)
    - `Md_` (1, nM, xyz)
    *OPTIONALS*
    - `w_`  (1, nM)
    *OUTPUTS*
    - `err` (1,)
    """
    Me_ = (Mr_[..., :2] - Md_[..., :2])
    err = (Me_ if w_ is None else Me_*w_[..., None]).norm()**2
    #err = torch.abs(torch.norm(Me_, float('inf')))
    return err


def err_ml2xy(Mr_: Tensor, Md_: Tensor, w_: Optional[Tensor] = None) -> Tensor:
    """
    *INPUTS*
    - `Mr_` (1, nM, xyz)
    - `Md_` (1, nM, xyz)
    *OPTIONALS*
    - `w_`  (1, nM)
    *OUTPUTS*
    - `err` (1,)
    """
    Me_ = Mr_[..., :2].norm(dim=-1) - Md_[..., :2].norm(dim=-1)
    err = (Me_ if w_ is None else Me_*w_).norm()**2
    return err


def err_stdmxy(Mr_, w_):
    Mr_norm = Mr_[..., :2].norm(dim=-1) * w_
    err = torch.std(Mr_norm)
    return -err