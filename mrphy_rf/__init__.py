r"""A MRI excitation physics module.

About MRphy.py:
===============

``MRphy.py`` provides the following constant and submodules:

Constant:

    - ``γH``: water proton gyro frequency, "4257.6 Hz/Gauss".

Submodules:

    - :mod:`~mrphy.utils`
    - :mod:`~mrphy.beffective`
    - :mod:`~mrphy.sims`
    - :mod:`~mrphy.slowsims`
    - :mod:`~mrphy.mobjs`

General Comments:
=================

Variable naming convention:
---------------------------

A trailing ``_`` in a variable/attribute name indicates compactness, i.e.
instead of size ``(N, *Nd, ...)``, the variable/attributes size
``(N, nM, ...)``.
For instance:
A field map variable ``b0map`` may be of size ``(N, nx, ny, nz)``, while its
compact countarpart ``b0map_`` has a size of ``(N, nM)``.

Special keywords used in documentations:
----------------------------------------

- ``N``:   batch size
- ``nM``:  the number of spins
- ``Nd``:  A int-tuple for array size, e.g.: ``Nd = (nx, (ny, (nz)))``. \
  In python convention, ``*`` unpacks a tuple. \
  Therefore, ``(N, *Nd) === (N, nx, ny, nz)``.
- ``nT``:  the number of time points
- ``xy``:  the dimension has length of ``2``
- ``xyz``: the dimension has length of ``3``
- ``⊻``: **Either or**. \
  When used in enumerating function keywords arguments, it means the function \
  accepts at most one of the keywords in a set as an input, e.g., \
  ``b0map ⊻ b0map_`` means accepting either ``b0map`` or ``b0map_``. \
  When used in specifying variable dimensions, it means the dimension tuple \
  can be one of the size tuple set, e.g. ``(N, nM ⊻ *Nd, xyz)`` means \
  accepting dimension either ``(N, nM, xyz)`` or ``(N, *Nd, xyz)``.
"""

from math import pi as π, inf  # noqa: F401
import torch
from torch import tensor

# This is not used.
γH = tensor(4257.6, dtype=torch.double)  # Hz/Gauss, water proton gyro freq.
T1G = tensor(1.47, dtype=torch.double)   # Sec, T1 for gray matter
T2G = tensor(0.07, dtype=torch.double)   # Sec, T2 for gray matter

dt0 = tensor(4e-6, dtype=torch.double)   # Sec, default dwell time
gmax0 = tensor(5, dtype=torch.double)    # Gauss/cm
smax0 = tensor(12e3, dtype=torch.double)  # Gauss/cm/Sec
rfmax0 = tensor(0.25, dtype=torch.double)  # Gauss

_slice = slice(None)

#from mrphy_rf import (utils, beffective, sims, slowsims, mobjs)  # noqa: E402

__all__ = ['γH', 'utils', 'beffective', 'sims', 'slowsims', 'mobjs']
