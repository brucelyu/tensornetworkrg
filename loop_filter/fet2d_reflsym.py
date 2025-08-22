#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : fet2d_reflsym.py
# Author            : Xinliang (Bruce) Lyu <xlyu@ihes.fr>
# Date              : 21.08.2025
# Last Modified Date: 21.08.2025
# Last Modified By  : Xinliang (Bruce) Lyu <xlyu@ihes.fr>
"""
This module implements a Entanglement Filtering;
the lattice reflection symmetry is exploited

The scheme is a combination of
- Evenbly's full environment truncation (FET)
- Graph-Independent Local Truncations (Gilt) by
Hauru, Delcamp, and Mizera

Due to the lattice reflection symmetry,
there are only two filtering matrix sx, sy

The relevant environment is constructed in the file `./env2d_reflsym.py`
Some functions in the file `./env2d_reflsym.py` that only exploits
the lattice reflection are also reused here.
"""
from .. import u1ten
from . import env2d_rotsym, env2d_reflsym
from ncon import ncon


# I. For the initialization of the filtering matriices sx and sy
def init_s(A, chis, chienv, epsilon, epsilon_inv=1e-10,
           bond="x"):
    """Initialization of the filtering matrix s

    Args:
        A (TensorCommon): 4-leg tensor
        chis (int): χs in the filtering
        chienv (int): χenv for taking the pinv of Upsilon0
        epsilon (float): for the pinv

    Returns:
        s (TensorCommon): the filtering matrix
        Lr (TensorCommon): the low-rank matrix, Lr = s @ s*
        Upsilon0 (TensorCommon): bone environment tensor
        dbA (TensorCommon): contraction of two copies of A:
            useful for later optimization of s

    """
    Upsilon0 = env2d_reflsym(A, bond=bond)
    # determine the low-rank matrix (like Gilt without recursion)
    # Step 1. Take the Moore-Penrose inverse of the Upsilon tensor
    Upsilon0_pinv = u1ten.pinv(
        Upsilon0, [0, 1], [2, 3],
        eps_mach=epsilon_inv, chiCut=chienv
    )
    # Step 2. Obtain the low-rank matrix Lr
    Lr = ncon([Upsilon0_pinv, Upsilon0], [[-1, -2, 1, 2], [1, 2, 3, 3]])
    # Step 3. Split Lr to get s: Lr = s @ s*
    s = Lr.split([0], [1], chis=[k+1 for k in range(chis)],
                 eps=epsilon)[0]
    # make the direction of s as [-1, 1]
    s = s.flip_dir(1)
    return s, Lr, Upsilon0



# end of file
