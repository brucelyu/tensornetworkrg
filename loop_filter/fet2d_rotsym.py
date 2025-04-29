#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : fet2d_rotsym.py
# Author            : Xinliang (Bruce) Lyu <xlyu@ihes.fr>
# Date              : 29.04.2025
# Last Modified Date: 29.04.2025
# Last Modified By  : Xinliang (Bruce) Lyu <xlyu@ihes.fr>
"""
This module implements a Entanglement Filtering;
both the lattice reflection and rotational symmetries are exploited

The scheme is a combination of
- Evenbly's full environment truncation (FET)
- Graph-Independent Local Truncations (Gilt) by
Hauru, Delcamp, and Mizera

Due to the lattice reflection and rotation symmetry,
there is only one filtering matrix s

The relevant environment is constructed in the file `./env2d_rotsym.py`
"""
from .. import u1ten
from . import env2d_rotsym
from ncon import ncon


# I. For the initialization of the filtering matrix s
def init_s(A, chis, chienv, epsilon):
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
    dbA = env2d_rotsym.contr2A(A)
    Upsilon0 = env2d_rotsym.dbA2Upsilon0(dbA)
    # determine the low-rank matrix (like Gilt without recursion)
    # Step 1. Take the Moore-Penrose inverse of the Upsilon tensor
    Upsilon0_pinv = u1ten.pinv(
        Upsilon0, [0, 1], [2, 3],
        eps_mach=epsilon, chiCut=chienv
    )
    # Step 2. Obtain the low-rank matrix Lr
    Lr = ncon([Upsilon0_pinv, Upsilon0], [[-1, -2, 1, 2], [1, 2, 3, 3]])
    # Step 3. Split Lr to get s: Lr = s @ s*
    s = Lr.split([0], [1], chis=[k+1 for k in range(chis)],
                 eps=epsilon)[0]
    # make the direction of s as [-1, 1]
    s = s.flip_dir(1)
    return s, Lr, Upsilon0, dbA

# II. For optimization of s matrix


def fidelity(A, s, PsiPsi):
    """fidelity of the Entanglement Filtering

    Args:
        A (TensorCommon): 4-leg tensor
        s (TensorCommon): the filtering matrix
        PsiPsi (float): the overlap <ψ|ψ>

    """
    dbA = env2d_rotsym.contr2A(A)
    Upsilon, P = env2d_rotsym.dbA2UpsilonP(dbA, s)
    f, err, PhiPhi = env2d_rotsym.plaqFidelity(s, P, Upsilon, PsiPsi)
    return f, err, PhiPhi


# end of the file
