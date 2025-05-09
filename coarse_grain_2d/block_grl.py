#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : block_grl.py
# Author            : Xinliang (Bruce) Lyu <xlyu@ihes.fr>
# Date              : 09.05.2025
# Last Modified Date: 09.05.2025
# Last Modified By  : Xinliang (Bruce) Lyu <xlyu@ihes.fr>
"""
Block-tensor map without considering any lattice symmetry

Tensor leg order convention is
A[x, y, x', y']
      y
      |
  x'--A--x
      |
      y'

"""
from .hotrg_grl import optProj
from ncon import ncon


def block4ten(A, px, py):
    """contraction of a 2x2 block of tensors

    Args:
        A (TensorCommon): 4-leg tensor with real values
        px (TensorCommon): 3-leg isometric tensor p[ij o]
        px (TensorCommon): 3-leg isometric tensor p[ij o]


    Returns:
        Ap (TensorCommon): the coarse-grained tensor

    """
    Ap = ncon([A, A, A, A,
               px.conj(), py.conj(), px, py],
              [[2, 4, 9, 1], [9, 11, 7, 5], [10, 5, 6, 8], [3, 1, 10, 12],
               [2, 3, -1], [4, 11, -2], [7, 6, -3], [12, 8, -4]])
    return Ap


def cgTen(A, chi, cg_eps=1e-8):
    """a single RG step for block-tensor map

    Args:
        A (TensorCommon): 4-leg tensor
        chi (int): the bond dimension

    Kwargs:
        cg_eps (float):
            a numerical precision control for coarse graining process
            see function `optProj` for details

    Returns:
        Ap (TensorCommon):
            corase-grained tensor after 1 RG step
            with rescaling factor b=2

    """
    # first determine the two isometric tensors
    px, errx, eigvx = optProj(A, A, chi, direction="y", cg_eps=cg_eps)
    py, erry, eigvy = optProj(A, A, chi, direction="x", cg_eps=cg_eps)
    # do the block-tensor contraction
    Ap = block4ten(A, px, py)
    return Ap, px, py, errx, erry, eigvx, eigvy
