#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : trg.py
# Author            : Xinliang (Bruce) Lyu <xlyu@ihes.fr>
# Date              : 24.03.2025
# Last Modified Date: 24.03.2025
# Last Modified By  : Xinliang (Bruce) Lyu <xlyu@ihes.fr>
"""
An implementation of Levin and Nave's TRG algorithm:
https://arxiv.org/abs/cond-mat/0611687

Tensor leg order convention is
A[x, y, x', y']
      y
      |
  x'--A--x
      |
      y'

For tensors with complex values,
the conjugate direction of the tensor is
assumed to be [1, 1, -1, -1]
"""

from ncon import ncon


def splitTen(A1, A2, chi, cg_eps=1e-8):
    """split the tensor A diagonally
                   |                      |
     |             t3--         |       --t4
   --A1-- -->    /       ,    --A2--  -->   \
     |       --t1               |            t2--
                |                            |
    Args:
        A1 (TensorCommon): 4-leg tensor
        A2 (TensorCommon): 4-leg tensor
        chi (int): bond dimension

    Kwargs:
        direction (str): direction of the splitting
            choose in ["13", "24"]
        cg_eps (float):
            a numerical precision control for coarse graining process
            singular values that are smaller than it will
            be truncated regardless of χ

    Returns:
        t1, t2, t3, t4 (TensorCommon):
            3-leg tensor after splitting
            The 3rd leg is the truncated leg
        err1 (float), err2 (float): errors of the SVD splittings

    """
    t3, t1, err1 = A1.split(
        [0, 1], [2, 3], chis=[i+1 for i in range(chi)], eps=cg_eps,
        return_rel_err=True
    )
    # put the truncated leg into the last position
    t1 = t1.transpose([1, 2, 0])
    t4, t2, err2 = A2.split(
        [2, 1], [0, 3], chis=[i+1 for i in range(chi)], eps=cg_eps,
        return_rel_err=True
    )
    # put the truncated leg into the last position
    t2 = t2.transpose([1, 2, 0])
    return t1, t2, t3, t4, err1, err2


def contr4pieces(t1, t2, t3, t4):
    """Contract 4 pieces of 3-leg tensors
              \      /
      \ /      t2--t1
       A'  =    |   |
      / \      t3--t4
              /      \
    Args:
        t1 (TensorCommon): 3-leg tensor
        t2 (TensorCommon): same
        t3 (TensorCommon): same
        t4 (TensorCommon): same

    Returns:
        A (TensorCommon): 4-leg coarse-grained tensor

    """
    A = ncon([t1, t2, t3, t4],
             [[1, 11, -1], [1, 12, -2], [2, 12, -3], [2, 11, -4]])
    return A
