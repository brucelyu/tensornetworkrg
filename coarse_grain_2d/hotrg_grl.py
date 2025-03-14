#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : hotrg_grl.py
# Author            : Xinliang (Bruce) Lyu <xlyu@ihes.fr>
# Date              : 14.03.2025
# Last Modified Date: 14.03.2025
# Last Modified By  : Xinliang (Bruce) Lyu <xlyu@ihes.fr>
"""
A general implementation of the HOTRG that does not assume any
lattice symmetry of the underlying model.
This is what was proposed in the original HOTRG paper:
https://arxiv.org/abs/1201.1144

Tensor leg order convention is
A[x, y, x', y']
      y
      |
  x'--A--x
      |
      y'

For tensors with complex values,
the conjugate direction of the tensor is
assumed to be [1, -1, 1, -1]
"""

import numpy as np
from ncon import ncon


def optProj(A, B, chi, leg="x", cg_eps=1e-8):
    """determine the isometry p for fusing two legs

    Args:
        A (TensorCommon): 4-leg tensor
        B (TensorCommon): 4-leg tensor
        chi (int): bond dimension χ

    Kwargs:
        cg_eps (float):
            a numerical precision control for coarse graining process
            eigenvalues small than cg_eps will be thrown away
            regardless of χ

    Returns:
        px (TensorCommon): 3-leg tensor

    """
    # leg should be either x or y
    assert leg in ["x", "y"]
    # The "density matrix" ρ
    if leg == "x":
        rho = ncon([A, B, A.conjugate(), B.conjugate()],
                   [[-1, 1, 2, 5], [-2, 3, 5, 4],
                    [-3, 1, 2, 6], [-4, 3, 6, 4]]
                   )
    elif leg == "y":
        rho = ncon([A, B, A.conjugate(), B.conjugate()],
                   [[1, 5, -1, 2], [5, 3, -2, 4],
                    [1, 6, -3, 2], [6, 3, -4, 4]]
                   )
    # eigenvalue decomposition and truncation
    eigv, p, err = rho.eig(
        [0, 1], [2, 3], hermitian=True,
        chis=[i+1 for i in range(chi)], eps=cg_eps,
        trunc_err_func=trunc_err_func,
        return_rel_err=True
    )
    return p, err, eigv


def trunc_err_func(eigv, chi):
    """
    No need to take square since we work with M M'
    whose eigenvalues are themself square of singular values of M
    """
    res = np.sum(eigv[chi:]) / np.sum(eigv)
    return res
