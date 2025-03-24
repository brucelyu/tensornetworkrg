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
assumed to be [1, 1, -1, -1]
"""

import numpy as np
from ncon import ncon


def optProj(A, B, chi, direction="x", cg_eps=1e-8):
    """determine the isometry p for fusing two legs

    Args:
        A (TensorCommon): 4-leg tensor
        B (TensorCommon): 4-leg tensor
        chi (int): bond dimension χ

    Kwargs:
        direction (str): direction of the coarse-graining collapse
            For direction = "y", two x legs are fused
            For direction = "x", two y legs are fused
        cg_eps (float):
            a numerical precision control for coarse graining process
            eigenvalues small than cg_eps will be thrown away
            regardless of χ

    Returns:
        px (TensorCommon): 3-leg tensor

    """
    # `direction` should be either x or y
    assert direction in ["x", "y"]
    # The "density matrix" ρ
    if direction == "y":
        rho = ncon([A, B, A.conjugate(), B.conjugate()],
                   [[-1, 2, 1, 5], [-2, 5, 3, 4],
                    [-3, 2, 1, 6], [-4, 6, 3, 4]]
                   )
    elif direction == "x":
        rho = ncon([A, B, A.conjugate(), B.conjugate()],
                   [[1, -1, 5, 2], [5, -2, 3, 4],
                    [1, -3, 6, 2], [6, -4, 3, 4]]
                   )
    # eigenvalue decomposition and truncation
    eigv, p, err = rho.eig(
        [0, 1], [2, 3], hermitian=True,
        chis=[i+1 for i in range(chi)], eps=cg_eps,
        trunc_err_func=trunc_err_func,
        return_rel_err=True
    )
    return p, err, eigv


def collap2ten(A, B, pout, pin, direction="y"):
    """HOTRG collpase of two tensors along a direction

    Args:
        A (TensorCommon): 4-leg tensor
        B (TensorCommon): 4-leg tensor
        pout (TensorCommon): 3-leg isometric tensor
            for outer legs of A and B
        pin (TensorCommon): 3-leg isometric tensor
            for inner legs of A and B

    Kwargs:
        direction (str): direction of the coarse-graining collapse
            For direction = "y", two x legs are fused
            For direction = "x", two y legs are fused

    Returns:
        Ap (TensorCommon): the coarse-grained tensor

    """
    # `direction` should be either x or y
    assert direction in ["x", "y"]
    # contraction A and B using two isometric tensors
    if direction == "y":
        Ap = ncon([A, B, pout, pin],
                  [[1, -2, 5, 3], [4, 3, 2, -4],
                   [1, 4, -1], [5, 2, -3]])
    elif direction == "x":
        Ap = ncon([A, B, pout, pin],
                  [[-1, 1, 3, 5], [3, 4, -3, 2],
                   [1, 4, -2], [5, 2, -4]])
    return Ap


def cgTen(A, chi, cg_eps=1e-8):
    """a single RG step for coarse-graining the tensor A

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
    # First do the y-collapse (fuse x legs)
    px, errx, eigvx = optProj(A, A, chi, direction="y", cg_eps=cg_eps)
    Ay = collap2ten(A, A, px.conjugate(), px, direction="y")
    # and then the x-collapse (fuse y legs)
    py, erry, eigvy = optProj(Ay, Ay, chi, direction="x", cg_eps=cg_eps)
    Ap = collap2ten(Ay, Ay, py.conjugate(), py, direction="x")
    # store isometric tensors and RG errors
    return Ap, px, py, errx, erry


def trunc_err_func(eigv, chi):
    """
    No need to take square since we work with M M'
    whose eigenvalues are themself square of singular values of M
    """
    res = np.sum(eigv[chi:]) / np.sum(eigv)
    return res
