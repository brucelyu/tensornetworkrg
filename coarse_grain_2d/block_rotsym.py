#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : block_rotsym.py
# Author            : Xinliang (Bruce) Lyu <xlyu@ihes.fr>
# Date              : 08.04.2025
# Last Modified Date: 08.04.2025
# Last Modified By  : Xinliang (Bruce) Lyu <xlyu@ihes.fr>
"""
Block-tensor map with the following two lattice symmetry exploited
- lattice reflection
- rotational
The 4-leg tensor A is assumed to be real.
We have not worked out the complex-value tensor yet

Tensor leg order convention is
A[x, y, x', y']
      y
      |
  x'--A--x
      |
      y'

"""

import numpy as np
from ncon import ncon


def densityM(A):
    """Construct the density matrix for the isometry

    Args:
        A (TensorCommon): 4-leg tensor with real values

    Returns:
        rho (TensorCommon): 4-leg density matrix

    """
    rho = ncon([A, A.conjugate(), A.conjugate(), A],
               [[-1, 2, 1, 5], [-2, 4, 3, 5],
                [-3, 2, 1, 6], [-4, 4, 3, 6]]
               )
    # This matrix has the symmetry under the SWAP operation:
    #   rho[i,j,m,n] = rho[j,i,n,m]
    return rho


def findProj(rho, chi, cg_eps=1e-8):
    """determine the isometry p

    Args:
        rho (TensorCommon): density matrix for isometry
        chi (int): the bond dimension χ

    Kwargs:
        cg_eps (float):
            a numerical precision control for coarse graining process
            eigenvalues small than cg_eps will be thrown away
            regardless of χ

    Returns:
        p (TensorCommon): 3-leg tensor
        g (array): SWAP sign
        err (float): error for truncation
        eigv (TensorCommon): eigenvalues

    """
    # simontaneously diagonalize rho and the SWAP matrix
    # since SWAP @ rho = rho @ SWAP.
    # SWAP @ rho[ij mn] = rho[ji mn]
    # Eigenvalues are the multiplication of the two
    # (The second pair is [3, 2] instead of [2, 3] due to SWAP)
    eigv, p, err = rho.eig(
        [0, 1], [3, 2],
        chis=[i+1 for i in range(chi)], eps=cg_eps,
        trunc_err_func=trunc_err_func,
        return_rel_err=True
    )
    # calculate the SWAP eigenvalues
    g = np.sign(eigv)
    return p, g, err, eigv.abs()


def block4ten(A, p):
    """contraction of a 2x2 block of tensors

    Args:
        A (TODO): TODO
        p (TODO): TODO

    Returns: TODO

    """
    return Ap


def trunc_err_func(eigv, chi):
    """
    No need to take square since we work with M M'
    whose eigenvalues are themself square of singular values of M
    """
    res = np.sum(np.abs(eigv[chi:])) / np.sum(np.abs(eigv))
    return res
