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

from ncon import ncon


def densityM(A):
    """determine the isometric

    Args:
        A (TensorCommon): 4-leg tensor with real values
        chi (int): the bond dimension χ

    Kwargs:
        cg_eps (float):
            a numerical precision control for coarse graining process
            eigenvalues small than cg_eps will be thrown away
            regardless of χ

    Returns:
        p (TensorCommon): 3-leg tensor
        g (array): SWAP sign matrix

    """
    rho = ncon([A, A.conjugate(), A.conjugate(), A],
               [[-1, 2, 1, 5], [-2, 4, 3, 5],
                [-3, 2, 1, 6], [-4, 4, 3, 6]]
               )
    # This matrix has the symmetry under the SWAP operation:
    #   rho[i,j,m,n] = rho[j,i,n,m]
    return rho
