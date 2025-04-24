#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : env2d_rotsym.py
# Author            : Xinliang (Bruce) Lyu <xlyu@ihes.fr>
# Date              : 24.04.2025
# Last Modified Date: 24.04.2025
# Last Modified By  : Xinliang (Bruce) Lyu <xlyu@ihes.fr>
"""
Environment of the plaquette entanglement filtering.
Lattice reflection and rotation symmetries are exploited.
The scheme is a combination of two previous techniques:
- Gilt in https://arxiv.org/abs/1709.07460
- FET in https://arxiv.org/abs/1801.05390

The transposition trick is used here to exploit the lattice reflection symmetry

The order convention of the tensor leg is
                       y
                       |
A[x, y, x', y'] =  x'--A--x
                       |
                       y'
"""
from ncon import ncon


def contr2A(A):
    """contract two A tensors to get a double-A tensor

    Args:
        A (TensorCommon): 4-leg tensor

    Returns:
        dbA (TensorCommon): 4-leg tensor
        dbA[x, xd, y, yd] =
          yd | | y
       xd -- dbA -- x

    """
    # transposition trick is applied here (A.conj() is transposed)
    dbA = ncon([A, A.conj()], [[-1, -3, 1, 2], [-2, -4, 1, 2]])
    return dbA


def contr4dbA(dbA0, dbA1):
    """contract four double-A tensors

    Args:
        dbA0 (TensorCommon): 4-leg tensor
            constructed from the `contr2A` function
        dbA1 (TensorCommon): 4-leg tensor
            constructed from the `contr2A` function

    Returns:
        Upsilon (TensorCommon): 4-leg tensor
        Upsilon[x, xf, x', xf'] =
           x ---    --- x'
               Upsilon
           xf---    ---xf'
    """
    Upsilon = ncon([dbA0, dbA1.conj(), dbA1.conj(), dbA1],
                   [[-3, -1, 1, 2], [5, 6, 1, 2],
                    [-4, -2, 3, 4], [5, 6, 3, 4]])
    return Upsilon


def dbA2Upsilon0(dbA):
    """Upsilon for the initialization of the filtering matrix

    Args:
        dbA (TensorCommon): 4-leg tensor
            constructed from `constr2A` function

    Returns:
        Upsilon0 (TensorCommon): 4-leg tensor
            for initialization of the filtering matrix

    """
    Upsilon0 = contr4dbA(dbA, dbA)
    return Upsilon0


def dbA2UpsilonP(dbA, s):
    """Construct Upsilon and P
    For the optimization of the filtering matrix

    Args:
        dbA (TensorCommon): 4-leg tensor
            constructed from `constr2A` function
        s (TensorCommon): 2-leg tensor
            the filtering matrix

    Returns:
        Upsilon (TensorCommon): 4-leg tensor
        P (TensorCommon): 2-leg tensor

    """
    # these two are for contructing tensor P
    dbA1s = ncon([dbA, s],
                 [[-1, -2, -3, 1], [1, -4]]
                 )
    dbA2s = ncon([dbA1s, s.conj()],
                 [[-1, 1, -3, -4], [1, -2]]
                 )
    # these two are for contructing tensor Upsilon
    dbA2sb = ncon([[dbA1s, s.conj()]],
                  [[-1, -2, 1, -4], [1, -3]])
    dbA4s = ncon([dbA2sb, s, s.conj()],
                 [[1, 2, -3, -4], [1, -1], [2, -2]])
    # construct Upsilon and P
    Upsilon = contr4dbA(dbA2sb, dbA4s)
    Pten = contr4dbA(dbA1s, dbA2s)
    P = ncon([Pten], [[-1, -2, 1, 1]])
    return Upsilon, P




# end of the file
