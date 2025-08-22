#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : env2d_reflsym.py
# Author            : Xinliang (Bruce) Lyu <xlyu@ihes.fr>
# Date              : 21.08.2025
# Last Modified Date: 21.08.2025
# Last Modified By  : Xinliang (Bruce) Lyu <xlyu@ihes.fr>
"""
Environment of the plaquette entanglement filtering.
Lattice reflection is exploited.

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

Some functions in the file `./env2d_reflsym.py` that only exploits
the lattice reflection are also reused here.
"""
from ncon import ncon
from . import env2d_rotsym


def build_Upsilon0(A, bond="x"):
    """build Upsilon0 for initialization of sx and sy

    Args:
        A (TensorCommon): 4-leg tensor

    Kwargs:
        bond (str): choose in ["x", "y"]

    Returns:
        Upsilon0 (TensorCommon)

    """
    assert bond in ["x", "y"], "choose `bond` in x or y"
    # The function `env2d_rotsym.dbA2Upsilon0` is for bond "x"
    # For the bond "y", we simply transpose the leg of A according to
    # A[x, y, x', y'] -- > A[y, x, y', x']
    if bond == "y":
        Arel = A.transpose([1, 0, 3, 2])
    else:
        Arel = A * 1.0

    dbA = env2d_rotsym.contr2A(Arel)
    Upsilon0 = env2d_rotsym.dbA2Upsilon0(dbA)
    return Upsilon0


def build_UpsilonQ(A, sx, sy, bond="x"):
    """Build Upsilon and Q
    For the optimization of the filtering matrices

    Args:
        A (TensorCommon): 6-leg tensor
        sx,sy (TensorCommon): 2-leg tensor
            the filtering matrices

    Kwargs:
        bond (str): choose in ["x", "y"]

    Returns:
        Upsilon (TensorCommon): 4-leg tensor
        Q (TensorCommon): 2-leg tensor

    """
    assert bond in ["x", "y"]
    # The prototype is for x bond
    if bond == "x":
        Arel = A * 1.0
        sxrel = sx * 1.0
        syrel = sy * 1.0
    else:
        Arel = A.transpose([1, 0, 3, 2])
        sxrel = sy * 1.0
        syrel = sx * 1.0

    dbA = env2d_rotsym.contr2A(Arel)
    # these two are for contructing tensor Q
    dbAsy = ncon([dbA, syrel.conj()],
                 [[-1, -2, -3, 1], [1, -4]])
    dbAsyx = ncon([dbAsy, sxrel.conj()],
                  [[-1, 1, -3, -4], [1, -2]])
    # these two are for contructing tensor Upsilon
    dbAsyy = ncon([dbAsy, syrel],
                  [[-1, -2, 1, -4], [1, -3]])
    dbA4s = ncon([dbAsyy, sxrel, sxrel.conj()],
                 [[1, 2, -3, -4], [1, -1], [2, -2]])
    # construct Upsilon and Q
    Upsilon = env2d_rotsym.contr4dbA(dbAsyy, dbA4s)
    Qten = env2d_rotsym.contr4dbA(dbAsy, dbAsyx)
    Q = ncon([Qten], [[-1, -2, 1, 1]])
    return Upsilon, Q


# end of file
