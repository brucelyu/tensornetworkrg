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

# end of file
