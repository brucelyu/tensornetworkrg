#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : env3d.py
# Author            : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
# Date              : 05.06.2023
# Last Modified Date: 05.06.2023
# Last Modified By  : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
"""
Constrcut environment for 3D entanglement-filtering process based on
1) full-environment trunction (FET)
2) Graph-independent local truncations (GILT)

--- For cubic environments ---
There are three filtering matrices:,
    sx, sy, sz
for each directions.
The prototyical case is the environments after linearizing the sy.
Other two cases should be obtained by rotating the inital input 6-leg tensor A
_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
"""
import numpy as np
from ncon import ncon

"""
Part 0: cube environments
The computational cost is O(χ^{12})
Order of the tensor leg is A[x, x', y, y', z, z']
"""

# I. First are functions for constructing <ψ|ψ> given the input tensor A
# Those functions will be the building blocks for other
# linearized environments containing sx, sy, sz


def contr2A(A):
    """contract (A, A_dagger) pair

    Args:
        A (TensorCommon): reference (+++) 6-leg tensor
             z  x'
             | /
         y'--A--y
           / |
         x   z'

    Returns:
        dbA (TensorCommon): a 6-leg tensor
            dbA[x, xd, y, yd, z, zd]
              zd  z
              |  |
         yd-- dbA --y
             /  /
           xd  x

    """
    # Three legs are contracted; the cost is 3+3+3=9, so O(χ^9)
    dbA = ncon([A, A.conj()],
               [[-1, 1, -3, 2, -5, 3],
                [-2, 1, -4, 2, -6, 3]])
    return dbA


def contrx(dbA):
    """contract (dbA, dbA_dagger) pair in x-direction

    Args:
        dbA (TensorCommon): 6-leg tensor
            dbA[x, xd, y, yd, z, zd]

    Returns:
        quadrA (TensorCommon): an 8-leg tensor
            quadrA[y, yd, yd', y', z, zd, zd', z']

    """
    # Two legs are contracted; the cost is 4 + 2 + 4 = 10, so O(χ^{10})
    quadrA = ncon([dbA, dbA.conj()],
                  [[1, 2, -1, -2, -5, -6],
                   [1, 2, -3, -4, -7, -8]])
    return quadrA


# This step is the bottleneck of the computational efficiency.
# It has computational costs like O(χ^{12}).
def contrz(quadrA):
    """contract (quadrA, quadrA_dagger) pair in z-direction

    Args:
        quadrA (TensorCommon): 8-leg tensor

    Returns:
        octuA (TensorCommon): an 8-leg tensor
            octA[y1, y1d, y2d, y2,
                 y3d, y3, y4, y4d]

    """
    # Four legs are contracted; the cost is 4 + 4 + 4 = 12, so O(χ^{12})
    octuA = ncon([quadrA, quadrA.conj()],
                 [[-1, -2, -3, -4, 1, 2, 3, 4],
                  [-5, -6, -7, -8, 1, 2, 3, 4]])
    return octuA


def contr2psipsi(octuA):
    """modulus of the wavefunction of the 8-tensor cube

    Args:
        octuA (TensorCommon): 8-leg tensor

    Returns:
        psipsi (float): modulus of the wavefunction of the 8-tensor cube

    """
    res = ncon([octuA, octuA.conj()],
               [[1, 2, 3, 4, 5, 6, 7, 8],
                [1, 2, 3, 4, 5, 6, 7, 8]])
    return res


# This function is the combination of the previous 4 steps
# so the input is A and output is the desired <ψ|ψ>
def cubePsiPsi(A):
    """Construct the <ψ|ψ> given the input A

    """
    dbA = contr2A(A)
    quadrA = contrx(dbA)
    octuA = contrz(quadrA)
    res = contr2psipsi(octuA)
    return res


# II. Next are functions for constructing linearlized environments.
# The implementation is based on a prototypical case,
# where some chosen sy's vary,
# while all other sx, sy, and sz's are treated as constant

# II.1 Helper function for constructing Psy
def absbs2AA(dbA, sx, sz):
    """absorb sx, sz matrices to double-A tensor

    Args:
        dbA (TensorCommon): 6-leg tensor
              zd  z
              |  |
         yd-- dbA --y
             /  /
           xd  x

    Returns: dbAs
        Similar to dbA, but with its x and z legs
        squeezed using sx and sz

    """
    dbAs = ncon([dbA, sx, sz],
                [[1, -2, -3, -4, 2, -6],
                 [1, -1], [2, -5]])
    return dbAs


def absbs2octuA(octuA, sy):
    """absorb three copies of sy to octupole-A tensor

    Args:
        octuA (TensorCommon): 8-leg tensor

    Returns: octuAs
        Similar to octuA, but with

    """
    octuAs = ncon([octuA, sy.conj(), sy.conj(), sy],
                  [[-1, -2, 1, -4, 2, -6, 3, -8],
                   [1, -3], [2, -5], [3, -7]])
    return octuAs


def contr2Psy(octuAs, sy):
    """construct the environment Psy

    Args:
        octuAs (TensorCommon): 8-leg tensor
            output of the function `absbs2octuA`

    Returns: Psy

    """
    octuAssy = ncon([octuAs, sy],
                    [[1, -2, -3, -4, -5, -6, -7, -8],
                     [1, -1]])
    Psy_dagger = ncon(
        [octuAs, octuAssy.conj()],
        [[-1, 1, 5, 2, 6, 3, 7, 4],
         [-2, 1, 5, 2, 6, 3, 7, 4]]
    )
    return Psy_dagger.conj()

# II.2 Helper function for constructing γsy
