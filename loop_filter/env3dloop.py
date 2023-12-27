#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : env3dloop.py
# Author            : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
# Date              : 25.12.2023
# Last Modified Date: 25.12.2023
# Last Modified By  : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
"""
Construct plaquette (or loop) environment for
a 3D classical system on a cube lattice.

This is essentially the same as 2D loop filtering, just with
one additional pairs of leg attached.

The environments are for the following
two entanglement-filtering schemes:
1) Full-environment trunction (FET)
2) Graph-independent local truncations (GILT)

There are two filtering matrices:,
    sx, sy
for each directions.
The prototyical case is the environments after linearizing the sy.
Other two cases should be obtained by
rotating the inital input 6-leg tensor A.

The computational cost is O(χ^{8}), which is much smaller
than the cube filtering and block-tensor process
Order of the tensor leg is A[x, x', y, y', z, z']
_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
"""
from ncon import ncon


# I. For initial P, γ (for GILT) and calculating <ψ|ψ>
# The functions are also useful for FET optimization environments
def contrInLeg(A, B):
    """contract 4 legs other than the 2 to be squeeze
    This contraction is the bottleneck of
    the loop-environment construction.
    Its cost is O(χ^8)

    Args:
        A (TensorCommon): 6-leg tensor
        B (TensorCommon): 6-leg tensor

    Returns:
        dbA (TensorCommon), a 4-leg tensor
        dbA[y, yd, x, xd]
         yd-- dbA --y
              |  |
              xd x
    """
    dbA = ncon([A, B],
               [[-3, 1, -1, 2, 3, 4],
                [-4, 1, -2, 2, 3, 4]])
    return dbA


def contrx(dbA, dbB):
    """contract (dbA, dbB) pair in x direction
    Thus the only remaining legs are 4 y-direction ones

    Args:
        dbA (TensorCommon): 4-leg tensor
        dbB (TensorCommon): 4-leg tensor

    Returns:
        quadrA (TensorCommon): an 4-leg tensor

    """
    quadrA = ncon([dbA, dbB],
                  [[-1, -2, 1, 2],
                   [-3, -4, 1, 2]])
    return quadrA


def loopGamma(A, direction="y"):
    """loop-filtering γ environment for initialization (GILT)

    Args:
        A (TensorCommon): 6-leg tensor

    Returns:
        Gammay (TensorCommon): 4-leg tensor

    """
    # Cost is χ^8
    dbA = contrInLeg(A, A.conj())
    # Cost is χ^6
    quadrA = contrx(dbA, dbA.conj())
    # Cost is χ^6
    Gammay = ncon(
        [quadrA, quadrA.conj()],
        [[-3, -1, 1, 2], [-4, -2, 1, 2]]
    )
    return Gammay


# II. Absorb mx matrix
# Note: mx is treated as constant matrix
# during my optimization
def mxAssemble(dbA, mx):
    """Assemble mx matrix into the doubleA matrix

    Args:
        dbA (TensorCommon): 4-leg tensor
        mx (TensorCommon): 2-leg tensor
            loop-filtering matrix

    Returns:
        dbAp (TensorCommon): 4-leg tensor
            Similar to dbA, with mx matrix assembled for P environment
        dbAgm (TensorCommon): 4-leg tensor
            Similar to dbA, with mx matrix assembled for γ environment

    """
    dbAp = ncon([dbA, mx],
                [[-1, -2, 1, -4], [1, -3]])
    dbAgm = ncon([dbAp, mx.conj()],
                 [[-1, -2, -3, 1], [1, -4]])
    return dbAp, dbAgm


# III. Absorb my matrix
def myAssemble(dbAp, dbAgm, my):
    """Assemble my matrix into the doubleA matrix

    Args:
        dbAp (TensorCommon): 4-leg tensor
        dbAgm (TensorCommon): 4-leg tensor
        my (TensorCommon): 2-leg tensor
            loop-filtering matrix

    Returns: TODO

    """
    # Absorb my into one side of dbAp
    dbApy = ncon([dbAp, my],
                 [[1, -2, -3, -4], [1, -1]])
    # Absorb my into two sides of dbAgm
    dbAgmy = ncon([dbAgm, my, my.conj()],
                  [[1, 2, -3, -4], [1, -1], [2, -2]])
    return dbApy, dbAgmy


# IV. Quadrupole tensor to P and γ
def quadr2P(quadrA, my):
    """Contract quadrupole tensor to P
    The cost is roughly χ^5

    Args:
        quadrA (TensorCommon): an 4-leg tensor
            resultant tensor after `contrx`
        my (TensorCommon): 2-leg tensor
            loop-filtering matrix

    Returns:
        Py (TensorCommon): 2-leg tensor

    """
    quadrAy = ncon([quadrA, my],
                   [[1, -2, -3, -4], [1, -1]])
    Py = ncon(
        [quadrA, quadrAy.conj()],
        [[-1, 2, 3, 4], [-2, 2, 3, 4]]
    ).conj()
    return Py


def quadr2gm(quadrA, my):
    """Contract quadrupole tensor to γ
    The cost is roughly χ^6

    Args:
        quadrA (TensorCommon): an 4-leg tensor
            resultant tensor after `contrx`
        my (TensorCommon): 2-leg tensor
            loop-filtering matrix

    Returns:
        Gammay (TensorCommon): 4-leg tensor

    """
    quadrAy = ncon([quadrA, my, my.conj()],
                   [[1, 2, -3, -4], [1, -1], [2, -2]])
    Gammay = ncon(
        [quadrA, quadrAy.conj()],
        [[-3, -1, 1, 2], [-4, -2, 1, 2]]
    )
    return Gammay


# V. Combine the above functions to construct P and γ for loop filtering
def dbA2FETenv(dbAp, dbAgm, my):
    """Build P from doubleA tensors and my

    Args:
        dbAp (TensorCommon): 4-leg tensor
        dbAgm (TensorCommon): 4-leg tensor
        mx (TensorCommon): 2-leg tensor
        my (TensorCommon): 2-leg tensor

    Returns:
        Py (TensorCommon): 2-leg tensor
        Gammay (TensorCommon): 4-leg tensor

    """
    # Absorbing my into doubleA tensors
    # Cost is roughly χ^5
    dbApy, dbAgmy = myAssemble(
        dbAp, dbAgm, my
    )
    # Plug two doubleA tensors to form quadrupole tensors
    # Cost is roughly χ^6
    quadrAp = contrx(dbAp, dbApy.conj())
    quadrAgm = contrx(dbAgm, dbAgmy.conj())
    # quadrupole tensors to P and γ
    # Cost is rouphly χ^5
    Py = quadr2P(quadrAp, my)
    # Cost is rouphly χ^6
    Gammay = quadr2gm(quadrAgm, my)
    return Py, Gammay


# VI. Rotate the tensor to other directions, including
# - x leg in z-loop
def swapxy(A, mx, my):
    """ Swap x and y direction
    """
    Ap = A.transpose([2, 3, 0, 1, 4, 5])
    mxp = my * 1.0
    myp = mx * 1.0
    return Ap, mxp, myp
