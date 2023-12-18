#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : env3dcube.py
# Author            : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
# Date              : 15.12.2023
# Last Modified Date: 15.12.2023
# Last Modified By  : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
"""
Construct cube (or membrane) environments for
a 3D lattice system on a cubic lattice.

The environments are for the following
two entanglement-filtering schemes:
1) Full-environment trunction (FET)
2) Graph-independent local truncations (GILT)

There are three filtering matrices:,
    sx, sy, sz
for each directions.
The prototyical case is the environments after linearizing the sy.
Other two cases should be obtained by
rotating the inital input 6-leg tensor A.

The computational cost is O(χ^{12})
Order of the tensor leg is A[x, x', y, y', z, z']
_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
"""
from ncon import ncon

# I. For constructing <ψ|ψ> and absorbing sx, sz matrices

def sOnA(A, sx, sz):
    """absorb sx, sz matrices into main tensor A

    Args:
        A (TensorCommon): reference (+++) 6-leg tensor
             z  x'
             | /
         y'--A--y
           / |
         x   z'
        sx (TensorCommon): 2-leg tensor for filtering (x direction)
         χ -- sx -- χs
        sz (TensorCommon): 2-leg tensor for filtering (z direction)

    Returns: 6-leg tensor after absorbing sx and sz
             |
             sz
             | /
           --A--
           / |
         sx
        /
    """
    Axz = ncon([A, sx, sz], [[1, -2, -3, -4, 2, -6], [1, -1], [2, -5]])
    return Axz


def contrInLeg(A, B):
    """contract the three inner legs of two 6-leg tensors

    Args:
        A (TensorCommon): 6-leg tensor
        B (TensorCommon): 6-leg tensor

    Returns:
        dbA (TensorCommon), a 6-leg tensor
        dbA[y, yd, z, zd, x, xd]
              zd  z
              |  |
         yd-- dbA --y
             /  /
           xd  x

    """
    dbA = ncon([A, B],
               [[-5, 1, -1, 2, -3, 3],
                [-6, 1, -2, 2, -4, 3]])
    return dbA


def contrx(dbA, dbB):
    """contract (dbA, dbB) pair in x direction

    Args:
        dbA (TensorCommon): 6-leg tensor
        dbB (TensorCommon): 6-leg tensor

    Returns:
        quadrA (TensorCommon): an 8-leg tensor
    """
    quadrA = ncon([dbA, dbB],
                  [[-1, -2, -5, -6, 1, 2],
                   [-3, -4, -7, -8, 1, 2]])
    return quadrA


def contrz(quadrA, quadrB):
    """contract (quadrA, quadrB) pair in z direction

    Args:
        quadrA (TensorCommon): 8-leg tensor
        quadrB (TensorCommon): 8-leg tensor

    Returns:
        octuA (TensorCommon): 8-leg tensor

    """
    octuA = ncon([quadrA, quadrB],
                 [[-1, -2, -3, -4, 1, 2, 3, 4],
                  [-5, -6, -7, -8, 1, 2, 3, 4]])
    return octuA


# II. For absorbing sy matrices

def syOn1Leg(dbAp, sy):
    """absorb sy matrix into the doubleA tensor on one leg
    This is used for constructing P

    Args:
        dbAp (TensorCommon): 6-leg tensor
        sy (TensorCommon): 2-leg tensor

    Returns:
        dbApy (TensorCommon)

    """
    dbApy = ncon([dbAp, sy], [[1, -2, -3, -4, -5, -6], [1, -1]])
    return dbApy


def syOn2Leg(dbAgm, sy):
    """absorb sy matrix into the doubleA tensor on two legs
    This is used for constructing γ

    Args:
        dbAgm (TensorCommon): 6-leg tensor
        sy (TensorCommon): 2-leg tensor

    Returns:
        dbAgmy (TensorCommon)

    """
    dbAgmy = ncon([dbAgm, sy, sy.conj()],
                  [[1, 2, -3, -4, -5, -6],
                   [1, -1], [2, -2]])
    return dbAgmy


def syOn1LegQuad(quadrA, sy):
    """absorb sy matrix into the first leg of quadrA tensor
    This is used for constructing P

    Args:
        quadrA (TensorCommon): 8-leg tensor
        sy (TensorCommon): 2-leg tensor

    Returns:
        quadrAy (TensorCommon)

    """
    quadrAy = ncon([quadrA, sy],
                   [[1, -2, -3, -4, -5, -6, -7, -8], [1, -1]]
                   )
    return quadrAy


def syOn2LegQuad(quadrA, sy):
    """absorb (sy, sy_dagger) into the first two legs of quadrA tensor
    This is used for constructing γ

    Args:
        quadrA (TensorCommon): 8-leg tensor
        sy (TensorCommon): 2-leg tensor

    Returns:
        quadrAy (TensorCommon)

    """
    quadrAy = ncon([quadrA, sy, sy.conj()],
                   [[1, 2, -3, -4, -5, -6, -7, -8],
                    [1, -1], [2, -2]])
    return quadrAy


# III. Octuple tensor to P and γ
def octu2P(octuA, sy):
    """contract (octuA, octuA.conj()) pair and and sy
    Do the octuA contraction first is faster than
    do the sy contraction first

    Args:
        octuA (TensorCommon): 8-leg tensor
        sy (TensorCommon): 2-leg tensor

    Returns:
        Psy (TensorCommon)

    """
    Psy0 = ncon([octuA, octuA.conj()],
                [[-1, 1, 2, 3, 4, 5, 6, 7],
                 [-2, 1, 2, 3, 4, 5, 6, 7]])
    Psy_dagger = ncon([Psy0, sy.conj()], [[-1, 1], [1, -2]])
    return Psy_dagger.conj()


def octu2gm(octuA, sy):
    """construct γ matrix from octuple tensor

    Args:
        octuA (TensorCommon): 8-leg tensor
        sy (TensorCommon): 2-leg tensor

    Returns:
        Gamma (TensorCommon)

    """
    Gamma0 = ncon([octuA, octuA.conj()],
                  [[-3, -1, 1, 2, 3, 4, 5, 6],
                   [-4, -2, 1, 2, 3, 4, 5, 6]]
                  )
    Gamma = ncon([Gamma0, sy, sy.conj()],
                 [[-1, 1, -3, 2], [1, -2], [2, -4]])
    return Gamma
