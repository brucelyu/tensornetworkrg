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

# II.1 Helper function for constructing Psy (Prototype)
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
    # Seven legs are contracted; the cost is 1 + 7 + 1, so O(χ^9)
    Psy_dagger = ncon(
        [octuAs, octuAssy.conj()],
        [[-1, 1, 5, 2, 6, 3, 7, 4],
         [-2, 1, 5, 2, 6, 3, 7, 4]]
    )
    return Psy_dagger.conj()


# II.2 Helper function for constructing γsy (Prototype)
def absbs2AA_left(dbA, sx, sz):
    """absorb sx*, sz* to the left legs of double-A tensor

    Args:
        dbA (TensorCommon): 6-leg tensor
              zd  z
              |  |
         yd-- dbA --y
             /  /
           xd  x

    Returns: dbAs

    """
    dbAs = ncon([dbA, sx.conj(), sz.conj()],
                [[-1, 1, -3, -4, -5, 2],
                 [1, -2], [2, -6]])
    return dbAs


def absbs2octuA_left(octuA, sy):
    """absorb three copies of sy to the left of octupole-A tensor

    Args:
        octuA (TensorCommon): 8-leg tensor

    Returns: octuAs

    """
    octuAs = ncon([octuA, sy, sy, sy.conj()],
                  [[-1, -2, -3, 1, -5, 2, -7, 3],
                   [1, -4], [2, -6], [3, -8]])
    return octuAs


def contr2Gammasy(octuAss, sy):
    """construct the environment γsy
    """
    octuAsssy = ncon([octuAss, sy, sy.conj()],
                     [[1, 2, -3, -4, -5, -6, -7, -8],
                      [1, -1], [2, -2]])
    # Six legs are contracted; the cost is 2 + 6 + 2, so O(χ^{10})
    Gammasy = ncon(
        [octuAss, octuAsssy.conj()],
        [[-3, -1, 4, 1, 5, 2, 6, 3],
         [-4, -2, 4, 1, 5, 2, 6, 3]]
    )
    return Gammasy


# II.3 Combine the previous two sets of helper functions
# to construct general Ps and γs
def cubePermute(A, sx, sy, sz,
                direction="y"):
    # For constructing sz, sx environments
    # from the prototypical sy one
    if direction == "y":
        # do nothing for prototypical case
        return A, sx, sy, sz
    # 1) Permute legs of A
    # 2) Permuate sx, sy, sz
    elif direction == "z":
        # (xyz) --> (yzx)
        Ap = A.transpose([2, 3, 4, 5, 0, 1])
        sxp = sy * 1.0
        syp = sz * 1.0
        szp = sx * 1.0
    elif direction == "x":
        # (xyz) --> (zxy)
        Ap = A.transpose([4, 5, 0, 1, 2, 3])
        sxp = sz * 1.0
        syp = sx * 1.0
        szp = sy * 1.0
    else:
        errMsg = "direction should be x, y or z"
        raise ValueError(errMsg)
    return Ap, sxp, syp, szp


def cubePs(A, sx, sy, sz,
           direction="y"):
    """
    Construct `Ps` environment in smaller steps.
    The bottleneck for the computational efficiency is
    the step `contrz` with cost O(χ^{12})
    """
    # rotate A and permute sx, sy, sz
    # according to the direction
    (Ap, sxp, syp, szp) = cubePermute(
         A, sx, sy, sz,
         direction
    )
    # construct Ps in smaller steps
    # Psy is the protypical case
    # Step #1: contract two copies of A
    # Cost: O(χ^9)
    dbA = contr2A(Ap)
    # Step #2: absorb sx and sz to dbA
    dbAs = absbs2AA(dbA, sxp, szp)
    # Step #3: contract two copies of dbAs
    # Cost: O(χ^{10})
    quadrA = contrx(dbAs)
    # Step #4: contract two copies of quadrA
    # Cost: O(χ^{12}) <--- This is the bottleneck of the efficiency
    octuA = contrz(quadrA)

    # Step #5: absorb sy
    octuAs = absbs2octuA(octuA, syp)
    # Step #6: contract octuAs and sy to get Psy
    # Cost: O(χ^9)
    Ps = contr2Psy(octuAs, syp)
    return Ps


def cubeGammas(A, sx, sy, sz,
               direction="y"):
    """
    Construct `γ` environment in smaller steps.
    The bottleneck for the computational efficiency is
    the step `contrz` with cost O(χ^{12})
    """
    # rotate A and permute sx, sy, sz
    # according to the direction
    (Ap, sxp, syp, szp) = cubePermute(
         A, sx, sy, sz,
         direction
    )
    # construct γs in smaller steps
    # γs is the protypical case
    # Step #1: contract two copies of A
    # (same as Ps)
    # Cost: O(χ^9)
    dbA = contr2A(Ap)
    # Step #2.i: absorb sx and sz on the right of dbA
    # (same as Ps)
    dbAs = absbs2AA(dbA, sxp, szp)
    # Step #2.ii: absorb sx and sz on the left of dbAs
    # (*new for γs)
    dbAss = absbs2AA_left(dbAs, sxp, szp)
    # Step #3 and #4: get quadra- and octu-A
    # (same as Ps)
    quadrA = contrx(dbAss)

    # Cost: O(χ^{12}) <--- This is the bottleneck of the efficiency
    octuA = contrz(quadrA)
    # Step #5.i: absorb sy on the right of octuA
    # (same as Ps)
    octuAs = absbs2octuA(octuA, syp)
    # Step #5.ii: absorb sy on the left of uctA
    # (*new for γs)
    octuAss = absbs2octuA_left(octuAs, syp)
    # Step #6: contract octuAss and sy to get γsy
    Gammas = contr2Gammasy(octuAss, syp)
    return Gammas


# III. For initialization of s matrices
def cubeGamma(A, direction="y"):
    """
    Almost the same as `cubeGammas`, except
    all s matrices are identity
    """
    # rotate A and permute sx, sy, sz
    # according to the direction
    (Ap, sxp, syp, szp) = cubePermute(
         A, 1, 1, 1,
         direction
    )
    dbA = contr2A(Ap)
    quadrA = contrx(dbA)
    octuA = contrz(quadrA)
    Gammasy = ncon(
        [octuA, octuA.conj()],
        [[-3, -1, 4, 1, 5, 2, 6, 3],
         [-4, -2, 4, 1, 5, 2, 6, 3]]
    )
    return Gammasy


# IV. Fidelity
def cubeFidelity(s, Ps, Gammas, PsiPsi=1):
    """
    Fidelity of FET approximation:
    0 <= f <= 1
    The process is exact if fidelity f = 1
    """
    # <ψ|φ>
    PsiPhi = ncon([Ps.conj(), s], [[1, 2], [1, 2]])
    # <φ|φ>
    PhiPhi = ncon([Gammas, s.conj(), s],
                  [[1, 2, 3, 4], [1, 2], [3, 4]])
    # fidelity = |<ψ|φ>|^2 / (<φ|φ> <ψ|ψ>)
    f = PsiPhi * PsiPhi.conj() / (PhiPhi * PsiPsi)
    f = f.norm()
    return f, 1 - f, PhiPhi
