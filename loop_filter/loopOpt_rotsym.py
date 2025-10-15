#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : loopOpt_rotsym.py
# Author            : Xinliang (Bruce) Lyu <xlyu@ihes.fr>
# Date              : 14.10.2025
# Last Modified Date: 14.10.2025
# Last Modified By  : Xinliang (Bruce) Lyu <xlyu@ihes.fr>
"""
An entanglement filtering scheme based on loop optimization
as is proposed in the loop-TNR paper:
https://arxiv.org/abs/1512.04938

Lattice reflection and rotation symmetries are exploited.

However, instead of using variational MPS method,
we reformulate to optimization problem using an idea
proposed in Evenbly's FET paper:
https://arxiv.org/abs/1801.05390
We suspect that these two optimization procedure are equivalent to each other.

The order convention of the tensor leg is
                       y
                       |
A[x, y, x', y'] =  x'--A--x ,
                       |
                       y'

               | j
vL[ijα] = i -- vL           ,
                 \ α

while the order of ij for vL depends on the direction of
the arrow on its TN diagram.
"""
from ncon import ncon


def multizp2vL(vL, zp):
    """multiply zp on the 3rd leg of vL

    Args:
        vL (TensorCommon): 3-leg tensor
        zp (TensorCommon): 1D vector
            the renormalized bond matrix

    Returns:
        vLzp (TensorCommon): 3-leg tensor
            with zp multiplied on the 3rd leg of vL

    """
    vLzp = vL * 1.0
    if zp is not None:
        vLzp = vLzp.multiply_diag(zp, 2, direction="r")
    return vLzp


def vL2Av(vL, zp):
    """combine vL and zp back to a 4-leg tensor Av
    It is the reverse process of splitting of A using EVD
    in the function `.trg_rotsym.init_trg`

    Args:
        vL (TensorCommon): 3-leg tensor
        zp (TensorCommon): 1D vector
            the renormalized bond matrix

    Returns:
        Av (TensorCommon): 4-leg tensor
                           |
               |         --vL
             --Av--  =       \
               |              zp
                               \
                                vL*--
                                |
    """
    vLzp = multizp2vL(vL, zp)
    Av = ncon([vLzp, vL.conj()], [[-3, -2, 1], [-4, -1, 1]])
    return Av


def contr2A(Ar, Al):
    """contract two 4-leg tensors

    Args:
        Ar (TensorCommon): 4-leg tensor
        Al (TensorCommon): 4-leg tensor

    Returns:
        dbA (TensorCommon): 4-leg tensor

    """
    dbA = ncon([Ar, Al.conj()], [[-1, -3, 1, 2], [-2, -4, 1, 2]])
    return dbA


def multiz2dbA(dbA, z):
    """multiply the bond matrix z on dbA

    Args:
        dbA (TensorCommon): 4-leg tensor
            result of the above `contr2A`
        z (TensorCommon): 1D vector
            the bond matrix

    Returns:
        dbAz (TensorCommon):

    """
    dbAz = dbA * 1.0
    # apply z on the four legs of dbA
    if z is not None:
        dbAz = dbAz.multiply_diag(z, 0, direction="r")
        dbAz = dbAz.multiply_diag(z, 3, direction="r")
        dbAz = dbAz.multiply_diag(z, 1, direction="l")
        dbAz = dbAz.multiply_diag(z, 2, direction="l")
    return dbAz


def vL2Upsilon(vL, zp, z):
    """build the Υ tensor from vL and bond matrices

    Args:
        vL (TensorCommon): 3-leg tensor
        zp (TensorCommon): 1D vector
            the renormalized bond matrix
        z (TensorCommon): 1D vector
            the bond matrix

    Returns:
        Upsilon (TensorCommon): 4-leg tensor
            The Υ tensor for updating vL
            in loop optimization

    """
    Av = vL2Av(vL, zp)       # costs: O(χ^5)
    dbAv = contr2A(Av, Av)   # costs: O(χ^6)
    # multiply z on dbAv
    dbAvz = multiz2dbA(dbAv, z)
    # multiply zp on vL
    vLzp = multizp2vL(vL, zp)
    # build Upsilon            costs: O(3χ^6)
    Upsilon = ncon([dbAvz, dbAv.conj(), dbAvz, vLzp, vLzp.conj()],
                   [[12, 13, 1, 2], [3, 4, 1, 2], [3, 4, -3, -1],
                    [11, 12, -4], [11, 13, -2]])
    return Upsilon


def vL2Q(vL, zp, z, A):
    """build the Q tensor

    Args:
        vL (TensorCommon): 3-leg tensor
        zp (TensorCommon): 1D vector
            the renormalized bond matrix
        z (TensorCommon): 1D vector
            the bond matrix
        A (TensorCommon): 4-leg tensor
            the bulk tensor in TN

    Returns:
        Q (TensorCommon): 3-leg tensor

    """
    Av = vL2Av(vL, zp)       # costs: O(χ^5)
    AAv = contr2A(A, Av)     # costs: O(χ^6)
    # multiply z on AAv
    AAvz = multiz2dbA(AAv, z)
    # multiply zp on vL
    vLzp = multizp2vL(vL, zp)
    # build Q
    Qstar = ncon([AAvz, AAv.conj(), AAvz, A.conj(), vLzp.conj()],
                 [[11, 14, 1, 2], [3, 4, 1, 2], [3, 4, 12, -1],
                  [11, 12, -3, 13], [13, 14, -2]])
    Q = Qstar.conj()
    return Q


def A2PsiPsi(A, z):
    """calculate the overlap <ψ|ψ>
    For computing the fidelity of the loop optimization

    Args:
        A (TensorCommon): 4-leg tensor
            the bulk tensor in TN
        z (TensorCommon): 1D vector
            the bond matrix

    Returns:
        PsiPsi (float): the overlap

    """
    dbA = contr2A(A, A)    # costs: O(χ^6)
    dbAz = multiz2dbA(dbA, z)
    # build PsiPsi         # costs: O(χ^6)
    quadrAz = ncon([dbAz, dbA.conj()], [[1, 2, -1, -2], [1, 2, -3, -4]])
    PsiPsi = ncon([quadrAz]*2, [[1, 2, 3, 4], [3, 4, 1, 2]])
    return PsiPsi


# end of file
