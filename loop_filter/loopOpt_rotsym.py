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
               |              z
                               \
                                vL*--
                                |
    """
    # apply the bond matrix zp to vL
    vLz = vL * 1.0
    if zp is not None:
        vLz = vLz.multiply_diag(zp, 2, direction="r")
    Av = ncon([vLz, vL.conj()], [[-3, -2, 1], [-4, -1, 1]])
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


def multizdbA(dbA, z):
    """multiply the bond matrix z on dbA

    Args:
        dbA (TensorCommon): 4-leg tensor
            result of the above `contr2A`
        z (TensorCommon): 1D vector
            the bond matrix

    Returns:
        dbAz (TensorCommon):

    """
    return dbAz




# end of file
