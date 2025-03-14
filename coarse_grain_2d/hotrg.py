#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : hotrg.py
# Author            : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
# Date              : 18.10.2022
# Last Modified Date: 18.10.2022
# Last Modified By  : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
"""
Various 2D HOTRG implementations
that exploiting lattice symmetry of the underlying model.
Until Septebmer 2024, when I graduated from my PhD,
we had explored
1) lattice reflection
2) lattice rotation

Notice that this file does not implement the usual HOTRG
without exploiting the symmetry.
See another file `hotrg_grl.py` for a more general
implementation of the HOTRG

Tensor leg order convention is
A[x, y, x', y']
      y
      |
   x--A--x'
      |
      y'

For the totally-isotropic version, the
conjugate direction of the tensor is
assumed to be [-1, 1, -1, 1].

For the reflection-symmetric version, the
conjugate direction of the tensor is
assumed to be [1, 1, -1, -1].
"""

import numpy as np
from ncon import ncon


# This is for isotropic block-tensor RG in 2D
def opt_v(Ain, chi, dtol=1e-10):
    """
    Determine the squeezer v
    A trick is used to strictly preserve
    the reflection symmetry
    """
    B = Ain.conjugate()
    env_v = ncon([Ain, B, Ain.conjugate(), B.conjugate()],
                 [[-2, 1, 2, 5], [-1, 3, 4, 5],
                  [-4, 1, 2, 6], [-3, 3, 4, 6]
                  ])
    (d,
     v,
     SPerr
     ) = env_v.eig([0, 1], [2, 3],
                   hermitian=True,
                   chis=[i+1 for i in range(chi)],
                   trunc_err_func=trunc_err_func,
                   eps=dtol, return_rel_err=True)
    return v, SPerr


# --- The following three functions are for                     ---|
# --- the reflection-symmetric implmentation of the 2d HOTRG --    |
def opt_vDown(Ain, chi, dtol=1e-10):
    """determine the isometric tensor for a 2d reflection-symmetric HOTRG.
    Basically the same as `opt_v`;
    the difference is the convention
    of gauge in the design

    Args:
        Ain (TensorCommon): 4-leg tensor A[i, j, k, l]
        chi (int): squeezed bond dimension

    Kwargs:
        dtol (float): truncate eigenvectors with
            eigenvalues smaller than dtol

    Returns:
        vDown (TensorCommon): 3-leg isometric tensor

    """
    B = Ain.conjugate()
    env_v = ncon([Ain, B, Ain.conjugate(), B.conjugate()],
                 [[-1, 1, 2, 5], [-2, 3, 4, 5],
                  [-3, 1, 2, 6], [-4, 3, 4, 6]
                  ])
    (d,
     v,
     SPerr
     ) = env_v.eig([0, 1], [2, 3],
                   hermitian=True,
                   chis=[i+1 for i in range(chi)],
                   trunc_err_func=trunc_err_func,
                   eps=dtol, return_rel_err=True)
    return v, SPerr


def block2tensors(A, v, vin):
    """block A and its reflection in vertical direction a la HOTRG

    Args:
        A (TensorCommon): 4-leg tensor
        v (TensorCommon): outer isometry (left)
        vin (TensorCommon): inner isometry (right)

    Returns:
        Aout (TensorCommon): 4-leg coarser tensor
    """
    Aout = ncon([A, A.conj(), v.conj(), vin],
                [[1, -2, 3, 4], [5, -4, 2, 4],
                 [1, 5, -1], [3, 2, -3]])
    return Aout


def blockPlaq(Ain, v, w, vin):
    """block a 2-by-2 plaquette made of A and its reflection
    using isometric tensors `v, w, vin` a la HOTRG

    Args:
        Ain (TensorCommon): 4-leg tensor A[i, j, k, l]
        v (TensorCommon): 3-leg isometric tensor
        w (TensorCommon): 3-leg isometric tensor
        vin (TensorCommon): 3-leg isoemtric tensor

    Returns:
        Aout (TensorCommon): 4-leg coarser tensor

    """
    # Block vertically
    Avc = block2tensors(Ain, v, vin)
    # Block horizontally
    Avcreflv = Avc.conj().transpose([2, 1, 0, 3])
    Aout = block2tensors(Avcreflv.transpose([1, 2, 3, 0]),
                         w, w)
    # rotate back
    Aout = Aout.transpose([3, 0, 1, 2])
    return Aout


def reflSymHOTRG(Ain, chi, chiI=None, dtol=1e-10, horiSym=True):
    """coarse graining of 2D HOTRG that preserves reflection symmetry
    Block the following plaquette
        |     |
     ---A----Areflv*--
        |     |
        |     |
 --Areflh*----Areflhv--
        |     |
    For the blocking part, we
    1) first block two tensors in the vertical direction, and then
    2) blocking the resultant tensor in horizontal direction.
    However, notice both two outer isometric tensors
    are determined using initial tensor as environment.

    Args:
        Ain (TensorCommon): 4-leg tensor A[i, j, k, l]
        chi (int): bond dimension
        dtol (float): truncate eigenvectors with
            eigvenvalues smaller than dtol
        horiSym (boolean): whether does `Ain` have
            horizontal-reflection symmetry

    Returns:
        Aout (TensorCommon): 4-leg coarser tensor
        v (TensorCommon): outer isometry for blocking
            two tensors vertically
        vin (TensorCommon): inner isometry for blocking
            two tensors vertically
        w (TensorCommon): outer isometry for blocking
            two tensors horizontally
    """
    # determine one outer isometry
    v, errv = opt_vDown(Ain, chi, dtol)
    # determine the other outer isometry
    Areflvstar = Ain.conj().transpose([2, 1, 0, 3])
    # input the rotated-legs-by-90-degrees tensor
    w, errw = opt_vDown(Areflvstar.transpose([1, 2, 3, 0]),
                        chi, dtol)
    # determine the inner isometry
    if horiSym:
        # vin is exactly the same as v
        vin = v * 1.0
        errvin = errv
    else:
        # determine vin by input Areflv*
        (vin,
         errvin
         ) = opt_vDown(Areflvstar, chiI, dtol)
    # finally we block Ain and v, vin, w and
    # get the coarser tensor
    Aout = blockPlaq(Ain, v, w, vin)
    SPerrList = [errv, errw, errvin]
    return Aout, v, w, vin, SPerrList

# --- Here is the end of the                                    ---|
# --- the reflection-symmetric implmentation of the 2d HOTRG --- __|

# --------------------------------------------------------------------- #
# --------------------------------------------------------------------- #

# --- The following three functions are for                       ---|
# --- the rotationally-symmetric implmentation of the 2d HOTRG --    |


def block4tensor(Ain, v):
    """
    Block 4 tensors to get a coarser tensor

    The block is totally isotropic but
    the computation cost is high here, which
    goes like O(χ^8)
    """
    # rotation A 90° to get B
    B1 = Ain.transpose([1, 2, 3, 0])
    B2 = Ain.transpose([2, 3, 0, 1])
    B3 = Ain.transpose([3, 0, 1, 2])
    # 90° rotation symmetry is imposed here
    Aout = ncon([Ain, B1, B2, B3,
                 v, v.conjugate(), v, v.conjugate()],
                [[2, 4, 9, 1], [9, 11, 7, 5],
                 [10, 5, 6, 8], [3, 1, 10, 12],
                 [2, 3, -1], [4, 11, -2], [6, 7, -3], [8, 12, -4]
                 ]
                )
    msg = "mirror symmetry and rotation symmetry not compatible"
    assert Aout.transpose([1, 2, 3, 0]).allclose(Aout.transpose([0, 3, 2, 1]).conjugate()), msg
    assert Aout.transpose([1, 2, 3, 0]).allclose(Aout.transpose([2, 1, 0, 3]).conjugate()), msg
    # symmetrize just in case for machine error
    Aout = 1/3 * (Aout + Aout.transpose([3, 2, 1, 0]).conjugate() + Aout.transpose([1, 0, 3, 2]).conjugate())
    return Aout


def isotroHOTRG(Ain, chi, dtol=1e-10):
    v, SPerr = opt_v(Ain, chi)
    Aout = block4tensor(Ain, v)
    return Aout, v, SPerr


def getAuvu(AList, vList, rgstep=5):
    """
    Map from bipartitle A, B tensor network
    to a uniform single-Au tensor network
    """
    vformal = vList[rgstep - 1]
    vcur = vList[rgstep]
    Acur = AList[rgstep]
    gauge = ncon([vformal]*2, [[1, 2, -1], [2, 1, -2]])
    # perform the gauge transformation
    Au = ncon([Acur, gauge, gauge.conjugate()],
              [[-1, 2, 3, -4], [2, -2], [3, -3]])
    vu = ncon([vcur, gauge.conjugate()],
              [[-1, 2, -3], [2, -2]])
    return Au, vu

# --- Here is the end of the                                   ---   |
# --- the rotationally-symmetric implmentation of the 2d HOTRG --- __|


def trunc_err_func(eigv, chi):
    """
    No need to take square since we work with M M'
    whose eigenvalues are themself square of singular values of M
    """
    res = np.sum(eigv[chi:]) / np.sum(eigv)
    return res
