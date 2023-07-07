#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : toymodels.py
# Author            : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
# Date              : 24.10.2022
# Last Modified Date: 24.10.2022
# Last Modified By  : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
"""
Construct various toy models representing
the short-range correlations in tensor network.
"""
from ncon import ncon
from abeliantensors import Tensor, TensorZ2


# I. For 2D: CDL structure:
# four copies of 2-leg corner tensors.
def cdlten(cornerChi=2, isZ2=False):
    """construct the corner-double-line tensor
          -3 -4
           ||
      -1 --oo-- -5
      -2 --oo-- -6
           ||
         -7 -8
    Kwargs:
        cornerChi (int):
        bond dimension of the corner matrix

        isZ2 (boolean):
        whether to use Z2-symmetryc tensor

    Returns:
        cdl (TensorCommon): the CDL 4-leg tensor
        loopNumber (?): the loop represented by the CDL
        cmat (TensorCommon): the corner matrix

    """
    if not isZ2:
        cmat = Tensor.random([cornerChi]*2)
    else:
        n = [cornerChi // 2, cornerChi - cornerChi // 2]
        shape = [n] * 2
        qhape = [[0, 1]] * 2
        dirs = [1, -1]
        cmat = TensorZ2.random(shape=shape, qhape=qhape, dirs=dirs)
    # make corner matrix symmetric
    cmat = 0.5 * (cmat + cmat.transpose().conj())
    # construct the CDL structure
    cdl = ncon([cmat, cmat.conj(), cmat, cmat.conj()],
               [[-1, -3], [-5, -4], [-6, -8], [-2, -7]])
    cdl = cdl.join_indices([0, 1], [2, 3], [4, 5], [6, 7],
                           dirs=[1, 1, -1, -1])
    # the loop represented by the CDL structure
    loopNumber = ncon([cmat, cmat.conj(), cmat, cmat.conj()],
                      [[1, 2], [3, 2], [3, 4], [1, 4]])
    return cdl, loopNumber, cmat


# II. For 3D
# II.1 Edge-double-line (EDL) structure:
# 12 copies of 2-leg edge tensors.
def edlten(cornerChi=2, singleYSide=True):
    """
    The leg order convention:
    edl[x, x', y, y', z, z']
    if singleYside=True, we only use 4 copies of edge matrix
    that connect to y leg
    """
    # create edge matrix
    ematx, ematy, ematz = [Tensor.random([cornerChi]*2) for i in range(3)]
    EYE11 = Tensor.eye(1)
    # make corner matrix symmetric
    ematx = 0.5 * (ematx + ematx.transpose().conj())
    ematy = 0.5 * (ematy + ematy.transpose().conj())
    ematz = 0.5 * (ematz + ematz.transpose().conj())

    # construct the EDL structure
    # A "direct product" of 12 edge matrix
    # In each plane, the order of tensors are according to
    # right-hand rule along the normal vector of the face
    if singleYSide:
        tenList = [
            ematx, EYE11.conj(), EYE11, ematx.conj(),  # x-plane
            EYE11, EYE11, EYE11, EYE11,  # y-plane
            ematz, ematz.conj(), EYE11, EYE11   # z-plane
        ]
    else:
        tenList = [
            ematx, ematx.conj(), ematx, ematx.conj(),  # x-plane
            ematy, ematy.conj(), ematy, ematy.conj(),  # y-plane
            ematz, ematz.conj(), ematz, ematz.conj()   # z-plane
        ]
    edl = ncon(
        tenList,
        [[-9, -18], [-13, -20], [-15, -24], [-11, -22],
         [-2, -17], [-4, -21], [-8, -23], [-6, -19],
         [-1, -10], [-5, -12], [-7, -16], [-3, -14],
         ]
    )
    edl = edl.join_indices([0, 1, 2, 3], [4, 5, 6, 7],
                           [8, 9, 10, 11], [12, 13, 14, 15],
                           [16, 17, 18, 19], [20, 21, 22, 23],
                           dirs=[1, -1, 1, -1, 1, -1])
    return edl, ematx, ematy, ematz


# II.2 Corner-triple-line (CTL) structure:
# 8 copies of 3-leg edge tensors.
def ctlten(cornerChi=2, singleYSide=True):
    """
    I'm not very sure about the Z2 version now...
    See the notes for the tensor network diagrams...
    """
    # create 3D corner matrix
    cmat = Tensor.random([cornerChi]*3)
    EYE111 = Tensor.ones([1]*3)
    # construct the CTL structure
    # A "direct product" of 8 copies of 3-leg coner matrix
    if singleYSide:
        tenList = [
            cmat, EYE111.conj(), EYE111, cmat.conj(),
            cmat.conj(), EYE111, EYE111.conj(), cmat
        ]
    else:
        tenList = [
            cmat, cmat.conj(), cmat, cmat.conj(),
            cmat.conj(), cmat, cmat.conj(), cmat
        ]
    ctl = ncon(
        tenList,
        [[-1, -9, -17], [-2, -13, -20],
         [-3, -14, -24], [-4, -10, -21],
         #
         [-5, -12, -18], [-6, -16, -19],
         [-7, -15, -23], [-8, -11, -22]
         ]
    )
    ctl = ctl.join_indices([0, 1, 2, 3], [4, 5, 6, 7],
                           [8, 9, 10, 11], [12, 13, 14, 15],
                           [16, 17, 18, 19], [20, 21, 22, 23],
                           dirs=[1, -1, 1, -1, 1, -1])
    return ctl, cmat
