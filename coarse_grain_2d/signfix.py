#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : signfix.py
# Author            : Xinliang (Bruce) Lyu <xlyu@ihes.fr>
# Date              : 25.08.2025
# Last Modified Date: 25.08.2025
# Last Modified By  : Xinliang (Bruce) Lyu <xlyu@ihes.fr>
"""
Functions for sign fixing for
4-leg tensors in 2d TNRG
"""
import numpy as np
from abeliantensors import TensorZ2


def findSigns(A, Aold):
    def applyZ2Signs(A, signx, signy):
        """apply Z2-symmetric sign to A
        signx vector before converting to TensorZ2
           signx = [dx, dy]
        """
        Afix = A.copy()
        for key in Afix.sects.keys():
            x, y, xp, yp = key
            Afix[key] = (
                Afix[key] *
                signx[x][:, None, None, None] *
                signy[y][None, :, None, None] *
                signx[xp][None, None, :, None] *
                signy[yp][None, None, None, :]
            )
        return Afix

    errMsg = "Input tensors should be TensorZ2"
    assert (type(A)) is TensorZ2, errMsg
    assert (type(Aold)) is TensorZ2, errMsg
    if not A.shape == Aold.shape:
        return None

    chosenSector = (1, 1, 0, 0)
    # We set (dx0)_0 = (dy0)_0 = (dx1)_0 = 1
    # first determine dy1
    dy1 = np.sign(A[chosenSector][0, :, 0, 0] *
                  Aold[chosenSector][0, :, 0, 0])
    # determine dx0
    dx0 = np.sign(A[chosenSector][0, 0, :, 0] *
                  Aold[chosenSector][0, 0, :, 0]) * dy1[0]
    # determine dy0
    dy0 = np.sign(A[chosenSector][0, 0, 0, :] *
                  Aold[chosenSector][0, 0, 0, :]) * dy1[0]
    # dtermine dx1
    dx1 = np.sign(A[chosenSector][:, 0, 0, 0] *
                  Aold[chosenSector][:, 0, 0, 0]) * dy1[0]

    printConnectness(A)

    # assemble sign vectors according to its charge
    signx = [dx0, dx1]
    signy = [dy0, dy1]

    # fix the sign of A
    Afix = applyZ2Signs(A, signx, signy)

    # Covert signx, signy, signz to Z2-symmetric tensors
    qhape = [[0, 1]]
    dxShape = [len(sect) for sect in signx]
    dyShape = [len(sect) for sect in signy]
    signx = TensorZ2.from_ndarray(
        np.concatenate(signx), dirs=[1], invar=False,
        shape=[dxShape], qhape=qhape
    )
    signy = TensorZ2.from_ndarray(
        np.concatenate(signy), dirs=[1], invar=False,
        shape=[dyShape], qhape=qhape
    )
    return Afix, signx, signy


def printConnectness(A):
    """reliabily of the sign fixing procedure
    - For a matrix M, if |M[i, j]| not vanishing, we say
    index i and j are connected.
     - In order to reliably determine the sign, we assume that
    matrix elements should be well connect , we print out the
    connectness of the matrix elements below as debug message
    """
    # normalized A
    A = A / A.norm()
    # chosen sector
    chosenSect0x = (1, 1, 0, 0)

    # print connectness
    print("For dx0,")
    with np.printoptions(precision=2, suppress=True):
        print(np.log10(np.abs(
            A[chosenSect0x][0, 0, :, 0]
        )))
        print("For dy0,")
    with np.printoptions(precision=2, suppress=True):
        print(np.log10(np.abs(
            A[chosenSect0x][0, 0, 0, :]
        )))
    print("-----")
    print("For dx1,")
    with np.printoptions(precision=2, suppress=True):
        print(np.log10(np.abs(
            A[chosenSect0x][:, 0, 0, 0]
        )))
        print("For dy1,")
    with np.printoptions(precision=2, suppress=True):
        print(np.log10(np.abs(
            A[chosenSect0x][0, :, 0, 0]
        )))


def signOnp(px, py, signx, signy):
    px = px.multiply_diag(signx, axis=2, direction='r')
    py = py.multiply_diag(signy, axis=2, direction='r')
    return px, py
