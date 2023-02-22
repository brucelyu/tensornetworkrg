#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : signfix.py
# Author            : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
# Date              : 22.02.2023
# Last Modified Date: 22.02.2023
# Last Modified By  : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
"""
Functions for sign fixing for
6-leg tensors in 3d TNRG
"""
import numpy as np
from abeliantensors import TensorZ2


def findSigns(Aout, Aold, verbose=True):
    """find the sign ambiguities between two 6-leg tensors
    - It is a special case for more general gauge ambiguities;
      for svd-based tnrg on real tensors with reflection symmetry,
      we have only sign ambiguities
    - It is for Z2-symmetric tensors;
    - The chosen of sectors is the rule of thumb for 3D Ising.

    Args:
        Aout (TensorZ2): 6-leg coarse tensor
        Aold (TensorZ2): 6-leg old tensor

    Kwargs:
        verbose (boolean): print information

    Returns:
        Aout: after sign fixing
        signx, signy, signz

    """
    def applyZ2Signs(A, signx, signy, signz):
        """apply Z2-symmetric sign to A
        signx vector before converting to TensorZ2
           signx = [dx, dy]
        """
        Afix = A.copy()
        for key in Afix.sects.keys():
            x, xp, y, yp, z, zp = key
            Afix[key] = (
                Afix[key] *
                signx[x][:, None, None, None, None, None] *
                signx[xp][None, :, None, None, None, None] *
                signy[y][None, None, :, None, None, None] *
                signy[yp][None, None, None, :, None, None] *
                signz[z][None, None, None, None, :, None] *
                signz[zp][None, None, None, None, None, :]
             )
        return Afix

    errMsg = "Input tensors should be TensorZ2"
    assert (type(Aout)) is TensorZ2, errMsg
    assert (type(Aold)) is TensorZ2, errMsg
    errMessg = "The shape of A and Aold must be the same"
    assert Aout.shape == Aold.shape, errMessg
    # Choose different sectors to determine the sign
    # for charge=1 sector, the very first component
    # under the convention that dx1[0]=1, determine dy1[0], dz1[0]
    chosenSect1fory0 = (1, 0, 1, 0, 0, 0)
    dy10 = np.sign(Aout[chosenSect1fory0][0, 0, 0, 0, 0, 0] *
                   Aold[chosenSect1fory0][0, 0, 0, 0, 0, 0])
    chosenSect1forz0 = (1, 0, 0, 0, 1, 0)
    dz10 = np.sign(Aout[chosenSect1forz0][0, 0, 0, 0, 0, 0] *
                   Aold[chosenSect1forz0][0, 0, 0, 0, 0, 0])

    # for charge=0 sector
    chosenSect0x = (0, 0, 1, 0, 1, 0)
    dx0 = np.sign(dy10 * dz10 *
                  Aout[chosenSect0x][:, 0, 0, 0, 0, 0] *
                  Aold[chosenSect0x][:, 0, 0, 0, 0, 0]
                  )
    chosenSect0y = (1, 0, 0, 0, 1, 0)
    dy0 = np.sign(dz10 *
                  Aout[chosenSect0y][0, 0, :, 0, 0, 0] *
                  Aold[chosenSect0y][0, 0, :, 0, 0, 0]
                  )
    chosenSect0z = (1, 0, 1, 0, 0, 0)
    dz0 = np.sign(dy10 *
                  Aout[chosenSect0z][0, 0, 0, 0, :, 0] *
                  Aold[chosenSect0z][0, 0, 0, 0, :, 0])
    # determine other components in charge=1 sector
    chosenSect1x = (1, 1, 1, 0, 1, 0)
    dx1 = np.sign(dy10 * dz10 *
                  Aout[chosenSect1x][:, 0, 0, 0, 0, 0] *
                  Aold[chosenSect1x][:, 0, 0, 0, 0, 0])
    chosenSect1y = (1, 0, 1, 1, 1, 0)
    dy1 = np.sign(dy10 * dz10 *
                  Aout[chosenSect1y][0, 0, :, 0, 0, 0] *
                  Aold[chosenSect1y][0, 0, :, 0, 0, 0])
    chosenSect1z = (1, 0, 1, 0, 1, 1)
    dz1 = np.sign(dy10 * dz10 *
                  Aout[chosenSect1z][0, 0, 0, 0, :, 0] *
                  Aold[chosenSect1z][0, 0, 0, 0, :, 0])
    # print connectness
    if verbose:
        printConnectness(Aold)
    # assemble sign vectors according to its charge
    signx = [dx0, dx1]
    signy = [dy0, dy1]
    signz = [dz0, dz1]
    # fix the sign of Aout
    Afix = applyZ2Signs(Aout, signx, signy, signz)
    # Covert signx, signy, signz to Z2-symmetric tensors
    qhape = [[0, 1]]
    dxShape = [len(sect) for sect in signx]
    dyShape = [len(sect) for sect in signy]
    dzShape = [len(sect) for sect in signz]

    signx = TensorZ2.from_ndarray(
        np.concatenate(signx), dirs=[1], invar=False,
        shape=[dxShape], qhape=qhape
    )
    signy = TensorZ2.from_ndarray(
        np.concatenate(signy), dirs=[1], invar=False,
        shape=[dyShape], qhape=qhape
    )
    signz = TensorZ2.from_ndarray(
        np.concatenate(signz), dirs=[1], invar=False,
        shape=[dzShape], qhape=qhape
    )
    return Afix, signx, signy, signz


def printConnectness(Ain):
    """reliabily of the sign fixing procedure
    - For a matrix M, if |M[i, j]| not vanishing, we say
    index i and j are connected.
    - In order to reliably determine the sign, we assume that
    matrix elements should be well connect , we print out the
    connectness of the matrix elements below as debug message
    """
    # normalized A
    A = Ain / Ain.norm()
    # for charge=1 sector, the very first component
    chosenSect1fory0 = (1, 0, 1, 0, 0, 0)
    chosenSect1forz0 = (1, 0, 0, 0, 1, 0)
    # for charge=0 sector
    chosenSect0x = (0, 0, 1, 0, 1, 0)
    chosenSect0y = (1, 0, 0, 0, 1, 0)
    chosenSect0z = (1, 0, 1, 0, 0, 0)
    # for charge=1 sector
    chosenSect1x = (1, 1, 1, 0, 1, 0)
    chosenSect1y = (1, 0, 1, 1, 1, 0)
    chosenSect1z = (1, 0, 1, 0, 1, 1)
    # print connectness
    print("For dy10,")
    with np.printoptions(precision=2, suppress=True):
        print(np.log10(np.abs(
            A[chosenSect1fory0][0, 0, 0, 0, 0, 0]
        )))
    print("For dz10,")
    with np.printoptions(precision=2, suppress=True):
        print(np.log10(np.abs(
            A[chosenSect1forz0][0, 0, 0, 0, 0, 0]
        )))
    print("-----")
    print("For dx0,")
    with np.printoptions(precision=2, suppress=True):
        print(np.log10(np.abs(
            A[chosenSect0x][:, 0, 0, 0, 0, 0]
        )))
    print("For dy0,")
    with np.printoptions(precision=2, suppress=True):
        print(np.log10(np.abs(
            A[chosenSect0y][0, 0, :, 0, 0, 0]
        )))
    print("For dz0,")
    with np.printoptions(precision=2, suppress=True):
        print(np.log10(np.abs(
            A[chosenSect0z][0, 0, 0, 0, :, 0]
        )))
    print("-----")
    print("For dx1,")
    with np.printoptions(precision=2, suppress=True):
        print(np.log10(np.abs(
            A[chosenSect1x][:, 0, 0, 0, 0, 0]
        )))
    print("For dy1,")
    with np.printoptions(precision=2, suppress=True):
        print(np.log10(np.abs(
            A[chosenSect1y][0, 0, :, 0, 0, 0]
        )))
    print("For dz1,")
    with np.printoptions(precision=2, suppress=True):
        print(np.log10(np.abs(
            A[chosenSect1z][0, 0, 0, 0, :, 0]
        )))
