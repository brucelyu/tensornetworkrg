#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : fet3dloop.py
# Author            : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
# Date              : 04.01.2024
# Last Modified Date: 04.01.2024
# Last Modified By  : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
"""
3D Generalization of
- Evenbly's full environment truncation (FET)
- Graph-Independent Local Truncations (GILT) by
Hauru, Delcamp, and Mizera

We generalize it to 3D, by approximating
a plaquette of 4 copies of 6-leg tensors.
All the relevant environments are construction in `./env3dloop.py`

The initialization and FET is implemented with the following
3D HOTRG-like block-tensor scheme in mind:
(CG stands for coarse graining and LF for loop filtering)
- CG1: z-direction so x and y legs have increased bond dimensions
χm and χin for outer and inner orientations.
- LF1: z-loop filtering of the outer x and y legs to truncate
the bond dimension χm --> χs
- CG2: y-direction. Now the outer x leg is the outermost leg
with bond dimension changes χs --> χs^2 --> χ;
but the z leg has increased bond dimension χm
- LF2 : y-loop filtering of the outer z leg to truncate
the bond dimension χm --> χs.
The outmost x leg is left untouched in this step.
- CG3: x-direction. Both y and z legs becomes outermost legs
with bond dimension changes χs --> χs^2 --> χ.
_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
"""
from . import env3dloop, fet3d
from ncon import ncon


def init_zloopm(A, chis, chienv, epsilon):
    """Initialization of m matrices for z-loop
    It is a (mx, my) pair

    Args:
        A (TensorCommon): 6-leg tensor
            It is the tensor located at
            - (x+y+)-position for z-loop filtering after z corase graining
            It has shape
            - A[x, xp, y, yp, z, zp]
        chis (int): squeezed bond dimension of m matrices
            such that m is a χ-by-χs matrix.
            It is the bond dimension for
            the truncated SVD of low-rank Lr matrix.
        chienv (int): for pseudo-inverse of leg environment
            To avoid the trivial solution of Lr matrix,
            we should truncate the singular values
            when taking the pseudo-inverse of leg environment
        epsilon (float): for psudo-inverse of leg environment
            if further truncation to χ < chienv has
            an error < epsilon, then we truncate to
            the smallest possible χ.

    Returns: loop-filtering matrices pair
        mx (TensorCommon): 2-leg tensor
        my (TensorCommon): 2-leg tensor

    """
    # Construct the γ environment for initialization of m matrices
    Gammay = env3dloop.loopGamma(A)
    # (Swap xy leg)
    A4x = env3dloop.swapxy(A, 1, 1)[0]
    Gammax = env3dloop.loopGamma(A4x)
    # Find initial low-rank matrix Lr and split it to get m matrix
    my, Lry = fet3d.init_s_gilt(Gammay, chis, chienv, epsilon,
                                init_soft=False)
    mx, Lrx = fet3d.init_s_gilt(Gammax, chis, chienv, epsilon,
                                init_soft=False)
    return mx, my, Lrx, Lry, Gammay


def init_yloopm(A, chis, chienv, epsilon):
    """
    Similar to the above `init_zloopm` function but
    the input tensor A located at
        - (x+)-position for y-loop filtering after z+y corase graining
    It has shape
        - A[x, xp, y, yp, z, zp]
    """
    # Construct the γ environment for initialization of m matrices
    # (rotate A leg: xyz --> zxy)
    Ar = A.transpose([4, 5, 0, 1, 2, 3])
    # (Swap zx leg)
    A4z = env3dloop.swapxy(Ar, 1, 1)[0]
    Gammaz = env3dloop.loopGamma(A4z)
    # Find initial low-rank matrix Lr and split it to get m matrix
    mz, Lrz = fet3d.init_s_gilt(Gammaz, chis, chienv, epsilon,
                                init_soft=False)
    return mz, Lrz, Gammaz


# ------------------------------------------------\\
# Optimize y -> x -> y -> x -> ... in a cyclic way
def optimize_zloop_v0(A, mx, my, PsiPsi, epsilon=1e-10,
                   iter_max=100, display=True,
                   swapList=[False, True], checkStep=10):
    """find mx, my matricees that maximize FET fidelity
    z-loop is approximated here

    Args:
        A (TensorCommon): 6-leg tensor
            It is the tensor located at
            - (x+y+)-position for z-loop filtering after z corase graining
            It has shape
            - A[x, xp, y, yp, z, zp]
        mx (TensorCommon): 2-leg tensor
        my (TensorCommon): 2-leg tensor
        PsiPsi (float):
            <ψ|ψ> the norm square of the wavefunction |ψ>
            to be optimized

    Kwargs:
        epsilon (float):
            For pseudo-inverse of γs
            Just for safety. Not so crucial I think
            I set it to be the same as coarse graining one
        iter_max (int): maximal iteration for a single direction
        display (boolean):
            print out the error evoluation during the optimization

    Returns: new m matrices, mx, my

    """
    errList = []
    legDic = {False: "y", True: "x"}
    doneLegs = {False: False, True: False}
    nleg = len(swapList)
    for k in range(iter_max):
        for isSwap in swapList:
            Ap, mxp, myp = env3dloop.swapxy(A, mx, my, isSwap)
            # This is the bottleneck of the computational cost
            # in this loop-filtering process.
            # The cost is χ^8
            dbA = env3dloop.contrInLeg(Ap, Ap.conj())
            # absorb the constant m matrix (mxp) into dbA tensor
            dbAp, dbAgm = env3dloop.mxAssemble(dbA, mxp)
            # using FET to optimize the other m matrix (myp)
            mnew, err = optimize_single(dbAp, dbAgm, myp, PsiPsi, epsilon)[:2]
            # update the other m matrix
            if isSwap:
                mx = mnew * 1.0
            else:
                my = mnew * 1.0
            # record FET error
            errList.append(err)
            # check the change of FET error
            if (k % checkStep == 0 or k == iter_max - 1):
                # if the FET error is small,
                # the optimization for the leg is done
                doneLegs[isSwap] = (abs(errList[-1]) < epsilon)
                # if the change of FET error is small,
                # the optimization for the leg is done
                if k > 0:
                    smallChange = (
                        abs((errList[-1] - errList[-1 - nleg*checkStep]) / (
                            errList[-1 - nleg * checkStep] + epsilon)
                            ) < (0.01 * nleg * checkStep / 100)
                    )
                    doneLegs[isSwap] = (
                        smallChange or doneLegs[isSwap]
                    )
                # print out the change of FET error if needed
                if display:
                    print("Iteration {:d}.".format(k+1),
                          "FET Error (loop) is {:.3e} (leg {:s})".format(
                              err, legDic[isSwap])
                          )
        if all(doneLegs.values()):
            break
    return mx, my, errList


def optimize_yloop_v0(A, mz, PsiPsi, epsilon=1e-10,
                   iter_max=100, display=True,
                   checkStep=10):
    """find mz matrices that maximize FET fidelity
    y-loop is approximated here
    Similar to the above `init_zloopm` function but
    the input tensor A located at
        - (x+)-position for y-loop filtering after z+y corase graining
    Also, there is only one m matrx to be inserted in the y-loop

    Returns: mz

    """
    errList = []
    doneLeg = False
    # rotate the tensor leg: xyz --> zxy
    Ap = A.transpose([4, 5, 0, 1, 2, 3])
    # (Swap zx leg)
    Ap = env3dloop.swapxy(Ap, 1, 1)[0]
    # number of leg to be optimized
    nleg = 1
    for k in range(iter_max):
        dbA = env3dloop.contrInLeg(Ap, Ap.conj())
        # using FET to optimize the mz matrix
        mz, err = optimize_single(dbA, dbA, mz, PsiPsi, epsilon)[:2]
        # record FET error
        errList.append(err)
        # check the change of FET error
        if (k % checkStep == 0 or k == iter_max - 1):
            # if the FET error is small,
            # the optimization for the leg is done
            doneLeg = (abs(errList[-1]) < epsilon)
            # if the change of FET error is small,
            # the optimization for the leg is done
            if k > 0:
                smallChange = (
                    abs((errList[-1] - errList[-1 - nleg*checkStep]) / (
                        errList[-1 - nleg * checkStep] + epsilon)
                        ) < (0.01 * nleg * checkStep / 100)
                )
                doneLeg = (
                    smallChange or doneLeg
                )
            if display:
                print("Iteration {:d}.".format(k+1),
                      "FET Error (y-loop) is {:.3e} (leg z)".format(
                          err)
                      )
        if doneLeg:
            break
    return mz, errList
# ------------------------------------------------//


# ------------------------------------------------\\
# Optimize round by round:
# for each round, a single leg is updated a few times
def optimize_zloop(
    A, mx, my, PsiPsi, epsilon=1e-10,
    iter_max=20, n_round=2, display=True
):
    """find mx, my matricees that maximize FET fidelity
    """
    leg_list = ["y", "x"]
    legSwapDic = {"y": False, "x": True}
    errList = []
    for m in range(n_round):
        doneLegs = {k: False for k in leg_list}
        for leg in leg_list:
            Ap, mxp, myp = env3dloop.swapxy(
                A, mx, my, legSwapDic[leg]
            )
            # This is the bottleneck of the computational cost
            # in this loop-filtering process.
            # The cost is χ^8
            dbA = env3dloop.contrInLeg(Ap, Ap.conj())
            # absorb the constant m matrix (mxp) into dbA tensor
            dbAp, dbAgm = env3dloop.mxAssemble(dbA, mxp)
            # using FET to optimize the other m matrix (myp)
            mnew, err = opt_1s(
                dbAp, dbAgm, myp, PsiPsi, epsilon, iter_max
            )
            if leg == "y":
                my = mnew * 1.0
            elif leg == "x":
                mx = mnew * 1.0
            # record error
            errList.append(err)
            # print change of FET error in a round
            if display and (m % 5 == 0 or m == n_round - 1):
                print("This is round {:d} for leg {:s}:".format(m+1, leg),
                      "FET-loop Error {:.3e} ---> {:.3e}".format(
                          err[1], err[-1]),
                      "(in {:d} iterations)".format(len(err) - 1)
                      )
            # if the change of FET error is small, or the FET itself is small,
            # the optimization for the leg is done
            doneLegs[leg] = (
                abs((err[-1] - err[1]) / (err[1] + epsilon)
                    ) < (0.01 * iter_max/100)
            ) or (abs(err[-1]) < epsilon)

        # Optimization for all legs done!
        if all(doneLegs.values()):
            if display:
                print("  Final round {:d} for leg {:s}:".format(m+1, leg),
                      "FET-loop Error {:.3e} ---> {:.3e}".format(
                          err[1], err[-1]),
                      "(in {:d} iterations)".format(len(err) - 1)
                      )
            break
    return mx, my, errList


def optimize_yloop(
    A, mz, PsiPsi, epsilon=1e-10,
    iter_max=20, n_round=2, display=True
):
    """find mz matrices that maximize FET fidelity
    """
    errList = []
    # rotate the tensor leg: xyz --> zxy
    Ap = A.transpose([4, 5, 0, 1, 2, 3])
    # (Swap zx leg)
    Ap = env3dloop.swapxy(Ap, 1, 1)[0]
    # construct double A tensor
    dbA = env3dloop.contrInLeg(Ap, Ap.conj())
    for m in range(n_round):
        doneLeg = False
        # using FET to optimize the mz matrix
        mz, err = opt_1s(
            dbA, dbA, mz, PsiPsi, epsilon, iter_max
        )
        # record error
        errList.append(err)
        if display and (m % 5 == 0 or m == n_round - 1):
            print("This is round {:d} for leg {:s}:".format(m+1, "z"),
                  "FET-loop Error {:.3e} ---> {:.3e}".format(
                      err[1], err[-1]),
                  "(in {:d} iterations)".format(len(err) - 1)
                  )
        # if the change of FET error is small, or the FET itself is small,
        # the optimization for the leg is done
        doneLeg = (
            abs((err[-1] - err[1]) / (err[1] + epsilon)
                ) < (0.01 * iter_max/100)
        ) or (abs(err[-1]) < epsilon)

        # Optimization for the leg done!
        if doneLeg:
            if display:
                print("  Final round {:d} for leg {:s}:".format(m+1, "z"),
                      "FET-loop Error {:.3e} ---> {:.3e}".format(
                          err[1], err[-1]),
                      "(in {:d} iterations)".format(len(err) - 1)
                      )
            break
    return mz, errList
# ------------------------------------------------//


def opt_1s(dbAp, dbAgm, sold, PsiPsi,
           epsilon=1e-10, iter_max=20):
    """
    This is the same as `.fet3d.opt_1s` function
    """
    snew = sold * 1.0
    err = []
    for k in range(iter_max):
        # update s matrix
        snew, errNew, errOld = optimize_single(
            dbAp, dbAgm, snew, PsiPsi, epsilon
        )
        # record FET error
        if k == 0:
            err.append(errOld)
            err.append(errNew)
        else:
            err.append(errNew)
        # if FET error is very small, stop iteration
        if errNew < epsilon:
            break
    return snew, err


def optimize_single(dbAp, dbAgm, mold, PsiPsi, epsilon):
    """Optimize a single m matrix using `fet3d.updateMats`

    Args:
        dbAp (TensorCommon): 4-leg tensor
            doubleA for P with mx absorbed
        dbAgm (TensorCommon): 4-leg tensor
            doubleA for γ with mx absorbed
        mold (TensorCommon): 2-leg tensor
            filtering matrix to be optimized
        PsiPsi (float):
            <ψ|ψ> the norm square of the wavefunction |ψ>
            to be optimized
        epsilon (float):
            For pseudo-inverse of γs
            Just for safety. Not so crucial I think

    Returns:
        mnew (TensorCommon): 2-leg tensor
            the updated sy matrix
        err (float): FET error of snew

    """
    # connstruct P and γ enviornment by absorbing `mold`
    Pm, Gammam = env3dloop.dbA2FETenv(dbAp, dbAgm, mold)
    # old FET error from P and γ environments
    errOld = fet3d.cubeFidelity(mold, Pm, Gammam, PsiPsi)[1]
    # no need for optimization if the error is already very small
    if errOld < epsilon:
        return mold, errOld, errOld
    # propose a candidate s
    mtemp = fet3d.updateMats(Pm, Gammam, epsilon=epsilon)
    # normalized stemp (for the convex combination)
    mtemp = mtemp / mtemp.norm()
    # try all the convex combination of old s and new s
    # make sure the approximation error go down
    # (This idea is taken from Evenbly's TNR codes)
    for p in range(11):
        mnew = (1 - 0.1*p) * mtemp + 0.1 * p * mold
        # connstruct P and γ enviornment by absorbing `mnew`
        PmN, GammamN = env3dloop.dbA2FETenv(dbAp, dbAgm, mnew)
        # calculate 1 - fidelity from new P and γ environments
        errNew = fet3d.cubeFidelity(mnew, PmN, GammamN, PsiPsi)[1]
        # once the FET error reduces, we update s matrix
        if (errNew <= errOld) or (errNew < epsilon) or (p == 10):
            # make sure the returned s matrix is normalized
            mnew = mnew / mnew.norm()
            break
    return mnew, errNew, errOld


# For absorbing mx, my, mz matrices into the intermediate main tensors
# A1. For z-loop filtering: mx and my
def absb_mloopz(Az, mx, my):
    """
    Apply mx, my matricx to
        Az: intermediate tensor after the first z-collapse
    """
    Azm = ncon(
        [Az, mx, my],
        [[1, -2, 2, -4, -5, -6], [1, -1], [2, -3]]
    )
    return Azm


# A2. For y-loop filtering: mz
def absb_mloopy(Azy, mz):
    """
    Apply mx, my matricx to
        Az: intermediate tensor after the first z-collapse
    """
    Azym = ncon(
        [Azy, mz, mz.conj()],
        [[-1, -2, -3, -4, 1, 2], [1, -5], [2, -6]]
    )
    return Azym


# Fidelity of z-loop and y-loop
def fidelityLPZ(Az, mx, my, PsiPsi):
    dbA = env3dloop.contrInLeg(Az, Az.conj())
    dbAp, dbAgm = env3dloop.mxAssemble(dbA, mx)
    Pmy, Gammamy = env3dloop.dbA2FETenv(dbAp, dbAgm, my)
    f, err, PhiPhi = fet3d.cubeFidelity(my, Pmy, Gammamy, PsiPsi)
    return f, err, PhiPhi


def fidelityLPY(Azy, mz, PsiPsi):
    # rotate the tensor leg: xyz --> zxy
    Azyr = Azy.transpose([4, 5, 0, 1, 2, 3])
    # swap (zx) legs
    Azyr = env3dloop.swapxy(Azyr, 1, 1)[0]
    dbA = env3dloop.contrInLeg(Azyr, Azyr.conj())
    Pmz, Gammamz = env3dloop.dbA2FETenv(dbA, dbA, mz)
    f, err, PhiPhi = fet3d.cubeFidelity(mz, Pmz, Gammamz, PsiPsi)
    return f, err, PhiPhi
