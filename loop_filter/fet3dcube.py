#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : fet3dcube.py
# Author            : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
# Date              : 20.12.2023
# Last Modified Date: 20.12.2023
# Last Modified By  : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
"""
This module is almost the same as `./fet3d.py` except that
- we call the cube environment construction from `./env3dcube.py`
instead of `./env3d.py`

3D Generalization of
- Evenbly's full environment truncation (FET)
- Graph-Independent Local Truncations (GILT) by
Hauru, Delcamp, and Mizera

We generalize it to 3D, by approximating
a cube of 8 copies of 6-leg tensors.

Most environments are construction in `./env3dcube.py`
Some useful functions in `./env3d.py` and `./fet3d.py` are called
"""
from . import env3dcube, env3d, fet3d


def optimize_alls(A, sx, sy, sz, PsiPsi, epsilon=1e-10,
                  iter_max=100, display=True,
                  checkStep=20):
    """find sx, sy, sz that maximize FET fidelity

    Args:
        A (TensorCommon): 6-leg tensor
            It is the tensor located at (+++) corner
            of the block-tensor RG with shape
            A[x, xp, y, yp, z, zp]
        sx (TensorCommon): x-leg entanglement filtering
        sy (TensorCommon): y-leg entanglement filtering
        sz (TensorCommon): z-leg entanglement filtering
        PsiPsi (float):
            <ψ|ψ> the norm square of the wavefunction |ψ>
            to be optimized

    Kwargs:
        epsilon (float):
            For pseudo-inverse of γs
            Just for safety
            Not so crucial I think
        iter_max (int): maximal iteration for a single direction
        display (boolean):
            print out the error evoluation during the optimization

    Returns:
        new s matrices
            sx, sy, sz

    """
    leg_list = ["y", "z", "x"]
    errList = []
    doneLegs = {k: False for k in leg_list}
    for k in range(iter_max):
        for leg in leg_list:
            # permute the legs of main tensor and among s matrcies
            Ap, sxp, syp, szp = env3d.cubePermute(
                A, sx, sy, sz, direction=leg)
            # absorb sx, sz (prototype direction) into main tensor
            # and construct two doubleA tensors
            Axz = env3dcube.sOnA(Ap, sxp, szp)
            dbAp = env3dcube.contrInLeg(Axz, Ap.conj())
            dbAgm = env3dcube.contrInLeg(Axz, Axz.conj())
            snew, err = optimize_single(dbAp, dbAgm, syp, PsiPsi, epsilon)
            # update s
            if leg == "y":
                sy = snew * 1.0
            elif leg == "z":
                sz = snew * 1.0
            elif leg == "x":
                sx = snew * 1.0
            # record FET error
            errList.append(err)
            # check the change of FET error
            if (k % checkStep == 0 or k == iter_max - 1):
                # if the FET error is small,
                # the optimization for the leg is done
                doneLegs[leg] = (abs(errList[-1]) < epsilon)
                # if the change of FET error is small,
                # the optimization for the leg is done
                if k > 0:
                    doneLegs[leg] = (
                        abs((errList[-1] - errList[-1 - 3 * checkStep]) / (
                            errList[-1 - 3 * checkStep] + 1e-8)
                            ) < (0.01 * 3 * checkStep / 100)
                        or doneLegs[leg]
                    )
                # print out the change of FET error if needed
                if display:
                    print("Iteration {:d}.".format(k+1),
                          "FET Error is {:.3e} (leg {:s})".format(err, leg)
                          )
        if all(doneLegs.values()):
            break

    return sx, sy, sz, errList


def optimize_single(dbAp, dbAgm, sold, PsiPsi, epsilon):
    """Optimize a single s matrix using `fet3d.updateMats`
    A trick (used in Evenbly's TNR implementation) is used
    to ensure the FET error is not increasing

    Args:
        dbAp (TensorCommon): 6-leg tensor
            doubleA for P with sx, sz absorbed
        dbAgm (TensorCommon): 6-leg tensor
            doubleA for γ with sx, sz absorbed
        sold (TensorCommon): 2-leg tensor
        PsiPsi (float):
            <ψ|ψ> the norm square of the wavefunction |ψ>
            to be optimized
        epsilon (float):
            For pseudo-inverse of γs
            Just for safety
            Not so crucial I think

    Returns:
        snew (TensorCommon): 2-leg tensor
            the updated sy matrix
        err (float): FET error of snew

    """
    # old FET error
    Ps = env3dcube.dbA2P(dbAp, sold)
    Gammas = env3dcube.dbA2gm(dbAgm, sold)
    # calculate 1 - fidelity from P and γ environments
    errOld = fet3d.cubeFidelity(sold, Ps, Gammas, PsiPsi)[1]
    # no need for optimization if the error is already very small
    if errOld < epsilon:
        return sold, errOld
    # propose a candidate s
    stemp = fet3d.updateMats(Ps, Gammas, epsilon=epsilon)
    # normalized stemp (for the convex combination)
    stemp = stemp / stemp.norm()
    # try all the convex combination of old s and new s
    # make sure the approximation error go down
    # (This idea is taken from Evenbly's TNR codes)
    for p in range(11):
        snew = (1 - 0.1*p) * stemp + 0.1 * p * sold
        Pnew = env3dcube.dbA2P(dbAp, snew)
        Gammanew = env3dcube.dbA2gm(dbAgm, snew)
        # calculate 1 - fidelity from P and γ environments
        errNew = fet3d.cubeFidelity(snew, Pnew, Gammanew, PsiPsi)[1]
        # once the FET error reduces, we update s matrix
        if (errNew <= errOld) or (errNew < epsilon) or (p == 10):
            # make sure the returned s matrix is normalized
            snew = snew / snew.norm()
            break
    return snew, errNew


def fidelity(A, sx, sy, sz, PsiPsi):
    Ps = env3d.cubePs(A, sx, sy, sz, direction="y")
    Gammas = env3d.cubeGammas(A, sx, sy, sz, direction="y")
    f, err, PhiPhi = fet3d.cubeFidelity(sy, Ps, Gammas, PsiPsi)
    return f, err, PhiPhi
