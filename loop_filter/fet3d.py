#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : fet3d.py
# Author            : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
# Date              : 29.06.2023
# Last Modified Date: 29.06.2023
# Last Modified By  : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
"""
Generalization of Evenbly's full environment truncation (FET) in 3D.
- The original paper of Evenbly is explained in 2D, see
Title: Gauge fixing, canonical forms, and optimal truncations
in tensor networks with closed loops
Author: Glen Evenbly
Phys. Rev. B 98, 085155 – Published 31 August 2018
url: https://journals.aps.org/prb/abstract/10.1103/PhysRevB.98.085155

- We generalize it to 3D, by approximating a cube
of 8 copies of 6-leg tensors.
All the relevant environments are construction in `./env3d.py`
"""
from .. import u1ten
from ncon import ncon
from . import env3d


# I. For initialization of s matrix
def findLr(Gamma, epsilon=1e-10,
           soft=False, chiCut=None):
    """
    Determine the low-rank matrix a la FET.
    This version is almost the same GILT without recursion.
    We will use it to initialize the low-rank matrix
    See function `.fet.findLR` for more details.
    """
    Gamma_pinv = u1ten.pinv(Gamma, eps_mach=epsilon,
                            soft=soft, chiCut=chiCut)
    Lr = ncon([Gamma_pinv, Gamma], [[-1, -2, 1, 2], [1, 2, 3, 3]])
    return Lr


def init_s_gilt(Gamma, chis, chienv, epsilon_init,
                init_soft=False):
    """
    Initialize s matrix using `findLr`
    """
    # use baby FET to find low-rank matrix
    Lr = findLr(Gamma, epsilon=epsilon_init,
                soft=init_soft, chiCut=chienv)
    # split and truncate Lr to find s
    s = Lr.split([0], [1], chis=[k+1 for k in range(chis)],
                 eps=epsilon_init)[0]
    # make the direction of s as [-1, 1]
    s = s.flip_dir(1)
    return s, Lr


def init_alls(A, chis, chienv, epsilon):
    """Initialization of s matrices in 3 directions

    Args:
        A (TensorCommon): 6-leg tensor
            It is the tensor located at (+++) corner
            of the block-tensor RG with shape
            A[x, xp, y, yp, z, zp]
        chis (int): squeezed bond dimension of s matrices
            such that s is a χ-by-χs matrix.
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

    Returns: sx, sy, sz
        s matrices in 3 directions

    """
    # Construct environments for initialization of s matrices
    # all have cost O(χ^12)
    Gammay = env3d.cubeGamma(A, direction="y")
    Gammaz = env3d.cubeGamma(A, direction="z")
    Gammax = env3d.cubeGamma(A, direction="x")
    # find initial Lr and s
    sy, Lry = init_s_gilt(Gammay, chis, chienv, epsilon,
                          init_soft=False)
    sz, Lrz = init_s_gilt(Gammaz, chis, chienv, epsilon,
                          init_soft=False)
    sx, Lrx = init_s_gilt(Gammax, chis, chienv, epsilon,
                          init_soft=False)
    return sx, sy, sz, Lrx, Lry, Lrz, Gammay


# II. For optimization of s matrix
# II.1 Single update step for s matrix
def updateMats(Ps, Gammas, epsilon=1e-10):
    """update squzeezer matrix s
    following Evenbly's FET proposal
    """
    Gammas_pinv = u1ten.pinv(Gammas, eps_mach=epsilon)
    sp = ncon([Gammas_pinv, Ps], [[-1, -2, 1, 2], [1, 2]])
    return sp


# II.2 Add a trick by Evenbly for a single update
def linUpdateMats(octuP, octuGamma, sold, PsiPsi,
                  epsilon=1e-10):
    """Based on `updateMats`, but adding one more trick
    Still single update

    Args:
        octuP (TensorCommon):
            8-leg octupole tensor for Ps
        octuGamma (TensorCommon):
            8-leg octupole tensor for γs
        sold (TensorCommon): old s matrix
        PsiPsi (float): <ψ|ψ>
            For calculating FET error

    Kwargs:
        epsilon (float):
            For pseudo-inverse of γs
            Just for safety
            Not so crucial I think

    Returns: new s matrix, sM

    """
    # old FET error
    Ps, Gammas = env3d.secondContr(octuP, octuGamma, sold)
    errOld = cubeFidelity(sold, Ps, Gammas, PsiPsi)[1]
    # propose a candidate s
    stemp = updateMats(Ps, Gammas, epsilon=epsilon)
    # normalized stemp (for the convex combination)
    stemp = stemp / stemp.norm()
    # try all the convex combination of old s and new s
    # make sure the approximation error go down
    for p in range(11):
        snew = (1 - 0.1*p) * stemp + 0.1 * p * sold
        # Pnew, Gammanew = env3d.secondContr(octuP, octuGamma, snew)
        # errNew = cubeFidelity(snew, Pnew, Gammanew, PsiPsi)[1]
        errNew = cubeError(octuP, octuGamma, snew, PsiPsi)
        # once the FET error reduces, we update s matrix
        if (errNew <= errOld) or (errNew < epsilon) or (p == 10):
            sM = snew / snew.norm()
            break
    return sM, errNew, errOld


# II.3 Iterative optimization of a single direction
def opt_1s(octuP, octuGamma, sold, PsiPsi,
           epsilon=1e-10, iter_max=20):
    snew = sold * 1.0
    err = []
    for k in range(iter_max):
        # update s matrix
        snew, errNew, errOld = linUpdateMats(
            octuP, octuGamma, snew, PsiPsi,
            epsilon=epsilon)
        # record FET error
        if k == 0:
            err.append(errOld)
            err.append(errNew)
        else:
            err.append(errNew)
        if errNew < epsilon:
            break
    return snew, err


# II.4 Put all optimization functions together
def opt_alls(A, sx, sy, sz, PsiPsi, epsilon=1e-10,
             iter_max=20, n_round=2, display=True):
    """find sx, sy, sz to maximaize FET fidelity

    Args:
        A (TensorCommon): 6-leg tensor
            It is the tensor located at (+++) corner
            of the block-tensor RG with shape
            A[x, xp, y, yp, z, zp]
        sx (TensorCommon): x-leg entanglement filtering
        sy (TensorCommon): y-leg entanglement filtering
        sz (TensorCommon): z-leg entanglement filtering

    Kwargs:
        iter_max (int): maximal iteration for a single direction
        n_round (int): number of y-z-x rounds

    Returns:
        new s matrices
            sx, sy, sz
        Error changes during optimization
            errList
    """
    leg_list = ["y", "z", "x"]
    errList = []
    for m in range(n_round):
        doneLegs = {k: False for k in leg_list}
        for leg in leg_list:
            Ap, sxp, syp, szp = env3d.cubePermute(
                A, sx, sy, sz, direction=leg)
            # bottleneck of computational cost: O(χ^12)
            # we will do it 3 x n_round times
            octu4P, octu4Gamma = env3d.firstContr(Ap, sxp, szp)
            if display:
                print("χ^12 construction of envrionment finished!")
            # update the s-matrix for the given direction
            snew, err = opt_1s(
                octu4P, octu4Gamma, syp, PsiPsi,
                epsilon, iter_max)
            # update s
            if leg == "y":
                sy = snew * 1.0
            elif leg == "z":
                sz = snew * 1.0
            elif leg == "x":
                sx = snew * 1.0
            # record error
            errList.append(err)
            if display:
                print("This is round {:d} for leg {:s}:".format(m+1, leg),
                      "FET Error {:.3e} ---> {:.3e}".format(err[1], err[-1]),
                      "(in {:d} iterations)".format(len(err) - 1)
                      )
            # if the change of FET error is small, the opt for the leg is done
            doneLegs[leg] = (
                abs((err[-1] - err[1]) / (err[1] + 1e-8)
                    ) < (0.01 * iter_max/100)
            )
        if all(doneLegs.values()):
            break

    return sx, sy, sz, errList


# FET Fidelity (or error)
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


def cubeError(octuP, octuGamma, s, PsiPsi):
    """
    Much cheaper way to calculate FET error
    Very similar to `.env3d.contr2Psy` and
    `.env3d.contr2Gammasy`
    """
    # <ψ|φ>
    octuPs = ncon([octuP, s, s.conj(), s.conj(), s],
                  [[1, -2, 2, -4, 3, -6, 4, -8],
                   [1, -1], [2, -3], [3, -5], [4, -7]])
    PsiPhi = ncon([octuPs, octuPs.conj()],
                  [[1, 2, 3, 4, 5, 6, 7, 8],
                   [1, 2, 3, 4, 5, 6, 7, 8]])
    # <φ|φ>
    octuGammas = ncon(
        [octuGamma,
         s, s.conj(), s.conj(), s,
         s.conj(), s, s, s.conj()],
        [[1, 2, 3, 4, 5, 6, 7, 8],
         [1, -1], [2, -2], [3, -3], [4, -4],
         [5, -5], [6, -6], [7, -7], [8, -8]])
    PhiPhi = ncon([octuGammas, octuGammas.conj()],
                  [[1, 2, 3, 4, 5, 6, 7, 8],
                   [1, 2, 3, 4, 5, 6, 7, 8]])
    # fidelity = |<ψ|φ>|^2 / (<φ|φ> <ψ|ψ>)
    f = PsiPhi * PsiPhi.conj() / (PhiPhi * PsiPsi)
    err = 1 - f.norm()
    return err


def initFidelity(Lr, Gamma):
    """
    Fidlity for initial low-rank matrix Lr
    """
    PsiPsi = ncon([Gamma], [[1, 1, 2, 2]])
    P = ncon([Gamma], [[-1, -2, 1, 1]])
    f, err, PhiPhi = cubeFidelity(Lr, P, Gamma, PsiPsi)
    return f, err


# Combine all the above functions
def absbs(A, sx, sy, sz):
    """
    Apply sx, sy, and sz to the main tensor A
    """
    As = ncon(
        [A, sx, sy, sz],
        [[1, -2, 2, -4, 3, -6], [1, -1], [2, -3], [3, -5]]
    )
    return As
