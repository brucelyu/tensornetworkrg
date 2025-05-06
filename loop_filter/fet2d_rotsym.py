#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : fet2d_rotsym.py
# Author            : Xinliang (Bruce) Lyu <xlyu@ihes.fr>
# Date              : 29.04.2025
# Last Modified Date: 29.04.2025
# Last Modified By  : Xinliang (Bruce) Lyu <xlyu@ihes.fr>
"""
This module implements a Entanglement Filtering;
both the lattice reflection and rotational symmetries are exploited

The scheme is a combination of
- Evenbly's full environment truncation (FET)
- Graph-Independent Local Truncations (Gilt) by
Hauru, Delcamp, and Mizera

Due to the lattice reflection and rotation symmetry,
there is only one filtering matrix s

The relevant environment is constructed in the file `./env2d_rotsym.py`
"""
from .. import u1ten
from . import env2d_rotsym
from ncon import ncon


# I. For the initialization of the filtering matrix s
def init_s(A, chis, chienv, epsilon, epsilon_inv=1e-10):
    """Initialization of the filtering matrix s

    Args:
        A (TensorCommon): 4-leg tensor
        chis (int): χs in the filtering
        chienv (int): χenv for taking the pinv of Upsilon0
        epsilon (float): for the pinv

    Returns:
        s (TensorCommon): the filtering matrix
        Lr (TensorCommon): the low-rank matrix, Lr = s @ s*
        Upsilon0 (TensorCommon): bone environment tensor
        dbA (TensorCommon): contraction of two copies of A:
            useful for later optimization of s

    """
    dbA = env2d_rotsym.contr2A(A)
    Upsilon0 = env2d_rotsym.dbA2Upsilon0(dbA)
    # determine the low-rank matrix (like Gilt without recursion)
    # Step 1. Take the Moore-Penrose inverse of the Upsilon tensor
    Upsilon0_pinv = u1ten.pinv(
        Upsilon0, [0, 1], [2, 3],
        eps_mach=epsilon_inv, chiCut=chienv
    )
    # Step 2. Obtain the low-rank matrix Lr
    Lr = ncon([Upsilon0_pinv, Upsilon0], [[-1, -2, 1, 2], [1, 2, 3, 3]])
    # Step 3. Split Lr to get s: Lr = s @ s*
    s = Lr.split([0], [1], chis=[k+1 for k in range(chis)],
                 eps=epsilon)[0]
    # make the direction of s as [-1, 1]
    s = s.flip_dir(1)
    return s, Lr, Upsilon0, dbA


# II. For optimization of s matrix
def propose_s(P, Upsilon, epsilon=1e-10):
    """propose an updated filtering matrix s
    in order to maximaze the fidelity

    Args:
        P (TensorCommon): 2-leg tensor
        Upsilon (TensorCommon): 4-leg tensor

    Kwargs:
        epsilon (float):
            for avoding the numerical problem of taking
            the inverse of smaller numbers.
            It is not essential in the update of the s matrix.

    Returns:
        sp (TensorCommon): the proposed filtering matrix

    """
    # take the inverse of Upsilon
    # (pinv here is just for numerical safety)
    Upsilon_inv = u1ten.pinv(
        Upsilon, [0, 1], [2, 3], eps_mach=epsilon
    )
    sp = ncon([Upsilon_inv, P], [[-1, -2, 1, 2], [1, 2]])
    return sp


def update_s(dbA, sold, PsiPsi, epsilon=1e-10):
    """update the filtering matrix s
    A trick that is used in Evenbly's TNR implementation is employed
    here to improve the convergence of the fidelity

    Args:
        dbA (TensorCommon): contraction of 2 copies of A
        sold (TensorCommon): the old filtering matrix
        PsiPsi (float): the overlap <ψ|ψ>

    Returns:
        snew (TensorCommon): the updated filtering matrix
        errNew (float): error after the update
        errOld (float): error before the update

    """
    # construct Υ and P
    Upsilon, P = env2d_rotsym.dbA2UpsilonP(dbA, sold)
    # calculate the old error
    errOld = env2d_rotsym.plaqFidelity(sold, P, Upsilon, PsiPsi)[1]
    # no need for optimization if the error is already very small
    if errOld < epsilon:
        return sold, errOld, errOld
    # propose a candidate filtering matrix s
    stemp = propose_s(P, Upsilon, epsilon=epsilon)
    # normalized stemp (for the convex combination)
    stemp = stemp / stemp.norm()
    # try 10 convex combinations of old s and new s
    # make sure the approximation error go down
    # (This idea is taken from Evenbly's TNR codes)
    for p in range(11):
        # take the convex combination
        snew = (1 - 0.1 * p) * stemp + 0.1 * p * sold
        # calculate the fidelity
        errNew = fidelity(dbA, snew, PsiPsi)[1]
        # once the FET error reduces, we update s matrix
        if (errNew <= errOld) or (errNew < epsilon) or (p == 10):
            # make sure the returned s matrix is normalized
            snew = snew / snew.norm()
            break

    return snew, errNew, errOld


def opt_s(dbA, s_init, PsiPsi, epsilon=1e-10, iter_max=20,
          display=True):
    """iteratively update s to maximaize the fidelity

    Args:
        dbA (TensorCommon): contraction of 2 copies of A
        s_init (TensorCommon): the initial filtering matrix
        PsiPsi (float): the overlap <ψ|ψ>

    Kwargs:
        iter_max (int): maximal iteration

    Returns:
        s (TensorCommon): the final filtering matrix
        err (float): the final error

    """
    s = s_init * 1.0
    errs = []
    for k in range(iter_max):
        if k == 0:
            # normalized s
            s = s / s.norm()
        # update s matrix
        s, errNew, errOld = update_s(dbA, s, PsiPsi, epsilon)
        # record evolution of errors
        if k == 0:
            errs.append(errOld)
            errs.append(errNew)
            if display:
                print("EF Error (initial) = {:.3e}".format(errs[0]))
        else:
            errs.append(errNew)

        # exit the iteration if the error is very small
        if errNew < epsilon:
            break
        # exit the iteration if the error converges
        if k > 49 and k % 50 == 0:
            if display:
                print("EF Error (iter {:2d}) = {:.3e}".format(k, errs[-1]))
            isConverge = (
                abs((errs[-1] - errs[-51]) / (errs[-51] + 1e-8)
                    ) < 0.005
            )
            if isConverge:
                break

    return s, errs


def fidelity(dbA, s, PsiPsi):
    """fidelity of the Entanglement Filtering

    Args:
        dbA (TensorCommon): contraction of 2 copies of A
        s (TensorCommon): the filtering matrix
        PsiPsi (float): the overlap <ψ|ψ>

    """
    Upsilon, P = env2d_rotsym.dbA2UpsilonP(dbA, s)
    f, err, PhiPhi = env2d_rotsym.plaqFidelity(s, P, Upsilon, PsiPsi)
    return f, err, PhiPhi


# III. Apply the filtering matrix to the 4-leg tensor A
def absorb(A, s):
    """A absorbs the filtering matrix s

    Args:
        A (TensorCommon): 4-leg tensor
        s (TensorCommon): filtering matrix

    """
    As = ncon([A, s, s.conj()],
              [[1, 2, -3, -4], [1, -1], [2, -2]]
              )
    return As

# end of the file
