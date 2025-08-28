#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : fet2d_reflsym.py
# Author            : Xinliang (Bruce) Lyu <xlyu@ihes.fr>
# Date              : 21.08.2025
# Last Modified Date: 21.08.2025
# Last Modified By  : Xinliang (Bruce) Lyu <xlyu@ihes.fr>
"""
This module implements a Entanglement Filtering;
the lattice reflection symmetry is exploited

The scheme is a combination of
- Evenbly's full environment truncation (FET)
- Graph-Independent Local Truncations (Gilt) by
Hauru, Delcamp, and Mizera

Due to the lattice reflection symmetry,
there are only two filtering matrix sx, sy

The relevant environment is constructed in the file `./env2d_reflsym.py`
Some functions in the file `./env2d_reflsym.py` that only exploits
the lattice reflection are also reused here.
"""
from .. import u1ten
from . import env2d_reflsym, env2d_rotsym, fet2d_rotsym
from ncon import ncon
import itertools


# I. For the initialization of the filtering matriices sx and sy
def init_s(A, chis, chienv, epsilon, epsilon_inv=1e-10,
           bond="x"):
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
    Upsilon0 = env2d_reflsym.build_Upsilon0(A, bond=bond)
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
    return s, Lr, Upsilon0


# II. For optimization of sx and sy
def opt_alls(A, sx, sy, PsiPsi, epsilon=1e-10,
             iter_max=5, n_round=40, display=True):
    """optimized filtering matrices sx and sy
    """
    bond_list = ["x", "y"]
    errHistory = []
    for m in range(n_round):
        doneLegs = {k: False for k in bond_list}
        for bond in bond_list:
            # find an updated s
            snew, err = opt_1s(
                A, sx, sy, PsiPsi, epsilon, bond, iter_max
            )
            # update s
            if bond == "x":
                sx = snew * 1.0
            elif bond == "y":
                sy = snew * 1.0
            # record error
            errHistory.append(err)
            if display and (m % 5 == 0 or m == n_round - 1):
                print("This is round {:d} for leg {:s}:".format(m+1, bond),
                      "EF Error {:.3e} ---> {:.3e}".format(err[0], err[-1]),
                      "(in {:d} iterations)".format(len(err) - 1)
                      )
            # if the change of FET error is small, or the FET itself is small,
            # the optimization for the leg is done
            doneLegs[bond] = (
                abs((err[-1] - err[1]) / (err[1] + epsilon)
                    ) < (0.005 * iter_max/10)
            ) or (abs(err[-1]) < epsilon)

        # exit the round iteration if both legs are done
        if all(doneLegs.values()):
            if display:
                print("  Final round {:d} for leg {:s}:".format(m+1, bond),
                      "EF Error {:.3e} ---> {:.3e}".format(err[1], err[-1]),
                      "(in {:d} iterations)".format(len(err) - 1)
                      )
            break
    # flatten the errHistory list
    errHistory = list(
        itertools.chain.from_iterable(errHistory)
    )
    return sx, sy, errHistory


def opt_1s(A, sx, sy, PsiPsi, epsilon, bond="x", iter_max=5):
    """
    Update a single filtering matrix s iteratively for `iter_max` times
    using `update_s`
    """
    # the old filtering matrix
    if bond == "x":
        sold = sx * 1.0
    else:
        sold = sy * 1.0

    snew = sold * 1.0
    err = []

    for k in range(iter_max):
        # update s matrix on `bond`
        snew, errNew, errOld = update_s(
            A, sx, sy, PsiPsi, epsilon, bond=bond
        )
        # update sx, sy matrices
        if bond == "x":
            sx = snew * 1.0
        else:
            sy = snew * 1.0
        # record the EF error
        if k == 0:
            err.append(errOld)
            err.append(errNew)
        else:
            err.append(errNew)
        # step the iteration if the error is small
        if errNew < epsilon:
            break

    return snew, err


def update_s(A, sx, sy, PsiPsi, epsilon, bond="x"):
    """Update a filtering matrix
    """
    # the old filtering matrix
    if bond == "x":
        sold = sx * 1.0
    else:
        sold = sy * 1.0
    # construct Υ and Q
    Upsilon, Q = env2d_reflsym.build_UpsilonQ(A, sx, sy, bond=bond)
    # calculate the old error
    errOld = env2d_rotsym.plaqFidelity(sold, Q, Upsilon, PsiPsi)[1]

    # propose a candidate filtering matrix s
    stemp = fet2d_rotsym.propose_s(Q, Upsilon, epsilon=epsilon)
    # normalized stemp (for the convex combination)
    stemp = stemp / stemp.norm()
    # try 10 convex combinations of sold and stemp
    # make sure the approximation error go down
    # (This idea is taken from Evenbly's TNR codes)
    for p in range(11):
        # take the convex combination
        snew = (1 - 0.1 * p) * stemp + 0.1 * p * sold
        # calculate the fidelity
        if bond == "x":
            UpsilonN, QN = env2d_reflsym.build_UpsilonQ(A, snew, sy, bond=bond)
        else:
            UpsilonN, QN = env2d_reflsym.build_UpsilonQ(A, sx, snew, bond=bond)
        errNew = env2d_rotsym.plaqFidelity(snew, QN, UpsilonN, PsiPsi)[1]
        # once the FET error reduces, we update s matrix
        if (errNew <= errOld) or (errNew < epsilon) or (p == 10):
            # make sure the returned s matrix is normalized
            snew = snew / snew.norm()
            break
    return snew, errNew, errOld


def opt_s_cycl(A, sx0, sy0, PsiPsi,
               epsilon=1e-10, iter_max=1000, display=True):
    """iteratively update sx and sy to maximaize the fidelity

    Args:
        A (TensorCommon): the 4-leg tensor
        sx0 (TensorCommon): the initial filtering matrix on x bond
        sy0 (TensorCommon): the initial filtering matrix on y bond
        PsiPsi (TensorCommon): the overlap <ψ|ψ>

    Kwargs:
        epsilon (float): safety for the inverse matrix (not essential here)
        iter_max (int): maximal iteration
        display (boolean): printout info

    Returns:
        sx, sy (TensorCommon): the optimized filtering matrices
        err (float): the final error

    """
    # normalized filtering matrices
    sx = sx0 / sx0.norm()
    sy = sy0 / sy0.norm()
    errs = []

    # the initial EF error
    Upsilonx, Qx = env2d_reflsym.build_UpsilonQ(A, sx, sy, bond="x")
    err0 = env2d_rotsym.plaqFidelity(sx, Qx, Upsilonx, PsiPsi)[1]
    errs.append(err0)
    if display:
        print("EF Error (initial) = {:.3e}".format(err0))
    # return the initial solution if the EF error is already small
    if err0 < epsilon:
        return sx, sy, errs

    # enter the optimization iteration
    bondCycle = itertools.cycle(["x", "y"])
    for k in range(iter_max):
        bond = next(bondCycle)
        # deterine the new filtering matrix
        snew, errNew = update_s(
            A, sx, sy, PsiPsi, epsilon, bond=bond
        )[:2]
        # update sx, sy matrices
        if bond == "x":
            sx = snew * 1.0
        else:
            sy = snew * 1.0
        # record the EF error
        errs.append(errNew)

        # exit the iteration if the error is very small
        if errNew < epsilon:
            break
        # exit the iteration if the error converges
        if k > 99 and k % 100 == 0:
            if display:
                print("EF Error (iter {:2d}) = {:.3e}".format(k, errs[-1]))
            isConverge = (
                abs((errs[-1] - errs[-101]) / (errs[-101] + 1e-8)
                    ) < 0.010
            )
            if isConverge:
                break
    return sx, sy, errs


def fidelity(A, sx, sy, PsiPsi):
    Upsilonx, Qx = env2d_reflsym.build_UpsilonQ(A, sx, sy, bond="x")
    f, err, PhiPhi = env2d_rotsym.plaqFidelity(sx, Qx, Upsilonx, PsiPsi)
    return f, err, PhiPhi


# III. Apply the sx and sy to the 4-leg tensor A
def absorb(A, sx, sy):
    As = ncon([A, sx, sy],
              [[1, 2, -3, -4], [1, -1], [2, -2]]
              )
    return As

# end of file
