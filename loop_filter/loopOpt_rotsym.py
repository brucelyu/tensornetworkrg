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
                 \α

while the order of ij for vL depends on the direction of
the arrow on its TN diagram.
"""
from ncon import ncon
from .. import u1ten


# Part I: Functions for building environment tensors
# for updating the vL tensor and calculating fidelity F

def multizp2vL(vL, zp):
    """multiply zp on the 3rd leg of vL

    Args:
        vL (TensorCommon): 3-leg tensor
        zp (TensorCommon): 1D vector
            the renormalized bond matrix

    Returns:
        vLzp (TensorCommon): 3-leg tensor
            with zp multiplied on the 3rd leg of vL

    """
    vLzp = vL * 1.0
    if zp is not None:
        vLzp = vLzp.multiply_diag(zp, 2, direction="r")
    return vLzp


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
               |              zp
                               \
                                vL*--
                                |
    """
    vLzp = multizp2vL(vL, zp)
    Av = ncon([vLzp, vL.conj()], [[-3, -2, 1], [-4, -1, 1]])
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


def multiz2dbA(dbA, z):
    """multiply the bond matrix z on dbA

    Args:
        dbA (TensorCommon): 4-leg tensor
            result of the above `contr2A`
        z (TensorCommon): 1D vector
            the bond matrix

    Returns:
        dbAz (TensorCommon):

    """
    dbAz = dbA * 1.0
    # apply z on the four legs of dbA
    if z is not None:
        dbAz = dbAz.multiply_diag(z, 0, direction="r")
        dbAz = dbAz.multiply_diag(z, 3, direction="r")
        dbAz = dbAz.multiply_diag(z, 1, direction="l")
        dbAz = dbAz.multiply_diag(z, 2, direction="l")
    return dbAz


# -- Build Υ tensor
def vL2Upsilon(vL, zp, z):
    """build the Υ tensor from vL and bond matrices

    Args:
        vL (TensorCommon): 3-leg tensor
        zp (TensorCommon): 1D vector
            the renormalized bond matrix
        z (TensorCommon): 1D vector
            the bond matrix

    Returns:
        Upsilon (TensorCommon): 4-leg tensor
            The Υ tensor for updating vL
            in loop optimization

    """
    Av = vL2Av(vL, zp)       # costs: O(χ^5)
    dbAv = contr2A(Av, Av)   # costs: O(χ^6)
    # multiply z on dbAv
    dbAvz = multiz2dbA(dbAv, z)
    # multiply zp on vL
    vLzp = multizp2vL(vL, zp)
    # build Upsilon            costs: O(3χ^6)
    Upsilon = ncon([dbAvz, dbAv.conj(), dbAvz, vLzp, vLzp.conj()],
                   [[12, 13, 1, 2], [3, 4, 1, 2], [3, 4, -3, -1],
                    [11, 12, -4], [11, 13, -2]])
    return Upsilon


# -- Build Q tensor
def vL2Q(vL, zp, z, A):
    """build the Q tensor

    Args:
        vL (TensorCommon): 3-leg tensor
        zp (TensorCommon): 1D vector
            the renormalized bond matrix
        z (TensorCommon): 1D vector
            the bond matrix
        A (TensorCommon): 4-leg tensor
            the bulk tensor in TN

    Returns:
        Q (TensorCommon): 3-leg tensor

    """
    Av = vL2Av(vL, zp)       # costs: O(χ^5)
    AAv = contr2A(A, Av)     # costs: O(χ^6)
    # multiply z on AAv
    AAvz = multiz2dbA(AAv, z)
    # multiply zp on vL
    vLzp = multizp2vL(vL, zp)
    # build Q
    Qstar = ncon([AAvz, AAv.conj(), AAvz, A.conj(), vLzp.conj()],
                 [[11, 14, 1, 2], [3, 4, 1, 2], [3, 4, 12, -1],
                  [11, 12, -3, 13], [13, 14, -2]])
    Q = Qstar.conj()
    return Q


# -- Calculate the overlap of the original plaquette: <ψ|ψ>
def A2PsiPsi(A, z):
    """calculate the overlap <ψ|ψ>
    For computing the fidelity of the loop optimization

    Args:
        A (TensorCommon): 4-leg tensor
            the bulk tensor in TN
        z (TensorCommon): 1D vector
            the bond matrix

    Returns:
        PsiPsi (float): the overlap

    """
    dbA = contr2A(A, A)    # costs: O(χ^6)
    dbAz = multiz2dbA(dbA, z)
    # build PsiPsi         # costs: O(χ^6)
    quadrAz = ncon([dbAz, dbA.conj()], [[1, 2, -1, -2], [1, 2, -3, -4]])
    PsiPsi = ncon([quadrAz]*2, [[1, 2, 3, 4], [3, 4, 1, 2]])
    return PsiPsi


# -- Calculate the fidelity
def fidelity(vL, zp, z, A, return_overlap=False):
    Upsilon = vL2Upsilon(vL, zp, z)
    Q = vL2Q(vL, zp, z, A)
    f, PhiPhi, PsiPsi = UQ2fid(vL, Upsilon, Q, z, A)
    if not return_overlap:
        return f, 1 - f, Upsilon, Q
    else:
        return f, 1 - f, PhiPhi, PsiPsi


def UQ2fid(vL, Upsilon, Q, z, A):
    # calculate <Φ|Φ>
    PhiPhi = ncon([Upsilon, vL, vL.conj()],
                  [[1, 2, 3, 4], [5, 1, 2], [5, 3, 4]])
    # calculate <Φ|Ψ>
    PhiPsi = ncon([Q.conj(), vL],
                  [[1, 2, 3], [3, 1, 2]])
    # calculate <Ψ|Ψ>
    PsiPsi = A2PsiPsi(A, z)
    # calculate fidelity = |<Φ|Ψ>|^2 / (<Φ|Φ> <Ψ|Ψ>)
    f = PhiPsi * PhiPsi.conj() / (PhiPhi * PsiPsi)
    f = f.norm()
    return f, PhiPhi, PsiPsi


# Part II: Functions for the optimization of vL tensor
def propose_vL(Upsilon, Q, eps_pinv=1e-8):
    """propose a candidate as an updated vL
    for optimizing the fideltiy

    Args:
        Upsilon (TensorCommon): 4-leg tensor
        Q (TensorCommon): 3-leg tensor

    Kwargs:
        eps_pinv (float):
            threshold below which the eigenvalues
            are considered as zero

    Returns:
        vLp (TensorCommon): 3-leg tensor
            the candidate as an updated vL

    """
    Upsilon_inv = u1ten.pinv(
        Upsilon, [0, 1], [2, 3], eps_mach=eps_pinv
    )
    vLpstar = ncon([Upsilon_inv, Q.conj()], [[-2, -3, 1, 2], [1, 2, -1]])
    vLp = vLpstar.conj()
    return vLp


def update_vL(vL, zp, z, A, eps_pinv=1e-8,
              eps_errEF=1e-10):
    """update vL for optimizing the fidelity

    Args:
        vL (TensorCommon): 3-leg tensor
        zp (TensorCommon): 1D vector
            the renormalized bond matrix
        z (TensorCommon): 1D vector
            the bond matrix
        A (TensorCommon): 4-leg tensor
            the bulk tensor in TN

    Kwargs:
        eps_pinv (float):
            threshold below which the eigenvalues
            are considered as zero
        eps_errEF (float):
            threshold below which the EF erro
            is considered small enough

    Returns:
        vLnew (TensorCommon): 3-leg tensor
            the updated vL
        errNew (float): loop error after the update
        errOld (float): loop error before the update

    """
    # calculate old fidelity and construct Υ and P
    # -- Normalized vL
    vLold = vL / vL.norm()
    _, errOld, Upsilon, Q = fidelity(vLold, zp, z, A)
    # propose a candidate vL
    vLp = propose_vL(Upsilon, Q, eps_pinv)
    # --- Normalized vLp
    vLp = vLp / vLp.norm()

    # try 10 convex combinations of old vL and the proposed vL
    # to make sure the error goes down
    # (This idea is taken from Evenbly's TNR codes)
    for p in range(11):
        vLnew = (1 - 0.1 * p) * vLp + 0.1 * p * vLold
        # newPortion = (1 - 0.1 * p)
        # once the error reduces, we exit
        errNew = fidelity(vLnew, zp, z, A)[1]
        if (errNew <= errOld) or (errNew < eps_errEF):
            # print("{:.0%} of new solution is mixed!".format(newPortion))
            vLnew = vLnew / vLnew.norm()
            break
    return vLnew, errNew, errOld


def opt_vL(vL_init, zp, z, A, eps_pinv=1e-8,
           eps_errEF=1e-10, iter_max=50,
           display=True, iter_min=5):
    """find vL that maximalize the fidelity

    Args:
        vL_initi (TensorCommon): 3-leg tensor
        zp (TensorCommon): 1D vector
            the renormalized bond matrix
        z (TensorCommon): 1D vector
            the bond matrix
        A (TensorCommon): 4-leg tensor
            the bulk tensor in TN

    Kwargs:
        eps_pinv (float):
            threshold below which the eigenvalues
            are considered as zero
        eps_errEF (float):
            threshold below which the EF erro
            is considered small enough

    Returns:
        vL (TensorCommon): 3-leg tensor
            the optimal vL
    """
    vL = vL_init * 1.0
    # record error (1 - fidelity) during the iteration
    err_hist = []
    for k in range(iter_max):
        # update vL
        vL, errNew, errOld = update_vL(vL, zp, z, A, eps_pinv, eps_errEF)
        # record evolution of errors
        if k == 0:
            err_hist.append(errOld)
            err_hist.append(errNew)
            if display:
                print("  --> Loop Optimization Error (initial) = {:.3e}".format(
                    err_hist[0]))
        else:
            err_hist.append(errNew)

        # exit the iteration if the error is very small
        isMinIter = (k + 1 >= iter_min)
        if isMinIter and (errNew < eps_errEF):
            if display:
                print("  --> Loop Optimization Error (iter {:2d}) = {:.3e}".format(
                    k+1, err_hist[-1]))
            break

        # exit the iteration if the error converges
        if (k + 1) % iter_min == 0:
            if display:
                print("  --> Loop Optimization Error (iter {:2d}) = {:.3e}".format(
                    k+1, err_hist[-1]))
            isConverge = (
                abs((err_hist[-1] - err_hist[-1 - iter_min]) /
                    (err_hist[-1 - iter_min] + eps_errEF)
                    ) < (0.001 * iter_min)
            )
            if isConverge:
                break

    return vL, err_hist

# end of file
