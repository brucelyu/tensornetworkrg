#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : gilt.py
# Author            : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
# Date              : 21.10.2022
# Last Modified Date: 21.10.2022
# Last Modified By  : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
"""
This is a implementation of the graph-independent local truncations
described in paper,
Markus Hauru, Clement Delcamp, and Sebastian Mizera,
Phys. Rev. B 97, 045111 – Published 10 January 2018
url: https://journals.aps.org/prb/abstract/10.1103/PhysRevB.97.045111
"""
from ncon import ncon


# -----------------------|
# -----------------------|
# This is the key process of the implementation of the GILT.
# Given an environment of a leg containing a loop, this
# function general a low-rank R that detects the number represent this loop
# and truncate the loop through this leg.
def Ropt(U, s, epsilon=1e-7, convergence_eps=1e-2,
         counter=1, verbose=False):
    """Determine the low-rank matrix of a bond in a tensor network
    Recursively inserting low-rank R matrix into a bond and absorb
    the matrix into its two nearby tensors.

    Current implementation is designed with CDL structure in mind.
    It works specifically well for pure CDL structure,
    especially when FIRST truncating an environment.

    - TODOs
    1) It is important to know more about its limitations
    for dealing with real statistical models.
    2) Compare its performance with other iteration methods, like
    full environment truncation used in implicit-TNR, which might
    be more general and applicable for real statistical models.

    Args:
        U (TensorCommon): U in the svd of environment, env = USV.
        Since we usually choose environment of a single leg, this
        tensor will have a shape like U[i,j, α]
           |α
        i--U--j

        s (TensorCommon): singular values of the environment

    Kwargs:
        epsilon (float):
        A small number below which we think the singular value is zero.
        For a pure CDL tensor, it can be as small as the machine precise
        in principle (what about in practice?)

        convergence_eps (float):
        To determine how close the final R is to a projector.
        Again, it is based on the consideration of a pure
        CDL structure

        counter (int): count the recursion depth
        verbose (boolean): print out information

    Returns:
        Rprime (TensorCommon): desired low-rank matrix
        counter (int): depth of the recursion
    """
    Rp = findRp(U, s, epsilon)
    # small number for trucated svd
    spliteps = epsilon * 1e-3
    # we assume reflectioin symmetry of the environment here
    # the splitting of the R matrix should reflect this
    Rhalf, Rd = symmetricSplit(Rp, spliteps=spliteps)
    done_recursing = (Rd - 1).abs().max() < convergence_eps

    # enter the recursion
    if (not done_recursing):
        URR = ncon([U, Rhalf, Rhalf.conj()],
                   [[1, 2, -3], [1, -1], [2, -2]])
        URRs = URR.multiply_diag(s.flip_dir(0), 2, direction="left")
        Uinner, sinner = URRs.svd([0, 1], [2])[:2]
        # recursively find the inner one
        Rhalfinner, counter = Ropt(
            Uinner, sinner, epsilon, convergence_eps,
            counter=counter+1, verbose=verbose
        )
        # combine the inner one and the current one
        Rhalf = ncon([Rhalf, Rhalfinner],
                     [[-1, 1], [1, -2]])
    return Rhalf, counter


def symmetricSplit(mat, eps=1e-8, spliteps=1e-8):
    """
    Split a positive semi-definite matrix `mat`:
    mat = matHalfL @ matHalfL.transpose().conjugate()

    Return
    matHalfL: left half piece of the mat
    d: eigenvalues of mat
    """
    # ensure that the input matrix is symmetric
    diffmmt = (mat - mat.transpose((1, 0)).conj()).norm() / mat.norm()
    errMsg = "Diff. of mat and mat.T is {:.2e}".format(diffmmt)
    assert diffmmt < eps, errMsg
    # eigenvalue decomposition of mat
    d, U = mat.eig([0], [1], hermitian=True, eps=eps)
    # make sure all eigenvalues are positive
    errMsg = "All the eigenvalues of the matrix should be non-negative"
    assert (d > -eps).all(), errMsg
    d_sqrt = d.sqrt()
    matHalfL = U.multiply_diag(d_sqrt, -1, direction="right")
    return matHalfL, d


def findRp(U, s, epsilon=1e-7):
    """
    A single step of determing the low-rank matrix R
    """
    t = tvec(U)
    tprime = topt(t, s, epsilon)
    Rprime = Rtensor(U, tprime)
    # symmetrize
    Rprime = 0.5 * (Rprime + Rprime.transpose().conj())
    return Rprime


def tvec(U):
    """
    Calculate t vector from U tesnor
    Parameters
    t--k =    |k
            --U--
            |____|
    ----------
    U : three leg tensor U[i,j,k]
        singular vectors (eigenvectors of an environment matrix).
    Returns
    -------
    t : t[k]
    """
    t = ncon([U], [[1, 1, -1]])
    return t


def topt(t, s, epsilon=1e-7):
    """
    One optimization method to choose t' such that the rank of
    the resultant Rp is low
    Parameters
    ----------
    t : vector t[k]
        constructed from tensor U
           |k
         --U--
        |____|
    s : vector s[i]
        singular spectrum of an eivironment matrix.
    epsilon : float, optional
         a small number below which we think the singular value is zero.
         The default is 1e-7.
    Returns
    -------
    tprime : vector tprime[k]
        optimized t.
    """
    # properly normalize the environment spectrum if it is not
    # s = s / s.norm()
    s = s / s.max()
    # For Z2 symmetric tensor
    t = t.flip_dir(0)
    # choose t' according to eq.(31) in the paper
    if epsilon != 0:
        ratio = s / epsilon
        weight = ratio**(2) / (1 + ratio**(2))
        tprime = t.multiply_diag(weight, 0, direction="left")
    else:
        tprime = t * 1.0
    return tprime


def Rtensor(U, t):
    """
    Calculate R tensor from U and t
                 t
    i--R--j =    |
              i--U--j
    Parameters
    ----------
    U : three leg tensor U[ijk]
           |k
        i--U--j
    t : vector t[k].
    Returns
    -------
    R : matrix R[i,j]
        a low rank matrix to truncate a leg..
    """
    R = ncon([U.conjugate(), t.conjugate()], [[-1, -2, 1], [1]])
    return R
