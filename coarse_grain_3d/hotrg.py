#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : hotrg.py
# Author            : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
# Date              : 15.10.2022
# Last Modified Date: 15.10.2022
# Last Modified By  : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
"""
Various 3D HOTRG implementations
Tensor leg order convention is
A[x, x', y, y', z, z']
     z  x'
     | /
 y'--A--y
   / |
 x   z'

For the reflection-symmetric version, the
conjugate direction of the tensor is
assumed to be [1, -1, 1, -1, 1, -1].
"""

import numpy as np
from ncon import ncon
from . import signfix as sf
from .. import u1ten


# The original scheme of HOTRG: one direction at a time
def dirHOTRG(A, chi, direction,
             cg_eps=1e-8, comm=None):
    """HOTRG in one direction

    Args:
        A (TensorCommon): 6-leg tensor
        chi (int): bond dimension
        direction (str): direction of HOTRG

    Kwargs:
        cg_eps (float):
            if eigenvalues are smaller than cg_eps,
            the corresponding eigenvectors will be truncated
        comm (MPI.COMM_WOLRD):
            for parallel computation

    Returns:
        Aout (TensorCommon): 6-leg coarser tensor
        isometries (list): the pair of isometries [pzx, pzy]
        errs (list): approximation error in two directions
    """
    # determine isometries
    (pzx, pzy,
     errzx, errzy,
     dzx, dzy
     ) = optProjector(A, A, chi, direction=direction, cg_eps=1e-8)
    # contraction step
    Aout = block2tensors(A, A, pzx, pzy, direction, comm=comm)
    # isometries and errors
    isometries = [pzx, pzy]
    errs = [errzx, errzy]
    return Aout, isometries, errs


# general functions
def block2tensors(A, B, pzx, pzy, direction,
                  comm=None):
    """Block-transform two tensors along a given direction
    using two pairs of isometric tensors

    The prototype is the z-direction block-transformation;
    thus this function is rotational of tensor netowrk +
    calling of `blockAlongZ` function.

    Args:
        A (TensorCommon): +(z, y, x) tensor
        B (TensorCommon): -(z, y, x) tensor
        pzx (TensorCommon): 3-leg isometry for
            squeezing two legs in
            1) x direction for z-direction block transformation
            2) z direction for y-direction block transformation
            3) y direction for x-direction block transformation
        pzy (TensorCommon): 3-leg isometry for
            squeezing two legs in
            1) y direction for z-direction block transformation
            2) x direction for y-direction block transformation
            3) z direction for x-direction block transformation
        direction (str): direction for block transformation
            chosen among ["x", "y", "z"]

    Kwargs:
        comm (MPI.COMM_WOLRD):
            for parallel computation

    Returns:
        Aout (TensorCommon):
            Aout = A.B.pzx.pzx*.pzy.pzy*
            where the . schematically denote
            tensor contraction
    """
    # For Parallel computation, broadcast the input tensors: A, B, pzx, pzy
    if comm is not None:
        A = comm.bcast(A, root=0)
        B = comm.bcast(B, root=0)
        pzx = comm.bcast(pzx, root=0)
        pzy = comm.bcast(pzy, root=0)
    # Rotate the order of the legs of A and B to the right position
    perm, inv_perm = rotInd(direction)
    Arot = A.transpose(perm)
    Brot = B.transpose(perm)
    # block transformation
    Aout = blockAlongZ(Arot, Brot, pzx, pzy, comm=comm)
    # rotate the order of tensor legs back
    Aout = Aout.transpose(inv_perm)
    return Aout


def optProjector(A, B, chi, direction="z", cg_eps=1e-8):
    """
    Determine the projector operator for the 3d HOTRG
    The trick is to implement a prototype one: contraction in z direction,
    squeeze two x legs.
    Other cases can be achieved by permutating legs

    Parameters
    ----------
    A : 6-leg tensor, A[x, x', y, y', z, z']
        |z
        A--y
      x/
    B : 6-leg tensor, B[x, x', y, y', z, z']
        the same as A.
    chi : int
        upper bound of the output bond dimension.
    direction : str, optional
        direction of the hotrg, choice is among ["x", "y", "z"].
        The default is "z".
    cg_eps : float, optional
        normalized eigenvalues smaller that cg_eps will be thrown away.
        The default is 1e-15.
    Returns
    -------
    pzx : 3-leg tensor
            /x
           /
          |
          |
          | /x
        / |/
       /
       x_out
    pzy : 3-leg tensor
        similiar to pzx, but with x <--> y.
    dzx : array
        eigenvalues of M M^+ matrix corresponding to pzx.
    dzy : array
        eigenvalues of M M^+ matrix corresponding to pzy.
    errzx : float
        error of approximation for squeezing in z direction for two x legs.
    errzy : float
        error of approximation for squeezing in z direction for two y legs.
    """
    # Rotate the A and B to the right position
    perm = rotInd(direction)[0]
    Arot = A.transpose(perm)
    Brot = B.transpose(perm)

    # The following code is written using z direction as prototype.
    # For y direction, (xyz) --> (zxy), so pzx --> pyz, pzy --> pyx
    # For x direction, (xyz) --> (yzx), so pzx --> pxy, pzy --> pxz
    # determine the isometry pzx for z-contraction and x-squeeze
    pzx, errzx, dzx = zCollapseXproj(Arot, Brot,
                                     chi=chi, cg_eps=cg_eps
                                     )

    # permutation x, x', y, y' <---> y, y', x', x
    perm = [2, 3, 1, 0, 4, 5]
    Aperm = Arot.transpose(perm)
    Bperm = Brot.transpose(perm)
    # determine the isometry pxy for z-contraction and y-squeeze
    pzy, errzy, dzy = zCollapseXproj(Aperm, Bperm,
                                     chi=chi, cg_eps=cg_eps
                                     )
    return pzx, pzy, errzx, errzy, dzx, dzy


def rotInd(direction):
    """
    Rotate the leg of the 6-leg tensor A[x, x', y, y', z, z']

    Parameters
    ----------
    direction : str
        direction for contraction. The default is "z".
        "z": do nothing, so we have identity permutation [0, 1, 2, 3, 4, 5]
        "y": (xyz) --> (zxy), so we have permutation [4, 5, 0, 1, 2, 3]
        "x": (xyz) --> (yzx), so we have permutation [2, 3, 4, 5, 0, 1]
    Returns
    -------
    perm : list
        representation of a permutation.
    inv_perm : list
        representation of a inverse permutation to undo perm
    """
    Err_messg = "direction should be choisen among [x,y,z]"
    assert direction in ["x", "y", "z"], Err_messg
    if direction == "z":
        perm = [0, 1, 2, 3, 4, 5]
        inv_perm = [0, 1, 2, 3, 4, 5]
    elif direction == "y":
        perm = [4, 5, 0, 1, 2, 3]
        inv_perm = [2, 3, 4, 5, 0, 1]
    elif direction == "x":
        perm = [2, 3, 4, 5, 0, 1]
        inv_perm = [4, 5, 0, 1, 2, 3]
    return perm, inv_perm


def zCollapseXproj(A, B, chi, cg_eps=1e-8):
    """construct the x projector for z-direction coarse graining

    Args:
        A (TensorCommon): upper tensor
        B (TensorCommon): lower tensor
        chi (int): bond dimension

    Returns:
        p (TensorCommon): 3-leg projector
        d (TensorCommon): eigenvalues of environment
        err (float) : error for the projective truncation
    """
    env_M = ncon([A, B, A.conjugate(), B.conjugate()],
                 [[-1, 2, 3, 4, 1, 9], [-2, 6, 7, 8, 9, 5],
                  [-3, 2, 3, 4, 1, 10], [-4, 6, 7, 8, 10, 5]
                  ])
    (d,
     p,
     err
     ) = env_M.eig([0, 1], [2, 3],
                   hermitian=True,
                   chis=[i+1 for i in range(chi)], eps=cg_eps,
                   trunc_err_func=trunc_err_func,
                   return_rel_err=True)
    return p, err, d


def blockAlongZ(A, B, pzx, pzy, comm=None):
    """Block-transform two tensors along z direction
    using two pairs of isometric tensors

    Args:
        A (TensorCommon): upper tensor
        B (TensorCommon): lower tensor
        pzx (TensorCommon): 3-leg isometry for
            squeezing two legs in x direction
        pzy (TensorCommon): 3-leg isometry for
            squeezing two legs in y direction

    Kwargs:
        comm (MPI.COMM_WOLRD):
            for parallel computation

    Returns:
        Aout (TensorCommon):
            Aout = A.B.pzx.pzx*.pzy.pzy*
            where the . schematically denote
            tensor contraction
    """
    # contraction for coarse graining process
    # This version has computation cost O(chi^11) and
    # memory cost O(chi^8). The memory cost can be reduced to O(chi^6).
    if comm is None:
        Aout = ncon([A, B, pzx.conjugate(), pzy.conjugate(), pzx, pzy],
                    [[1, 2, 6, 7, -5, 5], [8, 9, 3, 4, 5, -6],
                     [1, 8, -1], [6, 3, -3], [2, 9, -2], [7, 4, -4]
                     ])

    #  parallel computation start --
    else:
        # broadcase the input tensors
        A = comm.bcast(A, root=0)
        B = comm.bcast(B, root=0)
        pzx = comm.bcast(pzx, root=0)
        pzy = comm.bcast(pzy, root=0)
        # we fix two indices
        # A   is fixed to   A[:, :, :, j, :, i]
        # B   is fixed to   B[:, :, :, :, i, :]
        # pzy is fixed to   pzy[j, :, :]

        # initialize the output tensor after contraction
        Aout = 0
        rank = comm.Get_rank()
        size = comm.Get_size()
        # job indicator
        jobind = 0
        for i in u1ten.loopleg(A, 5):
            # fix one leg to i
            # Ai = u1ten.fixleg(A, 5, i)
            # Bi = u1ten.fixleg(B, 4, i)
            for j in u1ten.loopleg(A, 3):
                # TAKE CARE OF THE PARALLEL COMPUTATION
                if jobind % size != rank:
                    jobind += 1
                    continue
                # fix one leg to i
                Ai = u1ten.fixleg(A, 5, i)
                Bi = u1ten.fixleg(B, 4, i)
                # fix the other leg to j
                Aij = u1ten.fixleg(Ai, 3, j)
                pzyj = u1ten.fixleg(pzy, 0, j)
                # the following two contraction has
                # memory cost:        O(chi^6), and
                # computational cost: O(chi^8)
                blockup = ncon([Aij, pzx.conjugate(), pzy.conjugate()],
                               [[1, -4, 2, -3], [1, -5, -1], [2, -6, -2]])
                blockdown = ncon([Bi, pzx, pzyj],
                                 [[-5, 1, -6, 2, -3],
                                  [-4, 1, -1], [2, -2]])
                # the final contraction has
                # memory cost:         O(chi^6), and
                # comuputational cost: O(chi^9)
                Aout += ncon([blockup, blockdown],
                             [[-1, -3, -5, 1, 2, 3], [-2, -4, -6, 1, 2, 3]])
                # increase the job indicator for parallel computation
                jobind += 1
        # collective reducing sum operation
        Aout = comm.allreduce(Aout)

    #  parallel computation end
    return Aout


def trunc_err_func(eigv, chi):
    """
    No need to take square since we work with M M'
    whose eigenvalues are themself square of singular values of M
    """
    res = np.sum(eigv[chi:]) / np.sum(eigv)
    return res


# fucntions for sign fixing
def signFix(Aout, isom3dir, Aold, cg_dirs,
            verbose=True):
    """sign fixing procedure for Z2-symmetric tensor

    Args:
        Aout (TensorZ2): 6-leg coarse tensor
        isom3dir (dict): isometries
            - isom3dir["z"] is a list [pzx, pzy] for
            two isometries during blocking in z direction;
            - pzx, pzy are both 3-leg TensorZ2 object
        Aold (TensorZ2): 6-leg old tensor
        cg_dirs (list): order of the hotrg contraction
            - for example ["z", "y", "x"] stands for hotrg
            in z -> y -> x order

    Returns:
        Aout (TensorZ2): after gauge fixing
        isom3dir (dict): after gauge fixing

    """
    # only do the fixing if the shape if
    # 1) Aout and A have the same shape
    if (Aout.shape == Aold.shape):
        if verbose:
            print("---------------")
            print("Sign fixing...")
        (
            Aout,
            signx, signy, signz
        ) = sf.findSigns(Aout, Aold, verbose=verbose)
        isom3dir = signOnIsoms(isom3dir, signx, signy, signz,
                               cg_dirs)
    return Aout, isom3dir


def signOnIsoms(isom3dir, signx, signy, signz,
                cg_dirs):
    """apply the sign vectors to the outer-most isometries
    Currently only written for cg_dirs=["z", "y", "x"]

    Args:
        isom3dir (dict): isometries
        signx (TensorZ2): sign vector in x-direction
        cg_dirs (list): order of the hotrg contraction
            - for example ["z", "y", "x"] stands for hotrg
            in z -> y -> x order

    Returns:
        isom3dir: updated version

    """
    assert cg_dirs == ["z", "y", "x"]
    isom3dir["y"][1] = (isom3dir["y"][1]).multiply_diag(
        signx, axis=2, direction="r"
    )
    isom3dir["x"][0] = (isom3dir["x"][0]).multiply_diag(
        signy, axis=2, direction="r"
    )
    isom3dir["x"][1] = (isom3dir["x"][1]).multiply_diag(
        signz, axis=2, direction="r"
    )
    return isom3dir


# functions for linearization of hotrg
def fullContr(A, isom3dir, cg_dirs=["z", "y", "x"],
              comm=None):
    """One hotrg step, including contractions in 3 directions

    Args:
        A (TensorCommon): input 6-leg tensor
        isom3dir (dict): isometries

    Kwargs:
        comm (MPI.COMM_WORLD): for parallelization

    Returns:
        Aps (List): [A, A', A'', A''']
        - Coarse tensors after z-direction,
        y-direction, and x-direction contractions
        if we choose z -> y -> x order
        - final output tensor is Aout = A'''

    """
    Ap = A * 1.0
    Aps = []
    Aps.append(Ap)
    for direction in cg_dirs:
        pzx, pzy = isom3dir[direction]
        Ap = blockAlongZ(Ap, Ap, pzx, pzy, comm=comm)
        Aps.append(Ap)
    return Aps
