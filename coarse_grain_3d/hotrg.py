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

    #  parallel computation
    else:
        # we fix two indices
        # A   is fixed to   A[:, :, :, j, :, i]
        # B   is fixed to   B[:, :, :, :, i, :]
        # pzy is fixed to   pzy[j, :, :]

        # initialize the output tensor after contraction
        Aout = 0
        # TAKE CARE OF PARALLEL COMPUTATION
        rank = comm.Get_rank()
        size = comm.Get_size()
        # job indicator
        jobind = 0
        # TODO
        raise NotImplementedError("Not implemented yet")
    return Aout


def trunc_err_func(eigv, chi):
    """
    No need to take square since we work with M M'
    whose eigenvalues are themself square of singular values of M
    """
    res = np.sum(eigv[chi:]) / np.sum(eigv)
    return res
