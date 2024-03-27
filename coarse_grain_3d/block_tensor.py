#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : block_tensor.py
# Author            : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
# Date              : 14.07.2023
# Last Modified Date: 14.07.2023
# Last Modified By  : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
"""
A HOTRG-like block-tensor coarse graining method
- It is adapted from the usual 3D HOTRG to
    suit our entanglement-filtering process
- It is more general than the usual HOTRG. By adjust the parameter, this method
    interpolates between usual HOTRG and the full block-tensor transformation,
    which applies 4-to-1 isometric tensors to 6 faces
    of a cube of 8 copies of input tensor.
What's more, the reflection symmetry is exactly imposed here.
"""
from ncon import ncon
import numpy as np
from .hotrg import zCollapseXproj
from . import signfix as sf
from .. import u1ten


# I. For determing isometric tensors
def zfind1p(A, chi, which="pmx", cg_eps=1e-8,
            chiSet=None):
    # 0. For leg permutation
    # 0.1 For leg permutation in z direction
    permzleg = [0, 1, 2, 3, 5, 4]
    # 0.2 For leg permutation x, x', y, y' <---> y, y', x, x'
    permxy = [2, 3, 0, 1, 4, 5]
    # 0.3 For leg permutation in x direction
    permxleg = [1, 0, 2, 3, 4, 5]

    if which == "pmx":
        Arela = A * 1.0
    elif which == "pmy":
        # rotate in xy plane
        Arela = A.transpose(permxy)
    elif which == "pix":
        # reflect x legs + conjugate
        Arela = A.transpose(permxleg).conj()
    elif which == "piy":
        # rotate + reflect + conjugate
        Arela = A.transpose(permxy).transpose(permxleg).conj()
    else:
        raise ValueError("`which` value not correct!")

    (
        p, err, d
    ) = zCollapseXproj(
        Arela, Arela.transpose(permzleg).conj(),
        chi, cg_eps=cg_eps,
        chiSet=chiSet
    )
    return p, err, d


def zfindp(A, chiM, chiI, cg_eps=1e-8):
    """
    Determine all 4 projectors for coarse graining
    in z direction
    """
    # I. For two intermediate outer projectors
    # I.1 pmx
    (
        pmx, errmx, dmx
    ) = zfind1p(A, chiM, which="pmx", cg_eps=cg_eps)
    # I.2 pmy
    (
        pmy, errmy, dmy
    ) = zfind1p(A, chiM, which="pmy", cg_eps=cg_eps)

    # II. For two inner projectors
    # II.1 pix
    (
        pix, errix, dix
    ) = zfind1p(A, chiI, which="pix", cg_eps=cg_eps)
    # II.2 piy
    (
        piy, erriy, diy
    ) = zfind1p(A, chiI, which="piy", cg_eps=cg_eps)

    # group projectors
    zprojs = [pmx, pix, pmy, piy]
    zerrs = [errmx, errix, errmy, erriy]
    zds = [dmx, dix, dmy, diy]
    return zprojs, zerrs, zds


def yfindp(Az, chi, chiM, chiII, cg_eps=1e-8,
           chiSet=None):
    """
    Projectors for y-direction coarse graining
    after z-direction block-tensor RG.
    """
    # rotation xyz --> zxy
    Ar = Az.transpose([4, 5, 0, 1, 2, 3])
    # I. pmz
    (
        pmz, errmz, dmz
    ) = zfind1p(Ar, chiM, which="pmx", cg_eps=cg_eps)
    # II. pox
    (
        pox, errox, dox
    ) = zfind1p(Ar, chi, which="pmy", cg_eps=cg_eps,
                chiSet=chiSet)
    # III. piix
    (
        piix, erriix, diix
    ) = zfind1p(Ar, chiII, which="piy", cg_eps=cg_eps)

    # group projectors
    yprojs = [pmz, pox, piix]
    yerrs = [errmz, errox, erriix]
    yds = [dmz, dox, diix]
    return yprojs, yerrs, yds


def xfindp(Azy, chi, cg_eps=1e-8,
           chiSet=None):
    """
    Projectors for x-direction coarse graining
    after z- and y-direction block-tensor RG.
    """
    # rotation xyz --> yzx
    Ar = Azy.transpose([2, 3, 4, 5, 0, 1])
    # I. poy
    (
        poy, erroy, doy
    ) = zfind1p(Ar, chi, which="pmx", cg_eps=cg_eps,
                chiSet=chiSet)
    # II. poz
    (
        poz, erroz, doz
    ) = zfind1p(Ar, chi, which="pmy", cg_eps=cg_eps,
                chiSet=chiSet)
    # group projectors
    xprojs = [poy, poz]
    xerrs = [erroy, erroz]
    xds = [doy, doz]
    return xprojs, xerrs, xds


# II. For block-tensor transformation along a single direction

# /// ------ This contraction reuse the usual HOTRG idea -------- \\\
def zblock2tenN(A, B, pxc, pyc, px, py, comm=None):
    # contraction for coarse graining process
    # This version has computation cost O(chi^11) and
    # memory cost O(chi^8). The memory cost can be reduced to O(chi^6).
    if comm is None:
        Aout = ncon([A, B, pxc, pyc, px, py],
                    [[1, 2, 6, 7, -5, 5], [8, 9, 3, 4, 5, -6],
                     [1, 8, -1], [6, 3, -3], [2, 9, -2], [7, 4, -4]
                     ])
    return Aout


def yblock2tenN(A, B, pzc, pxc, pz, px, comm=None):
    perm = [4, 5, 0, 1, 2, 3]
    inv_perm = [2, 3, 4, 5, 0, 1]
    # Rotate to prototypical position
    Ar = A.transpose(perm)
    Br = B.transpose(perm)
    # Call `zblock2ten` to contract
    AB = zblock2tenN(Ar, Br, pzc, pxc, pz, px, comm)
    # Rotate back to absolute position
    AB = AB.transpose(inv_perm)
    return AB


def xblock2tenN(A, B, pyc, pzc, py, pz, comm=None):
    perm = [2, 3, 4, 5, 0, 1]
    inv_perm = [4, 5, 0, 1, 2, 3]
    # Rotate to prototypical position
    Ar = A.transpose(perm)
    Br = B.transpose(perm)
    # Call `zblock2ten` to contract
    AB = zblock2tenN(Ar, Br, pyc, pzc, py, pz, comm)
    # Rotate back to absolute position
    AB = AB.transpose(inv_perm)
    return AB
# \\\ ------ ------------------------------------------- -------- ///


# - A more efficient contraction for `block2ten` functions
def zblock2ten(A, B, pxc, pyc, px, py, comm=None):
    # z-direction collapse of HOTRG-like block-tensor map
    # The relative bond dimensions are χ, χm, xi
    # Computational cost: χ^9 χm^2, which is χ^11 for χm ~ χ
    # Memory imprint    : χ^6 χm^2, which is χ^8  for χm ~ χ
    if comm is None:
        Aout = ncon([A, B, pxc, pyc, px, py],
                    [
                        [1, 6, 2, 8, -5, 3], [4, 7, 5, 9, 3, -6],
                        [1, 4, -1], [2, 5, -3], [6, 7, -2], [8, 9, -4]
                    ])
    else:
        # parallelization codes
        # 0. broadcase the input tensors
        A = comm.bcast(A, root=0)
        B = comm.bcast(B, root=0)
        pxc = comm.bcast(pxc, root=0)
        pyc = comm.bcast(pyc, root=0)
        px = comm.bcast(px, root=0)
        py = comm.bcast(py, root=0)
        # 1. initialize the output tensor after contraction
        Aout = 0  # for idle process, the contribution is 0
        # 2. determine the job of each process
        rank = comm.Get_rank()   # rank of current process
        size = comm.Get_size()   # size of process
        jobind = 0               # indicator of the job in the for loop
        for i in u1ten.loopleg(A, 3):
            for j in u1ten.loopleg(B, 1):
                # parallelization process: χ^2
                # check whether the job belongs to current process
                if jobind % size != rank:
                    jobind += 1
                    continue
                # fix legs to i
                Ai = u1ten.fixleg(A, 3, i)
                pyi = u1ten.fixleg(py, 0, i)
                # fix legs to j
                Bj = u1ten.fixleg(B, 1, j)
                pxj = u1ten.fixleg(px, 1, j)
                # The following contraction has
                # CPU cost: χs^4 χ^3 χm^2
                # Mem cost: χs^2 χm^2 χi^2
                Aout += ncon([Ai, Bj, pxc, pyc, pxj, pyi],
                             [
                                 [1, 6, 2, -5, 3], [4, 5, 9, 3, -6],
                                 [1, 4, -1], [2, 5, -3], [6, -2], [9, -4]
                             ])
                # increase the job indicator for parallel computation
                jobind += 1
        # 3. collective reducing sum operation
        Aout = comm.allreduce(Aout)
        # parallelization codes end
    return Aout


def yblock2ten(A, B, pzc, pxc, pz, px, comm=None):
    # y-direction collapse of HOTRG-like block-tensor map
    # The relative bond dimensions are χ, χm, χi, χii
    # Computational cost: χ^3 χm^5 χi^3, which is χ^8 χi^3 for χm ~ χ
    # Memory imprint    : χ^2 χm^4 χi^2, which is χ^6 χi^2 for χm ~ χ
    if comm is None:
        Aout = ncon([A, B, pzc, pxc, pz, px],
                    [
                        [7, 8, -3, 4, 1, 2], [3, 9, 4, -4, 5, 6],
                        [1, 5, -5], [7, 3, -1], [2, 6, -6], [8, 9, -2]
                    ])
    else:
        # parallelization codes
        # 0. broadcase the input tensors
        A = comm.bcast(A, root=0)
        B = comm.bcast(B, root=0)
        pzc = comm.bcast(pzc, root=0)
        pxc = comm.bcast(pxc, root=0)
        pz = comm.bcast(pz, root=0)
        px = comm.bcast(px, root=0)
        # 1. initialize the output tensor after contraction
        Aout = 0  # for idle process, the contribution is 0
        # 2. determine the job of each process
        rank = comm.Get_rank()   # rank of current process
        size = comm.Get_size()   # size of process
        jobind = 0               # indicator of the job in the for loop
        for i in u1ten.loopleg(A, 1):
            for j in u1ten.loopleg(B, 1):
                # parallelization process: χi^2
                # check whether the job belongs to current process
                if jobind % size != rank:
                    jobind += 1
                    continue
                # fix legs to i
                Ai = u1ten.fixleg(A, 1, i)
                pxi = u1ten.fixleg(px, 0, i)
                # fix legs to j
                Bj = u1ten.fixleg(B, 1, j)
                pxij = u1ten.fixleg(pxi, 0, j)
                # The following contraction has
                # CPU cost: χs^5 χ^1 χm^2 χi
                # Mem cost: χs^4 χm^2 χi^1 + χ^1 χs^2 χm^2 χii^1
                Aout += ncon([Ai, Bj, pzc, pxc, pz, pxij],
                             [
                                 [7, -3, 4, 1, 2], [3, 4, -4, 5, 6],
                                 [1, 5, -5], [7, 3, -1], [2, 6, -6], [-2]
                             ])
                # increase the job indicator for parallel computation
                jobind += 1

        # 3. collective reducing sum operation
        Aout = comm.allreduce(Aout)
        # parallelization codes end
    return Aout


def xblock2ten(A, B, pyc, pzc, py, pz, comm=None):
    # x-direction collapse of HOTRG-like block-tensor map
    # The relative bond dimensions are χ, χm, χii
    # Computational cost: χ^6 χm^4 χii, which is χ^10 χii for χm ~ χ
    # Memory imprint    : χ^3 χm^4 χii, which is χ^7 χii for χm ~ χ
    if comm is None:
        Aout = ncon([A, B, pyc, pzc, py, pz],
                    [
                        [-1, 5, 1, 2, 6, 7], [5, -2, 8, 9, 3, 4],
                        [1, 8, -3], [6, 3, -5], [2, 9, -4], [7, 4, -6]
                    ])
    else:
        # parallelization codes
        # 0. broadcase the input tensors
        A = comm.bcast(A, root=0)
        B = comm.bcast(B, root=0)
        pyc = comm.bcast(pyc, root=0)
        pzc = comm.bcast(pzc, root=0)
        py = comm.bcast(py, root=0)
        pz = comm.bcast(pz, root=0)
        # 1. initialize the output tensor after contraction
        Aout = 0  # for idle process, the contribution is 0
        # 2. determine the job of each process
        rank = comm.Get_rank()   # rank of current process
        size = comm.Get_size()   # size of process
        jobind = 0               # indicator of the job in the for loop
        for i in u1ten.loopleg(A, 1):
            # parallelization process: χii
            # check whether the job belongs to current process
            if jobind % size != rank:
                jobind += 1
                continue
            # fix legs to i
            Ai = u1ten.fixleg(A, 1, i)
            Bi = u1ten.fixleg(B, 0, i)
            # The following contraction has
            # CPU cost: χ^6 χMs^4
            # Mem cost: χ^3 χMs^4 + χ^6
            Aout += ncon([Ai, Bi, pyc, pzc, py, pz],
                         [
                             [-1, 1, 2, 6, 7], [-2, 8, 9, 3, 4],
                             [1, 8, -3], [6, 3, -5], [2, 9, -4], [7, 4, -6]
                         ])
            # increase the job indicator for parallel computation
            jobind += 1

        # 3. collective reducing sum operation
        Aout = comm.allreduce(Aout)
        # parallelization codes end
    return Aout


# /// ------ This contraction reuse the usual HOTRG idea -------- \\\
# All 1-direction tensor contraction is calling this as prototype
# including `yblock` and `xblock`
def zblockN(A, pxc, pyc, px, py, comm=None):
    # contraction for coarse graining process
    # This version has computation cost O(chi^11) and
    # memory cost O(chi^8). The memory cost can be reduced to O(chi^6).
    Aout = zblock2tenN(A, A.transpose([0, 1, 2, 3, 5, 4]).conj(),
                       pxc, pyc, px, py, comm)
    return Aout


# This calls `zblockN`
def yblockN(Az, pzc, pxc, pz, px, comm=None):
    # rotate to the prototpye position
    # rotation xyz --> zxy
    Ar = Az.transpose([4, 5, 0, 1, 2, 3])
    # call `zblock` to contract
    Azy = zblockN(Ar, pzc, pxc, pz, px, comm=comm)
    # rotate back to absolute position
    Azy = Azy.transpose([2, 3, 4, 5, 0, 1])
    return Azy


# This calls `zblockN`
def xblockN(Azy, pyc, pzc, py, pz, comm=None):
    # rotate to the prototpye position
    # rotation xyz --> yzx
    Ar = Azy.transpose([2, 3, 4, 5, 0, 1])
    # call `zblock` to contract
    Ac = zblockN(Ar, pyc, pzc, py, pz, comm=comm)
    # rotate back
    Ac = Ac.transpose([4, 5, 0, 1, 2, 3])
    return Ac
# \\\ ------ ------------------------------------------- -------- ///


# - A more efficient contraction for `block` functions
# - they just call the corresponding `block2ten` functions
def zblock(A, pxc, pyc, px, py, comm=None):
    Aout = zblock2ten(
        A, A.transpose([0, 1, 2, 3, 5, 4]).conj(),
        pxc, pyc, px, py, comm
    )
    return Aout


def yblock(Az, pzc, pxc, pz, px, comm=None):
    Aout = yblock2ten(
        Az, Az.transpose([0, 1, 3, 2, 4, 5]).conj(),
        pzc, pxc, pz, px, comm
    )
    return Aout


def xblock(Azy, pyc, pzc, py, pz, comm=None):
    Aout = xblock2ten(
        Azy, Azy.transpose([1, 0, 2, 3, 4, 5]).conj(),
        pyc, pzc, py, pz, comm
    )
    return Aout


# III. Full block-tensor transformation
def blockrg(A, chi, chiM, chiI, chiII,
            cg_eps=1e-10, display=True,
            chiSet=None):
    """tensor coarse grain

    Args:
        A (TensorCommon): 6-leg tensor
        chi (int): output bond dimension
        chiM (int): intermediate bond dimension
        chiI (int): first inner bond dimension
        chiII (int): second inner bond dimension

    Kwargs:
        cg_eps (float):
        display (boolean):

    Returns: Aout, 6-leg coarser tensor

    """
    if display:
        print("--------------------")
        print("--------------------")
    # I.1 z direction
    zpjs, zerrs, zds = zfindp(A, chiM, chiI,
                              cg_eps=cg_eps)
    pmx, pix, pmy, piy = zpjs
    A = zblock(
        A, pmx.conj(), pmy.conj(), pix, piy
    )
    # I.2 y direction
    ypjs, yerrs, yds = yfindp(A, chi, chiM, chiII,
                              cg_eps=cg_eps,
                              chiSet=chiSet)
    pmz, pox, piix = ypjs
    A = yblock(
        A, pmz.conj(), pox.conj(), pmz, piix
    )
    # I.3 x direction
    xpjs, xerrs, xds = xfindp(A, chi,
                              cg_eps=cg_eps,
                              chiSet=chiSet)
    poy, poz = xpjs
    A = xblock(
        A, poy.conj(), poz.conj(), poy, poz
    )
    if display:
        print("Brief summary of block-tensor RG errors...")
        print("I. Outmost errors: (χ = {:d})".format(chi))
        print("x = {:.2e}, y = {:.2e}, z = {:.2e}".format(
                  yerrs[1], xerrs[0], xerrs[1]
              ))
        print("II. Intermediate errors: (χm = {:d})".format(chiM))
        print("   Actually χmx = {}, χmy = {}, χmz = {}".format(
            pmx.shape[2], pmy.shape[2], pmz.shape[2]
        ))
        print("x = {:.2e}, y = {:.2e}, z = {:.2e}".format(
                  zerrs[0], zerrs[2], yerrs[0]
              ))
        print("III. Inner-cube errors:",
              "(χi = {:d}, χii = {:d})".format(chiI, chiII))
        print("   Actually χix = {}, χiy = {}, χiix = {}".format(
            pix.shape[2], piy.shape[2], piix.shape[2]
        ))
        print("xin = {:.2e}, yin = {:.2e}, xinin = {:.2e}".format(
                  zerrs[1], zerrs[3], yerrs[2]
              ))
        print("x-direction RG spectrum is")
        xarr = -np.sort(-yds[1].to_ndarray())
        print(xarr/xarr[0])
    return (A, pox, poy, poz, pmx, pmy, pmz,
            pix, piy, piix, xerrs, yerrs, zerrs)


# IV. Sign fixing
def signFix(Aout, Aold, pox, poy, poz,
            verbose=True):
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
        # absorb three sign matrices into the
        # outmost isometric tensors
        pox, poy, poz = signOnPout(
            pox, poy, poz,
            signx, signy, signz
        )
    return Aout, pox, poy, poz


def signOnPout(pox, poy, poz, signx, signy, signz):
    pox = pox.multiply_diag(
        signx, axis=2, direction="r"
    )
    poy = poy.multiply_diag(
        signy, axis=2, direction="r"
    )
    poz = poz.multiply_diag(
        signz, axis=2, direction="r"
    )
    return pox, poy, poz


# V. Linearization of the block-tensor RG transformation
# In this procedure, all isometric tensors are treated as known objects
def fullContr(A, isom_all, comm=None):
    """Full contraction of block-tensor transformation
    This is almost the same as `blockrg`

    Args:
        A (TensorCommon): 6-leg main tensor
        isom_all (List): list of isometric tensors

    Kwargs:
        comm (MPI.COMM_WORLD): for parallelization

    Returns:
        Aps (List): [A, A', A'', A''']
        - Coarse tensors after z-direction,
        y-direction, and x-direction contractions
        if we choose z -> y -> x order
        - final output tensor is Aout = A'''

    """
    # read all the isometric tensors
    [pox, poy, poz, pmx, pmy, pmz, pix, piy, piix] = isom_all
    Az = zblock(
                A, pmx.conj(), pmy.conj(), pix, piy
    )
    Azy = yblock(
                Az, pmz.conj(), pox.conj(), pmz, piix
    )
    Azyx = xblock(
                Azy, poy.conj(), poz.conj(), poy, poz
    )
    return A, Az, Azy, Azyx


def linrgmap_OLD(deltaA, Astar_all, isom_all,
             refl_c=[0, 0, 0], comm=None):
    """Construct linearized block-tensor RG equation
    Lattice-reflection symmetry is utilized here to
    build linearized RG in different charge sectors.

    Args:
        deltaA (TensorCommon): 6-leg tensor
            perturbation to the fixed-point tensor
        Astar_all (List): [A, Az, Azy, Azyx]
            Fix-point tensor and its intermediate
            renormalized ones
        isom_all (List): list of isometric tensors

    Kwargs:
        refl_c (list): lattice-reflection charge
            Choose among: [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]
                          [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]
        comm (MPI.COMM_WORLD): for parallelization

    Returns:
        deltaAc (TensorCommon): 6-leg tensor
            renormalized perturbation
    """
    [pox, poy, poz, pmx, pmy, pmz, pix, piy, piix] = isom_all
    [A, Az, Azy, Azyx] = Astar_all
    cX, cY, cZ = refl_c
    c2sign = {0: 1, 1: -1}
    # I. z-direction linearization
    # call `zblock2ten` to contract
    deltaAz = (
        zblock2ten(
            deltaA, A, pmx.conj(), pmy.conj(), pix, piy, comm
        ) +
        c2sign[cZ] * zblock2ten(
            A, deltaA, pmx.conj(), pmy.conj(), pix, piy, comm
        )
    )
    # II. y-direction linearization
    # rotate to the prototpye position
    # rotation xyz --> zxy
    Azr = Az.transpose([4, 5, 0, 1, 2, 3])
    deltaAzr = deltaAz.transpose([4, 5, 0, 1, 2, 3])
    # call `zblock2ten` to contract
    deltaAzy = (
        zblock2ten(
            deltaAzr, Azr, pmz.conj(), pox.conj(), pmx, piix, comm
        ) +
        c2sign[cY] * zblock2ten(
            Azr, deltaAzr, pmz.conj(), pox.conj(), pmx, piix, comm
        )
    )
    # III. x-direction linearization
    # rotate to the prototpye position
    Azyr = Azy.transpose([2, 3, 4, 5, 0, 1])
    deltaAzyr = deltaAzy.transpose([4, 5, 0, 1, 2, 3])
    # call `zblock2ten` to contract
    deltaAc = (
        zblock2ten(
            deltaAzyr, Azyr, poy.conj(), poz.conj(), poy, poz, comm
        ) +
        c2sign[cX] * zblock2ten(
            Azyr, deltaAzyr, poy.conj(), poz.conj(), poy, poz, comm
        )
    )
    # rotate back to absolute position
    deltaAc = deltaAc.transpose([4, 5, 0, 1, 2, 3])
    return deltaAc


def linrgmap(dA, Astar_all, isom_all,
             refl_c=[0, 0, 0], comm=None):
    """Construct linearized block-tensor RG equation
    Lattice-reflection symmetry is utilized here to
    build linearized RG in different charge sectors.

    Args:
        deltaA (TensorCommon): 6-leg tensor
            perturbation to the fixed-point tensor
        Astar_all (List): [A, Az, Azy, Azyx]
            Fix-point tensor and its intermediate
            renormalized ones
        isom_all (List): list of isometric tensors

    Kwargs:
        refl_c (list): lattice-reflection charge
            Choose among: [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]
                          [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]
        comm (MPI.COMM_WORLD): for parallelization

    Returns:
        deltaAc (TensorCommon): 6-leg tensor
            renormalized perturbation
    """
    [pox, poy, poz, pmx, pmy, pmz, pix, piy, piix] = isom_all
    [A, Az, Azy, Azyx] = Astar_all
    cX, cY, cZ = refl_c
    c2sign = {0: 1, 1: -1}
    # I. z-direction linearization
    dAz = (
        zblock2ten(
            dA, A.transpose([0, 1, 2, 3, 5, 4]).conj(),
            pmx.conj(), pmy.conj(), pix, piy, comm=comm
        ) +
        zblock2ten(
            A, dA.transpose([0, 1, 2, 3, 5, 4]).conj(),
            pmx.conj(), pmy.conj(), pix, piy, comm=comm
        ) * c2sign[cZ]
    )
    # II. y-direction linearization
    dAzy = (
        yblock2ten(
            dAz, Az.transpose([0, 1, 3, 2, 4, 5]).conj(),
            pmz.conj(), pox.conj(), pmz, piix, comm=comm
        ) +
        yblock2ten(
            Az, dAz.transpose([0, 1, 3, 2, 4, 5]).conj(),
            pmz.conj(), pox.conj(), pmz, piix, comm=comm
        ) * c2sign[cY]
    )
    # III. x-direction linearization
    dAc = (
        xblock2ten(
            dAzy, Azy.transpose([1, 0, 2, 3, 4, 5]).conj(),
            poy.conj(), poz.conj(), poy, poz, comm=comm
        ) +
        xblock2ten(
            Azy, dAzy.transpose([1, 0, 2, 3, 4, 5]).conj(),
            poy.conj(), poz.conj(), poy, poz, comm=comm
        ) * c2sign[cX]
    )
    return dAc

# Useful for algorithm development
def moErrs(A, chi, chiM, chiI, chiII,
           firsterr=True):
    """
    Various RG errors if we coarse grain A
    """
    # determine z projectors
    zpjs, zerr = zfindp(A, chiM, chiI)[:2]
    if firsterr:
        return zerr
    else:
        pmx, pix, pmy, piy = zpjs
        # contract z direction
        Az = zblock(A, pmx.conj(), pmy.conj(), pix, piy)
        # determine y projectors
        ypjs, yerr = yfindp(Az, chi, chiM, chiII)[:2]
        errmx, errix, errmy, erriy = zerr
        errmz, errox, erriix = yerr
        return errmx, errmy, errmz, errox, errix, erriy, erriix
