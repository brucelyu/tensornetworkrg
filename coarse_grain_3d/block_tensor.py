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


# I. For determing isometric tensors
def zfind1p(A, chi, which="pmx", cg_eps=1e-8):
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
        chi)
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


def yfindp(Az, chi, chiM, chiII, cg_eps=1e-8):
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
    ) = zfind1p(Ar, chi, which="pmy", cg_eps=cg_eps)
    # III. piix
    (
        piix, erriix, diix
    ) = zfind1p(Ar, chiII, which="piy", cg_eps=cg_eps)

    # group projectors
    yprojs = [pmz, pox, piix]
    yerrs = [errmz, errox, erriix]
    yds = [dmz, dox, diix]
    return yprojs, yerrs, yds


def xfindp(Azy, chi, cg_eps=1e-8):
    """
    Projectors for x-direction coarse graining
    after z- and y-direction block-tensor RG.
    """
    # rotation xyz --> yzx
    Ar = Azy.transpose([2, 3, 4, 5, 0, 1])
    # I. poy
    (
        poy, erroy, doy
    ) = zfind1p(Ar, chi, which="pmx", cg_eps=cg_eps)
    # II. poz
    (
        poz, erroz, doz
    ) = zfind1p(Ar, chi, which="pmy", cg_eps=cg_eps)
    # group projectors
    xprojs = [poy, poz]
    xerrs = [erroy, erroz]
    xds = [doy, doz]
    return xprojs, xerrs, xds


# II. For block-tensor transformation along a single direction

# All 1-direction tensor contraction is calling this as prototype
# including `yblock` and `xblock`
def zblock(A, pxc, pyc, px, py, comm=None):
    # contraction for coarse graining process
    # This version has computation cost O(chi^11) and
    # memory cost O(chi^8). The memory cost can be reduced to O(chi^6).
    if comm is None:
        Aout = ncon([A, A.conj(), pxc, pyc, px, py],
                    [[1, 2, 6, 7, -5, 5], [8, 9, 3, 4, -6, 5],
                     [1, 8, -1], [6, 3, -3], [2, 9, -2], [7, 4, -4]
                     ])
    return Aout


# This calls `zblock`
def yblock(Az, pzc, pxc, pz, px, comm=None):
    # rotate to the prototpye position
    # rotation xyz --> zxy
    Ar = Az.transpose([4, 5, 0, 1, 2, 3])
    # call `zblock` to contract
    Azy = zblock(Ar, pzc, pxc, pz, px, comm=comm)
    # rotate back to absolute position
    Azy = Azy.transpose([2, 3, 4, 5, 0, 1])
    return Azy


# This calls `zblock`
def xblock(Azy, pyc, pzc, py, pz, comm=None):
    # rotate to the prototpye position
    # rotation xyz --> yzx
    Ar = Azy.transpose([2, 3, 4, 5, 0, 1])
    # call `zblock` to contract
    Ac = zblock(Ar, pyc, pzc, py, pz, comm=comm)
    # rotate back
    Ac = Ac.transpose([4, 5, 0, 1, 2, 3])
    return Ac


# III. Full block-tensor transformation
def blockrg(A, chi, chiM, chiI, chiII,
            cg_eps=1e-10, display=True):
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
                              cg_eps=cg_eps)
    pmz, pox, piix = ypjs
    A = yblock(
        A, pmz.conj(), pox.conj(), pmz, piix
    )
    # I.3 x direction
    xpjs, xerrs, xds = xfindp(A, chi,
                              cg_eps=cg_eps)
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
        print("x = {:.2e}, y = {:.2e}, z = {:.2e}".format(
                  zerrs[0], zerrs[2], yerrs[0]
              ))
        print("III. Inner-cube errors:",
              "(χi = {:d}, χii = {:d})".format(chiI, chiII))
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
