#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : efrg.py
# Author            : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
# Date              : 27.09.2023
# Last Modified Date: 27.09.2023
# Last Modified By  : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
"""
Functions for linearzation of entanglement-free RG (efrg)
"""
from ..loop_filter import fet3d, fet3dloop
from . import block_tensor as bkten3d


def fullContr(A, cg_tens, comm=None,
              ver="base",
              cubeFilter=True,
              loopFilter=True):
    """
    Basically the same as the `.block_tensor.fullContr`
    See the documentation there
    """
    if ver == "base":
        [
            pox, poy, poz, pmx, pmy, pmz, pix, piy, piix,
            sx, sy, sz
        ] = cg_tens
    elif ver == "bistage":
        [
            pox, poy, poz, pmx, pmy, pmz, pix, piy, piix,
            sx, sy, sz, mx, my, mz
        ] = cg_tens
    # (E.1) Absorbing cube-filtering matrices
    # into the main tensor A, such that
    # the 3 outer legs of A is squeezed
    if (ver == "base") or (ver == "bistage" and cubeFilter):
        As = fet3d.absbs(A, sx, sy, sz)
    else:
        As = A * 1.0

    # (C). Apply the hotrg-like block-tensor RG
    # (
    #    As, Asz, Aszy, Aszyx
    # ) = bkten3d.fullContr(As, cg_tens[:-3], comm)

    # (Below the same as `bkten3d.fullContr`)
    # ------------------------------\
    # (C.1) z-direction collapse
    Asz = bkten3d.zblock(
        As, pmx.conj(), pmy.conj(), pix, piy,
        comm=comm
    )
    # (E.2.1) Absorbing loop-filtering matrices
    # into x and y legs of z-collapsed tensor `Az`
    if ver == "bistage" and loopFilter and (mx is not None):
        Asz = fet3dloop.absb_mloopz(Asz, mx, my)

    # (C.2) y-direction collapse
    Aszy = bkten3d.yblock(
        Asz, pmz.conj(), pox.conj(), pmz, piix,
        comm=comm
    )
    # (E.2.2) Absorbing loop-filtering matrices
    # into z legs of y-collapsed tensor `Azy`
    if ver == "bistage" and loopFilter and (mz is not None):
        Aszy = fet3dloop.absb_mloopy(Aszy, mz)

    # (C.3) x-direction collapse
    Aszyx = bkten3d.xblock(
        Aszy, poy.conj(), poz.conj(), poy, poz,
        comm=comm
    )
    # ------------------------------/
    return A, As, Asz, Aszy, Aszyx


def linrgmap(dA, Astar_all, cg_tens,
             refl_c=[0, 0, 0], comm=None,
             ver="base",
             cubeFilter=True,
             loopFilter=True):
    """
    Basically the same as the `.block_tensor.linrgmap`
    See the documentation there
    """
    if ver == "base":
        [
            pox, poy, poz, pmx, pmy, pmz, pix, piy, piix,
            sx, sy, sz
        ] = cg_tens
    elif ver == "bistage":
        [
            pox, poy, poz, pmx, pmy, pmz, pix, piy, piix,
            sx, sy, sz, mx, my, mz
        ] = cg_tens

    [A, As, Asz, Aszy, Aszyx] = Astar_all
    cX, cY, cZ = refl_c
    c2sign = {0: 1, 1: -1}

    # (E.1) cube-filtering matrices linearization
    if (ver == "base") or (ver == "bistage" and cubeFilter):
        dAs = fet3d.absbs(dA, sx, sy, sz)
    else:
        dAs = dA * 1.0

    # (C) Call the linearization of block-tensor RG
    # ------------------------------\
    # dAc = bkten3d.linrgmap(
    #     dAs, Astar_all[1:], cg_tens[:-3],
    #     refl_c, comm
    # )
    # ------------------------------/

    # below are detailed implementation
    # ------------------------------\
    # (C.1) z-direction linearization
    dAsz = (
        bkten3d.zblock2ten(
            dAs, As.transpose([0, 1, 2, 3, 5, 4]).conj(),
            pmx.conj(), pmy.conj(), pix, piy, comm=comm
        ) +
        bkten3d.zblock2ten(
            As, dAs.transpose([0, 1, 2, 3, 5, 4]).conj(),
            pmx.conj(), pmy.conj(), pix, piy, comm=comm
        ) * c2sign[cZ]
    )
    # (E.2.1) Z-loop-filtering matrices linearization
    if ver == "bistage" and loopFilter and (mx is not None):
        dAsz = fet3dloop.absb_mloopz(dAsz, mx, my)

    # (C.2). y-direction linearization
    dAszy = (
        bkten3d.yblock2ten(
            dAsz, Asz.transpose([0, 1, 3, 2, 4, 5]).conj(),
            pmz.conj(), pox.conj(), pmz, piix, comm=comm
        ) +
        bkten3d.yblock2ten(
            Asz, dAsz.transpose([0, 1, 3, 2, 4, 5]).conj(),
            pmz.conj(), pox.conj(), pmz, piix, comm=comm
        ) * c2sign[cY]
    )
    # (E.2.2) Y-loop-filtering matrices linearization
    if ver == "bistage" and loopFilter and (mz is not None):
        dAszy = fet3dloop.absb_mloopy(dAszy, mz)

    # (C.3) x-direction linearization
    dAc = (
        bkten3d.xblock2ten(
            dAszy, Aszy.transpose([1, 0, 2, 3, 4, 5]).conj(),
            poy.conj(), poz.conj(), poy, poz, comm=comm
        ) +
        bkten3d.xblock2ten(
            Aszy, dAszy.transpose([1, 0, 2, 3, 4, 5]).conj(),
            poy.conj(), poz.conj(), poy, poz, comm=comm
        ) * c2sign[cX]
    )
    # ------------------------------/
    return dAc
