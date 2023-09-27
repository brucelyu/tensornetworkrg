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
from ..loop_filter import fet3d
from . import block_tensor as bkten3d


def fullContr(A, cg_tens, comm=None):
    """
    Basically the same as the `.block_tensor.fullContr`
    See the documentation there
    """
    [
        pox, poy, poz, pmx, pmy, pmz, pix, piy, piix,
        sx, sy, sz
    ] = cg_tens
    # I. Absorbing entanglement-filtering matrices
    # into the main tensor A, such that
    # the 3 outer legs of A is squeezed
    As = fet3d.absbs(A, sx, sy, sz)

    # II. Apply the hotrg-like block-tensor RG
    # (
    #    As, Asz, Aszy, Aszyx
    # ) = bkten3d.fullContr(As, cg_tens[:-3], comm)

    # (Below the same as `bkten3d.fullContr`)
    # ------------------------------\
    Asz = bkten3d.zblock(
        As, pmx.conj(), pmy.conj(), pix, piy
    )
    Aszy = bkten3d.yblock(
        Asz, pmz.conj(), pox.conj(), pmz, piix
    )
    Aszyx = bkten3d.xblock(
        Aszy, poy.conj(), poz.conj(), poy, poz
    )
    # ------------------------------/
    return A, As, Asz, Aszy, Aszyx


def linrgmap(dA, Astar_all, cg_tens,
             refl_c=[0, 0, 0], comm=None):
    """
    Basically the same as the `.block_tensor.linrgmap`
    See the documentation there
    """
    [
        pox, poy, poz, pmx, pmy, pmz, pix, piy, piix,
        sx, sy, sz
    ] = cg_tens
    [A, As, Asz, Aszy, Aszyx] = Astar_all
    cX, cY, cZ = refl_c
    c2sign = {0: 1, 1: -1}

    # 0. entanglement-filtering matrices linearization
    dAs = fet3d.absbs(dA, sx, sy, sz)

    # 1. Call the linearization of block-tensor RG
    # ------------------------------\
    # dAc = bkten3d.linrgmap(
    #     dAs, Astar_all[1:], cg_tens[:-3],
    #     refl_c, comm
    # )
    # ------------------------------/

    # below are detailed implementation
    # ------------------------------\
    # I. z-direction linearization
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
    # II. y-direction linearization
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
    # III. x-direction linearization
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
