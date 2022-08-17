#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : trg_evenbly.py
# Author            : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
# Date              : 17.08.2022
# Last Modified Date: 17.08.2022
# Last Modified By  : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>

from ncon import ncon


def dotrg(Ain, chiM, chiH, chiV, dtol=1e-16):
    """
    Block 4 tensors into 1 coarser tensor according to
    the modified version of trg by Evenbly.

    We assume reflection symmetry for the input tensor Ain
    See Evenbly's website here: https://www.tensors.net/p-trg

    Input:
    ------
    Ain: abeliantensors
        4-leg tensor
    chiM, chiH, chiV: integer
        - chiH and chiV are horizontal and vertical output
        bond dimension
        - chiM is bond dimension in middle step, designed
        to reduce the computation cost from HOTRG's χ^7 to χ^6
    dtol: float
        a small number,
        eigenvalue threshold for automatic truncate indication of indices

    Output:
    ------
    Aout: abeliantensor
        4-leg coarser tensor, with Frobenius norm 1
    q, v, w: abeliantensor
        3-leg isometric tensors to squeeze the legs
    SPerr_list: list
        length-3 list, square of approximation errors
        for three projective truncations like
        |P @ A - A|^2
    """
    q, SPerrq = opt_q(Ain, chiM, dtol=dtol)
    v, SPerrv = opt_v(Ain, q, chiH, dtol=dtol)
    w, SPerrw = opt_w(Ain, q, chiV, dtol=dtol)
    Aout = block_4tensor(Ain, q, v, w)
    # pull out the norm (define a new function)
    Anorm = LA.norm(Aout)
    Aout = Aout / Anorm
    SPerr_list = [SPerrq, SPerrv, SPerrw]
    return Aout, q, v, w, Anorm, SPerr_list
