#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : cleanLoop.py
# Author            : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
# Date              : 25.10.2022
# Last Modified Date: 25.10.2022
# Last Modified By  : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
"""
Graph-independent way of cutting out loops in tensor network
"""
from ncon import ncon
from .env2d import envUs, envRefSym
from .gilt import Ropt


def gilts2dReflSym(A, epsilon=1e-6, convergence_eps=1e-2,
                   bothSides=False):
    """impose reflection symmetry when performing
    the Markus's technique of GILT on 2D plaquettes.

    The end result is that we only need to determine
    one single low-rank matrix R, which just
    truncates the horizontal bonds.
      j|
     i--A--k,
       l|
     and the plaquette looks like
        | (I) |
     ---A--R--Arefl---
    (IV)|     |(II)
        |     |
     ---A-----Arefl---
        |(III)|

    Args:
        A (TensorCommon): 4-leg tensor A[i, j, k, l]

    Returns: TODO

    """
    legenv = envRefSym(A)
    U, s = envUs(legenv, eps=1e-14)
    Rhalf, counter = Ropt(
        U, s, epsilon=epsilon, convergence_eps=convergence_eps,
        verbose=False
    )
    # loop-free tensor A
    if not bothSides:
        Alf = ncon([A, Rhalf], [[-1, -2, 1, -4], [1, -3]])
    else:
        Alf = ncon([A, Rhalf, Rhalf], [[1, -2, 2, -4], [1, -1], [2, -3]])
    return Alf, Rhalf
