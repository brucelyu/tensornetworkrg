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
import numpy as np
from ncon import ncon
from .env2d import envUs, envRefSym, envHalfRefSym
from .gilt import Ropt
from . import fet


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


def fet2dReflSym(A, chis, epsilon=1e-13, iter_max=20,
                 epsilon_init=1e-16, bothSides=True,
                 display=False):
    """Reflection-symmetric FET for HOTRG in 2D
    The subsequent HOTRG acts on 4-tensor plaquette
        |     |
     ---A----Areflv*--
        |     |
        |     |
 --Areflh*----Areflhv--
        |     |
    so the FET should act on 4-tensor plaquette
        |        |
 --Areflhv--Lr--Areflh*--
        |        |
        |        |
 --Areflv*--Lr*--A--
        |        |
    where Lr = s @ s*

    We first find and return s, and also return
     |            |
  --Alf-- =  --s--A--s*-- if bothSide=True
     |            |
     or
     |            |
  --Alf-- =  --s--A----   if bothSide=False
     |            |

    Args:
        A (TensorCommon): 4-leg tensor A[i, j, k, l]
        chis (int): squeezed bond dimension

    Kwargs:
        epsilon (float): for truncation during pseudoinverse
        iter_max (int): maximal iteration number
        epsilon_init (float): for truncation during psedoinverse
            during the first baby version FET
        bothSides (boolean): whether to truncate both
            sides of tensor A
        display (boolean): whether to print out information

    Returns:
        Alf (TensorCommon): loop-free tensor
    """
    # The FET should act on a plaquette with
    # A with both horizontal and vertical reflected
    Areflhv = A.transpose([2, 3, 0, 1])
    # construct the half environment for
    # reflection-symmetric FET
    Gamma_h = envHalfRefSym(Areflhv)
    # find s
    s = fet.optMats(Gamma_h, chis, epsilon=epsilon, iter_max=iter_max,
                    epsilon_init=epsilon_init, display=True)
    # FET approximation error, or 1 - fidelity
    err = fet.fidelity2leg(Gamma_h, s)[1]
    if display:
        print("FET error (1 - fidelity) is {:.4e}".format(np.abs(err)))
    # apply s to input tensor A to truncate loops
    if bothSides:
        # truncate both the left and right legs of A
        Alf = ncon([A, s, s.conj()],
                   [[1, -2, 2, -4], [1, -1], [2, -3]])
    else:
        # only truncate the left leg of A
        Alf = ncon([A, s], [[1, -2, -3, -4], [1, -1]])
    return Alf, s, err
