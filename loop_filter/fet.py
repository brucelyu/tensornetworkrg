#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : fet.py
# Author            : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
# Date              : 31.10.2022
# Last Modified Date: 31.10.2022
# Last Modified By  : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
"""
This is a implementation of Evenbly's full environment truncation (FET)
See the follow paper,
Title: Gauge fixing, canonical forms, and optimal truncations
in tensor networks with closed loops
Author: Glen Evenbly
Phys. Rev. B 98, 085155 – Published 31 August 2018
url: https://journals.aps.org/prb/abstract/10.1103/PhysRevB.98.085155
"""
from .. import u1ten
from ncon import ncon


def findLR(Gamma, epsilon=1e-13):
    """determine the low-rank matrix from the bond environment
    This is just a prototype to demonstrate how FET works

    The input and output are the same as GILT.
    However, the strategy for finding the low-rank matrix is
    different.
    Also the ε here is the square of the GILT's ε

    Args:
        Gamma (TensorCommon): bond environment
            it is positive semi-definite

    Kwargs:
        epsilon (float): for truncation during pseudoinverse

    Returns:
        lr (TensorCommon): the low-rank matrix

    """
    Gamma_pinv = u1ten.pinv(Gamma, eps_mach=epsilon)
    lr = ncon([Gamma, Gamma_pinv], [[1, 1, 2, 3], [2, 3, -1, -2]])
    return lr


def findMats(Gamma, chis, epsilon=1e-13, iter_max=20):
    """determine half the piece of the low-rank matrix,
    given the bond environment and the squeezed bond dimension
    This is a iterative method, similar to GILTs's recursive approach,
    but it can be made more general to impose reflection symmetry

    Args:
        Gamma (TensorCommon): bond environment
            it is positive semi-definite
        chis (int): squeezed bond dimension

    Kwargs:
        epsilon (float): for truncation during pseudoinverse

    Returns:
        s (TensorCommon): the half piece of the low-rank matrix

    """
    # initialize s
    s = Gamma.eye(Gamma.shape[0])
    # TODO: take care of the u1-tensor slicing later
    # by specifying the indexOrder Array
    s = u1ten.slicing(s, (slice(None), slice(chis)),
                      indexOrder=None)
    # enter the iteration
    for k in range(iter_max):
        print(s.svd([0], [1])[1])
        Gammas = ncon([Gamma, s.conj(), s],
                      [[-1, 1, -3, 2], [1, -2], [2, -4]
                       ])
        Ps = ncon([Gamma, s], [[1, 1, -1, 2], [2, -2]])
        s = updateMats(Gammas, Ps, epsilon=epsilon)
        # hosvd-like truncation
        proj_s = s.svd([1], [0], eps=epsilon*100)[0]
        s = ncon([s, proj_s.conj()], [[-1, 1], [1, -2]])
    return s


def updateMats(Gammas, Ps, epsilon=1e-13):
    """update the squeezer matrix s,
    given the linearlized Gammas and Ps matrices

    Args:
        Gammas (TensorCommon): linearized Gamma
        Ps (TensorCommon): linearized Ps

    Kwargs:
        epsilon (float): for truncation during pseudoinverse

    Returns:
        s (TensorCommon): the half piece of the low-rank matrix

    """
    Gammas_pinv = u1ten.pinv(Gammas, eps_mach=epsilon)
    s = ncon([Ps, Gammas_pinv], [[1, 2], [1, 2, -1, -2]])
    return s
