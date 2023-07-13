#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : fet3d.py
# Author            : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
# Date              : 29.06.2023
# Last Modified Date: 29.06.2023
# Last Modified By  : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
"""
Generalization of Evenbly's full environment truncation (FET) in 3D.
- The original paper of Evenbly is explained in 2D, see
Title: Gauge fixing, canonical forms, and optimal truncations
in tensor networks with closed loops
Author: Glen Evenbly
Phys. Rev. B 98, 085155 – Published 31 August 2018
url: https://journals.aps.org/prb/abstract/10.1103/PhysRevB.98.085155

- We generalize it to 3D, by approximating a cube
of 8 copies of 6-leg tensors.
All the relevant environments are construction in `./env3d.py`
"""
from .. import u1ten
from ncon import ncon


def findLr(Gamma, epsilon=1e-10,
           soft=False, chiCut=None):
    """
    Determine the low-rank matrix a la FET.
    This version is almost the same GILT without recursion.
    We will use it to initialize the low-rank matrix
    See function `.fet.findLR` for more details.
    """
    Gamma_pinv = u1ten.pinv(Gamma, eps_mach=epsilon,
                            soft=soft, chiCut=chiCut)
    Lr = ncon([Gamma_pinv, Gamma], [[-1, -2, 1, 2], [1, 2, 3, 3]])
    return Lr


def init_s_gilt(Gamma, chis, epsilon_init,
                init_soft=False):
    """
    Initialize s matrix using `findLr`
    """
    # determine cut bond dimension for pseudo-inverse of Gamma
    if chis is None:
        chiCut = None
    else:
        chiCut = chis**2
    # use baby FET to find low-rank matrix
    Lr = findLr(Gamma, epsilon=epsilon_init,
                soft=init_soft, chiCut=chiCut)
    # split low-rank matrix to find s
    s = Lr.split([0], [1], chis=chis)[0]
    return s, Lr
