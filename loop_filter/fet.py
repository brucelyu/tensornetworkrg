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
import numpy as np


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


def findMats(Gamma, chis, epsilon=1e-13, iter_max=20,
             epsilon_init=1e-16, display=False):
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
    # initialize s matrix using the baby version FET
    # s = Gamma.eye(Gamma.shape[0])
    lr = findLR(Gamma, epsilon=epsilon_init)
    # split lr to find s
    s, ds = lr.split([0], [1], return_sings=True)[:2]
    s = u1ten.slicing(s, (slice(None), slice(chis)),
                      indexOrder=(ds, ds))
    # approximation metric
    # 1. fidelity and 1 - fidelity
    lr = s2lr(s)
    f, err = fidelity(Gamma, lr)
    # 2. entanglement entropy
    ee = entropy(lr)
    if display:
        print("The initial 1 - fidelity is {:.4e}".format(err))
        print("The initial entropy for",
              "the low-rank matrix is {:.2f}".format(ee))
    # enter the iteration
    for k in range(iter_max):
        # print(s.svd([0], [1])[1])
        # update s
        Gammas = ncon([Gamma, s.conj(), s],
                      [[-1, 1, -3, 2], [1, -2], [2, -4]
                       ])
        Ps = ncon([Gamma, s], [[1, 1, -1, 2], [2, -2]])
        s = updateMats(Gammas, Ps, epsilon=epsilon)
        # hosvd-like truncating right leg of s
        proj_s = s.svd([1], [0], eps=epsilon*100)[0]
        s = ncon([s, proj_s.conj()], [[-1, 1], [1, -2]])
        # approximation metric
        # 1. fidelity and 1 - fidelity
        lr = s2lr(s)
        f, err = fidelity(Gamma, lr)
        # 2. entanglement entropy
        ee = entropy(lr)
        if display:
            print("The 1 - fidelity is {:.4e}".format(err))
            print("The entropy for",
                  "the low-rank matrix is {:.2f}".format(ee))
    return s


def optMats(Gamma_h, chis, epsilon=1e-13, iter_max=20,
            epsilon_init=1e-16, display=False):
    """determine the half piece of the low-rank matrix
    This scheme preserves the reflection symmetry of the plaquette
    environment in both directions.

    Args:
        Gamma_h (TensorCommon): half environment tensor
        chis (int): squeezed bond dimension

    Kwargs:
        epsilon (float): for truncation during pseudoinverse
        iter_max (int): maximal iteration number
        display (boolean): print out information

    Returns:
        s (TensorCommon): the half piece of the low-rank matrix

    """
    # initialize s matrix using the baby version FET
    # s = Gamma_h.eye(Gamma_h.shape[0])
    GammaBaby = ncon([Gamma_h, Gamma_h.conj()],
                     [[-1, -3, 1, 2], [-2, -4, 1, 2]])
    lr = findLR(GammaBaby, epsilon=epsilon_init)
    # split lr to find s
    s, ds = lr.split([0], [1], return_sings=True)[:2]
    s = u1ten.slicing(s, (slice(None), slice(chis)),
                      indexOrder=(ds, ds))
    # approximation metric
    # 1. fidelity and 1 - fidelity
    f, err = fidelity2leg(Gamma_h, s)[:2]
    # 2. entanglement entropy
    lr = s2lr(s)
    ee = entropy(lr)
    if display:
        print("The initial 1 - fidelity is {:.4e}".format(err))
        print("The initial entropy for",
              "the low-rank matrix is {:.4f}".format(ee))
        print("The spectrum of the low-rank matrix is")
        print(s.svd([0], [1])[1])
    # enter the iteration
    for k in range(iter_max):
        # update s
        Gamma_hss = ncon([Gamma_h, s.conj(), s],
                         [[-1, -2, 1, 2], [1, -3], [2, -4]])
        Gammas = ncon([Gamma_hss, Gamma_hss.conj(), s.conj(), s],
                      [[-1, -3, 3, 4], [1, 2, 3, 4], [1, -2], [2, -4]])
        Gamma_hs = ncon([Gamma_h, s], [[-1, -2, -3, 1], [1, -4]])
        Ps = ncon([Gamma_hs, Gamma_hs.conj(), s],
                  [[1, -1, 3, 4], [1, 2, 3, 4], [2, -2]])
        s = updateMats(Gammas, Ps, epsilon=epsilon)
        # hosvd-like truncating right leg of s
        proj_s = s.svd([1], [0], eps=epsilon*100)[0]
        s = ncon([s, proj_s.conj()], [[-1, 1], [1, -2]])
        # approximation metric
        # 1. fidelity and 1 - fidelity
        f, err = fidelity2leg(Gamma_h, s)[:2]
        # 2. entanglement entropy
        lr = s2lr(s)
        ee = entropy(lr)
        if display:
            print("This is the {:d}-th iteration".format(k))
            print("The 1 - fidelity is {:.4e}".format(err))
            print("The entropy for",
                  "the low-rank matrix is {:.4f}".format(ee))
            print("The spectrum of the low-rank matrix is")
            print(s.svd([0], [1])[1])
        if chis is None:
            # for a GILT-like procedure
            # stop the iteration when we get a projector
            chisShould = np.exp(ee)
            chisCur = s.flatten_shape(s.shape)[1]
            if np.abs(chisCur - chisShould) < 1e-2:
                break
    # Since the optimization uses fidelity as the cost function,
    # the overall magnetitute of the tensor is not determined.
    # (this is like using minimizing the angle between two vectors)
    # In this final step, we fix the overall magnitute by demanding
    # <ψ|ψ> = <φ|φ>,
    psipsi, phiphi = fidelity2leg(Gamma_h, s)[2:4]
    psi2phi = (psipsi / phiphi).norm()
    s = s * (psi2phi)**(1/8)
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


def fidelity(Gamma, lr):
    """metric for approximation in FET

    Args:
        Gamma (TensorCommon): bond environment
        lr (TensorCommon): low-rank matrix

    Returns:
        f (float): fidelity

    """
    # this is just a constant for normalization <ψ|ψ>
    psipsi = ncon([Gamma], [[1, 1, 2, 2]])
    # <φ|φ>
    phiphi = ncon([Gamma, lr, lr.conj()], [[1, 2, 3, 4], [1, 2], [3, 4]])
    # <ψ|φ>
    psiphi = ncon([Gamma, lr], [[1, 2, 3, 3], [1, 2]])
    f = psiphi * psiphi.conj() / (psipsi * phiphi)
    return f, 1 - f


def fidelity2leg(Gamma_h, s):
    """metric for 2-leg reflection-symmetry FET

    Args:
        Gamma_h (TensorCommon): TODO
        lr (TensorCommon): low-rank matrix

    Returns
        f (float): fidelity

    """
    # this is just a constant for normalization <ψ|ψ>
    psipsi = ncon([Gamma_h, Gamma_h.conj()],
                  [[1, 2, 3, 4], [1, 2, 3, 4]])
    # <φ|φ>
    Gamma_h4s = ncon([Gamma_h, s, s.conj(), s.conj(), s],
                     [[1, 2, 3, 4], [1, -1], [2, -2], [3, -3], [4, -4]])
    phiphi = ncon([Gamma_h4s, Gamma_h4s.conj()],
                  [[1, 2, 3, 4], [1, 2, 3, 4]])

    # <ψ|φ>
    Gamma_h2s = ncon([Gamma_h, s, s.conj()],
                     [[1, -2, 2, -4], [1, -1], [2, -3]])
    psiphi = ncon([Gamma_h2s, Gamma_h2s.conj()],
                  [[1, 2, 3, 4], [1, 2, 3, 4]])
    # fidelity
    f = psiphi * psiphi.conj() / (psipsi * phiphi)
    return f, 1 - f, psipsi, phiphi


def entropy(lr):
    s = lr.svd([0], [1])[1]
    s = s / s.sum()
    s = s + 1e-15
    ee = -(s * s.log()).sum()
    return ee


def s2lr(s):
    """
    Glue the half piece `s` matrix into the
    low-rank matrix `lr`
    """
    lr = ncon([s, s.conj()], [[-1, 1], [-2, 1]])
    return lr