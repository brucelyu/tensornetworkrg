#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : env2d.py
# Author            : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
# Date              : 24.10.2022
# Last Modified Date: 24.10.2022
# Last Modified By  : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
"""
Construct one-leg environment using its adjacent plaquette.
"""
import numpy as np
from ncon import ncon


def envHalfRefSym(A):
    """2-leg reflection symmetric environment
    This environment goes beyond the GILT framework,
    where only a single leg is cut at a time.
    We move to the FET framework here.

    The environment here is roughly half of the GILT environment,
    consisting of 4 copies of A tensor.
    It is designed for cutting two legs in a plaquette at the same time,
    so the reflection symmetry in two directions are imposed explicitly.
      j|
     i--A--k,
       l|
     and the plaquette looks like
        |     |
     ---A--Lr--Areflv--
        |     |
        |     |
 --Areflh--Lr--Areflhv--
        |     |

    Args:
        A (TensorCommon): 4-leg tensor

    Returns:
        Gamma_h (TensorCommon):
            half environment tensor
        /---|
      /  ---A----
     |  /   |
     | | -Areflh--
     | | |  |
     | | ---A----
     | |    |
      \ \-Areflh--
        \---|
    """
    Gamma_h = ncon([A, A.conj(), A.conj(), A],
                   [[3, 4, -1, 5], [3, 4, -2, 6],
                    [1, 2, -3, 5], [1, 2, -4, 6]])
    return Gamma_h


def envUs(legenv, eps=1e-14):
    """find U, s from the plaquette environment of a leg

    Args:
        legenv (TODO): TODO

    Returns: TODO

    """
    d, U = legenv.eig([0, 1], [2, 3], hermitian=True,
                      eps=eps)
    # abs is taken for numerical errors around zero
    s = d.abs().sqrt()
    return U, s


def envRefSym(A):
    """Construct an environment that preserves reflection symmetry
    - Now the tensor network only has one-type of tensor A.
    - We construct a environment of leg "I" (see `envGeneralLeg`)
    such that the environment is symmetric under horizontal
    reflection w.r.t the middle verticle line.
       j|
     i--A--k,
       l|
     and the plaquette looks like
        | (I) |
     ---A----Arefl---
    (IV)|     |(II)
        |     |
     ---A----Arefl---
        |(III)|

    Args:
        A (TensorCommon): 4-leg tensor

    Returns:
        legenv (TensorCommon): environment of that leg
    """
    Arefl = A.transpose([2, 1, 0, 3]).conj()
    legenv = envGeneralLeg([A, Arefl, Arefl, A], Leg="I")
    # assert the assumed symmetry properties of legenv
    msg = "The environment doesn't have the assumed reflection symmetry"
    assert legenv.allclose(
        legenv.transpose([1, 0, 3, 2]).conj()
    ), msg
    return legenv


def envGeneralLeg(Alist, Leg="I"):
    """construct the environment of a leg in a plaquette
       j|
     i--A--k.
       l|
     A1,A2,A3,A4 = Alist, the plaquette looks like
        | (I) |
     ---A1----A2---
    (IV)|     |(II)
        |     |
     ---A4----A3---
        |(III)|

    No symmetry is assumed here

    Args:
        Alist (list): list of 4 tensors defining a plaquette
        Each element Alist[k] is a TensorCommon type

    Kwargs:
        Leg (str): choose among ("I","II","III","IV")

    Returns:
        legenv (TensorCommon): environment of that leg

    """
    # We just need to write the code for leg I. Leg II, III and IV are obtained
    # by rotating the diagram by 90, 180 and 270 degrees. Notice this will
    # change both the order of tensors and the order of their legs
    Arot1, Arot2, Arot3, Arot4 = rotateLattice(Alist, Leg)
    # calculate the environment matrix and eigenvalue decompose it.
    legenv = ncon([Arot1, Arot2, Arot3, Arot4,
                   Arot1.conjugate(), Arot2.conjugate(),
                   Arot3.conjugate(), Arot4.conjugate()],
                  [[7, 8, -1, 9], [-2, 12, 11, 13],
                   [5, 13, 4, 3], [2, 9, 5, 1],
                   [7, 8, -3, 10], [-4, 12, 11, 14],
                   [6, 14, 4, 3], [2, 10, 6, 1]]
                  )
    return legenv


def rotateLattice(Alist, Leg="I"):
    """
    rotate the lattice so that the leg goes to where the
    leg "I" used to stay
    Parameters
    ----------
    Alist : list
        containing four tensors in a plaquette.
    Leg : tr, choose among ("I","II","III","IV")
        the leg whose environment info is calculated. The default is "I".
    Returns
    -------
    Arotlist : list
        representing the rotated list.
    """
    # We just need to write the code for leg I. Leg II, III and IV are obtained
    # by rotating the diagram by 90, 180 and 270 degrees. Notice this will
    # change both the order of tensors and the order of their legs
    legdic = {"I": 0, "II": 1, "III": 2, "IV": 3}
    if Leg not in legdic.keys():
        raise ValueError("Leg should be choisen among (I,II,III,IV)!")
    perm = np.array([i for i in range(4)])
    curperm = list(np.mod(perm + legdic[Leg], 4))
    Arotlist = []
    # rotatoin both the lattice and the order of legs of the tensors
    for k in range(4):
        Arotlist.append(Alist[curperm[k]].transpose(curperm))
    return Arotlist


