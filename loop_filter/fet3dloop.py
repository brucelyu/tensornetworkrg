#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : fet3dloop.py
# Author            : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
# Date              : 04.01.2024
# Last Modified Date: 04.01.2024
# Last Modified By  : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
"""
3D Generalization of
- Evenbly's full environment truncation (FET)
- Graph-Independent Local Truncations (GILT) by
Hauru, Delcamp, and Mizera

We generalize it to 3D, by approximating
a plaquette of 4 copies of 6-leg tensors.
All the relevant environments are construction in `./env3dloop.py`

The initialization and FET is implemented with the following
3D HOTRG-like block-tensor scheme in mind:
(CG stands for coarse graining and LF for loop filtering)
- CG1: z-direction so x and y legs have increased bond dimensions
χm and χin for outer and inner orientations.
- LF1: z-loop filtering of the outer x and y legs to truncate
the bond dimension χm --> χs
- CG2: y-direction. Now the outer x leg is the outermost leg
with bond dimension changes χs --> χs^2 --> χ;
but the z leg has increased bond dimension χm
- LF2 : y-loop filtering of the outer z leg to truncate
the bond dimension χm --> χs.
The outmost x leg is left untouched in this step.
- CG3: x-direction. Both y and z legs becomes outermost legs
with bond dimension changes χs --> χs^2 --> χ.
_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
"""
from . import env3dloop, fet3d


def init_zloopm(A, chis, chienv, epsilon):
    """Initialization of m matrices for z-loop
    It is a (mx, my) pair

    Args:
        A (TensorCommon): 6-leg tensor
            It is the tensor located at
            - (x+y+)-position for z-loop filtering after z corase graining
            It has shape
            - A[x, xp, y, yp, z, zp]
        chis (int): squeezed bond dimension of m matrices
            such that m is a χ-by-χs matrix.
            It is the bond dimension for
            the truncated SVD of low-rank Lr matrix.
        chienv (int): for pseudo-inverse of leg environment
            To avoid the trivial solution of Lr matrix,
            we should truncate the singular values
            when taking the pseudo-inverse of leg environment
        epsilon (float): for psudo-inverse of leg environment
            if further truncation to χ < chienv has
            an error < epsilon, then we truncate to
            the smallest possible χ.

    Returns: loop-filtering matrices pair
        mx (TensorCommon): 2-leg tensor
        my (TensorCommon): 2-leg tensor

    """
    # Construct the γ environment for initialization of m matrices
    Gammay = env3dloop.loopGamma(A)
    # (Swap xy leg)
    A4x = env3dloop.swapxy(A, 1, 1)[0]
    Gammax = env3dloop.loopGamma(A4x)
    # Find initial low-rank matrix Lr and split it to get m matrix
    sy, Lry = fet3d.init_s_gilt(Gammay, chis, chienv, epsilon,
                                init_soft=False)
    sx, Lrx = fet3d.init_s_gilt(Gammax, chis, chienv, epsilon,
                                init_soft=False)
    return sx, sy, Lrx, Lry, Gammay


def init_yloopm(A, chis, chienv, epsilon):
    """
    Similar to the above `init_zloopm` function but
    the input tensor A located at
        - (x+)-position for y-loop filtering after z+y corase graining
    It has shape
        - A[x, xp, y, yp, z, zp]
    """
    # Construct the γ environment for initialization of m matrices
    # (rotate A leg: xyz --> zxy)
    Ar = A.transpose([4, 5, 0, 1, 2, 3])
    # (Swap zx leg)
    A4z = env3dloop.swapxy(Ar, 1, 1)[0]
    Gammaz = env3dloop.loopGamma(A4z)
    # Find initial low-rank matrix Lr and split it to get m matrix
    sz, Lrz = fet3d.init_s_gilt(Gammaz, chis, chienv, epsilon,
                                init_soft=False)
    return sz, Lrz, Gammaz
