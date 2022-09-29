#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : trg_evenbly.py
# Author            : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
# Date              : 17.08.2022
# Last Modified Date: 17.08.2022
# Last Modified By  : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>

import numpy as np
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
    # check the reflection symmetry of Ain,
    # it should be symmetric under reflection of
    # both horizontal and vertical directios
    err_messg = ("Input tensor Ain doesn't satisfy" +
                 " the reflection symmetry condition")
    assert ((Ain - Ain.transpose([2, 3, 0, 1])).norm() /
            Ain.norm() < 1e-10), err_messg
    q, SPerrq = opt_q(Ain, chiM, dtol=dtol)
    v, SPerrv = opt_v(Ain, q, chiH, dtol=dtol)
    w, SPerrw = opt_w(Ain, q, chiV, dtol=dtol)
    Aout = block_4tensor(Ain, q, v, w)
    SPerr_list = [SPerrq, SPerrv, SPerrw]
    return Aout, q, v, w, SPerr_list


def construct_q_env(Ain):
    """constrcut the environment of isometry q

    :Ain: abeliantensors

    ------
    returns
    env_q: abeliantensors
        - 4-leg tensor, which can be treated as
        a linear map from the first two
        to the last two.
        - environment tensor for squeezing
        the first two legs of the Ain
    """
    # for U(1)-symmetric tensors
    # the in-out direction for the two horizontal
    # legs should be flipped
    B = Ain.conjugate()
    env_q = ncon([Ain, B, B, Ain,
                  Ain.conjugate(), B.conjugate(),
                  B.conjugate(), Ain.conjugate()
                  ],
                 [[-1, -2, 11, 12], [7, 8, 11, 9],
                  [5, 12, 1, 2], [5, 9, 3, 4],
                  [-3, -4, 13, 14], [7, 8, 13, 10],
                  [6, 14, 1, 2], [6, 10, 3, 4]
                  ])
    return env_q


def opt_q(Ain, chiM, dtol=1e-10,
          return_d=False):
    """determine a good 3-leg isometry q

    :Ain: abeliantensors
        input 4-leg tensor
    :chiM: int
        squeezed bond dimension
    :dtol: float

    ------
    returns
    q: abeliantensors
        3-leg squeezer isometry
    SPerr: float
       square of the approximaion error
       for the projective truncation
    """
    # construct environment and cut bond dimension
    # according to eigenvalue decomposition
    env_q = construct_q_env(Ain)
    (d,
     q,
     SPerr
     ) = env_q.eig([0, 1], [2, 3], hermitian=True,
                   chis=[i+1 for i in range(chiM)],
                   trunc_err_func=trunc_err_func,
                   eps=dtol, return_rel_err=True)
    if return_d:
        return q, d, SPerr
    return q, SPerr


def build_qstarA(q, Ain):
    """group qstar and A to reduce computational cost

    Args:
        q: abeliantensors
            3-leg isometric tensor
        Ain: abeliantensors
            4-leg input tensor

    Returns:
        qstarA: abeliantensors
            3-leg tensor, dirs = [1, 1, 1]
            first two legs are the original horizontal and vertical
            ones; the last one is the squeezed leg in middle step.
    """
    qstarA = ncon([q.conjugate(), Ain], [[1, 2, -3], [1, 2, -2, -1]])
    return qstarA


def construct_v_env(qstarA, q):
    """construct the environment for the squeezing
    tensor v in verticle direction for producing a
    coarser horizontal leg

    :qstarA: abeliantensors
        3-leg tensor, basically a contraction of the input
        tensor A with a 3-leg tensor
    :q: abeliantensors

    ------
    returns:
    env_v: abeliantensors
        4-leg tensor

    """
    qB = qstarA.conjugate()
    qstar = q.conjugate()
    env_v = ncon([qB, qstarA, qstarA, qB,
                  qstar, q, q, qstar,
                  q, qstar, qstar, q,
                  qstarA, qB, qB, qstarA
                  ],
                 [[1, 3, 15], [2, 3, 16], [1, 4, 13], [2, 4, 14],
                  [5, 7, 15], [5, 8, 18], [6, 7, 16], [6, 8, 21],
                  [9, 11, 13], [9, 12, 19], [10, 11, 14], [10, 12, 22],
                  [17, -1, 18], [17, -2, 19], [20, -3, 21], [20, -4, 22]
                  ])
    return env_v


def opt_v(Ain, q, chiH, dtol=1e-10):
    """determine a good 3-leg isometry v
    v for squeezing legs in vertical direction
    to produce a horizontal coarser leg

    :Ain: abeliantensors
        input 4-leg tensor
    :q: abeliantensors
            3-leg isometric tensor
    :chiH: int
        horizontal squeezed bond dimension
    :dtol: float

    ------
    :returns:
    v: abeliantensors
        3-leg squeezer isometry to squeeze
        two horizontal legs to produce
        a coarser horizontal leg.
    SPerr: float
       square of the approximaion error
       for the projective truncation
    """
    qstarA = build_qstarA(q, Ain)
    env_v = construct_v_env(qstarA, q)
    (d,
     v,
     SPerr
     ) = env_v.eig([0, 1], [2, 3], hermitian=True,
                   chis=[i+1 for i in range(chiH)],
                   trunc_err_func=trunc_err_func,
                   eps=dtol, return_rel_err=True)
    return v, SPerr


def construct_w_env(qstarA, q):
    """construct the environment for the squeezing
    tensor w in horizontal direction for producing a
    coarser vertical leg

    Args:
        qstarA: abeliantensors
        q: abeliantensors

    ------
    Returns:
    env_w: abelianteosrs
        4-leg tensor

    """
    qB = qstarA.conjugate()
    qstar = q.conjugate()
    env_w = ncon([qB, qstarA, qstarA, qB,
                  q, qstar, qstar, q,
                  qstarA, qB, qB, qstarA,
                  qstar, q, q, qstar
                  ],
                 [[1, 3, 19], [1, 4, 13], [2, 3, 22], [2, 4, 14],
                  [5, 8, 13], [5, 7, 15], [6, 8, 14], [6, 7, 16],
                  [10, 12, 18], [10, 11, 15], [9, 12, 21], [9, 11, 16],
                  [20, -1, 21], [20, -2, 22], [17, -3, 18], [17, -4, 19]
                  ])
    return env_w


def opt_w(Ain, q, chiV, dtol=1e-10):
    """determine a good 3-leg isometry w
    w is for squeezing legs in horizontal direction
    to produce a vertical coarser leg

    :Ain: abeliantensors
        input 4-leg tensor
    :q: abeliantensors
            3-leg isometric tensor
    :chiV: int
        vertical squeezed bond dimension
    :dtol: float

    ------
    :returns:
    w: abeliantensors
        3-leg squeezer isometry to squeeze
        two vertical legs to produce
        a coarser vertical leg.
    SPerr: float
       square of the approximaion error
       for the projective truncation

    """
    qstarA = build_qstarA(q, Ain)
    env_w = construct_w_env(qstarA, q)
    (d,
     w,
     SPerr
     ) = env_w.eig([0, 1], [2, 3], hermitian=True,
                   chis=[i+1 for i in range(chiV)],
                   trunc_err_func=trunc_err_func,
                   eps=dtol, return_rel_err=True)
    return w, SPerr


def block_4tensor(Ain, q, v, w):
    """block 4 tensors
    - isometry v, w are for squeezing bond dimensions
    - isometry q is for reducing computational cost

    :Ain: abeliantensors
        4-leg input tensor
    :q: abeliantensors
        3-leg isometry for squeezing in middle step
    :v: abeliantensors
        3-leg isometry for vertical coarse graining
        to produce coarser horizontal legs
    :w: abeliantensors
        3-leg isometry for horizontal coarse graining
        to produce coarser vertical legs

    ------
    :returns:
    Aout: abeliantensors
        4-leg output tensor
    """
    qstarA = build_qstarA(q, Ain)
    qB = qstarA.conjugate()
    vqA = ncon([v.conjugate(), qstarA, qB],
               [[2, 3, -3], [1, 2, -1], [1, 3, -2]])
    wq = ncon([w.conjugate(), q.conjugate(), q],
              [[2, 3, -3], [1, 2, -1], [1, 3, -2]])
    Aout = ncon([vqA, wq, vqA, wq],
                [[3, 1, -1], [1, 4, -2],
                 [4, 2, -3], [2, 3, -4]])
    return Aout


def trunc_err_func(eigv, chi):
    """
    No need to take square since we work with M M'
    whose eigenvalues are themself square of singular values of M
    """
    res = np.sum(eigv[chi:]) / np.sum(eigv)
    return res
