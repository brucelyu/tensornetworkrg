#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : hotrg.py
# Author            : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
# Date              : 18.10.2022
# Last Modified Date: 18.10.2022
# Last Modified By  : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
"""
Various 2D HOTRG implementations

For the totally-isotropic version, the
conjugate direction of the tensor is
assumed to be [-1, 1, -1, 1].
"""

import numpy as np
from ncon import ncon


def opt_v(Ain, chi, dtol=1e-10):
    """
    Determine the squeezer v
    A trick is used to strictly preserve
    the reflection symmetry
    """
    B = Ain.conjugate()
    env_v = ncon([Ain, B, Ain.conjugate(), B.conjugate()],
                 [[-2, 1, 2, 5], [-1, 3, 4, 5],
                  [-4, 1, 2, 6], [-3, 3, 4, 6]
                  ])
    (d,
     v,
     SPerr
     ) = env_v.eig([0, 1], [2, 3],
                   hermitian=True,
                   chis=[i+1 for i in range(chi)],
                   trunc_err_func=trunc_err_func,
                   eps=dtol, return_rel_err=True)
    return v, SPerr


def block4tensor(Ain, v):
    """
    Block 4 tensors to get a coarser tensor

    The block is totally isotropic but
    the computation cost is high here, which
    goes like O(χ^8)
    """
    # rotation A 90° to get B
    B1 = Ain.transpose([1, 2, 3, 0])
    B2 = Ain.transpose([2, 3, 0, 1])
    B3 = Ain.transpose([3, 0, 1, 2])
    # 90° rotation symmetry is imposed here
    Aout = ncon([Ain, B1, B2, B3,
                 v, v.conjugate(), v, v.conjugate()],
                [[2, 4, 9, 1], [9, 11, 7, 5],
                 [10, 5, 6, 8], [3, 1, 10, 12],
                 [2, 3, -1], [4, 11, -2], [6, 7, -3], [8, 12, -4]
                 ]
                )
    msg = "mirror symmetry and rotation symmetry not compatible"
    assert Aout.transpose([1, 2, 3, 0]).allclose(Aout.transpose([0, 3, 2, 1]).conjugate()), msg
    assert Aout.transpose([1, 2, 3, 0]).allclose(Aout.transpose([2, 1, 0, 3]).conjugate()), msg
    # symmetrize just in case for machine error
    Aout = 1/3 * (Aout + Aout.transpose([3, 2, 1, 0]).conjugate() + Aout.transpose([1, 0, 3, 2]).conjugate())
    return Aout


def isotroHOTRG(Ain, chi, dtol=1e-10):
    v, SPerr = opt_v(Ain, chi)
    Aout = block4tensor(Ain, v)
    return Aout, v, SPerr


def getAuvu(AList, vList, rgstep=5):
    """
    Map from bipartitle A, B tensor network
    to a uniform single-Au tensor network
    """
    vformal = vList[rgstep - 1]
    vcur = vList[rgstep]
    Acur = AList[rgstep]
    gauge = ncon([vformal]*2, [[1, 2, -1], [2, 1, -2]])
    # perform the gauge transformation
    Au = ncon([Acur, gauge, gauge.conjugate()],
              [[-1, 2, 3, -4], [2, -2], [3, -3]])
    vu = ncon([vcur, gauge.conjugate()],
              [[-1, 2, -3], [2, -2]])
    return Au, vu


def trunc_err_func(eigv, chi):
    """
    No need to take square since we work with M M'
    whose eigenvalues are themself square of singular values of M
    """
    res = np.sum(eigv[chi:]) / np.sum(eigv)
    return res
