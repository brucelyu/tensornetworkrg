#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : toymodels.py
# Author            : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
# Date              : 24.10.2022
# Last Modified Date: 24.10.2022
# Last Modified By  : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
"""
Construct various toy models representing
the short-range correlations in tensor network.
"""
from ncon import ncon
from abeliantensors import Tensor


def cdlten(cornerChi=2, isZ2=False):
    """construct the corner-double-line tensor
          -3 -4
           ||
      -1 --oo-- -5
      -2 --oo-- -6
           ||
         -7 -8
    Kwargs:
        cornerChi (int):
        bond dimension of the corner matrix

        isZ2 (boolean):
        whether to use Z2-symmetryc tensor

    Returns:
        cdl (TensorCommon): the CDL 4-leg tensor
        loopNumber (?): the loop represented by the CDL
        cmat (TensorCommon): the corner matrix

    """
    if not isZ2:
        cmat = Tensor.random([cornerChi]*2)
    else:
        raise NotImplementedError("To be implemented...")
    # make corner matrix symmetric
    cmat = 0.5 * (cmat + cmat.transpose().conj())
    # construct the CDL structure
    cdl = ncon([cmat]*4,  [[-1, -3], [-5, -4], [-6, -8], [-2, -7]])
    cdl = cdl.join_indices([0, 1], [2, 3], [4, 5], [6, 7])
    # the loop represented by the CDL structure
    loopNumber = ncon([cmat]*4, [[1, 2], [3, 2], [3, 4], [1, 4]])
    return cdl, loopNumber, cmat
