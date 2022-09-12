#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : u1ten.py
# Author            : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
# Date              : 12.09.2022
# Last Modified Date: 12.09.2022
# Last Modified By  : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
"""
customized method for abeliantensors
maybe move to that package implementations eventually
"""
from ncon import ncon
import numpy as np


def pinv(B, a=[0, 1], b=[2, 3], eps_mach=1e-10, debug=False):
    """
    Calculate pesudo inverse of positive semi-definite matrix B.
    We first perform eigenvalue decomposition of B = U d Uh, and only keep
    eigenvalues with d > 1e-10. B^-1 = U d^-1 Uh

    Parameters
    ----------
    B : abeliantensors
    eps_mach : float, optional
        If the singular value is smaller than this, we set it to be 0.
        The default is 1e-10.

    Returns
    -------
    Binv : abeliantensors
        inverse of B
    """
    def invArray(tensor):
        """
        Invert every element in each block of tensor (Abeliean tensor)
        """
        if type(tensor).__module__.split(".")[1] == 'symmetrytensors':
            invtensor = tensor.copy()
            for mykey in tensor.sects.keys():
                invtensor[mykey] = 1 / tensor[mykey]
        else:
            invtensor = 1 / tensor
        return invtensor

    d, U = B.eig(a, b, hermitian=True, eps=eps_mach)
    if debug:
        print("Shape of d and U")
        print(d.shape)
        print(U.shape)
    dinv = invArray(d)
    contrLegU = list(-np.array(a) - 1) + [1]
    contrLegUh = list(-np.array(b) - 1) + [1]
    Ud = U.multiply_diag(dinv, axis=len(U.shape) - 1, direction='r')
    Binv = ncon([Ud, U.conjugate()], [contrLegU, contrLegUh])

    return Binv
