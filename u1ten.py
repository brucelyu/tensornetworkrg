#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : u1ten.py
# Author            : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
# Date              : 12.09.2022
# Last Modified Date: 27.09.2022
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


def sliceten(t, slc):
    """slice a U(1)-tensor

    Args:
    ------
    t (abeliantensors): tensor to be sliced
    slc (tuple): slicing range
        - slc[k] is a list specifying the slicing of the `k`-th leg
        - slc[k][c] is a `slice` object for slicing in charge-sector `c`

    Returns:
    ------
    tslc (abeliantensors): the tensor after slicing
    """
    # determine the shape of the tensor after slicing
    tnewShape = tenSlcShape(t, slc)
    tqhape = t.qhape
    tdirs = t.dirs
    tcharge = t.charge

    # Initialize the sliced tensor `tslc`
    # I follow Markus's matrix_eig implementation for
    # abeliantensors here, see
    # https://github.com/mhauru/abeliantensors/blob/master/src/abeliantensors/abeliantensor.py
    tslc = type(t)(
        tnewShape,
        qhape=tqhape,
        dirs=tdirs,
        charge=tcharge,
        dtype=t.dtype
    )

    # Set the blocks of `tslc` from the old tensor `t`
    for chargeKey in t.sects:
        keySlc = []
        for legSlc, legCharge in zip(slc, chargeKey):
            keySlc.append(legSlc[legCharge])
        keySlc = tuple(keySlc)
        tslc[chargeKey] = t[chargeKey][keySlc]
    msg = ("The keys of the sectors of the sliced tensor " +
           "should be the same as these of the input tensor")
    assert tslc.sects.keys() == t.sects.keys(), msg
    return tslc


def tenSlcShape(t, slc):
    """shape of a tensor after slicing

    Args:
    ------
    t (abeliantensors): tensor to be sliced
    slc (tuple): slicing range
        - slc[k] is a list specifying the slicing of the `k`-th leg
        - slc[k][c] is a `slice` object for slicing in charge-sector `c`

    Example Codes:
    ------
    ten = TensorZ2.random(shape=[[3, 4], [1, 7], [8, 9]],
                          qhape=[[0, 1]]*3, dirs=[1, 1, -1])
    slc = [[slice(2), slice(3)]] + [[slice(None), slice(None)]]*2
    tenSlcShape(ten, slc) will return [[2, 3], [1, 7], [8, 9]]

    Returns:
    ------
    tnewShape: list
        shape of the tensor after slicing
    """
    # determine the shape of the tensor after slicing
    toldShape = t.shape
    tnewShape = []
    for k in range(len(slc)):
        legShape = []
        for c in range(len(slc[k])):
            slcCharge = slc[k][c]
            if sliceLength(slcCharge) is None:
                curLength = toldShape[k][c]
            else:
                curLength = min(sliceLength(slcCharge),
                                toldShape[k][c])
            legShape.append(curLength)
        tnewShape.append(legShape)
    # check the shape
    msg = ("number of legs or charge sectors do not match between" +
           " the old and the sliced tensor")
    assert np.array(toldShape).shape == np.array(tnewShape).shape, msg
    return tnewShape


def sliceLength(slc):
    """length of a single slice object
    ceil((stop - start) / step)
    """
    if slc.start is None and slc.stop is None and slc.step is None:
        legLength = None
    else:
        # taking care of the step == None
        if slc.step is None:
            slc_step = 1
        else:
            slc_step = slc.step
        # taking care of the start == None
        if slc.start is None:
            slc_start = 0
        else:
            slc_start = slc.start
        # calcuate length of the slicing
        legLength = int(np.ceil((slc.stop - slc_start) / slc_step))
    return legLength
