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
import functools
from ncon import ncon
import numpy as np
from abeliantensors import Tensor, TensorCommon, AbelianTensor


def pinv(B, a=[0, 1], b=[2, 3], eps_mach=1e-10,
         soft=False, chiCut=None, debug=False):
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
    def invArray(tensor, softeps=0):
        """
        Invert every element in each block of tensor (Abeliean tensor)
        """
        if type(tensor).__module__.split(".")[1] == 'symmetrytensors':
            invtensor = tensor.copy()
            for mykey in tensor.sects.keys():
                invtensor[mykey] = 1 / (tensor[mykey] + softeps)
        else:
            invtensor = 1 / (tensor + softeps)
        return invtensor

    if not soft:
        d, U = B.eig(a, b, hermitian=True,
                     chis=[k+1 for k in range(chiCut)],
                     eps=eps_mach)
    else:
        # perform no truncation here but soft inverse below
        d, U = B.eig(a, b, hermitian=True)
    if debug:
        print("Shape of d and U")
        print(d.shape)
        print(U.shape)
    if not soft:
        dinv = invArray(d)
    else:
        # perform soft inversion
        dinv = invArray(d, softeps=eps_mach)
    contrLegU = list(-np.array(a) - 1) + [1]
    contrLegUh = list(-np.array(b) - 1) + [1]
    Ud = U.multiply_diag(dinv, axis=len(U.shape) - 1, direction='r')
    Binv = ncon([Ud, U.conjugate()], [contrLegU, contrLegUh])

    return Binv


# ------------------------- #
# All functions for slicing
def slicing(t, slc, indexOrder=None):
    """
    The common method for both the ordinary numpy Tensor
    and the symmetric tensor

    Args:
    ------
    t (Tensor or AbelianTensor): tensor to be sliced
    slc (tuple): slc[k] is a `slice` object for k-th leg
    indexOrder (tuple):
        indexOrder[k] is an 1-leg tensor (usually singular value
        or eigenvalue spectrum) specifying the importance of
        different index values of the k-th leg

    Returns:
    ------
    tslc (Tensor or AbelianTensor): tensor after slicing
    """
    # check input
    msg = "t should be either Tensor or AbelianTensor."
    assert issubclass(type(t), TensorCommon), msg
    if indexOrder is not None:
        msg = ("`slc` and `indexOrder` should both have" +
               "length equal to the number of leg of `t`")
        assert len(slc) == len(indexOrder), msg
    # slicing the tensor
    if type(t) is Tensor:
        tslc = t[slc]
    else:
        # for the case of AbelianTensor
        # First determine the slc according to indexOrder
        slc = slcu1(slc, indexOrder)
        # Then do the slice
        tslc = sliceten(t, slc)
    return tslc


def slcu1(slc, indexOrder):
    """
    Split the sliced bond dimensions in `slc` among charge sectors
    according to `indexOrder`

    Args:
    ------
    t (Tensor or AbelianTensor): tensor to be sliced
    slc (tuple): slc[k] is a `slice` object for k-th leg
    indexOrder (tuple):
        indexOrder[k] is an 1-leg tensor (usually singular value
        or eigenvalue spectrum) specifying the importance of
        different index values of the k-th leg

    Careful! I only implement for the cases where slc[k] is
          - slice(None)
          - slice(chi)

    Returns:
    ------
    slcnew (tuple):
        - slcnew[k] is a list specifying the slicing of the `k`-th leg
        - slcnew[k][c] is a `slice` object for slicing
        in charge-sector `indexOrder[k].qhape[0][c]`
    """
    slcNew = []
    for legslc, s in zip(slc, indexOrder):
        if legslc == slice(None):
            legslcNew = [slice(None)] * len(s.shape[0])
        elif (legslc.start) is None and (legslc.step is None):
            chi = legslc.stop
            # vectorize the index to charge function
            vecindArr2SymCharge = functools.partial(indArr2SymCharge,
                                                    s.shape[0], s.qhape[0])
            vecindArr2SymCharge = np.vectorize(vecindArr2SymCharge)
            retainCharge = vecindArr2SymCharge(
                (-1*s.to_ndarray()).argsort()
            )[:chi]
            # initialize
            legslcNew = []
            # determine the slice for each charge sector
            for c in range(len(s.shape[0])):
                curCharge = s.qhape[0][c]
                chiCharge = (retainCharge == curCharge).sum()
                legslcNew.append(slice(chiCharge))
        else:
            raise NotImplementedError
        slcNew.append(legslcNew)

    slcNew = tuple(slcNew)
    return slcNew


def indArr2SymCharge(legdim, qdim, arrind):
    """
    Convert a array index to its charge

    Example: Suppose we have
        legdim = [n0, n1, n2], qdim = [0, 1, 2];
        if arrind is in [0, n0), resturn 0;
        if arrind is in [n0, n0 + n1), return 1;
        if arrind is in [n0+n1, n0+n1+n2), return 2;
        else return None

        Notice the order in qdim doesn't mater
        since they will be sorted anyway.
    """
    startNum = 0
    res = None
    qdimSort = sorted(qdim)
    for nk, k in zip(legdim, qdimSort):
        endNum = startNum + nk
        if (arrind >= startNum) and (arrind < endNum):
            res = k
        startNum = endNum
    return res


def sliceten(t, slc):
    """slice a U(1)-tensor
    This is specifically for symmetric tensor class
    organized according to charge sectors.

    Args:
    ------
    t (abeliantensors): tensor to be sliced
    slc (tuple): slicing range
        - slc[k] is a list specifying the slicing of the `k`-th leg
        - slc[k][c] is a `slice` object for slicing
        in charge-sector `t.qhape[k][c]` of k-th leg

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
        for k, legSlcCharge in enumerate(zip(slc, chargeKey)):
            legSlc, legCharge = legSlcCharge
            chargeInd = tqhape[k].index(legCharge)
            keySlc.append(legSlc[chargeInd])
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
        - slc[k][c] is a `slice` object for slicing
        in charge-sector `t.qhape[k][c]`

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

# ------------------------- #


# ------------------------- #
# All functions for looping a leg
# and fixing a leg index
def fixleg(ten, posleg, ind):
    """fix one leg of a tensor

    Args:
        ten (TensorCommon): input tensor
        posleg (int): leg to be fixed
        ind (int or tuple): index value

    Returns:
        For a ordinary tensor, output is
        tensl = ten[:,..., ind, ..., :]
        where the posleg-th position is fixed to `ind`

    """
    assert type(posleg) is int
    # Read off number of legs of the tensor, ten
    legnum = len(ten.shape)
    slc = [slice(None)] * legnum
    if issubclass(type(ten), Tensor) or (type(ten) is np.ndarray):
        # for ordinary tensors
        slc[posleg] = ind
        tensl = ten[tuple(slc)]
    elif issubclass(type(ten), AbelianTensor):
        # Read off the charge and degenerate index
        partn, degind = ind
        # create properties for output tensor
        outshape = ten.shape[:posleg] + ten.shape[posleg+1:]
        outqhape = ten.qhape[:posleg] + ten.qhape[posleg+1:]
        outcharge = (ten.charge + partn * ten.dirs[posleg]) % ten.qodulus
        outdirs = ten.dirs[:posleg] + ten.dirs[posleg+1:]
        # create an empty Z2 tensor with the same type and
        # with one leg less (one leg is fixed)
        tensl = ten.zeros(shape=outshape, qhape=outqhape,
                          dirs=outdirs, charge=outcharge)
        # relate the tensl to ten
        slc[posleg] = degind
        # set the value of sliced tensor according to the input tensor
        for partsect in tensl.sects.keys():
            oldpartsect = partsect[:posleg] + (partn,) + partsect[posleg:]
            tensl[partsect] = ten[oldpartsect][tuple(slc)]
    else:
        raise TypeError("Input tensor type not supported")
    return tensl


def loopleg(ten, posleg):
    """loop through all indices of a leg of a tensor

    Args:
        ten (TensorCommon): input tensor
        posleg (int): leg to be fixed

    Returns:
        1) For ordinary tensors, return
            integer i
        from 0 till the size of the leg;
        2) For U1-symmetryc tensors, return
            tuple (curq, degind)
        with the first element the charge section number
        and the second element the degenerate index number

    """
    assert type(posleg) is int
    legShape = ten.shape[posleg]
    if issubclass(type(ten), Tensor) or (type(ten) is np.ndarray):
        # for ordinary tensors
        for i in range(legShape):
            yield i
    elif issubclass(type(ten), AbelianTensor):
        # for U1-symmetric tensors
        legQhape = ten.qhape[posleg]
        for curq, degN in zip(legQhape, legShape):
            for degind in range(degN):
                yield (curq, degind)


# ------------------------ #
# convert from TensorZ2 to ordinary tensor
# used for TNRG linearalization
def Z2toArray(Z2tensor):
    """Z2toArray.
    Reshape a Z2-symmetric tensor into a numpy Array
    Parameters
    ----------
    Z2tensor : abeliantensors
        The input tensor
    Returns
    -------
    tensorArray : numpy Array
    """
    # read off the charge of the tensor
    symmetryCharge = Z2tensor.charge
    # read off the number of legs of the tensor
    numLeg = len(Z2tensor.shape)
    # Merge all legs of Z2tensor to a single leg
    tensorArrayZ2 = Z2tensor.join_indices([i for i in range(numLeg)], dirs=1)
    # Read off the numpy Array from the corresponding sector
    tensorArray = tensorArrayZ2[(symmetryCharge,)]
    return tensorArray, tensorArrayZ2


def arraytoZ2(tensorArray, exemZ2tensor):
    """arraytoZ2.
    Reshape a numpy Array into a Z2-symmetric tensor with
    the same shape of the exemplary tensor exemZ2tensor
    Parameters
    ----------
    tensorArray : numpy Array
        input array to be reshaped
    exemZ2tensor : abeliantensors
        exemplary tensor
    Returns
    -------
    Z2tensor : abeliantensors
    """
    # read off related property from the exemplary tensor
    symmetryCharge = exemZ2tensor.charge
    symShape = exemZ2tensor.shape
    symQhape = exemZ2tensor.qhape
    symDirs = exemZ2tensor.dirs
    # produce an exemplary numpy array from exemZ2tensor
    exemArray, exemArrayZ2 = Z2toArray(exemZ2tensor)
    # impose the condition that len(tensorArray) == len(exemArray)
    errMessg = ("The length of tensorArray should match " +
                "the degrees of freedom in exemZ2tensor!")
    assert len(tensorArray) == len(exemArray), errMessg
    # Assign tensorArray to the corresponding sector in exemArrayZ2
    exemArrayZ2[(symmetryCharge,)] = tensorArray
    # split the leg
    Z2tensor = exemArrayZ2.split_indices(indices=[0], dims=[symShape],
                                         qims=[symQhape], dirs=[symDirs])
    return Z2tensor
