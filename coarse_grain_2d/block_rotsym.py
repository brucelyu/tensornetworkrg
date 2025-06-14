#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : block_rotsym.py
# Author            : Xinliang (Bruce) Lyu <xlyu@ihes.fr>
# Date              : 08.04.2025
# Last Modified Date: 08.04.2025
# Last Modified By  : Xinliang (Bruce) Lyu <xlyu@ihes.fr>
"""
Block-tensor map with the following two lattice symmetry exploited
- lattice reflection
- rotational
The 4-leg tensor A is assumed to be real.
We have not worked out the complex-value tensor yet

Tensor leg order convention is
A[x, y, x', y']
      y
      |
  x'--A--x
      |
      y'

"""

import numpy as np
from ncon import ncon


def densityM(A):
    """Construct the density matrix for the isometry

    Args:
        A (TensorCommon): 4-leg tensor with real values

    Returns:
        rho (TensorCommon): 4-leg density matrix

    """
    rho = ncon([A, A.conjugate(), A.conjugate(), A],
               [[-1, 2, 1, 5], [-2, 4, 3, 5],
                [-3, 2, 1, 6], [-4, 4, 3, 6]]
               )
    # This matrix has the symmetry under the SWAP operation:
    #   rho[i,j,m,n] = rho[j,i,n,m]
    return rho


def findProj(A, chi, cg_eps=1e-8):
    """determine the isometry p

    Args:
        rho (TensorCommon): density matrix for isometry
        chi (int): the bond dimension χ

    Kwargs:
        cg_eps (float):
            a numerical precision control for coarse graining process
            eigenvalues small than cg_eps will be thrown away
            regardless of χ

    Returns:
        p (TensorCommon): 3-leg isometric tensor p[ij o]
            The first two indices i,j are "in"
            The last one o is "out"
        g (array): SWAP sign
        err (float): error for truncation
        eigv (TensorCommon): eigenvalues

    """
    # simontaneously diagonalize rho and the SWAP matrix
    # since SWAP @ ρ = ρ @ SWAP,
    # where SWAP @ ρ[ij mn] = ρ[ji mn].
    # Our trick here is to diagonalize SWAP * ε + ρ,
    # where ε is a small number that doesn't change
    # the order of eigenvalues of ρ
    rho = densityM(A)
    # construct the SWAP matrix
    if rho.dirs is not None:
        # for abeliantensors
        eyeMat = rho.eye(
            rho.shape[0], qim=rho.qhape[0]
        )
        eyeMat.dirs = [1, 1]
    else:
        # for normal tensors
        eyeMat = rho.eye(rho.shape[0])
    SWAP = ncon([eyeMat, eyeMat.conj()], [[-1, -4], [-2, -3]])
    if SWAP.dirs is not None:
        SWAP.dirs = rho.dirs.copy()
    # perturb the density matrix ρ
    rhoP = rho + SWAP * cg_eps * 1e-2
    eigv, p, err = rhoP.eig(
        [0, 1], [2, 3], hermitian=True,
        chis=[i+1 for i in range(chi)], eps=cg_eps,
        trunc_err_func=trunc_err_func,
        return_rel_err=True
    )

    # calculate the SWAP eigenvalues
    g = ncon([p.conj(), p.conj()], [[1, 2, -1], [2, 1, -2]])
    g = g.diag()
    return p, g, err, eigv.abs()


def block4ten(A, p):
    """contraction of a 2x2 block of tensors
    Although both the rotation and reflection symmetries are preserved here,
    only the rotational symmetry is imposed by this map,
    while the reflection symmetry isn't imposed!

    Args:
        A (TensorCommon): 4-leg tensor with real values
        p (TensorCommon): 3-leg isometric tensor p[ij o]

    Returns:
        Ap (TensorCommon): the coarse-grained tensor

    """
    # The costs are 6χ^7 + χ^8
    Ap = ncon([A, A, A, A,
               p.conjugate(), p, p.conjugate(), p],
              [[2, 4, 9, 1], [11, 7, 5, 9], [6, 8, 10, 5], [12, 3, 1, 10],
               [2, 3, -1], [4, 11, -2], [6, 7, -3], [8, 12, -4]])
    return Ap


def l1Split(A, chiM, epsM):
    """Split A along l1 using EVD
    Tensor leg order convention is
    A[x, y, x', y'] and it will be split along l1
          y  l1
          |/
      x'--A--x
        / |
          y'
    """
    # Check the reflection symmetry along l1
    # A[x, y, x', y'] = A[y, x, y', x']
    errMsg = "No reflection symmetry along l1 for the input tensor A"
    assert A.allclose(A.transpose([1, 0, 3, 2]).conj()), errMsg

    # Peform the EVD of A along l1
    eigv, v, err = A.eig(
        [2, 1], [3, 0], hermitian=True,
        chis=[i+1 for i in range(chiM)], eps=epsM,
        return_rel_err=True
    )
    return eigv, v, err**2


def bktenTRG(v, eigv, p):
    """contraction of a 2x2 block of tensors with O(χ^6) costs
    We use an idea from the TRG to lower the costs of the block-tensor map
    from O(χ^8) to O(χ^6)

    """
    C = ncon([v.conj(), v, p.conj()],
             [[1, 2, -1], [1, 3, -2], [2, 3, -3]])
    Ct = C.multiply_diag(eigv, 0, direction='l')
    Ch = C.multiply_diag(eigv, 1, direction='r')
    Ap = ncon([Ct, Ch.conj(), Ct, Ch.conj()],
              [[3, 1, -1], [3, 2, -2], [4, 2, -3], [4, 1, -4]])
    return Ap


def trunc_err_func(eigv, chi):
    """
    No need to take square since we work with M M'
    whose eigenvalues are themself square of singular values of M
    """
    res = np.sum(np.abs(eigv[chi:])) / np.sum(np.abs(eigv))
    return res


# Functions related to the symmetry of the tensor
def reflect(A, axis, g=None):
    assert axis in ["x", "y"]
    if g is None:
        if axis == "x":
            Arefl = A.transpose([2, 1, 0, 3])
        elif axis == "y":
            Arefl = A.transpose([0, 3, 2, 1])
    else:
        # convert array g to a diagonal matrix with dirs=[1, 1]
        gM = g.diag()               # dirs=[1, -1] by default
        gM = gM.flip_dir(1)
        if axis == "x":
            Arefl = ncon(
                [A.transpose([2, 1, 0, 3]).conj(),
                 gM, gM.conj(), gM, gM.conj()],
                [[1, 2, 3, 4], [1, -1], [2, -2], [3, -3], [4, -4]])
        elif axis == "y":
            Arefl = ncon(
                [A.transpose([0, 3, 2, 1]).conj(),
                 gM, gM.conj(), gM, gM.conj()],
                [[1, 2, 3, 4], [1, -1], [2, -2], [3, -3], [4, -4]])
    return Arefl


def rotate(A, g=None):
    if g is None:
        # conjugate here is just for Z2-symmetric tensors dirs
        Arot = A.transpose([3, 0, 1, 2]).conj()
    else:
        # convert array g to a diagonal matrix with dirs=[1, 1]
        gM = g.diag()               # dirs=[1, -1] by default
        gM = gM.flip_dir(1)
        Arot = ncon(
            [A.transpose([3, 0, 1, 2]), gM, gM.conj(), gM, gM.conj()],
            [[1, 2, 3, 4], [1, -1], [2, -2], [3, -3], [4, -4]])
    return Arot


def isReflSym(A, g=None):
    """Check the reflection symmetry of the tensor

    Args:
        A (TensorCommon): 4-leg tensor with real values
        g (array): SWAP sign

    Returns:
        isX, isY (boolean)

    """
    Areflx = reflect(A, "x", g)
    Arefly = reflect(A, "y", g)
    isX = Areflx.allclose(A)
    isY = Arefly.allclose(A)
    return isX, isY


def isRotSym(A, g=None):
    """Check the rotation symmetry

    Args:
        A (TensorCommon): 4-leg tensor with real values

    Kwargs:
        g (array): SWAP sign

    Returns: isRot

    """
    Arot = rotate(A, g)
    isRot = Arot.allclose(A)
    return isRot


def makeReflSym(A, g=None):
    """symmetrize the tensor A w.r.t reflection

    Args:
        A (TensorCommon): 4-leg tensor with real values

    Returns:
        Asym (TensorCommon): symmetrized tensor

    """
    Ax = 0.5 * (A + reflect(A, "x", g))
    Axy = 0.5 * (Ax + reflect(Ax, "y", g))
    return Axy


def makeRotSym(A, g=None):
    A90 = rotate(A, g)
    A180 = rotate(A90, g)
    A270 = rotate(A180, g)
    Asym = 0.25 * (A + A90 + A180 + A270)
    return Asym


# end of file
