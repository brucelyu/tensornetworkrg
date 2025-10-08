#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : trg_rotsym.py
# Author            : Xinliang (Bruce) Lyu <xlyu@ihes.fr>
# Date              : 07.10.2025
# Last Modified Date: 07.10.2025
# Last Modified By  : Xinliang (Bruce) Lyu <xlyu@ihes.fr>
from ncon import ncon


def l1Split(A, chi, eps):
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
        chis=[i+1 for i in range(chi)], eps=eps,
        return_rel_err=True
    )
    errsq = err**2
    return eigv, v, errsq


def init_trg(A, chi, eps):
    """Initialize the tensors composing the loop
    """
    eigv, v, err = l1Split(A, chi, eps)
    # bond matrix is the sign of the eigenvalues
    bondzp = eigv.sign()
    # (Note: for a symmetric matrix, its singular value is
    # the absoluate value of its eigenvalue)
    s = eigv.abs()
    # distribute the absoluate value of the eigenvalues
    # to the isometric tensor v
    vL = v.copy()
    vL = vL.multiply_diag(s.sqrt(), 2, direction="r")
    return vL, bondzp, err, s, v


def contrvLz(vL, bondz=None):
    """contraction of 4 vL tensors
    along with the bond matrix z
    """
    vLz = vL * 1.0
    if bondz is not None:
        vLz = vLz.multiply_diag(bondz, 0, direction="r")
        vLz = vLz.multiply_diag(bondz, 1, direction="l")
    Ap = ncon([vLz, vL.conj(), vLz, vL.conj()],
              [[3, 1, -1], [4, 1, -2], [4, 2, -3], [3, 2, -4]])
    return Ap
