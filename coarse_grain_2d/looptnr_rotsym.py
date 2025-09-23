#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : looptnr_rotsym.py
# Author            : Xinliang (Bruce) Lyu <xlyu@ihes.fr>
# Date              : 12.09.2025
# Last Modified Date: 12.09.2025
# Last Modified By  : Xinliang (Bruce) Lyu <xlyu@ihes.fr>
"""
This file contains functions for a modified loop-TNR,
where the lattice reflection and rotational symmetries are exploited.

The method is based on the original scheme proposed in the following paper:
https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.118.110504
Although the above paper has mentioned how to exploit
the lattice rotation symmetry, it seems to us that their ansatz might
contain some additional assumptions about the main tensor being
positive semi-definite in a certain sense.

Our modified scheme is supposed to be more general.
Most of all, the meaning of lattice reflection and rotation symmetry
is more explict and clearer in our scheme.
Our scheme borrows ideas from the following two papers a lot
- Hauru et. al's Gilt:
https://journals.aps.org/prb/abstract/10.1103/PhysRevB.97.045111
- Evenbly's FET:
https://journals.aps.org/prb/abstract/10.1103/PhysRevB.98.085155

Tensor leg order convention is
A[x, y, x', y']
      y
      |
  x'--A--x
      |
      y'
"""
from .block_rotsym import l1Split
from ncon import ncon
import numpy as np


# I. Initialize the tensors composing the loop using a local TRG split
def init_trg(A, chi, eps):
    eigv, v, err = l1Split(A, chi, eps)
    # convert the eigvalues array into a
    # diagonal matrix Λ
    Lambda = eigv.diag()
    # Lambda connects with the last leg of v throught its first leg
    # like v @ Lambda
    return v, Lambda, err


# II. Determine the isometric tensor p
def densityM(v, Lambda, z=None):
    """Construct the density matrix for the isometry p

    Args:
        v (TensorCommon): 3-leg tensor with real values
        Lambda (TensorCommon): a matrix with real values

    Returns:
        rho (TensorCommon): 4-leg density matrix

    """
    vLmd = ncon([v, Lambda.conj()], [[-1, -2, 1], [-3, 1]])
    if z is None:
        # trivial bond matrix in the square-lattice TN
        rho = ncon([vLmd.conj(), vLmd, vLmd, vLmd.conj()],
                   [[1, -1, 5], [1, -2, 6], [3, -3, 5], [3, -4, 6]])
    else:
        # non-trivial bond matrix in the square-lattice TN
        rho = ncon([vLmd.conj(), vLmd, vLmd, vLmd.conj(), z, z.conj()],
                   [[1, -1, 5], [2, -2, 6], [3, -3, 5], [4, -4, 6],
                    [1, 2], [3, 4]]
                   )
    return rho


def findProj(rho, chi, cg_eps=1e-10):
    """find the isometry p

    Args:
        rho (TensorCommon): density matrix for p
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
    # Simontaneously diagonalize rho and the SWAP matrix
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
    rhoP = rho + SWAP * 1e-9
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


# III. Perform the TRG-like block-tensor map
def buildC(v, p, z=None):
    """build 3-leg tensor C
    for the TRG-like block-tensor map
    """
    if z is None:
        # trivial bond matrix in the square-lattice TN
        C = ncon([v.conj(), v, p.conj()],
                 [[1, 2, -1], [1, 3, -2], [2, 3, -3]]
                 )
    else:
        # non-trivial bond matrix in the square-lattice TN
        C = ncon([v.conj(), v, z, p.conj()],
                 [[1, 3, -1], [2, 4, -2], [1, 2], [3, 4, -3]]
                 )
    return C


def C2Ap(C, Lambda):
    """build A' from C and Λ
    """
    Ct = ncon([C, Lambda], [[1, -2, -3], [-1, 1]])
    Ch = ncon([C, Lambda.conj()], [[-1, 1, -3], [-2, 1]])
    Ap = ncon([Ct, Ch.conj(), Ct, Ch.conj()],
              [[3, 1, -1], [3, 2, -2], [4, 2, -3], [4, 1, -4]]
              )
    return Ap


def trunc_err_func(eigv, chi):
    """
    No need to take square since we work with M M'
    whose eigenvalues are themself square of singular values of M
    """
    res = np.sum(np.abs(eigv[chi:])) / np.sum(np.abs(eigv))
    return res


# IV. Coarse graining the bond matrix z
def cgz(z, p, g):
    znew = ncon([p.conj(), p, z, z],
                [[1, 4, -1], [3, 2, -2], [1, 3], [2, 4]]
                )
    errMsg = "bond matrix should be symmetric!"
    assert znew.transpose([1, 0]).conj().allclose(znew), errMsg
    # EVD znew
    # eigvz, Uz = znew.eig([0], [1], hermitian=True)
    P0, P1 = eig4g(g)
    eigvz, Uz, gnew = eig4Z(znew, P0, P1)
    # singular value part
    zps = eigvz.abs()
    Hz = Uz.copy()
    Hz = Hz.multiply_diag(zps.sqrt(), 1, direction="r")
    # Obtain zp
    zp = eigvz.sign().diag()    # dirs=[1, -1] by default
    return zp, Hz, zps, gnew


# IV.1 For diagonalzie z and g (SWAP sign)
def eig4g(g, isdiag=True):
    """Read off the eigenvector of g
    """
    if isdiag:
        idMat = g.eye(g.shape[0])
        P0 = idMat[:, g.sign() == 1]
        P1 = idMat[:, g.sign() == -1]
    else:
        pass
    return P0, P1


def eig4Z(z, P0, P1):
    """Diagonalize z in smaller blocks
    """
    ZM0 = P0.T @ z @ P0
    ZM1 = P1.T @ z @ P1
    eigvZ0, UZ0 = ZM0.eig([0], [1], hermitian=True)
    eigvZ1, UZ1 = ZM1.eig([0], [1], hermitian=True)
    # assemble eigenvectors
    U = z.zeros(z.shape)
    N0 = P0.shape[1]
    N1 = P1.shape[1]
    U[:, :N0] = P0 @ UZ0
    U[:, N0:] = P1 @ UZ1
    # assemble eigenvalues
    eigv = z.zeros(z.shape[0])
    eigv[:N0] = eigvZ0 * 1.0
    eigv[N0:] = eigvZ1 * 1.0
    # eigenvalues of g in this basis
    eigvg = z.zeros(z.shape[0])
    eigvg[:N0] = [1] * N0
    eigvg[N0:] = [-1] * N1
    return eigv, U, eigvg

# end of file
