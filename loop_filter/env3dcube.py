#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : env3dcube.py
# Author            : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
# Date              : 15.12.2023
# Last Modified Date: 15.12.2023
# Last Modified By  : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
"""
Construct cube (or membrane) environments for
a 3D lattice system on a cubic lattice.

The environments are for the following
two entanglement-filtering schemes:
1) Full-environment trunction (FET)
2) Graph-independent local truncations (GILT)

There are three filtering matrices:,
    sx, sy, sz
for each directions.
The prototyical case is the environments after linearizing the sy.
Other two cases should be obtained by
rotating the inital input 6-leg tensor A.

The computational cost is O(χ^{12})
Order of the tensor leg is A[x, x', y, y', z, z']
_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
"""
from ncon import ncon
from .. import u1ten

# I. For constructing <ψ|ψ> and absorbing sx, sz matrices

def sOnA(A, sx, sz):
    """absorb sx, sz matrices into main tensor A

    Args:
        A (TensorCommon): reference (+++) 6-leg tensor
             z  x'
             | /
         y'--A--y
           / |
         x   z'
        sx (TensorCommon): 2-leg tensor for filtering (x direction)
         χ -- sx -- χs
        sz (TensorCommon): 2-leg tensor for filtering (z direction)

    Returns: 6-leg tensor after absorbing sx and sz
             |
             sz
             | /
           --A--
           / |
         sx
        /
    """
    Axz = ncon([A, sx, sz], [[1, -2, -3, -4, 2, -6], [1, -1], [2, -5]])
    return Axz


def contrInLeg(A, B):
    """contract the three inner legs of two 6-leg tensors

    Args:
        A (TensorCommon): 6-leg tensor
        B (TensorCommon): 6-leg tensor

    Returns:
        dbA (TensorCommon), a 6-leg tensor
        dbA[y, yd, z, zd, x, xd]
              zd  z
              |  |
         yd-- dbA --y
             /  /
           xd  x

    """
    dbA = ncon([A, B],
               [[-5, 1, -1, 2, -3, 3],
                [-6, 1, -2, 2, -4, 3]])
    return dbA


def contrx(dbA, dbB, leg1=None, leg2=None, comm=None):
    """contract (dbA, dbB) pair in x direction

    Args:
        dbA (TensorCommon): 6-leg tensor
        dbB (TensorCommon): 6-leg tensor

    Returns:
        quadrA (TensorCommon): an 8-leg tensor
    """
    if comm is None:
        quadrA = ncon([dbA, dbB],
                      [[-1, -2, -5, -6, 1, 2],
                       [-3, -4, -7, -8, 1, 2]])
    else:
        # parallelization codes
        dbAi = u1ten.fixleg(dbA, 3, leg1)
        dbBj = u1ten.fixleg(dbB, 3, leg2)
        quadrA = ncon([dbAi, dbBj],
                      [[-1, -2, -5, 1, 2],
                       [-3, -4, -6, 1, 2]])
    return quadrA


def contrz(quadrA, quadrB, comm=None):
    """contract (quadrA, quadrB) pair in z direction

    Args:
        quadrA (TensorCommon): 8-leg tensor
        quadrB (TensorCommon): 8-leg tensor

    Returns:
        octuA (TensorCommon): 8-leg tensor

    """
    if comm is None:
        octuA = ncon([quadrA, quadrB],
                     [[-1, -2, -3, -4, 1, 2, 3, 4],
                      [-5, -6, -7, -8, 1, 2, 3, 4]])
    else:
        # parallelization codes
        octuA = ncon([quadrA, quadrB],
                     [[-1, -2, -3, -4, 1, 2],
                      [-5, -6, -7, -8, 1, 2]])
    return octuA


# II. For absorbing sy matrices

def syOn1Leg(dbAp, sy):
    """absorb sy matrix into the doubleA tensor on one leg
    This is used for constructing P

    Args:
        dbAp (TensorCommon): 6-leg tensor
        sy (TensorCommon): 2-leg tensor

    Returns:
        dbApy (TensorCommon)

    """
    dbApy = ncon([dbAp, sy], [[1, -2, -3, -4, -5, -6], [1, -1]])
    return dbApy


def syOn2Leg(dbAgm, sy):
    """absorb sy matrix into the doubleA tensor on two legs
    This is used for constructing γ

    Args:
        dbAgm (TensorCommon): 6-leg tensor
        sy (TensorCommon): 2-leg tensor

    Returns:
        dbAgmy (TensorCommon)

    """
    dbAgmy = ncon([dbAgm, sy, sy.conj()],
                  [[1, 2, -3, -4, -5, -6],
                   [1, -1], [2, -2]])
    return dbAgmy


def syOn1LegQuad(quadrA, sy):
    """absorb sy matrix into the first leg of quadrA tensor
    This is used for constructing P

    Args:
        quadrA (TensorCommon): 8-leg tensor
        sy (TensorCommon): 2-leg tensor

    Returns:
        quadrAy (TensorCommon)

    """
    quadrAy = ncon([quadrA, sy],
                   [[1, -2, -3, -4, -5, -6, -7, -8], [1, -1]]
                   )
    return quadrAy


def syOn2LegQuad(quadrA, sy):
    """absorb (sy, sy_dagger) into the first two legs of quadrA tensor
    This is used for constructing γ

    Args:
        quadrA (TensorCommon): 8-leg tensor
        sy (TensorCommon): 2-leg tensor

    Returns:
        quadrAy (TensorCommon)

    """
    quadrAy = ncon([quadrA, sy, sy.conj()],
                   [[1, 2, -3, -4, -5, -6, -7, -8],
                    [1, -1], [2, -2]])
    return quadrAy


# III. Octuple tensor to P and γ
def octu2P(octuA, sy):
    """contract (octuA, octuA.conj()) pair and and sy
    Do the octuA contraction first is faster than
    do the sy contraction first

    Args:
        octuA (TensorCommon): 8-leg tensor
        sy (TensorCommon): 2-leg tensor

    Returns:
        Psy (TensorCommon)

    """
    Psy0 = ncon([octuA, octuA.conj()],
                [[-1, 1, 2, 3, 4, 5, 6, 7],
                 [-2, 1, 2, 3, 4, 5, 6, 7]])
    Psy_dagger = ncon([Psy0, sy.conj()], [[-1, 1], [1, -2]])
    return Psy_dagger.conj()


def octu2gm(octuA, sy):
    """construct γ matrix from octuple tensor

    Args:
        octuA (TensorCommon): 8-leg tensor
        sy (TensorCommon): 2-leg tensor

    Returns:
        Gamma (TensorCommon)

    """
    Gamma0 = ncon([octuA, octuA.conj()],
                  [[-3, -1, 1, 2, 3, 4, 5, 6],
                   [-4, -2, 1, 2, 3, 4, 5, 6]]
                  )
    Gamma = ncon([Gamma0, sy, sy.conj()],
                 [[-1, 1, -3, 2], [1, -2], [2, -4]])
    return Gamma


# IV. Combine the above functions to construct P and γ
def dbA2P_old(dbAp, sy):
    """Build P from doubleA tensor and sy matrix
    This is simple a combination of functions in Parts I--III

    Args:
        dbAp (TensorCommon): 6-leg tensor
        sy (TensorCommon): 2-leg tensor

    Returns:
        P (TensorCommon): 2-leg tensor

    """
    # Cost is χ^4 χs^3
    dbApy = syOn1Leg(dbAp, sy)
    # Cost is χ^6 χs^4
    # For chi=[6,6], chis=[3,3], it takes about 0.4 seconds (CPU times)
    quadrAp = contrx(dbAp, dbApy.conj())
    # Cost is χ^6 χs^4 (it is faster than absorbing sy into `quadrAp`)
    # For chi=[6,6], chis=[3,3], it takes about 0.2 seconds (CPU times)
    quadrApy = contrx(dbApy, dbApy.conj())
    # Cost is χ^7 χs^5 <-- bottleneck of the computational cost
    # For chi=[6,6], chis=[3,3], it takes about 7 seconds (CPU times)
    # and 1.2 seconds of wall time using 8 threads
    octuAp = contrz(quadrAp, quadrApy.conj())
    # Cost is χ^6 χs^3 (it is faster than absorbing sy into `quadrAp`)
    # For chi=[6,6], chis=[3,3], it takes about 0.4 seconds (CPU times)
    Psy = octu2P(octuAp, sy)
    return Psy


def dbA2gm_old(dbAgm, sy):
    """Build γ from doubleA tensor and sy matrix
    This is simple a combination of functions in Parts I--III

    Args:
        dbAgm (TensorCommon): 6-leg tensor
        sy (TensorCommon): 2-leg tensor

    Returns:
        Gamma (TensorCommon): 4-leg tensor

    """
    # Cost is χ^2 χs^5
    dbAgmy = syOn2Leg(dbAgm, sy)
    # Cost is χ^2 χs^8
    # For chi=[6,6], chis=[3,3], it takes about 0.04 seconds (CPU times)
    quadrAgm = contrx(dbAgm, dbAgmy.conj())
    # Cost is χ^0 χs^10 (it is faster than absorbing sy into `quadrAgm`)
    # For chi=[6,6], chis=[3,3], it takes about 0.03 seconds (CPU times)
    quadrAgmy = contrx(dbAgmy, dbAgmy.conj())
    # Cost is χ^2 χs^10 <-- bottleneck in this function
    # For chi=[6,6], chis=[3,3], it takes about 0.37 seconds (CPU times)
    # and 0.08 seconds of wall time using 8 threads
    octuAgm = contrz(quadrAgm, quadrAgmy.conj())
    # Cost is χ^4 χs^6
    # For chi=[6,6], chis=[3,3], it takes about 0.06 seconds (CPU times)
    Gamma = octu2gm(octuAgm, sy)
    return Gamma


def dbA2octu(dbA, dbAy, comm=None):
    if comm is None:
        # Cost is χ^6 χs^4
        quadrA = contrx(dbA, dbAy.conj())
        # Cost is χ^6 χs^4 (it is faster than absorbing sy into `quadrAp`)
        quadrAy = contrx(dbAy, dbAy.conj())
        # Cost is χ^7 χs^5 <-- bottleneck of the computational cost
        octuA = contrz(quadrA, quadrAy.conj())
    else:
        # parallelization codes start
        # 0. broadcast the input tensors
        dbA = comm.bcast(dbA, root=0)
        dbAy = comm.bcast(dbAy, root=0)
        # 1. initialize the output tensor after contraction
        octuA = 0
        # 2. determine the job of each process
        rank = comm.Get_rank()   # rank of current process
        size = comm.Get_size()   # size of process
        jobind = 0               # indicator of the job in the for loop
        for i in u1ten.loopleg(dbA, 3):
            for j in u1ten.loopleg(dbA, 3):
                # parallelization process: χ^2 or χs^2
                # check whether the job belongs to current process
                if jobind % size != rank:
                    jobind += 1
                    continue
                # Cost is χ^4 χs^4
                quadrAij = contrx(
                    dbA, dbAy.conj(), leg1=i, leg2=j, comm=comm
                )
                # Cost is χ^4 χs^4
                quadrAyij = contrx(
                    dbAy, dbAy.conj(), leg1=i, leg2=j, comm=comm
                )
                # Cost is χ^5 χs^5 <-- bottleneck of the computational cost
                octuA += contrz(quadrAij, quadrAyij.conj(), comm=comm)
                # increase the job indicator for parallel computation
                jobind += 1
        # 3. collective reducing sum operation
        octuA = comm.allreduce(octuA)
        # parallelization codes end
    return octuA


def dbA2P(dbAp, sy, comm=None):
    """Build P from doubleA tensor and sy matrix
    This is simple a combination of functions in Parts I--III

    Args:
        dbAp (TensorCommon): 6-leg tensor
        sy (TensorCommon): 2-leg tensor

    Returns:
        P (TensorCommon): 2-leg tensor

    """
    # Cost is χ^4 χs^3
    dbApy = syOn1Leg(dbAp, sy)
    octuAp = dbA2octu(dbAp, dbApy, comm=comm)
    Psy = octu2P(octuAp, sy)
    return Psy


def dbA2gm(dbAgm, sy, comm=None):
    """Build γ from doubleA tensor and sy matrix
    This is simple a combination of functions in Parts I--III

    Args:
        dbAgm (TensorCommon): 6-leg tensor
        sy (TensorCommon): 2-leg tensor

    Returns:
        Gamma (TensorCommon): 4-leg tensor

    """
    # Cost is χ^2 χs^5
    dbAgmy = syOn2Leg(dbAgm, sy)
    octuAgm = dbA2octu(dbAgm, dbAgmy, comm=comm)
    Gamma = octu2gm(octuAgm, sy)
    return Gamma
