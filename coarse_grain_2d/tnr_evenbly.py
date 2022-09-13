#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : tnr_evenbly.py
# Author            : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
# Date              : 08.09.2022
# Last Modified Date: 08.09.2022
# Last Modified By  : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
import numpy as np
from ncon import ncon
from .. import u1ten
from .trg_evenbly import (opt_q, build_qstarA,
                          construct_v_env, construct_w_env,
                          trunc_err_func
                          )


def dotnr(Ain, chiM, chiS, chiU, chiH, chiV, dtol=1e-16,
          disiter=2000, miniter=100, convtol=0.01, is_display=True):
    """
    Block 4 tensors into 1 coarser tensor according to
    the tnr by Evenbly.

    We assume reflection symmetry for the input tensor Ain
    See Evenbly's website here: https://www.tensors.net/p-tnr

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
    ---
    Below are additional parameters for the disentangling
    process making tnr different from trg
    ---
    chiS, chiU: integer
        - bond dimensions for two furthur-squeezely legs,
        used to drive the fixed-point away from CDL tensors
    disiter, miniter: integer
        - maximal and minimal disentangling iteration step
    convtol: float
        - if the change of a certain "cost function" is small
        during the disentangling, then stop the iteration
    is_display: boolean
        printout informations during the disentangling process
    """
    err_messg = ("Input tensor Ain doesn't satisfy" +
                 " the reflection symmetry condition")
    assert ((Ain - Ain.transpose([2, 3, 0, 1])).norm() /
            Ain.norm() < 1e-10), err_messg
    # The first step is exactly the same as
    # the evenbly's trg implementation (see the import line)
    q, SPerrq = opt_q(Ain, chiM, dtol=dtol)
    # disentangling process, the key of tnr
    (s,
     y,
     u,
     SPerr_disent
     ) = opt_syu(Ain, q, chiS, chiU,
                 disiter, miniter, convtol,
                 dtol, is_display)
    # the final two step is similar to the trg
    # or the hotrg
    v, SPerrv = opt_v(Ain, q, s, y, chiH, dtol=dtol)
    w, SPerrw = opt_w(Ain, q, s, y, chiV, dtol=dtol)
    # block four input A tensors and other squeezing tensors
    Aout = block_4tensor(Ain, q, s, y, v, w)
    SPerr_list = [SPerrq, SPerr_disent, SPerrv, SPerrw]
    return Aout, q, s, u, y, v, w, SPerr_list


# ====== Crucial part: disentangling ======== |v|
def opt_syu(Ain, q, chiS, chiU,
            disiter=2000, miniter=100, convtol=0.01,
            dtol=1e-10, is_display=True):
    """Iteratively determine unitary u, isometric y
    and s matrix in the disentangling process.
    The solution is guided to drive away CDT tensors

    Args:
        Ain: abeliantensors
            input 4-leg tensor
        q (TODO): abeliantensors
            3-leg isometry for reducing
            the computational cost
        chiS, chiU: int
            - squeezed bond dimension
            during the disentangling process

    Kwargs:
        disiter: int
            maximal disentangling iteration step
        miniter: int
            minimal disentangling iteration step
        convtol: float
        dtol: float
        is_display: boolean
            print information during the disentangling

    Returns:
    ------
    s, y, u: abeliantensors
        used in the disentangling process.
        u is unitary for disentangling
        y is isometric
        s is a matrix
    """
    def prt_disent_err(k, disiter, SPerr, is_display):
        if is_display:
            print("Iteration: {:d} of {:d} disentangling,".format(k, disiter),
                  "Trunc. Error: {:.3e}".format(SPerr))

    SPerr = 1
    # initialize s, y, u
    s, y, u = init_syu(Ain, q, chiS, chiU)
    # update s, y, u iteratively
    for k in range(disiter + 1):
        # -- This first part is for printing out -------- #
        # -- approximation errors and break out --------- #
        # -- the iteration if the improvement is small -- #
        # For every 100 step, check the approximation error
        # for the disentangling
        if np.mod(k, 100) == 0:
            SPerrnew = SP_disent(Ain, q, s, y, u)[0]
            # if the improvement of the approximation error
            # is very small, exit the iteration
            if k > 50:
                errdelta = abs(SPerrnew - SPerr) / abs(SPerr)
                if (errdelta < convtol) or (abs(SPerrnew) < dtol):
                    SPerr = SPerrnew
                    prt_disent_err(k, disiter, SPerr, is_display)
                    break
            SPerr = SPerrnew
            prt_disent_err(k, disiter, SPerr, is_display)
        # ----------- The first part is done ------------ #

        # update s
        s = update_s_evenbly(Ain, q, s, y, u, dtol=dtol)
        # update y and u
        if k > 50:
            y = update_y(Ain, q, s, y, u)
            u = update_u(Ain, q, s, y, u)
    # ------- disentangling iteration finished ----- #

    # Since the optimization uses fidelity as the cost function,
    # the overall magnetitute of the tensor is not determined.
    # (this is like using minimizing the angle between two vectors)
    # In this final step, we fix the overall magnitute by demanding
    # the norm of the C tensor should not change (<ψ|ψ> = <φ|φ>),
    # the norm is absorbed into the matrix s.
    C = construct_C(Ain, q)
    psipsi = ncon([C, C.conjugate()], [[1, 2, 3, 4], [1, 2, 3, 4]])
    Cmod = ncon([C, s.conjugate(), s, s, s.conjugate()],
                [[1, 2, 3, 4],
                 [1, -1], [2, -2],
                 [3, -3], [4, -4]])
    phiphi = ncon([Cmod, Cmod.conjugate()], [[1, 2, 3, 4], [1, 2, 3, 4]])
    phi2psi = (phiphi / psipsi).norm()
    s = s / (phi2psi)**(1/8)
    return s, y, u, SPerr

# TODO
def init_syu(Ain, q, chiS, chiU):
    """
    A good way to initialize s, y, u
    """
    # TODO: `sliceten` to be implemented
    # the initial unitary u is
    # a direct product of two identity matrices
    # u = np.kron(np.eye(chiVI, chiU), np.eye(chiVI, chiU)).reshape(chiVI,chiVI,chiU,chiU)
    eye4u = Ain.eye()
    u = ncon([eye4u, eye4u], [[-1, -3], [-2, -3]])
    # the initial isometry y is taken from q
    # y = q[:, :u.shape[2], :chiS]
    y = sliceten(q, posleg=1, legrange=None)
    y = sliceten(y, posleg=2, legrange=None)
    # the initial matrix s is identity matrix
    s = q.eye(q.shape[2])
    s = sliceten(s, posleg=1, legrange=None)
    return s, y, u


# for updating the s matrix
# function for updating s, y, u
def update_s_evenbly(Ain, q, s, y, u, dtol=1e-10):
    # Square of projective truncation norms and error
    SPerrold, SPexact = SP_disent(Ain, q, s, y, u)
    # propose trial new s using its linearlized environment
    C = construct_C(Ain, q)
    CCstar = construct_CCstar(C)
    omega_s = construct_omega_s(CCstar, s)
    center_gamma_s = construct_center_gamma_s(CCstar, q, y, u)
    gamma_s = construct_gamma_s(center_gamma_s, s)
    stemp = ncon([u1ten.pinv(omega_s/omega_s.trace(), [0], [1],
                             eps_mach=dtol),
                  gamma_s],
                 [[-1, 1], [1, -2]])
    # we normalize the s during the optimization,
    # the proper norm is taken care of later
    stemp = stemp / stemp.norm()
    # mix stemp with old s
    for p in range(11):
        # take the convex combination
        snew = (1 - 0.1 * p) * stemp + (0.1 * p) * s
        # check the new approximation error, 1 - new_fidelity
        SPerrnew = SP_disent(Ain, q, snew, y, u)[0]
        if SPerrnew <= SPerrold:
            snew = snew / snew.norm()
            break
    return snew


def update_y(Ain, q, s, y, u):
    C = construct_C(Ain, q)
    CCstar = construct_CCstar(C)
    qstar = q.conjugate()
    sstar = s.conjugate()
    ystar = y.conjugate()
    ustar = u.conjugate()
    # environment of y
    env_y = ncon([CCstar, qstar, q,
                  ustar, ystar, sstar, s],
                 [[1, 2, 8, 4], [-1, 9, 8], [3, 6, 4],
                  [9, 6, -2, 7], [3, 7, 5], [1, -3], [2, 5]]
                 )
    ynew = linear_opt(env_y, leftlegs=[0, 1], rightlegs=[2])
    return ynew


def update_u(Ain, q, s, y, u):
    C = construct_C(Ain, q)
    CCstar = construct_CCstar(C)
    qstar = q.conjugate()
    sstar = s.conjugate()
    ystar = y.conjugate()
    # environment of ustar
    env_u = ncon([CCstar, qstar, q,
                  y, ystar, sstar, s],
                 [[1, 2, 4, 7], [3, -1, 4], [6, -2, 7],
                  [3, -3, 5], [6, -4, 8], [1, 5], [2, 8]]
                 )
    ustarnew = linear_opt(env_u, leftlegs=[0, 1], rightlegs=[2, 3])
    unew = ustarnew.conjugate()
    return unew


def linear_opt(env, leftlegs, rightlegs):
    """
    Evenbly-Vidal linearization optimization of
    isometric tensors

    Input:
    env: abeliantensors
        environment for the isometry to be optimized

    Return:
        isometry_new: abeliantensors
    """
    llegn = len(leftlegs)
    rlegn = len(rightlegs)
    # leg specified should equal to the
    # total leg of the environment
    assert len(env.shape) == (llegn + rlegn)
    # svd the environment
    u, s, vh = env.svd(leftlegs, rightlegs)
    # new isometry is constructed from u and vh
    u_ind = [-(k + 1) for k in range(llegn)]
    vh_ind = [-(k + 1 + llegn) for k in range(rlegn)]
    u_ind = u_ind + [1]
    vh_ind = [1] + vh_ind
    # remember to take the conjugate here
    isometry_new = ncon([u, vh], [u_ind, vh_ind]).conjugate()
    return isometry_new


# calculating the approximation error for the disentangling process
def SP_disent(Ain, q, s, y, u):
    """
    Approximation error for the disentangling process
    """
    # combine Ain and q for convenience
    C = construct_C(Ain, q)
    CCstar = construct_CCstar(C)
    # the scalar is refered to as
    # <ψ|ψ> in my notes
    # it is known and given
    SPexact = ncon([CCstar], [[1, 2, 1, 2]])
    # Square projection norm after approximation
    # prepare for construction of environments
    omega_s = construct_omega_s(CCstar, s)
    center_gamma_s = construct_center_gamma_s(CCstar, q, y, u)
    gamma_s = construct_gamma_s(center_gamma_s, s)
    # the error is defined as  1 - fidelity
    phiphi = ncon([omega_s, s.conjugate(), s], [[1, 2], [1, 3], [2, 3]])
    phipsi = ncon([gamma_s, s.conjugate()], [[1, 2], [1, 2]])
    SPerr = (1 -
             phipsi.norm()**2 / (phiphi * SPexact).norm()
             )
    return SPerr, SPexact


def construct_C(Ain, q):
    qstarA = build_qstarA(q, Ain)
    qB = qstarA.conjugate()
    C = ncon([qstarA, qB, qB, qstarA],
             [[1, 3, -1], [2, 3, -2],
              [1, 4, -3], [2, 4, -4]])
    return C


def construct_CCstar(C):
    CCstar = ncon([C, C.conjugate()],
                  [[-1, -2, 1, 2],
                   [-3, -4, 1, 2]])
    return CCstar


def construct_omega_s(CCstar, s):
    omega_s = ncon([CCstar, s, s.conjugate()],
                   [[-1, 1, -2, 2], [1, 3], [2, 3]])
    return omega_s


def construct_center_gamma_s(CCstar, q, y, u):
    qstar = q.conjugate()
    ystar = y.conjugate()
    ustar = u.conjugate()
    center_gamma_s = ncon([CCstar, qstar, q,
                           ustar, y, ystar],
                          [[-1, -3, 7, 8], [1, 3, 7], [4, 6, 8],
                           [3, 6, 2, 5], [1, 2, -2], [4, 5, -4]]
                          )
    return center_gamma_s


def construct_gamma_s(center_gamma_s, s):
    gamma_s = ncon([center_gamma_s, s],
                   [[-1, -2, 1, 2], [1, 2]])
    return gamma_s


# ====== End of the crucial part ============ |^|


# ====== Easy part: for determine ======== |v|
# = the horizontal and vertical squeezer = |v|
def build_sq(q, s):
    return ncon([q, s], [[-1, -2, 1], [1, -3]])


def opt_v(Ain, q, s, y, chiH, dtol=1e-10):
    """determine a good 3-leg isometry v
    v for squeezing legs in vertical direction
    to produce a horizontal coarser leg

    :Ain: abeliantensors
        input 4-leg tensor
    :q, s, y: abeliantensors
            q and y are two 3-leg isometric tensors,
            s is a matrix to squeeze out the CDL tensors
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
    # almost the same as Evenbly's trg construction
    # except 1) we use the contraction of s, q, A,
    # instead of just q, A; and 2) we replace some
    # q using y
    sq = build_sq(q, s)
    sqstarA = build_qstarA(sq, Ain)
    # remaining is exactly the same as Evenbly's
    # TRG implementation
    env_v = construct_v_env(sqstarA, y)
    (d,
     v,
     SPerr
     ) = env_v.eig([0, 1], [2, 3], hermitian=True,
                   chis=[i+1 for i in range(chiH)],
                   trunc_err_func=trunc_err_func,
                   eps=dtol, return_rel_err=True)
    return v, SPerr


def opt_w(Ain, q, s, y, chiV, dtol=1e-10):
    """determine a good 3-leg isometry w
    w is for squeezing legs in horizontal direction
    to produce a vertical coarser leg

    :Ain: abeliantensors
        input 4-leg tensor
    :q, s, y: abeliantensors
            q and y are two 3-leg isometric tensors,
            s is a matrix to squeeze out the CDL tensors
    :chiV: int
        horizontal squeezed bond dimension
    :dtol: float

    ------
    :returns:
    w: abeliantensors
        3-leg squeezer isometry to squeeze
        two horizontal legs to produce
        a coarser horizontal leg.
    SPerr: float
       square of the approximaion error
       for the projective truncation
    """
    # almost the same as Evenbly's trg construction
    # except 1) we use the contraction of s, q, A,
    # instead of just q, A; and 2) we replace some
    # q using y
    sq = build_sq(q, s)
    sqstarA = build_qstarA(sq, Ain)
    # remaining is exactly the same as Evenbly's
    # TRG implementation
    env_w = construct_w_env(sqstarA, y)
    (d,
     w,
     SPerr
     ) = env_w.eig([0, 1], [2, 3], hermitian=True,
                   chis=[i+1 for i in range(chiV)],
                   trunc_err_func=trunc_err_func,
                   eps=dtol, return_rel_err=True)
    return w, SPerr


def block_4tensor(Ain, q, s, y, v, w):
    """block 4 input tensors
    - isometry v, w are for squeezing bond dimensions
    - isometry q is for reducing computational cost
    - isometry y and matrix s comes from distengling process

    :Ain: abeliantensors
        4-leg input tensor
    :q: abeliantensors
        3-leg isometry for squeezing in middle step
    :s: abeliantensors
        matrix for squeezing out CDL structure
    :y: abeliantensors
        3-leg isometry corresponding to q;
        different from q due to s and implicit
        disentangling scheme proposed by Evenbly
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
    pass
# ====== End of the easy part =========== |^|
