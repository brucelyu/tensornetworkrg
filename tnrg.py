#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : tnrg.py
# Author            : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
# Date              : 20.05.2022
# Last Modified Date: 27.09.2023
# Last Modified By  : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>

from scipy.sparse.linalg import eigs, LinearOperator
import scipy.stats as spstats
import numpy as np
from abeliantensors import TensorZ2, Tensor
from ncon import ncon
from .initial_tensor import initial_tensor
from .coarse_grain_2d import trg_evenbly, tnr_evenbly, hotrg, block_rotsym
from .coarse_grain_2d import hotrg_grl as hotrg2d
from .coarse_grain_2d import trg as trg2d
from .coarse_grain_3d import hotrg as hotrg3d
from .coarse_grain_3d import block_tensor as bkten3d
from .coarse_grain_3d import efrg as efrg3d
from .loop_filter import (
    cleanLoop, toymodels, fet3d, env3d, fet3dcube, fet3dloop,
    fet2d_rotsym
)
from . import u1ten
from datetime import datetime
from dateutil.relativedelta import relativedelta


class TensorNetworkRG:
    """
    Tensor network renormalization group
    """
    model_choice = ("ising2d", "ising3d", "golden chain", "cdl2d",
                    "hardsquare1NN")
    ising_models = ("ising2d", "ising3d")

    def __init__(self, model):
        """__init__.
        Initialize a TensorNetworkRG instance.
        ----------
        `model`: string
            model name, default value as "isng2d"
            choose among `class.model_choice`.
        -----------
        Other initial data:
        `iter_n`: int
            current iteration step of RG
        `current_tensor`: tensor
            current tensor,
            the tensor magnitude is normalized to 1
        `tensor_magnitude`: list of float
            current and previous tensor magnitutde pulled
            out of the tesnor when normalizing
        """
        assert model in self.model_choice, "model name incorrect!"
        self.model = model
        self.model_parameters = {}
        self.iter_n = 0
        self.current_tensor = None
        self.tensor_magnitude = []
        self.isometry_applied = None
        self.exact_free_energy = 0

    # fetch instance properties #
    # ------------------------- #
    def get_model(self):
        """
        return model name
        """
        return self.model

    def get_iteration(self):
        """
        return current RG step
        """
        return self.iter_n

    def get_tensor(self):
        """
        return current tensor
        """
        return self.current_tensor.copy()

    def get_tensor_magnitude(self):
        """
        return list of tensor magnitudes
        """
        return self.tensor_magnitude.copy()

    def get_isom(self):
        return self.isometry_applied.copy()

    def get_model_parameters(self):
        """
        return model parameters
        """
        return self.model_parameters.copy()

    def get_exact_free_energy(self):
        errmsg = "The model should be 2d Ising"
        assert self.model == "ising2d", errmsg
        import scipy.integrate as integrate
        beta = 1 / self.model_parameters["temperature"]
        k = 1 / (np.sinh(2 * beta)**2)
        integrand = (
            lambda theta: np.log((np.cosh(2 * beta))**2 +
                                 1 / k * np.sqrt(1 + k**2-2*k*np.cos(2*theta))
                                 )
                     )
        self.exact_free_energy = (np.log(2) / 2 +
                                  1 / (2 * np.pi) * integrate.quad(integrand,
                                                                   0, np.pi)[0]
                                  )
        return self.exact_free_energy
    # ------------------------- #

    # set model parameters #
    # -------------------- #
    def set_model_parameters(self, temperature, ext_h):
        """
        Save the model parameter to `self.model_parameters`
        """
        self.model_parameters["temperature"] = temperature
        self.model_parameters["magnetic_field"] = ext_h

    def set_critical_model(self):
        """
        set the `self.model_parameters` to criticality
        """
        if self.model == "ising2d":
            self.model_parameters["magnetic_field"] = 0.0
            self.model_parameters["temperature"] = 2 / np.log(1 + np.sqrt(2))
        elif self.model == "ising3d":
            self.model_parameters["magnetic_field"] = 0.0
            self.model_parameters["temperature"] = 4.51152469
    # -------------------- #

    def generate_initial_tensor(self, onsite_symmetry=False, scheme="simple",
                                init_dirs=[1, 1, 1, 1]):
        """
        generate the initial tensor corresponding to the model
        `model_parameters`: dictionary
            - For the ising model in 2d and 3d, an exmaple is
                {"temperature": 1.0, "magnetic_field": 0.0,
                "onsite_symmetry": False}
        """
        init_ten = initial_tensor(self.model, self.model_parameters,
                                  onsite_symmetry, scheme)
        self.tensor_magnitude.append(init_ten.norm())
        self.current_tensor = init_ten / init_ten.norm()
        if self.model == "ising2d" and onsite_symmetry:
            self.current_tensor.dirs = init_dirs

    def generate_cdl(self, cornerChi=2, isZ2=False):
        cdl, loop, cmat = toymodels.cdlten(cornerChi, isZ2)
        self.tensor_magnitude.append(cdl.norm())
        self.current_tensor = cdl / cdl.norm()
        self.loop = loop
        self.cmat = cmat

    # pull out tensor magnitude #
    # ------------------------- #
    def pullout_magnitude(self):
        ten_mag = self.current_tensor.norm()
        self.current_tensor = self.current_tensor / ten_mag
        return ten_mag

    def save_tensor_magnitude(self, ten_mag):
        self.tensor_magnitude.append(ten_mag)
        self.iter_n += 1
        assert len(self.tensor_magnitude) == (self.iter_n + 1)
    # ------------------------- #

    @staticmethod
    def linearRG2scaleD(linearRG, dim_psiA, N_scaleD, baseEig=None,
                        b=2, d=3):
        """
        Diagonalize the linearized RG equation specified by responseMatFun
        to calculate the first N_scaleD scaling dimensions
        Parameters
        ----------
        responseMatFun : a linear map
            deltaPsiA --> deltaPsiAc, from a 1D array to another
        dim_psiA : int
            dimensinality of the 1D array deltaPsiA and deltaPsiAc:
                len(deltaPsiA)
        N_scaleD : int
            number of scaling dimensions to extract
        baseEig: float
            eigenvalue of the identity operator = b^d
        b: int
            rescaling factor of the RG map
        d: int
            spacial dimension
        """
        # diagonalize the linearized RG map
        hyperOp = LinearOperator((dim_psiA,) * 2, matvec=linearRG)
        # prepare the initial vector by applying the hyper-operator once
        v0 = hyperOp.matvec(
            np.random.rand(dim_psiA)
        )
        eigVals = np.sort(
            abs(
                eigs(hyperOp, k=N_scaleD, v0=v0,
                     which='LM', return_eigenvectors=False)
            )
        )
        eigVals = eigVals[::-1]

        # input fixed-point tensor has the correct norm (not used here)
        # see standard textbook for this formula
        # scDims = d - np.log(abs(eigVals)) / np.log(b)

        # We fixed the identity operator scaling dimension to 0
        if baseEig is None:
            scDims = -np.log(abs(eigVals/eigVals[0])) / np.log(b)
            return scDims, eigVals[0]
        else:
            scDims = -np.log(abs(eigVals/baseEig)) / np.log(b)
            return scDims


class TensorNetworkRG2D(TensorNetworkRG):
    """
    TNRG for 2D square lattice
    Tensor leg order convention is
    A[x, y, x', y']
      y
      |
   x--A--x'
      |
      y'
    """
    def block_tensor(self):
        """
        Block 4 tensors to be a single coarser tensor.
        No truncation
        """
        ten_cur = self.get_tensor()
        # block 4 tensors to for a new tensor (no truncation)
        ten_new = ncon([ten_cur]*4, [[-2, -3, 3, 1], [3, -4, -6, 2],
                                     [-1, 1, 4, -7], [4, 2, -5, -8]]
                       )
        ten_new = ten_new.join_indices((0, 1), (2, 3), (4, 5), (6, 7),
                                       dirs=[1, 1, -1, -1])
        # truncate the leg a la hosvd
        self.current_tensor = self.truncate_hosvd(ten_new)
        # pull out the tensor norm and save
        ten_mag = self.pullout_magnitude()
        self.save_tensor_magnitude(ten_mag)

    def trg(self, pars={"chiM": 4, "chiH": 4, "chiV": 4, "dtol": 1e-16}):
        if self.iter_n == 0:
            self.init_dw()
            self.boundary = "anti-parallel"
        ten_cur = self.get_tensor()
        chiM = pars["chiM"]
        chiH = pars["chiH"]
        chiV = pars["chiV"]
        dtol = pars["dtol"]
        # block 4 tensors according to evenbly's implementation
        (self.current_tensor,
         q, v, w,
         SPerr_list
         ) = trg_evenbly.dotrg(ten_cur, chiM, chiH, chiV, dtol)
        # pull out the tensor norm and save
        ten_mag = self.pullout_magnitude()
        self.save_tensor_magnitude(ten_mag)
        self.isometry_applied = [v, w]

    def tnr(self, pars={"chiM": 4, "chiH": 4, "chiV": 4, "dtol": 1e-16,
                        "chiS": 4, "chiU": 4,
                        "disiter": 2000, "miniter": 100, "convtol": 0.01,
                        "is_display": True}):
        if self.iter_n == 0:
            self.init_dw()
            self.boundary = "anti-parallel"
        ten_cur = self.get_tensor()
        d_w_prev = self.d_w
        chiM = pars["chiM"]
        chiH = pars["chiH"]
        chiV = pars["chiV"]
        chiS = pars["chiS"]
        chiU = pars["chiU"]
        dtol = pars["dtol"]
        disiter = pars["disiter"]
        miniter = pars["miniter"]
        convtol = pars["convtol"]
        is_display = pars["is_display"]
        (self.current_tensor,
         q, s, u, y, v, w,
         SPerr_list,
         self.d_w
         ) = tnr_evenbly.dotnr(ten_cur, chiM, chiS, chiU,
                               chiH, chiV, dtol, d_w_prev,
                               disiter, miniter, convtol,
                               is_display)
        if is_display:
            print("Truncation Errors:",
                  "{:.2e}, {:.2e}, {:.2e}, {:.2e}".format(
                      SPerr_list[0], SPerr_list[1],
                      SPerr_list[2], SPerr_list[3]
                  )
                  )
        # pull out the tensor norm and save
        ten_mag = self.pullout_magnitude()
        self.save_tensor_magnitude(ten_mag)
        self.isometry_applied = [v, w]
        # return local approximation errors
        lrerr = SPerr_list[1]
        return lrerr, SPerr_list[2:]

    def fet_hotrg(
        self,
        pars={"chi": 4, "dtol": 1e-16,
              "chis": None, "iter_max": 10,
              "epsilon": 1e-10, "epsilon_init": 1e-10,
              "bothSides": True, "display": True
              },
        # tentative
        init_stable=False
    ):
        if self.iter_n == 0:
            self.boundary = "parallel"
        ten_cur = self.get_tensor()
        chi = pars["chi"]
        dtol = pars["dtol"]
        chis = pars["chis"]
        iter_max = pars["iter_max"]
        epsilon = pars["epsilon"]
        epsilon_init = pars["epsilon_init"]
        bothSides = pars["bothSides"]
        display = pars["display"]
        # first apply the FET
        (Alf,
         s,
         errFET,
         d_debug
         ) = cleanLoop.fet2dReflSym(ten_cur, chis, epsilon, iter_max,
                                    epsilon_init, bothSides=bothSides,
                                    init_soft=False,
                                    display=display,
                                    return_init_s=True,
                                    init_stable=init_stable)
        if display:
            print("FET error (or rather 1 - fidelity)",
                  "is {:.4e}.".format(np.abs(errFET)))
            print("----------")
        # then use hotrg to coarse graining
        (self.current_tensor,
         v, w, vin,
         SPerrList
         ) = hotrg.reflSymHOTRG(Alf, chi, dtol,
                                horiSym=bothSides)
        if display:
            print("The two outer projection errors are")
            print("Vertical: {:.4e}".format(SPerrList[0]))
            print("Horizontal: {:.4e}".format(SPerrList[1]))
            print("The inner projection error is")
            print("{:.4e}".format(SPerrList[2]))
            print("----------")
        # pull out the tensor norm and save
        ten_mag = self.pullout_magnitude()
        self.save_tensor_magnitude(ten_mag)
        return d_debug, errFET, SPerrList

    def hotrg(
        self,
        pars={"chi": 4, "dtol": 1e-16,
              "display": True}
    ):
        """
        This scheme exploits the lattice reflection symmetry
        of the underlying model
        """
        if self.iter_n == 0:
            self.boundary = "parallel"
        # read hotrg parameters
        chi = pars["chi"]
        dtol = pars["dtol"]
        display = pars["display"]
        # use hotrg to coarse graining
        ten_cur = self.get_tensor()
        (self.current_tensor,
         v, w, vin,
         SPerrList
         ) = hotrg.reflSymHOTRG(ten_cur, chi, dtol,
                                horiSym=True)
        if display:
            print("The two outer projection errors are")
            print("Vertical: {:.4e}".format(SPerrList[0]))
            print("Horizontal: {:.4e}".format(SPerrList[1]))
            print("The inner projection error is")
            print("{:.4e}".format(SPerrList[2]))
            print("----------")
        # pull out the tensor norm and save
        ten_mag = self.pullout_magnitude()
        self.save_tensor_magnitude(ten_mag)

        # return approximation errors
        # no loop-filtering error
        lrerr = 0
        return lrerr, SPerrList

    def hotrg_grl(
        self,
        pars={"chi": 4, "dtol": 1e-16,
              "display": True}
    ):
        if self.iter_n == 0:
            self.boundary = "parallel"
        # read hotrg parameters
        chi = pars["chi"]
        cg_eps = pars["dtol"]
        display = pars["display"]
        # use hotrg to coarse-grain the tensor
        ten_cur = self.get_tensor()
        (self.current_tensor,
         px, py, errx, erry
         ) = hotrg2d.cgTen(
             ten_cur, chi, cg_eps=cg_eps
         )
        if display:
            print("///////////////////////////")
            print("The HOTRG errors are")
            print("Vertical:   {:.4e}".format(errx))
            print("Horizontal: {:.4e}".format(erry))
            print("===========================")

        # pull out the tensor norm and save
        ten_mag = self.pullout_magnitude()
        self.save_tensor_magnitude(ten_mag)

        # there is no entanglement filtering
        lrerr = 0
        # store the RG errors as a list
        SPerrList = [errx, erry]
        return lrerr, SPerrList

    def trg_grl(
        self,
        pars={"chi": 4, "dtol": 1e-16,
              "display": True}
    ):
        """
        Levin and Nave's TRG; it is general.
        Do it twice, so the rescaling factor is b=2
        """
        if self.iter_n == 0:
            self.boundary = "parallel"
        # read parameters for TRG
        chi = pars["chi"]
        cg_eps = pars["dtol"]
        display = pars["display"]
        # use TRG to coarse-grain the tensor twice
        ten_cur = self.get_tensor()
        # store the RG errors as a list
        SPerrList = []
        if display:
            print("///////////////////////////")
        for n in range(2):
            # 1. Splitting
            t1, t2, t3, t4, err1, err2 = trg2d.splitTen(
                ten_cur, ten_cur, chi, cg_eps
            )
            SPerrList.append(err1)
            SPerrList.append(err2)
            if display:
                print("TRG splitting...")
                print("err1 = {:.2e}".format(err1))
                print("err2 = {:.2e}".format(err2))
            # 2. Contraction
            ten_cur = trg2d.contr4pieces(t1, t2, t3, t4)
            if display:
                print("  TRG-{:d} done!".format(n+1))
                print("------")
        if display:
            scur = self.singlular_spectrum()
            print("The singular value spectrum of A is:")
            print(scur[:20])
            print("===========================")

        # update the current tensor
        self.current_tensor = ten_cur * 1.0
        # pull out the tensor norm and save
        ten_mag = self.pullout_magnitude()
        self.save_tensor_magnitude(ten_mag)

        # there is no entanglement filtering
        lrerr = 0
        return lrerr, SPerrList

    def bkten_rot(
        self,
        pars={"chi": 4, "dtol": 1e-16, "display": True}
    ):
        """
        Entanglement Filtering RG that preseves two lattice symmetries
        - reflection
        - rotation
        Furthermore, the RG map imposes the rotational symmetry
        """
        if self.iter_n == 0:
            # In this scheme, a pair of two isometries
            # has opposite arrow for input legs
            self.init_dw()
            self.boundary = "anti-parallel"

        # read parameters for the block-tensor part
        chi = pars["chi"]
        cg_eps = pars["dtol"]
        display = pars["display"]

        # Do the tensor RG map
        ten_cur = self.get_tensor()
        if display:
            print("///////////////////////////")
        # Step 1. Determine isometric tensors
        p, g, err, eigv = block_rotsym.findProj(
            ten_cur, chi, cg_eps=cg_eps
        )
        # Step 2. Contraction of the 2x2 block
        Aout = block_rotsym.block4ten(
            ten_cur, p
        )
        # print out info
        if display:
            print("The block-tensor map errors are")
            print("  - Error (out) = {:.2e}".format(err))
            print("The singular value spectrum of A is:")
            scur = self.singlular_spectrum()
            print(scur[:20])
            # Check the symmetry of the coarse-grained tensor
            print("Check symmetry of the coarse-grained tensor:")
            print("  - Reflection: {}".format(block_rotsym.isReflSym(Aout, g)))
            print("  - Rotation  : {}".format(block_rotsym.isRotSym(Aout, g)))
            print("===========================")
        # update the current tensor
        self.current_tensor = Aout * 1.0
        # pull out the tensor norm and save
        ten_mag = self.pullout_magnitude()
        self.save_tensor_magnitude(ten_mag)
        # save isometric tensors and SWAP signs
        self.isometry_applied = [p, p.conj()]  # x and y directions
        self.gSWAP = g * 1.0
        # return errors
        lrerr = 0    # no entanglement filtering error
        SPerrs = [err, err]
        return lrerr, SPerrs

    def efrg(
        self,
        pars={"chi": 6, "dtol": 1e-16, "display": True,
              "chis": 4, "chienv": 16, "epsilon": 1e-8}
    ):
        if self.iter_n == 0:
            # In this scheme, a pair of two isometries
            # has opposite arrow for input legs
            self.init_dw()
            self.boundary = "anti-parallel"
        # read parameters
        chi = pars["chi"]
        cg_eps = pars["dtol"]
        display = pars["display"]
        chis = pars["chis"]
        chienv = pars["chienv"]
        epsilon = pars["epsilon"]

        # read the input tensor
        Ain = self.get_tensor()

        # I. Entanglement filtering (EF)
        # I.1 Initialization of the filtering matrix
        (
            s, Lr, Upsilon0, dbA
        ) = fet2d_rotsym.init_s(
            Ain, chis, chienv, epsilon
        )
        if display:
            print("Shape of s is {}.".format(s.shape))
        # compute <ψ|ψ> for calculating fidelity
        PsiPsi = ncon([Upsilon0], [[1, 1, 2, 2]])
        # Fidelity (or error) of the EF
        errEF0 = fet2d_rotsym.fidelity(
            dbA, s, PsiPsi)[1]
        # I.2 Update the s matrix to minimize the EF error
        s, errsEF = fet2d_rotsym.opt_s(
            dbA, s, PsiPsi, epsilon=cg_eps,
            iter_max=200, display=True
        )
        # Fidelity (or error) of the EF after optimization
        errEF1, PhiPhi1 = fet2d_rotsym.fidelity(
            dbA, s, PsiPsi)[1:]
        # print out info of the EF
        if display:
            print("  Initial EF error is {:.3e}".format(errEF0))
            print("    Final EF error is {:.3e}".format(errEF1))
        # I.3 Take care the overall magnitude of s to
        # make sure that <ψ|ψ> = <φ|φ>
        PsiDivPhi = (PsiPsi / PhiPhi1).norm()
        s = s * (PsiDivPhi)**(1/16)
        # I.4 Act the filtering matrix s on the main tensor:
        # Ain -- > As
        As = fet2d_rotsym.absorb(Ain, s)

        # II. Block-tensor map
        # II.1 Determine the isometric tensors
        p, g, err, eigv = block_rotsym.findProj(
            As, chi, cg_eps=cg_eps
        )
        # II.2 Contraction of the 2x2 block
        Aout = block_rotsym.block4ten(
            As, p
        )
        # print out info of the block-tensor map
        if display:
            print("The block-tensor map error is")
            print("  - Error (out) = {:.2e}".format(err))
            print("The singular value spectrum of A is:")
            scur = self.singlular_spectrum()
            print(scur[:20])
            # Check the symmetry of the coarse-grained tensor
            print("Check symmetry of the coarse-grained tensor:")
            print("  - Reflection: {}".format(block_rotsym.isReflSym(Aout, g)))
            print("  - Rotation  : {}".format(block_rotsym.isRotSym(Aout, g)))
            print("===========================")

        # III. update the current tensor
        self.current_tensor = Aout * 1.0
        # pull out the tensor norm and save
        ten_mag = self.pullout_magnitude()
        self.save_tensor_magnitude(ten_mag)
        # save isometric tensors and SWAP signs
        self.isometry_applied = [p, p.conj()]  # x and y directions
        self.gSWAP = g * 1.0
        # return errors
        lrerr = [errEF0, errEF1]
        SPerrs = [err, err]

        return lrerr, SPerrs

    def rgmap(self, tnrg_pars,
              scheme="fet-hotrg", ver="base"):
        """
        coarse grain the tensors using schemes above
        - block tensor
        - trg
        - tnr
        - fet-hotrg
        - hotrg

        Return two kinds of local replacement errors
        1) loop-filtering process
        2) coarse graining process
        """
        if scheme == "fet-hotrg":
            if ver == "base":
                (d_debug,
                 lferrs,
                 SPerrs
                 ) = self.fet_hotrg(tnrg_pars,
                                    init_stable=False)
        elif scheme == "hotrg":
            if ver == "base":
                # This HOTRG exploits the lattice reflection symmetry
                (lferrs,
                 SPerrs
                 ) = self.hotrg(tnrg_pars)
            elif ver == "general":
                # This is the general HOTRG
                (lferrs,
                 SPerrs
                 ) = self.hotrg_grl(tnrg_pars)
        elif scheme == "tnr":
            if ver == "base":
                (lferrs,
                 SPerrs
                 ) = self.tnr(tnrg_pars)
        elif scheme == "trg":
            if ver == "general":
                (lferrs,
                 SPerrs
                 ) = self.trg_grl(tnrg_pars)
        elif scheme == "block":
            if ver == "rotsym":
                (lferrs,
                 SPerrs
                 ) = self.bkten_rot(tnrg_pars)
        elif scheme == "efrg":
            if ver == "rotsym":
                (lferrs,
                 SPerrs
                 ) = self.efrg(tnrg_pars)
        return lferrs, SPerrs

    def init_dw(self):
        ten_cur = self.get_tensor()
        if type(ten_cur) is Tensor:
            # for ordinary tensor
            self.d_w = None
        else:
            # for AbelianTensor
            vlegshape = ten_cur.shape[1]
            d_w = ten_cur.eye(vlegshape).diag()
            num = len(d_w)
            for charge in d_w.qhape[0]:
                chargeSect = []
                for k in range(len(d_w[(charge,)])):
                    chargeSect.append(num)
                    num = num - 1
                d_w[(charge,)] = np.array(chargeSect)
            self.d_w = d_w * 1.0

    @staticmethod
    def truncate_hosvd(ten, dtol=1e-8):
        """
        truncate a la hosvd
        """
        proj_x = ten.svd([0], [1, 2, 3], eps=dtol)[0]
        proj_y = ten.svd([1], [2, 3, 0], eps=dtol)[0]
        ten_new = ncon([ten, proj_x.conjugate(), proj_x,
                        proj_y.conjugate(), proj_y],
                       [[1, 2, 3, 4], [1, -1], [3, -3],
                        [2, -2], [4, -4]])
        return ten_new

    def gu_wen_cardy(self, aspect_ratio=1, num_scale=12):
        """
        Extract the central charge and scaling dimensions
        a la Gu, Wen and Cardy
        """
        # construct the transfer matrix
        contract_list = [[k + 1, -(aspect_ratio + k + 1),
                         (k + 1) % aspect_ratio + 1, -(k + 1)
                          ] for k in range(aspect_ratio)
                         ]
        ten_cur = self.get_tensor()
        ten_mag = self.get_tensor_magnitude()[-1]
        ten_inv = ten_mag**(-1/3) * ten_cur
        transfer_mat = ncon([ten_inv]*aspect_ratio,
                            contract_list)
        in_leg = [k for k in range(aspect_ratio)]
        out_leg = [k + aspect_ratio for k in range(aspect_ratio)]
        eig_val = transfer_mat.eig(in_leg, out_leg,
                                   sparse=True, chis=num_scale)[0]
        eig_val = eig_val.to_ndarray()
        eig_val = np.abs(eig_val)
        eig_val = -np.sort(-eig_val)
        central_charge = np.log(eig_val[0]) * 6 / np.pi * aspect_ratio
        scaling_dimensions = -np.log(eig_val/eig_val[0])/(2*np.pi)*aspect_ratio
        return central_charge, scaling_dimensions

    def eval_free_energy(self, initial_spin=2, b=2):
        """
        See my jupyter notebook on evently's trg
        for how to calculate the free energy

        initial_spin: int
            number of spins that the initial tensor corresponds to
        b: int
            ratio between the coarse lattice length and the length
            before RG.
            A coarser tensor corresponds to b^2 tensors before a RG step.
        """
        messg = "iter_n should be length of the tensor magnitute list plus 1"
        assert len(self.tensor_magnitude) == (self.iter_n + 1), messg
        # calculate free energy divided by the temperature
        ten_cur = self.get_tensor()
        ten_mag_arr = np.array(self.tensor_magnitude)
        weight = (
            (1 / initial_spin) *
            (1 / b**2)**np.array(range(0, self.iter_n + 1))
                  )
        # all contributions from the tensor magnitute
        g = (weight * np.log(ten_mag_arr)).sum()
        # the contribution from the tracing off the final normalized tensor
        # determine the proper gauge
        if self.boundary == "anti-parallel":
            # we need to take care of
            # the gauge matrices on bonds
            v, w = self.isometry_applied.copy()
            Hgauge = ncon([v, v], [[1, 2, -1], [2, 1, -2]])
            Vgauge = ncon([w, w], [[1, 2, -1], [2, 1, -2]])
            g += (
                (1 / (initial_spin * b**(2 * self.iter_n))) *
                np.log(ncon([ten_cur, Hgauge, Vgauge],
                            [[1, 3, 2, 4], [1, 2], [3, 4]]))
            )
        elif self.boundary == "parallel":
            # the gauge matrices on bonds is trivial
            g += (
                (1 / (initial_spin * b**(2 * self.iter_n))) *
                np.log(ncon([ten_cur], [[1, 2, 1, 2]]))
            )
        return g

    def singlular_spectrum(self):
        ten_cur = self.get_tensor()
        s = ten_cur.svd([0, 1], [2, 3])[1]
        s = s / s.max()
        s = s.to_ndarray()
        s = np.abs(s)
        s = -np.sort(-s)
        return s

    def ishermitian_tm(self):
        # check whether the transfer matrix is hermitian or not
        # If true, we gather an evidence that the individual tensor
        # has refelction symmetry
        ten_cur = self.get_tensor()
        if self.boundary == "parallel":
            tm_v = ncon([ten_cur], [[1, -1, 1, -2]])
            tm_h = ncon([ten_cur], [[-1, 1, -2, 1]])
        vsym = tm_v.allclose(
            tm_v.transpose().conj()
        )
        hsym = tm_h.allclose(
            tm_h.transpose().conj()
        )
        return vsym, hsym

    def generate_tm(self, direction="y"):
        # generate transfer matrix in x or y direction
        assert direction in ["x", "y"]
        ten_cur = self.get_tensor()
        if self.boundary == "anti-parallel":
            # we need to take care of
            # the gauge matrices on bonds
            v, w = self.isometry_applied.copy()
            Hgauge = ncon([v, v], [[1, 2, -1], [2, 1, -2]])
            Vgauge = ncon([w, w], [[1, 2, -1], [2, 1, -2]])
            ten_cur = ncon([ten_cur, Hgauge, Vgauge],
                           [[1, 2, -3, -4], [1, -1], [2, -2]])
        # construct the transfer matrix
        if direction == "x":
            tm = ncon([ten_cur], [[-1, 1, -2, 1]])
        elif direction == "y":
            tm = ncon([ten_cur], [[1, -1, 1, -2]])
        else:
            raise ValueError("The direction should be among x or y")
        return tm

    def degIndX(self, direction="y"):
        tm = self.generate_tm(direction=direction)
        trsq = (tm.trace()).norm()**2
        sqtr = (ncon([tm, tm], [[-1, 1], [1, -2]])).trace().norm()
        return trsq / sqtr


class TensorNetworkRG3D(TensorNetworkRG):
    """
    TNRG for 3D cubic lattice
    Tensor leg order convention is
    A[x, x', y, y', z, z']
         z  x'
         | /
     y'--A--y
       / |
     x   z'
    """
    # Play with the "spherical" transfer matrix in radial quantization
    def two_tensors_sphere_tm(self):
        """
        Construct a transfer matrix using sphere
        foliation by making a "pillow case" from
        two tensors
        """
        ten_cur = self.get_tensor()
        sphere_tm = ncon([ten_cur, ten_cur.conj()],
                         [[1, 2, 3, 4, -3, -1],
                          [1, 2, 3, 4, -4, -2]])
        return sphere_tm

    def six_tensors_sphere_tm(self):
        """
        Construct a transfer matrix using sphere
        foliation by making a box from six tensors
        """
        ten_cur = self.get_tensor()
        sphere_tm = ncon([ten_cur, ten_cur, ten_cur,
                          ten_cur.conj(), ten_cur.conj(), ten_cur.conj()],
                         [[1, 8, 2, 7, -7, -1], [9, 1, 3, 11, -8, -2],
                         [10, 2, 12, 3, -9, -3], [9, 5, 10, 4, -10, -4],
                         [5, 8, 12, 6, -11, -5], [4, 7, 6, 11, -12, -6]]
                         )
        return sphere_tm

    def generate_transfer_matrix(self, number_tensors=2):
        if number_tensors == 2:
            self.sphere_tm = self.two_tensors_sphere_tm()
        elif number_tensors == 6:
            self.sphere_tm = self.six_tensors_sphere_tm()
        else:
            raise ValueError("The number_tensors can only be 2 or 6.")

    def get_transfer_matrix(self):
        return self.sphere_tm * 1.0

    @staticmethod
    def spin_scaling_dimension():
        """
        The best-known scaling dimension for the spin operator
        """
        return 0.5181489

    def tm2x(self, number_tensors=2, sparse=False, num_state=60,
             fixed_field="spin"):
        """
        From the eigenvalues of the transfer matrix
        to scaling dimensions
        """
        self.generate_transfer_matrix(number_tensors=number_tensors)
        sph_tm = self.get_transfer_matrix()
        if number_tensors == 2:
            eig_val, eig_vec = sph_tm.eig([0, 1], [2, 3],
                                          sparse=sparse,
                                          chis=num_state)
        elif number_tensors == 6:
            eig_val, eig_vec = sph_tm.eig([0, 1, 2, 3, 4, 5],
                                          [6, 7, 8, 9, 10, 11],
                                          sparse=sparse,
                                          chis=num_state)
        else:
            raise ValueError("The number_tensors can only be 2 or 6.")
        # this is only for the numerical precision problem
        eig_val = np.abs(eig_val)
        if type(sph_tm) is TensorZ2:
            # normalize the largest to 1 and take -log() transformation
            log_ratio_even = -np.log(eig_val[(0,)] / eig_val[(0,)][0])
            log_ratio_odd = -np.log(eig_val[(1,)] / eig_val[(0,)][0])
            # determine the proportional constant
            if fixed_field == "spin":
                x_even = log_ratio_even / (
                    log_ratio_odd[0]) * self.spin_scaling_dimension()
                x_odd = log_ratio_odd / (
                    log_ratio_odd[0]) * self.spin_scaling_dimension()
            elif fixed_field == "stress-tensor":
                x_even = log_ratio_even / (
                    log_ratio_even[5]) * 3
                x_odd = log_ratio_odd / (
                    log_ratio_even[5]) * 3
            else:
                raise ValueError("fix_field is either spin or stress-tensor.")
            log_ratio = [log_ratio_even, log_ratio_odd]
            x = [x_even, x_odd]
        else:
            # normalize the largest to 1 and take -log() transformation
            log_ratio = -np.log(eig_val / eig_val[0])
            if fixed_field == "spin":
                x = log_ratio / (
                  log_ratio[1]) * self.spin_scaling_dimension()
            else:
                raise ValueError("fix_field should only be spin.")

        return x, log_ratio

    # Method for extracting various properties of the 6-leg tensor
    def generate_tm(self, direction="x"):
        # generate transfer matrix in x, y, or z direction
        ten_cur = self.get_tensor()
        if direction == "x":
            tm = ncon([ten_cur], [[-1, -2, 1, 1, 2, 2]])
        elif direction == "y":
            tm = ncon([ten_cur], [[1, 1, -1, -2, 2, 2]])
        elif direction == "z":
            tm = ncon([ten_cur], [[1, 1, 2, 2, -1, -2]])
        else:
            raise ValueError("The direction should be among x, y, or z")
        return tm

    def ishermitian_tm(self):
        # check whether the transfer matrix is hermitian or not
        # If true, we gather an evidence that the individual tensor
        # has refelction symmetry
        tm_x = self.generate_tm(direction="x")
        tm_y = self.generate_tm(direction="y")
        tm_z = self.generate_tm(direction="z")
        xsym = tm_x.allclose(
            tm_x.transpose().conj()
        )
        ysym = tm_y.allclose(
            tm_y.transpose().conj()
        )
        zsym = tm_z.allclose(
            tm_z.transpose().conj()
        )
        return xsym, ysym, zsym

    def degIndX(self, direction="z"):
        tm = self.generate_tm(direction=direction)
        trsq = (tm.trace()).norm()**2
        sqtr = (ncon([tm, tm], [[-1, 1], [1, -2]])).trace().norm()
        return trsq / sqtr

    def entangle(self, leg="x"):
        ten_cur = self.get_tensor()
        if leg == "xyz":
            s = ten_cur.svd([0, 2, 4], [1, 3, 5])[1]
        elif leg == "x":
            s = ten_cur.svd([0], [1, 2, 3, 4, 5])[1]
        elif leg == "y":
            s = ten_cur.svd([2], [0, 1, 3, 4, 5])[1]
        elif leg == "z":
            s = ten_cur.svd([4], [0, 1, 2, 3, 5])[1]
        else:
            errMsg = (
                "leg should be in the set [x, y, z, xyz]"
            )
            raise ValueError(errMsg)
        # maximal s normalized to 1 and sort
        s = s/s.max()
        s = s.to_ndarray()
        s = np.abs(s)
        s = -np.sort(-s)
        # calculate entanglement entropy
        pk = s**2
        # the following method will take care of the normalization
        ee = spstats.entropy(pk, base=2)
        return ee, s

    # I. coarse graining methods
    def hotrg(
        self,
        pars={"chi": 4, "cg_eps": 1e-16,
              "display": True},
        signFix=False,
        comm=None
    ):
        if self.iter_n == 0:
            self.boundary = "parallel"
        # read hotrg parameters
        chi = pars["chi"]
        cg_eps = pars["cg_eps"]
        display = pars["display"]
        # use hotrg to coarse graining
        Aold = self.get_tensor()
        Aout = Aold * 1.0
        # order of corase graining for three directions
        cg_dirs = ["z", "y", "x"]
        # a dictionary to save all isometric tensors in 3 directions
        isom3dir = {}
        SPerrList = []
        if display:
            print("--------------------")
            print("--------------------")
        for direction in cg_dirs:
            (Aout,
             isometries,
             errs
             ) = hotrg3d.dirHOTRG(Aout, chi, direction,
                                  cg_eps=cg_eps,
                                  comm=comm)
            isom3dir[direction] = isometries
            SPerrList.append(errs)
            if display:
                # dictionary for display of information
                pname = {"z": ["x", "y"], "y": ["z", "x"], "x": ["y", "z"]}
                print("This is tensor blocking in {:s} direction".format(
                    direction)
                      )
                print("Truncation in {:s} legs is {:.2e}".format(
                    pname[direction][0], errs[0])
                      )
                print("Truncation in {:s} legs is {:.2e}".format(
                    pname[direction][1], errs[1])
                      )
                print("----------")

        # (22 Feb. 2023) Sign fixing:
        if signFix:
            (
                Aout,
                isom3dir
            ) = hotrg3d.signFix(Aout, isom3dir,
                                Aold, cg_dirs,
                                verbose=display)

        # update current isometries
        self.isometry_applied = isom3dir.copy()
        # update the current tensor
        self.current_tensor = Aout * 1.0
        # pull out the tensor norm and save
        ten_mag = self.pullout_magnitude()
        self.save_tensor_magnitude(ten_mag)

        # return approximation errors
        # no loop-filtering error
        lrerr = 0
        return lrerr, SPerrList

    def block_hotrg(
        self,
        pars={"chi": 4, "chiM": 4, "chiI": 4, "chiII": 4,
              "cg_eps": 1e-16, "display": True},
        signFix=False,
        comm=None
    ):
        """
        HOTRG-like implementation of 3D block-tensor RG.
        It interpolates between the full block-tensor RG
        and the usual HOTRG by adjust bond dimensions:
        - Usual HOTRG:
            χ = χM = χI = χII
        - Full block-tensor:
            χM = χI = χ^2
            χII = χ^4
        Besides, the reflection symmetry is explicitly imposed
        in such a way to suit cube-entanglement filtering
        """
        if self.iter_n == 0:
            self.boundary = "parallel"
        # 0. read parameters
        chi = pars["chi"]
        chiM = pars["chiM"]
        chiI = pars["chiI"]
        chiII = pars["chiII"]
        cg_eps = pars["cg_eps"]
        display = pars["display"]
        # I. coarse graining
        Aold = self.get_tensor()
        (
            Aout, pox, poy, poz,
            pmx, pmy, pmz,
            pix, piy, piix,
            xerrs, yerrs, zerrs
         ) = bkten3d.blockrg(
            Aold, chi, chiM, chiI, chiII,
            cg_eps, display, comm=comm
        )

        # II. Sign fixing:
        if signFix:
            (
                Aout, pox, poy, poz
            ) = bkten3d.signFix(Aout, Aold,
                                pox, poy, poz,
                                verbose=display)

        # update the isometric tensors for block-tensor RG
        self.isometry_applied = [pox, poy, poz,
                                 pmx, pmy, pmz,
                                 pix, piy, piix]
        # update the current tensor
        self.current_tensor = Aout * 1.0
        # pull out the tensor norm and save
        ten_mag = self.pullout_magnitude()
        self.save_tensor_magnitude(ten_mag)

        # return approximation errors
        # no loop-filtering error
        lrerr = 0
        rgerr = [zerrs, yerrs, xerrs]
        return lrerr, rgerr

    def entfree_blockrg(
        self,
        pars={"chi": 6, "chiM": 6, "chiI": 15, "chiII": 36,
              "cg_eps": 1e-16, "display": True,
              "chis": 4, "chienv": 25, "epsilon": 1e-5},
        signFix=False,
        comm=None,
        chiSet=None
            ):
        if self.iter_n == 0:
            self.boundary = "parallel"
        # 0.1 read parameters for block-tensor
        chi = pars["chi"]
        chiM = pars["chiM"]
        chiI = pars["chiI"]
        chiII = pars["chiII"]
        cg_eps = pars["cg_eps"]
        display = pars["display"]
        # 0.2 read parameters for entanglement filtering
        chis = pars["chis"]
        chienv = pars["chienv"]
        init_epsilon = pars["epsilon"]
        # I. Entanglement filtering
        # I.1 Find initial sx, sy, sz matrices
        Aold = self.get_tensor()
        Aout = Aold * 1.0
        (
            sx, sy, sz,
            Lrx, Lry, Lrz, Gammay
        ) = fet3dcube.init_alls(Aout, chis, chienv, init_epsilon,
                                comm=comm)
        if display:
            print("Shape of initial sx is {}.".format(sx.shape))
            print("Shape of initial sy is {}.".format(sy.shape))
            print("Shape of initial sz is {}.".format(sz.shape))
        # compute <ψ|ψ> for calculating fidelity
        PsiPsi = ncon([Gammay], [[1, 1, 2, 2]])
        # FET fidelity of inserting initial s matrices
        errFET0 = fet3dcube.fidelity(Aout, sx, sy, sz, PsiPsi,
                                     comm=comm)[1]
        # I.2 Optimization of sx, sy, sz matrices
        (
            sx, sy, sz, fetErrList
         ) = fet3dcube.optimize_alls(
             Aout, sx, sy, sz, PsiPsi, epsilon=cg_eps,
             iter_max=5, n_round=40, display=display,
             comm=comm
         )
        # FET fidelity after optimization of s matrices
        (errFET1, PhiPhi1) = fet3dcube.fidelity(
            Aout, sx, sy, sz, PsiPsi, comm=comm)[1:]
        if display:
            print("  Initial FET error for insertion of",
                  "s matrices is {:.3e}".format(errFET0))
            print("    Final FET error for insertion of",
                  "s matrices is {:.3e}".format(errFET1))
        # II. coarse graining
        # II. Take care the overall magnitude of sx, sy, sz to
        # make sure that <ψ|ψ> = <φ|φ>
        PsiDivPhi = (PsiPsi / PhiPhi1).norm()
        sx = sx * (PsiDivPhi)**(1/48)
        sy = sy * (PsiDivPhi)**(1/48)
        sz = sz * (PsiDivPhi)**(1/48)
        # II.1 Absorb sx, sy, sz to (+++)-position tensor
        Aout = fet3d.absbs(Aout, sx, sy, sz)
        # II.2 Apply block-tensor transformation
        (
            Aout, pox, poy, poz,
            pmx, pmy, pmz,
            pix, piy, piix,
            xerrs, yerrs, zerrs
         ) = bkten3d.blockrg(
            Aout, chi, chiM, chiI, chiII,
            cg_eps, display,
             chiSet=chiSet
        )

        # III. Sign fixing:
        if signFix:
            (
                Aout, pox, poy, poz
            ) = bkten3d.signFix(Aout, Aold,
                                pox, poy, poz,
                                verbose=display)

        # update the isometric tensors for block-tensor RG
        self.isometry_applied = [pox, poy, poz,
                                 pmx, pmy, pmz,
                                 pix, piy, piix,
                                 sx, sy, sz]
        # update the current tensor
        self.current_tensor = Aout * 1.0
        # pull out the tensor norm and save
        ten_mag = self.pullout_magnitude()
        self.save_tensor_magnitude(ten_mag)

        # return approximation errors
        # no loop-filtering error
        lrerr = [errFET0, errFET1]
        rgerr = [zerrs, yerrs, xerrs]
        return lrerr, rgerr

    def ef2stp_blockrg(
        self,
        pars={"chi": 6, "chiM": 6, "chiI": 15, "chiII": 36,
              "cg_eps": 1e-16, "display": True,
              "chis": 4, "chienv": 25, "epsilon": 1e-5,
              "chiMs": 4, "chiMenv": 25, "epsilonM": 1e-5,
              "cubeFilter": True, "loopFilter": True,
              "cubeYZmore": False},
        signFix=False,
        comm=None,
        chiSet=None
            ):
        if self.iter_n == 0:
            self.boundary = "parallel"
        # 0.1 load the current tensor
        Aold = self.get_tensor()
        Aout = Aold * 1.0
        # 0.2 read parameters for block-tensor
        chi = pars["chi"]
        chiM = pars["chiM"]
        chiI = pars["chiI"]
        chiII = pars["chiII"]
        cg_eps = pars["cg_eps"]
        display = pars["display"]
        # 0.3 Switch for two filtering steps
        cubeFilter = pars.get("cubeFilter", True)
        loopFilter = pars.get("loopFilter", True)
        cubeYZmore = pars.get("cubeYZmore", 0)
        XloopF = pars.get("XloopF", False)

        if display:
            print("====================")
            print("Start {:d} RG step...".format(self.iter_n+1))

        # -----~~~>> Entanglement Filtering
        # (E.0) x-loop filtering on input tensor `A`
        if loopFilter and XloopF:
            if display:
                print("  >>~~~~X-Loop-Filter~~~~~~>>")
            # (E.0)S0: Set parameters for X-loop filtering
            # chiXs = int(np.ceil((pars["chi"] + 2 * (pars["chis"] - cubeYZmore)) / 3))
            chiXs = int(np.ceil((pars["chi"] + pars["chis"] - cubeYZmore) / 2))
            # chiXs = pars["chiMs"]
            chiXenv = chiXs**2
            epsilonX = pars["epsilon"]

            # (E.0)S1: Initialize mXy, mXz matrices (a GILT-like method)
            if display:
                timing0 = datetime.now()
            (
                mXy, mXz, mXLry, mXLrz, GammaLPXz
            ) = fet3dloop.init_xloopm(Aout, chiXs, chiXenv, epsilonX)
            if display:
                timing1 = datetime.now()
                diffT = relativedelta(timing1, timing0)
                print("--> X-Loop-filter initialization takes",
                      "{} minutes {:.3f} seconds <--".format(
                          diffT.minutes,
                          diffT.seconds + diffT.microseconds*1e-6
                      ))
                print("Shape of initial mXy is {}.".format(mXy.shape),
                      "(Qhape is {}).".format(mXy.qhape))
                print("Shape of initial mXz is {}.".format(mXz.shape),
                      "(Qhape is {}).".format(mXz.qhape))

            # (E.0)S2: Optimize mXy, mXz matrices using FET
            # - Compute <ψ|ψ> for calculating initialization fidelity
            PsiPsiLPX = ncon([GammaLPXz], [[1, 1, 2, 2]])
            # - FET fidelity of inserting initial s matrices
            err0LPX = fet3dloop.fidelityLPX(Aout, mXy, mXz, PsiPsiLPX)[1]
            # - Optimization of mXy, mXz matrices
            if display:
                timing0 = datetime.now()
            (
                mXy, mXz, ErrListLPX
            ) = fet3dloop.optimize_xloop(
                Aout, mXy, mXz, PsiPsiLPX, epsilon=cg_eps,
                iter_max=5, n_round=40, display=display
            )
            # - FET fidelity after optimization of mx, my matrices
            (err1LPX, PhiPhiLPX) = fet3dloop.fidelityLPX(
                Aout, mXy, mXz, PsiPsiLPX)[1:]
            if display:
                timing1 = datetime.now()
                diffT = relativedelta(timing1, timing0)
                print("--> Each X-Loop-FET iteration takes",
                      "{:.3f} seconds <--".format(
                          (diffT.minutes*60 +
                           diffT.seconds +
                           diffT.microseconds*1e-6) / (
                               len(flatten(ErrListLPX))
                           )
                      ))
                print("--> Total wall time is",
                      "{} minutes {:.3f} seconds <--".format(
                          diffT.minutes,
                          diffT.seconds + diffT.microseconds*1e-6
                      ))
                print("  Initial FET error for insertion of",
                      "mXy, mXz matrices is {:.3e}".format(err0LPX))
                print("    Final FET error for insertion of",
                      "mXy, mXz matrices is {:.3e}".format(err1LPX),
                      "(after {:d} rounds)".format(
                          int(len(ErrListLPX)/2)
                      ))

            # (E.0)S3: Absorb mXy, mXz into the tensor A
            # - Take care the overall magnitude of m matrices to
            # make sure that <ψ|ψ> = <φ|φ>.
            # - This magnitude contributes to the total free energy
            # but is not essential for the conformal data.
            PsiDivPhi = (PsiPsiLPX / PhiPhiLPX).norm()
            # - Factor 16 since there are 16 m matrices in <φ|φ>
            mXy = mXy * (PsiDivPhi)**(1/16)
            mXz = mXz * (PsiDivPhi)**(1/16)
            # - Absorb mXy, mXz to two outer legs of
            # (++)-position tensor A
            # so the two legs are squeezed due to filtering
            Aout = fet3dloop.absb_mloopx(Aout, mXy, mXz)
            if display:
                print("  <<~~~~~~~~~~<<")
        else:
            err0LPX, err1LPX = [0 for k in range(2)]
        # -----~~~<<

        # (E.1) cube-filtering on input tensor `A`
        # Approximate 8-copies of `A` forming a cube
        # determine sx, sy, sz matrices and apply them
        # on the 3 outer legs of the input tensor `A`
        if cubeFilter:
            if display:
                print("  >>~~~~Cube-Filter~~~~~~>>")
            # (E.1)S0: Read parameters for cube-filtering
            chis = pars["chis"]
            chienv = pars["chienv"]
            epsilonCube = pars["epsilon"]
            # (E.1)S1: Initialize sx, sy, sz matrices (a GILT-like method)
            if display:
                timing0 = datetime.now()
            (
                sx, sy, sz,
                Lrx, Lry, Lrz, Gammay
            ) = fet3dcube.init_alls(Aout, chis, chienv, epsilonCube,
                                    cubeYZmore=cubeYZmore, comm=comm)
            if display:
                timing1 = datetime.now()
                diffT = relativedelta(timing1, timing0)
                print("--> Cube-filter initialization takes",
                      "{} minutes {:.3f} seconds <--".format(
                          diffT.minutes,
                          diffT.seconds + diffT.microseconds*1e-6
                      ))
                print("Shape of initial sx is {}".format(sx.shape),
                      "(Qhape is {}).".format(sx.qhape))
                print("Shape of initial sy is {}".format(sy.shape),
                      "(Qhape is {}).".format(sy.qhape))
                print("Shape of initial sz is {}".format(sz.shape),
                      "(Qhape is {}).".format(sz.qhape))

            # (E.1)S2: Optimize sx, sy, sz matrices using FET
            # - Compute <ψ|ψ> for calculating initialization fidelity
            PsiPsiCube = ncon([Gammay], [[1, 1, 2, 2]])
            # - FET fidelity of inserting initial s matrices
            err0Cube = fet3dcube.fidelity(Aout, sx, sy, sz, PsiPsiCube,
                                          comm=comm)[1]
            # - Optimization of sx, sy, sz matrices
            if display:
                timing0 = datetime.now()
            (
                sx, sy, sz, cubeErrList
             ) = fet3dcube.optimize_alls(
                 Aout, sx, sy, sz, PsiPsiCube, epsilon=cg_eps,
                 iter_max=5, n_round=40, display=display,
                 comm=comm
             )
            # - FET fidelity after optimization of s matrices
            (err1Cube, PhiPhi1) = fet3dcube.fidelity(
                Aout, sx, sy, sz, PsiPsiCube, comm=comm)[1:]
            if display:
                timing1 = datetime.now()
                diffT = relativedelta(timing1, timing0)
                print("--> Each cube-FET iteration takes",
                      "{:.3f} seconds <--".format(
                          (diffT.minutes*60 +
                           diffT.seconds +
                           diffT.microseconds*1e-6) / (
                               len(flatten(cubeErrList))
                           )
                      ))
                print("--> Total wall time is",
                      "{} minutes {:.3f} seconds <--".format(
                          diffT.minutes,
                          diffT.seconds + diffT.microseconds*1e-6
                      ))
                print("  Initial FET error for insertion of",
                      "s matrices is {:.3e}".format(err0Cube))
                print("    Final FET error for insertion of",
                      "s matrices is {:.3e}".format(err1Cube),
                      "(after {:d} rounds)".format(
                          int(
                              len(cubeErrList) / 3
                          )
                      ))

            # (E.1)S3: Absorb sx, sy, sz into the initial tensor A
            # - Take care the overall magnitude of sx, sy, sz to
            # make sure that <ψ|ψ> = <φ|φ>.
            # - This magnitude contributes to the total free energy
            # but is not essential for the conformal data.
            PsiDivPhi = (PsiPsiCube / PhiPhi1).norm()
            # - Factor 48 since there are 48 s matrices in <φ|φ>
            sx = sx * (PsiDivPhi)**(1/48)
            sy = sy * (PsiDivPhi)**(1/48)
            sz = sz * (PsiDivPhi)**(1/48)
            # - Absorb sx, sy, sz to (+++)-position tensor
            # so its three outer legs are squeezed due to filtering
            Aout = fet3d.absbs(Aout, sx, sy, sz)
            if display:
                print("  <<~~~~~~~~~~<<")
        else:
            sx, sy, sz = [None for k in range(3)]
            err0Cube, err1Cube = [0 for k in range(2)]
        # -----~~~<<

        # -----\
        if display:
            timing0 = datetime.now()
        # (C.1) z-direction coarse graining
        # (C.1)S1: Determine 2-to-1 isometric tensors
        zpjs, zerrs, zds = bkten3d.zfindp(
            Aout, chiM, chiI, cg_eps=cg_eps
        )
        pmx, pix, pmy, piy = zpjs
        # (C.1)S2: Collapse two `A` tensor using isometric tensors
        Aout = bkten3d.zblock(
            Aout, pmx.conj(), pmy.conj(), pix, piy,
            comm=comm
        )
        if display:
            timing1 = datetime.now()
            diffT = relativedelta(timing1, timing0)
            print()
            print("--> z-direction(1) HOTRG takes",
                  "{} minutes {:.3f} seconds <--".format(
                      diffT.minutes,
                      diffT.seconds + diffT.microseconds*1e-6
                  ))
            print()
        # -----/

        # -----~~~>>
        # (E.2.1) loop-filtering on z-collapsed tensor `Az`
        # Approximate 4-copies of `Az` forming a plaquette (loop)
        # Determine mx, my matrices and apply them
        # on the outer x and y legs of `Az`
        if loopFilter:
            if display:
                print("  >>~~~~Z-Loop-Filter~~~~~~>>")
            # (E.2.1)S0: Read parameters for cube-filtering
            chiMs = pars["chiMs"]
            chiMenv = pars["chiMenv"]
            epsilonM = pars["epsilonM"]
            # (E.2.1)S1: Initialize mx, my matrices (a GILT-like method)
            if display:
                timing0 = datetime.now()
            (
                mx, my,
                mLrx, mLry, GammaLPZy
            ) = fet3dloop.init_zloopm(Aout, chiMs, chiMenv, epsilonM)
            if display:
                timing1 = datetime.now()
                diffT = relativedelta(timing1, timing0)
                print("--> Z-Loop-filter initialization takes",
                      "{} minutes {:.3f} seconds <--".format(
                          diffT.minutes,
                          diffT.seconds + diffT.microseconds*1e-6
                      ))
                print("Shape of initial mx is {}.".format(mx.shape),
                      "(Qhape is {}).".format(mx.qhape))
                print("Shape of initial my is {}.".format(my.shape),
                      "(Qhape is {}).".format(mx.qhape))

            # (E.2.1)S2: Optimize mx, my matrices using FET
            # - Compute <ψ|ψ> for calculating initialization fidelity
            PsiPsiLPZ = ncon([GammaLPZy], [[1, 1, 2, 2]])
            # - FET fidelity of inserting initial s matrices
            err0LPZ = fet3dloop.fidelityLPZ(Aout, mx, my, PsiPsiLPZ)[1]
            # - Optimization of mx, my matrices
            if display:
                timing0 = datetime.now()
            (
                mx, my, ErrListLPZ
            ) = fet3dloop.optimize_zloop(
                Aout, mx, my, PsiPsiLPZ, epsilon=cg_eps,
                iter_max=5, n_round=40, display=display
            )
            # - FET fidelity after optimization of mx, my matrices
            (err1LPZ, PhiPhiLPZ) = fet3dloop.fidelityLPZ(
                Aout, mx, my, PsiPsiLPZ)[1:]
            if display:
                timing1 = datetime.now()
                diffT = relativedelta(timing1, timing0)
                print("--> Each Z-Loop-FET iteration takes",
                      "{:.3f} seconds <--".format(
                          (diffT.minutes*60 +
                           diffT.seconds +
                           diffT.microseconds*1e-6) / (
                               len(flatten(ErrListLPZ))
                           )
                      ))
                print("--> Total wall time is",
                      "{} minutes {:.3f} seconds <--".format(
                          diffT.minutes,
                          diffT.seconds + diffT.microseconds*1e-6
                      ))
                print("  Initial FET error for insertion of",
                      "mx, my matrices is {:.3e}".format(err0LPZ))
                print("    Final FET error for insertion of",
                      "mx, my matrices is {:.3e}".format(err1LPZ),
                      "(after {:d} rounds)".format(
                          int(len(ErrListLPZ)/2)
                      ))

            # (E.2.1)S3: Absorb mx, my into the tensor Az
            # - Take care the overall magnitude of mx, my to
            # make sure that <ψ|ψ> = <φ|φ>.
            # - This magnitude contributes to the total free energy
            # but is not essential for the conformal data.
            PsiDivPhi = (PsiPsiLPZ / PhiPhiLPZ).norm()
            # - Factor 16 since there are 16 m matrices in <φ|φ>
            mx = mx * (PsiDivPhi)**(1/16)
            my = my * (PsiDivPhi)**(1/16)
            # - Absorb mx, my to two outer legs of
            # (++)-position tensor Az
            # so the two legs are squeezed due to filtering
            Aout = fet3dloop.absb_mloopz(Aout, mx, my)
            if display:
                print("  <<~~~~~~~~~~<<")
        else:
            mx, my = [None for k in range(2)]
            err0LPZ, err1LPZ = [0 for k in range(2)]
        # -----~~~<<

        # -----\
        if display:
            timing0 = datetime.now()
        # (C.2) y-direction coarse graining
        # (C.2) Step 1: Determine 2-to-1 isometric tensors
        ypjs, yerrs, yds = bkten3d.yfindp(
            Aout, chi, chiM, chiII, cg_eps=cg_eps, chiSet=chiSet
        )
        pmz, pox, piix = ypjs
        # (C.2) Step 2: Collapse two `A` tensor using isometric tensors
        Aout = bkten3d.yblock(
            Aout, pmz.conj(), pox.conj(), pmz, piix,
            comm=comm
        )
        if display:
            timing1 = datetime.now()
            diffT = relativedelta(timing1, timing0)
            print()
            print("--> y-direction(2) HOTRG takes",
                  "{} minutes {:.3f} seconds <--".format(
                      diffT.minutes,
                      diffT.seconds + diffT.microseconds*1e-6
                  ))
            print()
        # -----/

        # -----~~~>>
        # (E.2.2) loop-filtering on zy-collapsed tensor `Azy`
        # Approximate 4-copies of `Azy` forming a plaquette (loop)
        # Determine mz matrices and apply it
        # on both z-direction legs of `Azy`
        if loopFilter:
            if display:
                print("  >>~~~~Y-Loop-Filter~~~~~~>>")
                timing0 = datetime.now()
            # (E.2.2)S1: Initialize mz matrix (a GILT-like method)
            (
                mz, mLrz, GammaLPYz
            ) = fet3dloop.init_yloopm(Aout, chiMs, chiMenv, epsilonM)
            if display:
                timing1 = datetime.now()
                diffT = relativedelta(timing1, timing0)
                print("--> Y-Loop-filter initialization takes",
                      "{} minutes {:.3f} seconds <--".format(
                          diffT.minutes,
                          diffT.seconds + diffT.microseconds*1e-6
                      ))
                print("Shape of initial mz is {}.".format(mz.shape),
                      "(Qhape is {}).".format(mz.qhape))

            # (E.2.2)S2: Optimize mz matrix using FET
            # - Compute <ψ|ψ> for calculating initialization fidelity
            PsiPsiLPY = ncon([GammaLPYz], [[1, 1, 2, 2]])
            # - FET fidelity of inserting initial mz matrix
            err0LPY = fet3dloop.fidelityLPY(Aout, mz, PsiPsiLPY)[1]
            # - Optimization of mz matrix
            if display:
                timing0 = datetime.now()
            (
                mz, ErrListLPY
            ) = fet3dloop.optimize_yloop(
                Aout, mz, PsiPsiLPY, epsilon=cg_eps,
                iter_max=5, n_round=40, display=display
            )
            # - FET fidelity after optimization of mz matrix
            (err1LPY, PhiPhiLPY) = fet3dloop.fidelityLPY(
                Aout, mz, PsiPsiLPY)[1:]
            if display:
                timing1 = datetime.now()
                diffT = relativedelta(timing1, timing0)
                print("--> Each Y-Loop-FET iteration takes",
                      "{:.3f} seconds <--".format(
                          (diffT.minutes +
                           diffT.seconds +
                           diffT.microseconds*1e-6) / (
                               len(flatten(ErrListLPY))
                           )
                      ))
                print("--> Total wall time is",
                      "{} minutes {:.3f} seconds <--".format(
                          diffT.minutes,
                          diffT.seconds + diffT.microseconds*1e-6
                      ))
                print("  Initial FET error for insertion of",
                      "mz matrices is {:.3e}".format(err0LPY))
                print("    Final FET error for insertion of",
                      "mz matrices is {:.3e}".format(err1LPY),
                      "(after {:d} rounds)".format(
                          len(ErrListLPY)
                      ))

            # (E.2.2)S3: Absorb mz into the tensor Azy
            # - Take care the overall magnitude of mz to
            # make sure that <ψ|ψ> = <φ|φ>.
            # - This magnitude contributes to the total free energy
            # but is not essential for the conformal data.
            PsiDivPhi = (PsiPsiLPY / PhiPhiLPY).norm()
            # - Factor 8 since there are 8 mz matrices in <φ|φ>
            mz = mz * (PsiDivPhi)**(1/8)
            # - Absorb mz to two z legs of
            # (+)-position tensor Azy
            # so the two legs are squeezed due to filtering
            Aout = fet3dloop.absb_mloopy(Aout, mz)
            if display:
                print("  <<~~~~~~~~~~<<")
        else:
            mz = None
            err0LPY, err1LPY = [0 for k in range(2)]
        # -----~~~<<

        # -----\
        if display:
            timing0 = datetime.now()
        # (C.3) x-direction coarse graining
        # (C.3) Step 1: Determine 2-to-1 isometric tensors
        xpjs, xerrs, xds = bkten3d.xfindp(
            Aout, chi, cg_eps=cg_eps, chiSet=chiSet
        )
        poy, poz = xpjs
        # (C.3) Step 2: Collapse two `A` tensor using isometric tensors
        Aout = bkten3d.xblock(
            Aout, poy.conj(), poz.conj(), poy, poz,
            comm=comm
        )
        if display:
            timing1 = datetime.now()
            diffT = relativedelta(timing1, timing0)
            print()
            print("--> x-direction(3) HOTRG takes",
                  "{} minutes {:.3f} seconds <--".format(
                      diffT.minutes,
                      diffT.seconds + diffT.microseconds*1e-6
                  ))
            print()
        # -----/

        # print summary of block-tensor RG errors
        if display:
            print("Brief summary of block-tensor RG errors...")
            print("I. Outmost errors: (χ = {:d})".format(chi))
            print("x = {:.2e}, y = {:.2e}, z = {:.2e}".format(
                      yerrs[1], xerrs[0], xerrs[1]
                  ))
            print("II. Intermediate errors: (χm = {:d})".format(chiM))
            print("   Actually χmx = {} ({}), χmy = {} ({}),".format(
                pmx.shape[2], pmx.qhape[2],
                pmy.shape[2], pmy.qhape[2]
            ), "χmz = {} ({})".format(
                pmz.shape[2], pmz.qhape[2]
            ))
            print("x = {:.2e}, y = {:.2e}, z = {:.2e}".format(
                      zerrs[0], zerrs[2], yerrs[0]
                  ))
            print("III. Inner-cube errors:",
                  "(χi = {:d}, χii = {:d})".format(chiI, chiII))
            print("   Actually χix = {} ({}), χiy = {} ({}),".format(
                pix.shape[2], pix.qhape[2],
                piy.shape[2], piy.qhape[2]
            ), "χiix = {} ({})".format(
                 piix.shape[2], piix.qhape[2]
            ))
            print("xin = {:.2e}, yin = {:.2e}, xinin = {:.2e}".format(
                      zerrs[1], zerrs[3], yerrs[2]
                  ))
            print("x-direction RG spectrum is")
            xarr = -np.sort(-yds[1].to_ndarray())
            print(xarr/xarr[0])

        # (Gauge) Sign fixing:
        if signFix:
            (
                Aout, pox, poy, poz
            ) = bkten3d.signFix(Aout, Aold,
                                pox, poy, poz,
                                verbose=display)

        # update the isometric tensors for block-tensor RG
        if loopFilter and XloopF:
            # absorb X-loop filtering matrices into cube matrices
            sy = ncon([mXy, sy], [[-1, 1], [1, -2]])
            sz = ncon([mXz, sz], [[-1, 1], [1, -2]])
        self.isometry_applied = [
            pox, poy, poz, pmx, pmy, pmz, pix, piy, piix,
            sx, sy, sz, mx, my, mz
        ]
        # update the current tensor
        self.current_tensor = Aout * 1.0
        # pull out the tensor norm and save
        ten_mag = self.pullout_magnitude()
        self.save_tensor_magnitude(ten_mag)

        # return approximation errors
        # Entanglement filtering errors, including
        # - Cube filtering
        # - Loop filtering
        lrerr = [[err0Cube, err1Cube],
                 [err0LPZ, err1LPZ],
                 [err0LPY, err1LPY],
                 [err0LPX, err1LPX]]
        # block-tensor RG errors
        rgerr = [zerrs, yerrs, xerrs]
        return lrerr, rgerr

    def rgmap(self, tnrg_pars,
              scheme="hotrg3d", ver="base",
              gaugeFix=False,
              comm=None,
              chiSet=None,
              ):
        """
        coarse grain the tensors using schemes above
        - hotrg3d

        Return two kinds of local replacement errors
        1) loop-filtering process: it is zero for hotrg
        2) coarse graining process: x, y, z each; total three
        """
        if scheme == "hotrg3d":
            if ver == "base":
                (lferrs,
                 SPerrs
                 ) = self.hotrg(tnrg_pars,
                                signFix=gaugeFix,
                                comm=comm)
        elif scheme == "blockHOTRG":
            if ver == "base":
                (lferrs,
                 SPerrs
                 ) = self.block_hotrg(tnrg_pars,
                                      signFix=gaugeFix,
                                      comm=comm)
        elif scheme == "efrg":
            if ver == "base":
                (lferrs,
                 SPerrs
                 ) = self.entfree_blockrg(tnrg_pars,
                                          signFix=gaugeFix,
                                          comm=comm,
                                          chiSet=chiSet)
            elif ver == "bistage":
                (lferrs,
                 SPerrs
                 ) = self.ef2stp_blockrg(
                     tnrg_pars,
                     signFix=gaugeFix,
                     comm=comm)
        return lferrs, SPerrs

    # II. linearized RG maps
    @staticmethod
    def linear_hotrg(Astar, cgten,
                     comm=None):
        """
        Z2 symmetry imposed here:
        Astar and tensors in cgten should be Z2-symmetric
        """
        # I. generate all the middle tensors during a RG step
        AstarPiece = hotrg3d.fullContr(Astar, cgten, comm=comm)
        AstarPiece = AstarPiece[:-1]

        # II. construct linearized RG for both sectors
        # II.1 For change-0 (Even) sector
        AstarArray = u1ten.Z2toArray(Astar)[0]
        dim_ch0PsiA = AstarArray.shape[0]

        # define the response matrix for deltaA with charge = 0
        def linearRG0(deltaPsiA):
            # reshape the numpy array into Z2-symmetric tensor with charge = 0
            deltaAch0 = u1ten.arraytoZ2(deltaPsiA, Astar)
            # map from deltaAch0 --> deltaAcch0
            deltaAcch0 = hotrg3d.linrgmap(deltaAch0, AstarPiece, cgten,
                                          comm=comm)
            # reshape the output deltaAcch0 back into a 1D array
            deltaPsiAc = u1ten.Z2toArray(deltaAcch0)[0]
            return deltaPsiAc

        # II.2 For charge-1 (Odd) sector
        # create empty tensor with the same shape as Astar but
        # with charge = 1
        Aemptch1 = Astar.empty_like()
        Aemptch1.charge = 1
        Aemptch1Array = u1ten.Z2toArray(Aemptch1)[0]
        dim_ch1PsiA = Aemptch1Array.shape[0]

        # define the response matrix for deltaA with charge = 1
        def linearRG1(deltaPsiA):
            # reshape the numpy array into Z2-symmetric tensor with charge = 1
            deltaAch1 = u1ten.arraytoZ2(deltaPsiA, Aemptch1)
            # map from deltaAch1 --> deltaAcch1
            deltaAcch1 = hotrg3d.linrgmap(deltaAch1, AstarPiece, cgten,
                                          comm=comm)
            # reshape the output deltaAcch1 back into a 1D array
            deltaPsiAc = u1ten.Z2toArray(deltaAcch1)[0]
            return deltaPsiAc

        # III. Return linear RG maps and dimensions of the maps
        linearRGSet = [linearRG0, linearRG1]
        dims_PsiA = [dim_ch0PsiA, dim_ch1PsiA]
        return linearRGSet, dims_PsiA

    @staticmethod
    def linear_block_hotrg(
        Astar, cgten, refl_c=[0, 0, 0], comm=None, isEF=False,
        ver="base", cubeFilter=True, loopFilter=True
    ):
        # I. generate all the middle tensors during a RG step
        if not isEF:
            # Case 1: 3D hotrg-like block-tensor
            Astar_all = bkten3d.fullContr(Astar, cgten, comm=comm)
        else:
            # Case 2: add entanglement filtering
            Astar_all = efrg3d.fullContr(
                Astar, cgten, comm=comm,
                ver=ver, cubeFilter=cubeFilter, loopFilter=loopFilter
            )

        # II. construct linearized RG for both sectors
        # II.1 For Spin-flip charge-0 (Even) sector
        # dimensionality of the linear map
        AstarArray = u1ten.Z2toArray(Astar)[0]
        dim_ch0PsiA = AstarArray.shape[0]

        # define the response matrix for deltaA with charge = 0
        def linearRG0(deltaPsiA):
            # reshape the numpy array into Z2-symmetric tensor with charge = 0
            deltaAch0 = u1ten.arraytoZ2(deltaPsiA, Astar)
            # ------------------------------------\
            # Linear map: deltaAch0 --> deltaAcch0
            if not isEF:
                # Case 1: 3D hotrg-like block-tensor
                deltaAcch0 = bkten3d.linrgmap(
                    deltaAch0, Astar_all, cgten,
                    refl_c, comm=comm
                )
            else:
                # Case 2: add entanglement filtering
                deltaAcch0 = efrg3d.linrgmap(
                    deltaAch0, Astar_all, cgten,
                    refl_c, comm=comm,
                    ver=ver, cubeFilter=cubeFilter, loopFilter=loopFilter
                )
            # ------------------------------------/
            # reshape the output deltaAcch0 back into a 1D array
            deltaPsiAc = u1ten.Z2toArray(deltaAcch0)[0]
            return deltaPsiAc

        # II.2 For charge-1 (Odd) sector
        # create empty tensor with the same shape as Astar but
        # with charge = 1
        Aemptch1 = Astar.empty_like()
        Aemptch1.charge = 1
        Aemptch1Array = u1ten.Z2toArray(Aemptch1)[0]
        dim_ch1PsiA = Aemptch1Array.shape[0]

        # define the response matrix for deltaA with charge = 1
        def linearRG1(deltaPsiA):
            # reshape the numpy array into Z2-symmetric tensor with charge = 1
            deltaAch1 = u1ten.arraytoZ2(deltaPsiA, Aemptch1)
            # ------------------------------------\
            # Linear map from deltaAch1 --> deltaAcch1
            if not isEF:
                # Case 1: 3D hotrg-like block-tensor
                deltaAcch1 = bkten3d.linrgmap(
                    deltaAch1, Astar_all, cgten,
                    refl_c, comm=comm
                )
            else:
                # Case 2: add entanglement filtering
                deltaAcch1 = efrg3d.linrgmap(
                    deltaAch1, Astar_all, cgten,
                    refl_c, comm=comm,
                    ver=ver, cubeFilter=cubeFilter, loopFilter=loopFilter
                )
            # ------------------------------------/
            # reshape the output deltaAcch1 back into a 1D array
            deltaPsiAc = u1ten.Z2toArray(deltaAcch1)[0]
            return deltaPsiAc

        # III. Return linear RG maps and dimensions of the maps
        linearRGSet = [linearRG0, linearRG1]
        dims_PsiA = [dim_ch0PsiA, dim_ch1PsiA]
        return linearRGSet, dims_PsiA


    def eval_free_energy(self, initial_spin=1, b=2):
        """
        See my jupyter notebook on evently's trg
        for how to calculate the free energy

        initial_spin: int
            number of spins that the initial tensor corresponds to
        b: int
            ratio between the coarse lattice length and the length
            before RG.
            A coarser tensor corresponds to b^3 tensors before a RG step.
        """
        messg = "iter_n should be length of the tensor magnitute list plus 1"
        assert len(self.tensor_magnitude) == (self.iter_n + 1), messg
        # calculate free energy divided by the temperature
        ten_cur = self.get_tensor()
        ten_mag_arr = np.array(self.tensor_magnitude)
        weight = (
            (1 / initial_spin) *
            (1 / b**3)**np.array(range(0, self.iter_n + 1))
                  )
        # all contributions from the tensor magnitute
        g = (weight * np.log(ten_mag_arr)).sum()
        # the contribution from the tracing off the final normalized tensor
        g += (
            (1 / (initial_spin * b**(3 * self.iter_n))) *
            np.log(ncon([ten_cur], [[1, 1, 2, 2, 3, 3]]))
        )
        return g


def flatten(xss):
    return [x for xs in xss for x in xs]
