#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : tnrg.py
# Author            : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
# Date              : 20.05.2022
# Last Modified Date: 20.05.2022
# Last Modified By  : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>

import numpy as np
from abeliantensors import TensorZ2, Tensor
from ncon import ncon
from .initial_tensor import initial_tensor
from .coarse_grain_2d import trg_evenbly, tnr_evenbly, hotrg
from .loop_filter import cleanLoop, toymodels


class TensorNetworkRG:
    """
    Tensor network renormalization group
    """
    model_choice = ("ising2d", "ising3d", "golden chain", "cdl2d")
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


class TensorNetworkRG2D(TensorNetworkRG):
    """
    TNRG for 2D square lattice
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
        # pull out the tensor norm and save
        ten_mag = self.pullout_magnitude()
        self.save_tensor_magnitude(ten_mag)
        self.isometry_applied = [v, w]

    def fet_hotrg(
        self,
        pars={"chi": 4, "dtol": 1e-16,
              "chis": None, "iter_max": 10,
              "epsilon": 1e-10, "epsilon_init": 1e-10,
              "bothSides": True, "display": True
              }
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
         errFET
         ) = cleanLoop.fet2dReflSym(ten_cur, chis, epsilon, iter_max,
                                    epsilon_init, bothSides=bothSides,
                                    display=False)
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

    def hotrg(
        self,
        pars={"chi": 4, "dtol": 1e-16,
              "display": True}
    ):
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



class TensorNetworkRG3D(TensorNetworkRG):
    """
    TNRG for 3D cubic lattice
    """
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




class HOTRG2D(TensorNetworkRG2D):
    pass


class HOTRG3D(TensorNetworkRG3D):
    pass
