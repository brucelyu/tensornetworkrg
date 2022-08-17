#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : tnrg.py
# Author            : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
# Date              : 20.05.2022
# Last Modified Date: 20.05.2022
# Last Modified By  : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>

import numpy as np
from abeliantensors import TensorZ2
from ncon import ncon
from .initial_tensor import initial_tensor
from .coarse_grain_2d import trg_evenbly, tnr_evenbly


class TensorNetworkRG:
    """
    Tensor network renormalization group
    """
    model_choice = ("ising2d", "ising3d", "golden chain")
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

    def generate_initial_tensor(self, onsite_symmetry=False, scheme="simple"):
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
        ten_new = ten_new.join_indices((0, 1), (2, 3), (4, 5), (6, 7))
        # truncate the leg a la hosvd
        ten_new = self.truncate_hosvd(ten_new)
        # pull out the tensor norm
        self.tensor_magnitude.append(ten_new.norm())
        self.current_tensor = ten_new / ten_new.norm()
        self.iter_n += 1
        assert len(self.tensor_magnitude) == (self.iter_n + 1)

    @staticmethod
    def truncate_hosvd(ten):
        """
        truncate a la hosvd
        """
        proj_x = ten.svd([0], [1, 2, 3], eps=1e-15)[0]
        proj_y = ten.svd([1], [2, 3, 0], eps=1e-15)[0]
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
