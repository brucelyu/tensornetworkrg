#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : tnrg.py
# Author            : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
# Date              : 20.05.2022
# Last Modified Date: 20.05.2022
# Last Modified By  : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>

import numpy as np
from ncon import ncon
from .initial_tensor import initial_tensor


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

    def generate_initial_tensor(self, onsite_symmetry=False):
        """
        generate the initial tensor corresponding to the model
        `model_parameters`: dictionary
            - For the ising model in 2d and 3d, an exmaple is
                {"temperature": 1.0, "magnetic_field": 0.0,
                "onsite_symmetry": False}
        """
        init_ten = initial_tensor(self.model, self.model_parameters,
                                  onsite_symmetry)
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
        proj_x = ten_new.svd([0], [1, 2, 3], eps=1e-15)[0]
        proj_y = ten_new.svd([1], [2, 3, 0], eps=1e-15)[0]
        ten_new = ncon([ten_new, proj_x.conjugate(), proj_x,
                        proj_y.conjugate(), proj_y],
                       [[1, 2, 3, 4], [1, -1], [3, -3],
                        [2, -2], [4, -4]])
        # pull out the tensor norm
        self.tensor_magnitude.append(ten_new.norm())
        self.current_tensor = ten_new / ten_new.norm()
        self.iter_n += 1
        assert len(self.tensor_magnitude) == (self.iter_n + 1)

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
    pass


class HOTRG2D(TensorNetworkRG2D):
    pass


class HOTRG3D(TensorNetworkRG3D):
    pass
