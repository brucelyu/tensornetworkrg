#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : tnrg.py
# Author            : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
# Date              : 20.05.2022
# Last Modified Date: 20.05.2022
# Last Modified By  : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>

import numpy as np
from .initial_tensor import initial_tensor


class TensorNetworkRG:
    """
    Tensor network renormalization group
    """
    model_choice = ("ising2d", "ising3d")
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
        self.current_tensor = initial_tensor(self.model,
                                             self.model_parameters,
                                             onsite_symmetry)


class TensorNetworkRG2D(TensorNetworkRG):
    pass


class TensorNetworkRG3D(TensorNetworkRG):
    pass


class HOTRG2D(TensorNetworkRG2D):
    pass


class HOTRG3D(TensorNetworkRG3D):
    pass
