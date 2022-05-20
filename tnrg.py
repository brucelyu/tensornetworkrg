#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : tnrg.py
# Author            : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
# Date              : 20.05.2022
# Last Modified Date: 20.05.2022
# Last Modified By  : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>

class TensorNetworkRG:
    """
    Tensor network renormalization group
    """
    model_choice = ["ising2d", "ising3d"]
    scheme_choice = []

    def __init__(self, model, scheme):
        """__init__.
        Initialize a TensorNetworkRG instance.
        ----------
        model: string
            model name,
            choose among `class.model_choice`.
        scheme: string
            tensor network renormalization group scheme,
            choose among `class.scheme_choice`.

        """
        self.model = model
        self.scheme = scheme
