#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : InitialTensor.py
# Author            : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
# Date              : 20.05.2022
# Last Modified Date: 20.05.2022
# Last Modified By  : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
"""
Generate initial tensor A0 for different physical models
"""
from tntools.initialtensors import get_initial_tensor

MODEL_CHOICE = ("ising2d", "ising3d")


def initial_tensor(model, model_parameters, is_sym=False):
    """initial_tensor.
    Current support `model`:
        - "isng2d", "ising3d",
            with `model_parameters.keys()`:
            "temperature", "magnetic_field", "onsite_symmetry"

    Parameters
    ----------
    model : str
        model
    model_parameters : dictionary
        Key: model parameters
    """
    assert model in MODEL_CHOICE, "Model not supported yet!"
    if model in ("ising2d", "ising3d"):
        beta = 1 / model_parameters["temperature"]
        ext_h = model_parameters["magnetic_field"]
        if model == "ising2d":
            res = ising_2d(beta, ext_h, is_sym)
        elif model == "ising3d":
            res = ising_3d(beta, ext_h, is_sym)
    return res


def ising_2d(beta=0.4, ext_h=0, is_sym=False):
    """
    init_ten = ising_2d(beta,ext_h,is_sym).
    -------------------------
    Set up the initial tensor for 2d classical Ising model on a square lattice.

    Argument: J is defined to be beta * J = J / kT, and h is
    defined to be beta*h = h / kT,
    where J and h are conventional coupling constants.

    Return: a rank 4 tensor T[x, y, x', y'].
    Each index of the tensor represents
    physical classical spin, and the tensor T represents the Boltzmann weight
    for interaction on one plaquettes.
    """
    pars = {"model": "ising", "dtype": "float64",
            "J": 1, "H": ext_h, "beta": beta, "symmetry_tensors": is_sym}
    init_ten = get_initial_tensor(pars)
    return init_ten


def ising_3d(beta=0.4, ext_h=0, is_sym=False):
    """ising_3d.
    init_ten = ising_3d(beta, ext_h, is_sym)

    Parameters
    ----------
    beta : float
        inverse temperature
    ext_h : float
        external field
    is_sym : boolean
        impose Z2 symmetry

    Returns:
    ----------
    init_ten: abeliantensors
        A[x, x', y, y', z, z']
    """
    pars = {"model": "ising3d", "dtype": "float64",
            "J": 1, "H": ext_h, "beta": beta, "symmetry_tensors": is_sym}
    init_ten = get_initial_tensor(pars)
    # rotate the tensor A[x, y, x', y', z, z'] to
    # A[x, x', y, y', z, z']
    init_ten = init_ten.transpose([0, 2, 1, 3, 4, 5])
    return init_ten
