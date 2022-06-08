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
from abeliantensors import Tensor
import numpy as np
from ncon import ncon

MODEL_CHOICE = ("ising2d", "ising3d", "golden chain")


def initial_tensor(model, model_parameters, is_sym=False,
                   scheme="simple"):
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
    elif model == "golden chain":
        res = golden_chain(scheme=scheme)
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


def golden_chain(scheme="simple"):
    """golden_chain.
    init_ten = gold_chain()

    Initial tensor for the golden chain, or the fibonacci anyon model.

    The initial tensor actually corresponds to the A_4 ABF model,
    which is construction from the A_4 fusion categroy a la Aasen et al (2020).
    We choose the spectrual parameter to be lambda/2 to make the model 90-deg
    rotationally symmetric.

    Although it seems the initial local degrees of freedom has 4 values,
    the fusion rule breaks the square lattice into to two sublattice,
    local boltzamann weight has bond dimension 2.

    Parameters
    ----------
    scheme : str
        ["simple", "symmetric"]
        For "simple" scheme,
            the tensor itself doesn't have reflection
            or rotation symmetry, the bond dimension is 3.
        For "symmetric" scheme,
            the tensor itself has reflectoin and rotational
            symmetry, the bond dimension is 5.
    """
    # initialize the local plaquette Boltzmann weight
    boltz = Tensor.zeros(shape=[2, 2, 2, 2])
    d_0 = 1.0
    d_1 = 2 * np.cos(np.pi / 5)
    # set non-vanishing values for the Boltzmann weight
    overall_factor = np.sin(np.pi / 10)
    boltz[1, 0, 1, 0] = np.sqrt(d_1 / d_0) + np.sqrt(d_0 / d_1)
    boltz[0, 1, 0, 1] = np.sqrt(d_1 / d_0) + np.sqrt(d_0 / d_1)
    boltz[1, 1, 1, 1] = 2.0
    boltz[0, 1, 1, 1] = (d_0 / d_1)**(1/4)
    boltz[1, 0, 1, 1] = (d_0 / d_1)**(1/4)
    boltz[1, 1, 0, 1] = (d_0 / d_1)**(1/4)
    boltz[1, 1, 1, 0] = (d_0 / d_1)**(1/4)
    boltz = overall_factor * boltz

    if scheme == "simple":
        # define the gluing tensor
        gluing_ten = Tensor.zeros(shape=[2, 2, 2, 2])
        gluing_ten[0, 0, 0, 0] = 1
        gluing_ten[1, 1, 1, 1] = 1

        # contruct plaquette Boltzmann weight and
        # the gluing tensor to get the initial tensor
        init_ten = ncon([boltz, gluing_ten, boltz, gluing_ten],
                        [[-2, -3, 3, 1], [3, -4, -6, 2],
                         [4, 2, -5, -8], [-1, 1, 4, -7]])
        init_ten = init_ten.reshape(4, 4, 4, 4)
        # The 0-th dimension vanishing
        # init_ten[0, :, :, :].norm() == 0 and so on.
        init_ten = init_ten[1:, 1:, 1:, 1:]
    elif scheme == "symmetric":
        # first tensor
        ten_i = Tensor.zeros(shape=[3, 3, 3, 3])
        ten_i[0, 0, 0, 0] = boltz[1, 0, 1, 0]
        ten_i[1, 1, 1, 1] = boltz[0, 1, 0, 1]
        ten_i[2, 2, 2, 2] = boltz[1, 1, 1, 1]
        ten_i[1, 1, 2, 2] = boltz[0, 1, 1, 1]
        ten_i[2, 0, 0, 2] = boltz[1, 0, 1, 1]
        ten_i[2, 2, 1, 1] = boltz[1, 1, 0, 1]
        ten_i[0, 2, 2, 0] = boltz[1, 1, 1, 0]
        # second tensor
        swap_mat = Tensor.zeros(shape=[3, 3])
        swap_mat[2, 2] = 1
        swap_mat[0, 1] = 1
        swap_mat[1, 0] = 1
        ten_ii = ncon([ten_i, swap_mat, swap_mat, swap_mat, swap_mat],
                      [[1, 2, 3, 4], [1, -1], [2, -2],
                       [3, -3], [4, -4]])
        # contract two ten_i and two ten_ii
        init_ten = ncon([ten_i, ten_ii, ten_i, ten_ii],
                        [[-1, -3, 2, 1], [2, -4, -5, 3],
                         [4, 3, -6, -8], [-2, 1, 4, -7]])
        init_ten = init_ten.reshape(9, 9, 9, 9)
        null_indx = np.array([1, 2, 3, 6])
        for axis_n in range(4):
            assert init_ten.take(indices=null_indx,
                                 axis=axis_n).norm() < 1e-15
        image_indx = np.array([0, 4, 8, 5, 7])
        for axis_n in range(4):
            init_ten = init_ten.take(indices=image_indx,
                                     axis=axis_n)

    return init_ten
