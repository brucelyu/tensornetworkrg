#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : benchmark.py
# Author            : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
# Date              : 09.01.2023
# Last Modified Date: 09.01.2023
# Last Modified By  : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>

import numpy as np
import os
import pickle as pkl
from . import tnrg


def benm2DIsing(relT=1.0, h=0, isCrit=True,
                scheme="fet-hotrg", ver="base",
                pars={}):
    """
    Benchmark TNRG schemes on 2D Ising model by generating
    1) local approximation error RG flow
    2) approximation error of free energy

    pars = {"chi": 10, "dtol": 1e-10, "isZ2"=True, "rg_n": 18,
    "chis": 10/2, "iter_max": 2000,
    "outDir": "./out", "dataDir": "./data"}
    """
    # whether to impose Z2 symmetry
    isZ2 = pars["isZ2"]
    ising2d = tnrg.TensorNetworkRG2D("ising2d")
    if isCrit:
        ising2d.set_critical_model()
    else:
        Tc = (2 / np.log(1 + np.sqrt(2)))
        Tval = relT * Tc
        ising2d.set_model_parameters(Tval, h)
    if scheme in ["fet-hotrg", "hotrg"]:
        init_dirs = [1, 1, -1, -1]
    elif scheme == "tnr":
        init_dirs = [1, 1, 1, 1]
    else:
        raise NotImplementedError("Not implemented yet")
    # generate initial tensor
    ising2d.generate_initial_tensor(onsite_symmetry=isZ2,
                                    init_dirs=init_dirs)
    # exact free energ
    exact_g = ising2d.get_exact_free_energy()
    # read off tnrg parameters
    chi = pars["chi"]
    dtol = pars["dtol"]
    rg_n = pars["rg_n"]
    print("The two basic tnrg parameters are")
    print("TNRG bond dimension: --{:d}--".format(chi))
    print("TNRG epsilon: --{:.2e}--".format(dtol))
    print("TRNG iteration step: --{:d}--".format(rg_n))
    print("------")

    # Next is for hyper-parameters of
    # different blocking tensor schemes
    print("The TNRG scheme is --{:s}--,".format(scheme),
          "with version --{:s}--".format(ver))
    # fet-hotrg
    if scheme == "fet-hotrg":
        chis = pars["chis"]
        iter_max = pars["iter_max"]
        if ver == "base":
            epsilon = dtol * 1.0
            print("The additional hyper-parameters are")
            print("Entanglement-filtering squeezed bond dimension: ",
                  "--{:d}--".format(chis))
            print("The maximal FET iteration step: --{:d}--".format(iter_max))
            tnrg_pars = {"chi": chi, "dtol": dtol,
                         "chis": chis, "iter_max": iter_max,
                         "epsilon": epsilon, "epsilon_init": epsilon,
                         "bothSides": True, "display": True}
        else:
            raise NotImplementedError(
                "No such version for {:s}".format(scheme)
            )

    # HOTRG
    elif scheme == "hotrg":
        if ver == "base":
            tnrg_pars = {"chi": chi, "dtol": dtol, "display": True}
    elif scheme == "tnr":
        chis = pars["chis"]
        iter_max = pars["iter_max"]
        miniter = 200
        convtol = 0.01
        if ver == "base":
            print("The additional hyper-parameters are")
            print("Entanglement-filtering squeezed bond dimension: ",
                  "--{:d}--".format(chis))
            print("The maximal disentangling iteration step:",
                  "--{:d}--".format(iter_max))
            tnrg_pars = {
                "chiM": chi + 2, "chiH": chi, "chiV": chi, "dtol": dtol,
                "chiS": chis, "chiU": chis,
                "disiter": iter_max,
                "miniter": miniter, "convtol": convtol,
                "is_display": True}
    else:
        raise NotImplementedError("Not implemented yet")

    # enter the RG iteration
    (
        errFETList,
        errVList,
        errHList,
        gList
    ) = tnrgIterate(ising2d, rg_n, scheme, ver,
                    tnrg_pars, pars["dataDir"])
    gErrList = np.abs(np.array(gList) - exact_g) / np.abs(exact_g)
    # save the RG flow of various errors
    outDir = pars["outDir"]
    if outDir is not None:
        fname = "{:s}-{:s}-chi{:d}.pkl".format(scheme, ver, chi)
        saveData(outDir,
                 fname,
                 data=[chi, errFETList, errVList, errHList, gErrList]
                 )


def benm3DIsing(T=5.0, h=0, scheme="hotrg3d",
                ver="base",
                pars={}):
    """
    Benchmark TNRG schemes on 3D Ising model by generating
    1) local approximation error RG flow

    Kwargs:
        T (float): temperature
        h (float): magnetic field
        scheme (str): coarse graining scheme
            choice is ["hotrg3d"]
        ver (str): version of the scheme
        pars (dict): scheme parameter
            basic is hotrg + tensor-symmetric parameter:
            here is an example
            `pars={"isZ2": True, "rg_n": 12,
            "chi": 4, "cg_eps": 1e-8, "display": True,
            "dataDir": None, "determPhase": True}`

    Returns:
        various RG flows,XFlow, errMaxFlow, eeFlow, SPerrsFlow, lrerrsFlow
          - XFlow: degenerate index flow, useful for determining critical T
          - errMaxFlow: maximal RG replacement error in each RG step
          - eeFlow: entanglement entropies [eexyz, eex, eey, eez]
          - SPerrsFlow: Flow of RG repalcement errors in each RG step
          - lrerrsFlow: loop-reduction errors in each RG step

    """
    # create model instance
    ising3d = tnrg.TensorNetworkRG3D("ising3d")
    # set model parameter and create the initial tensor
    isZ2 = pars["isZ2"]
    ising3d.set_model_parameters(T, h)
    ising3d.generate_initial_tensor(onsite_symmetry=isZ2)
    # read off tnrg parameters
    chi = pars["chi"]
    dtol = pars["cg_eps"]
    rg_n = pars["rg_n"]
    display = pars["display"]

    if display:
        print("The basic 3d tnrg parameters are")
        print("TNRG bond dimension: --{:d}--".format(chi))
        print("TNRG epsilon: --{:.2e}--".format(dtol))
        print("TRNG iteration step: --{:d}--".format(rg_n))
        print("Impose Z2 symmetry? --{}--".format(isZ2))
        print("------")
    # generate degenerate index X flow
    (
        XFlow, errMaxFlow, eeFlow,
        SPerrsFlow, lrerrsFlow
    ) = tnrg3dIterate(ising3d, rg_n, scheme, ver,
                      tnrg_pars=pars, dataDir=pars["dataDir"],
                      determPhase=pars["determPhase"])
    return XFlow, errMaxFlow, eeFlow, SPerrsFlow, lrerrsFlow


def tnrg3dIterate(tnrg3dCase, rg_n=10, scheme="hotrg3d", ver="base",
                  tnrg_pars={}, dataDir=None, determPhase=True):
    """
    Perform the 3D TNRG iteration

    Args:
        tnrg3dCase (TensorNetworkRG3D):
            an instance for the class `tnrg.TensorNetworkRG3D`

    Kwargs:
        rg_n (int): maximal rg iteration step
        scheme (str): coarse graining scheme
            choose among ["hotrg3d"]
        ver (str): version of a given scheme
            default is ["base"]
        tnrg_pars (dict): parameters for the tnrg scheme
        dataDir (str):
            Path of directory to save data

    Returns:
        various RG flows,XFlow, errMaxFlow, eeFlow, SPerrsFlow, lrerrsFlow
          - XFlow: degenerate index flow, useful for determining critical T
          - errMaxFlow: maximal RG replacement error in each RG step
          - eeFlow: entanglement entropies [eexyz, eex, eey, eez]
          - SPerrsFlow: Flow of RG repalcement errors in each RG step
          - lrerrsFlow: loop-reduction errors in each RG step

    """
    assert tnrg3dCase.get_iteration() == 0
    # record rg flows
    XFlow = []
    lrerrsFlow = []
    SPerrsFlow = []
    errMaxFlow = []
    eeFlow = []
    for k in range(rg_n):
        (
         lrerrs,
         SPerrs
         ) = tnrg3dCase.rgmap(tnrg_pars,
                              scheme=scheme, ver=ver)
        # various properties of the current tensor
        # - Degenerate index:
        #   1 for trivial phase, 2 for Z2-symmetry breaking phase
        curX = tnrg3dCase.degIndX()
        # - Various entanglement entropy
        eex = tnrg3dCase.entangle(leg="x")[0]
        eey = tnrg3dCase.entangle(leg="y")[0]
        eez = tnrg3dCase.entangle(leg="z")[0]
        eexyz = tnrg3dCase.entangle(leg="xyz")[0]
        # maximal RG replacement errors of all 3*2=6 squeezers
        errMax = np.max(SPerrs)
        # cur_g = tnrg3dCase.eval_free_energy()
        # record rg flows
        XFlow.append(curX)
        lrerrsFlow.append(lrerrs)
        SPerrsFlow.append(SPerrs)
        errMaxFlow.append(errMax)
        eeFlow.append([eexyz, eex, eey, eez])
        if tnrg_pars["display"]:
            print("The RG step {:d} finished!".format(
                tnrg3dCase.get_iteration())
                  )
            print("Shape of the tensor is")
            print(tnrg3dCase.get_tensor().shape)
            print("Degnerate index X = {:.2f}".format(curX))
            print("Entanglement entropies:")
            print("Leg x: {:.2f}, Leg y: {:.2f}, Leg z: {:.2f}".format(
                eex, eey, eez
            )
                  )
            print("Leg xyz: {:.2f}".format(eexyz))
            print("Maximal truncation error is {:.2e}".format(errMax))
            print("----------")
            print("----------")
        # exit the RG iteration if already flowed to the trivial phase
        if determPhase:
            stop_eps = 0.01
            if (abs(curX - 1) < stop_eps) or (abs(curX - 2)) < stop_eps:
                break
    return XFlow, errMaxFlow, eeFlow, SPerrsFlow, lrerrsFlow


def tnrgIterate(tnrgCase, rg_n=21, scheme="fet-hotrg", ver="base",
                tnrg_pars={}, dataDir=None):
    assert tnrgCase.get_iteration() == 0
    lrerrList = []
    errVList = []
    errHList = []
    gList = []
    for k in range(rg_n):
        print("At RG step {:d}".format(tnrgCase.get_iteration()))
        (
         lrerrs,
         SPerrs
         ) = tnrgCase.rgmap(tnrg_pars,
                            scheme=scheme, ver=ver)
        cur_g = tnrgCase.eval_free_energy()
        # Append to list
        lrerrList.append(lrerrs)
        errVList.append(SPerrs[0])
        errHList.append(SPerrs[1])
        gList.append(cur_g.norm())
        print("Shape of the tensor is")
        print(tnrgCase.get_tensor().shape)
        print("----------")
        print("----------")

    return lrerrList, errVList, errHList, gList


def saveData(outDir, fname, data=None):
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    saveFile = outDir + "/" + fname
    with open(saveFile, "wb") as f:
        pkl.dump(data, f)


def plotErrs(chi, errFETList, errVList, errHList, gErrList,
             startn=1, endn=-1, outDir=None, fname=None):
    import matplotlib.pyplot as plt
    xArr = [k for k in range(startn, len(errVList) + endn)]
    plt.figure(figsize=(6, 8))
    ax1 = plt.subplot(311)
    ax1.plot(xArr, errFETList[startn:endn], "b.--", alpha=0.8,
             label="FET error")
    plt.ylabel("RG errors")
    plt.yscale("log")
    plt.legend()

    ax2 = plt.subplot(312)
    ax2.plot(xArr, errVList[startn:endn], "bx-", alpha=0.8,
             label="Vert. RG error")
    ax2.plot(xArr, errHList[startn:endn], "b+-", alpha=0.8,
             label="Hori. RG error")
    plt.ylabel("RG errors")
    plt.yscale("log")
    plt.legend()

    ax3 = plt.subplot(313)
    ax3.plot(gErrList, "ko-", alpha=0.6,
             label="stable error: {:.2e}".format(gErrList[-1]))
    plt.ylabel("Relative error of free energy")
    plt.yscale("log")
    plt.xlabel(r"RG step ($\chi$={:d})".format(chi))
    plt.legend()
    # save figure
    figFile = outDir + "/" + fname
    plt.savefig(figFile, bbox_inches='tight', dpi=300)


def plotErrs2(chi, errVList, errHList, gErrList,
              startn=1, endn=-1, outDir=None, fname=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 5))
    xArr = [k for k in range(startn, len(errVList) + endn)]
    ax2 = plt.subplot(211)
    ax2.plot(xArr, errVList[startn:endn], "bx-", alpha=0.8,
             label="Vert. RG error")
    ax2.plot(xArr, errHList[startn:endn], "b+-", alpha=0.8,
             label="Hori. RG error")
    plt.ylabel("RG errors")
    plt.yscale("log")
    plt.legend()
    plt.title("Baseline: HOTRG")

    ax3 = plt.subplot(212)
    ax3.plot(gErrList, "ko-", alpha=0.6,
             label="stable error: {:.2e}".format(gErrList[-1]))
    plt.ylabel("Relative error of free energy")
    plt.yscale("log")
    plt.xlabel(r"RG step ($\chi$={:d})".format(chi))
    plt.legend()
    # save figure
    figFile = outDir + "/" + fname
    plt.savefig(figFile, bbox_inches='tight', dpi=300)

