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
    if scheme == "fet-hotrg" or "hotrg":
        init_dirs = [1, 1, -1, -1]
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
            print("The additional hyper-parameters are")
            print("Entanglement-filtering squeezed bond dimension: ",
                  "--{:d}--".format(chis))
            print("The maximal FET iteration step: --{:d}--".format(iter_max))
            tnrg_pars = {"chi": chi, "dtol": dtol,
                         "chis": chis, "iter_max": iter_max,
                         "epsilon": dtol, "epsilon_init": dtol,
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
        raise NotImplementedError("Not implemented yet")
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

