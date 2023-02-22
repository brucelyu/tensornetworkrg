#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : rg_prescription.py
# Author            : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
# Date              : 20.02.2023
# Last Modified Date: 20.02.2023
# Last Modified By  : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
"""
Perform the textbook RG prescription according to the paper
Phys. Rev. Research 3, 023048 (2021)
by Xinliang Lyu, RuQing G. Xu, Naoki Kawashima
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import itertools
import pickle as pkl
from . import benchmark


def findTc(iter_n=15, Tlow=4.0, Thi=5.0,
           scheme="hotrg3d", ver="base",
           pars={}, outDir="./"):
    # helper function 1: Is hi-T phase or low-T phase?
    def ishiT(x):
        dist2hi = np.abs(x - 1)
        dist2low = np.abs(x - 2)
        if dist2hi < dist2low:
            res = True
        else:
            res = False
        return res

    # helper function 2: plot X flows
    def plotXFlows(Xlow, Xhi, Xtry, Tlow, Thi, Ttry,
                   savefile, linesty, alpha=1):
        dT = abs(Thi - Ttry) / Ttry
        # plt.figure()
        plt.title(("Difference from Tc = {:.2e}, ".format(dT)))
        plt.plot(Xlow[0:], "bo" + linesty, alpha=alpha,
                 label="low T = {:.5f}".format(Tlow))
        plt.plot(Xhi[0:], "k." + linesty, alpha=alpha,
                 label="hi T = {:.5f}".format(Thi))
        plt.plot(Xtry[0:], "gx" + linesty, alpha=alpha,
                 label="try T = {:.5f}".format(Ttry))
        plt.ylim([0.8, 2.6])
        plt.legend()
        plt.xlabel("RG step")
        plt.ylabel("Degenerate index $X$")
        plt.savefig(savefile, dpi=300)

    saveDir = (outDir + "{:s}_{:s}_out".format(scheme, ver) + "/" +
               "chi{:02d}".format(pars["chi"])
               )
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    # read Tc if exists
    TcFile = saveDir + "/Tc.pkl"
    if os.path.exists(TcFile):
        with open(TcFile, "rb") as f:
            Tlow, Thi = pkl.load(f)
    # enter the bisection
    Xlow = benchmark.benm3DIsing(Tlow, h=0,
                                 scheme=scheme, ver=ver,
                                 pars=pars)[0]
    Xhi = benchmark.benm3DIsing(Thi, h=0,
                                scheme=scheme, ver=ver,
                                pars=pars)[0]
    plt.figure()
    lines = itertools.cycle(("--", "-.", "-"))
    for k in range(iter_n + 1):
        # generate trial degenerate index X flow
        Ttry = 0.5 * (Tlow + Thi)
        Xtry = benchmark.benm3DIsing(Ttry, h=0,
                                     scheme=scheme, ver=ver,
                                     pars=pars)[0]
        # plot every 3 iteration
        if k % 3 == 0:
            curline = next(lines)
            saveFigName = saveDir + "/X_iterk{:02d}.png".format(k+1)
            plotXFlows(Xlow, Xhi, Xtry,
                       Tlow, Thi, Ttry,
                       saveFigName, curline, alpha=(k+1)/(iter_n+1))
        # update low and high bound
        if ishiT(Xtry[-1]):
            Thi = Ttry
            Xhi = Xtry.copy()
        else:
            Tlow = Ttry
            Xlow = Xtry.copy()
    # save Tc
    # save the lower and upper bound of Tc
    with open(TcFile, "wb") as f:
        pkl.dump([Tlow, Thi], f)
    # Append all figures
    orgfile = saveDir + '/X_iterk*.png'
    tarfile = saveDir + '/Xflow_all.png'
    syscommand = 'convert ' + orgfile + " -append " + tarfile
    if os.system(syscommand) != 0:
        print("Command, convert, not found in current os")
    else:
        os.system('rm ' + orgfile)


# TODO: generate RG flow at Tc + gauge (sign) fixing

# TODO: linearize RG map and extracting scaling dimensions
