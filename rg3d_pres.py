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
           pars={}, outDir="./",
           comm=None):
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

    # take care of PARAL CODE
    if comm is None:
        rank = 0
    else:
        rank = comm.Get_rank()
    # directory name for saving
    saveDir = saveDirName(scheme, ver, pars, outDir,
                          comm=comm)
    if (not os.path.exists(saveDir)) and (rank == 0):
        os.makedirs(saveDir)
    # read Tc if exists at rank-0 process
    TcFile = saveDir + "/Tc.pkl"
    if os.path.exists(TcFile) and (rank == 0):
        with open(TcFile, "rb") as f:
            Tlow, Thi = pkl.load(f)
    # PARAL CODE: broadcast Tlow and Thi
    if comm is not None:
        Tlow = comm.bcast(Tlow, root=0)
        Thi = comm.bcast(Thi, root=0)
    # enter the bisection
    Xlow = benchmark.benm3DIsing(Tlow, h=0,
                                 scheme=scheme, ver=ver,
                                 pars=pars,
                                 comm=comm)[0]
    Xhi = benchmark.benm3DIsing(Thi, h=0,
                                scheme=scheme, ver=ver,
                                pars=pars,
                                comm=comm)[0]
    # ---------------------\
    # plot at rank-0 process
    if rank == 0:
        plt.figure()
        lines = itertools.cycle(("--", "-.", "-"))
    # ---------------------/
    for k in range(iter_n + 1):
        # generate trial degenerate index X flow
        Ttry = 0.5 * (Tlow + Thi)
        Xtry = benchmark.benm3DIsing(Ttry, h=0,
                                     scheme=scheme, ver=ver,
                                     pars=pars,
                                     comm=comm)[0]
        # ------\
        # plot every 3 iteration at rank-0 process
        if (k % 3 == 0) and (rank == 0):
            curline = next(lines)
            saveFigName = saveDir + "/X_iterk{:02d}.png".format(k+1)
            plotXFlows(Xlow, Xhi, Xtry,
                       Tlow, Thi, Ttry,
                       saveFigName, curline, alpha=(k+1)/(iter_n+1))
        # ------/
        # update low and high bound
        if ishiT(Xtry[-1]):
            Thi = Ttry
            Xhi = Xtry.copy()
        else:
            Tlow = Ttry
            Xlow = Xtry.copy()
    # -------\
    # save Tc
    # save the lower and upper bound of Tc
    # at rank-0 process
    if rank == 0:
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
    # -------/


def generateRGflow(scheme="hotrg3d", ver="base",
                   pars={}, outDir="./", plotRGmax=15,
                   comm=None):
    """generate RG flow at critical temperature

    """
    # take care of PARAL CODE
    if comm is None:
        rank = 0
    else:
        rank = comm.Get_rank()

    saveDir = saveDirName(scheme, ver, pars, outDir,
                          comm=comm)
    # read Tc at rank-0 process
    TcFile = saveDir + "/Tc.pkl"
    Ttry = 0.0  # initialization
    if rank == 0:
        with open(TcFile, "rb") as f:
            Tlow, Thi = pkl.load(f)
        Ttry = 0.5 * (Tlow + Thi)
    # broadcast Ttry
    if comm is not None:
        Ttry = comm.bcast(Ttry, root=0)

    # take care of the save directory path
    if pars["dataDir"] is None:
        pars["dataDir"] = saveDir

    (
        XFlow, errMaxFlow,
        eeFlow, SPerrsFlow, lrerrsFlow
    ) = benchmark.benm3DIsing(Ttry, h=0,
                              scheme=scheme, ver=ver,
                              pars=pars,
                              gaugeFix=True,
                              comm=comm)
    # Read and plot the flow of tensor difference
    # generate in `benm3DIsing` process
    # at rank-0 process
    if rank == 0:
        tenDir = tensorsDir(pars["dataDir"])
        tenFile = tenDir + "/tenflows.pkl"
        with open(tenFile, "rb") as f:
            alltens = pkl.load(f)
            Amags, tenDiff, ten3diagDiff = alltens[1:]
        # plot data
        plotTenDiff(Amags, tenDiff, ten3diagDiff, saveDir,
                    Ttry, hiRG=plotRGmax)

# TODO: linearize RG map and extracting scaling dimensions
# functions...


# generate directory name for saving
def saveDirName(scheme, ver, pars, outDir="./",
                comm=None):
    """generate directory name for saving

    """
    if comm is None:
        endword = "out"
    else:
        endword = "paral"
    saveDir = (outDir + "{:s}_{:s}_{:s}".format(scheme, ver, endword) +
               "/" +
               "chi{:02d}".format(pars["chi"])
               )
    return saveDir


def tensorsDir(dataDir):
    return dataDir + "/tensors"


def plotTenDiff(Amags, tenDiff, ten3diagDiff, saveDir,
                Ttry, hiRG=15):
    Amags = np.array(Amags)
    AmagsDiff = np.abs(Amags[2:] - Amags[1:-1]) / Amags[1:-1]
    plt.figure(figsize=(10, 12))
    # 1) Difference of the tensor (tensor.norm()=1)
    ax1 = plt.subplot(311)
    ax1.plot(tenDiff[:hiRG], "ko--", alpha=0.6)
    plt.title("At temperature {:.5f}".format(Ttry))
    plt.yscale("log")
    plt.xlabel("RG step $n$")
    plt.ylabel(r"$\Vert \mathcal{A}^{(n+1)} - \mathcal{A}^{(n)} \Vert$")
    # 2) sign-independent tensor difference
    ax2 = plt.subplot(312)
    ax2.plot(ten3diagDiff[:hiRG], "k.--", alpha=0.6)
    # plt.title("Sign-independent difference of adjacent tensors")
    plt.yscale("log")
    plt.xlabel("RG step $n$")
    plt.ylabel(r"$\sqrt{\sum_{abc}(\mathcal{A}^{n+1}_{aabbcc} - \mathcal{A}^{n}_{aabbcc})^2} $")
    # 3) Flow of diff(Anorm)
    xvalm = min(hiRG, len(Amags[1:]))
    ax3 = plt.subplot(313)
    ax3.plot(AmagsDiff[:hiRG], "bx--", alpha=0.6)
    plt.yscale("log")
    plt.xticks(np.arange(0, xvalm - 1, 2),
               np.arange(1, xvalm, 2))
    plt.ylabel(r"diff($\Vert A^{(n)}\Vert$)")
    plt.xlabel("RG step $n$")
    # save
    plt.savefig(saveDir + "/tenDiffs.png", bbox_inches="tight", dpi=300)
