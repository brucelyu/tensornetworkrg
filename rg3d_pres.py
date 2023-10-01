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
from datetime import datetime
import time
from . import benchmark, tnrg
from .coarse_grain_3d import hotrg as hotrg3d
from .coarse_grain_3d import block_tensor as bkten3d
from .coarse_grain_3d import efrg as efrg3d


def findTc(iter_n=15, Tlow=4.0, Thi=5.0,
           scheme="hotrg3d", ver="base",
           pars={}, outDir="./",
           comm=None, chiSet=None):
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
                                 comm=comm, noEE=True,
                                 chiSet=chiSet)[0]
    Xhi = benchmark.benm3DIsing(Thi, h=0,
                                scheme=scheme, ver=ver,
                                pars=pars,
                                comm=comm, noEE=True,
                                chiSet=chiSet)[0]
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
                                     comm=comm, noEE=True)[0]
        # ------\
        # plot every 3 iteration at rank-0 process
        if (k % 3 == 0) and (rank == 0):
            curline = next(lines)
            saveFigName = saveDir + "/X_iterk{:02d}.png".format(k+1)
            plotXFlows(Xlow, Xhi, Xtry,
                       Tlow, Thi, Ttry,
                       saveFigName, curline, alpha=(k+1)/(iter_n+1))
            print("This is the {:d}-th bisection step...".format(k + 1))
            print("Critical Temperature is bounded by",
                  "{:.5f} and {:.5f}".format(Tlow, Thi))
            print("---")
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
                   comm=None, chiSet=None):
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
                              comm=comm,
                              chiSet=chiSet)
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


# linearize RG map and extracting scaling dimensions
def linRG2scaleD(scheme="hotrg3d", ver="base", pars={},
                 rgstart=1, rgend=9,
                 evenN=10, oddN=10,
                 outDir="./", comm=None):
    def fixMagTen(A, Amag):
        # fix the tensor norm (optional)
        return A * (Amag**(-1/7))

    # take care of PARAL CODE
    if comm is None:
        rank = 0
    else:
        rank = comm.Get_rank()

    saveDir = saveDirName(scheme, ver, pars, outDir,
                          comm=comm)

    # take care of the save directory path
    if pars["dataDir"] is None:
        pars["dataDir"] = saveDir
    # read isometry tensors and magnituites of A at rank-0 process
    tenDir = tensorsDir(pars["dataDir"])
    isom, Amags = 0, 0
    if rank == 0:
        scDEvenExt = [0, 1.413, 2.413, 2.413, 2.413,
                      3, 3, 3, 3, 3]
        scDOddExt = [0.518, 1.518, 1.518, 1.518,
                     2.518, 2.518, 2.518, 2.518, 2.518, 2.518]
        tenFile = tenDir + "/tenflows.pkl"
        with open(tenFile, "rb") as f:
            alltens = pkl.load(f)
            isom, Amags = alltens[:2]
    # broadcast isom and Amags
    if comm is not None:
        isom = comm.bcast(isom, root=0)
        Amags = comm.bcast(Amags, root=0)

    # %%%%%%%%%%%%%%%%%%%%%%%%%% \
    # calculate scaling dimensions \
    if rank == 0:
        # Print out the time when the script is executed
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d. %H:%M:%S")
        print("Running Time =", current_time)
        print("Perform the tensor RG prescription using",
              "{:s}-{:s}".format(scheme, ver))
        print("Bond dimension is χ={:d}".format(pars["chi"]))
    # list to save scaling dimensions
    rgsteps = []
    scDList = []
    for k in range(rgstart, rgend):
        if rank == 0:
            print("/--------------------\\")
            print("For {}-th to {}-th RG step...".format(k, k+1))
            # start time
            startT = time.time()
        # read out the fixed-point tensor
        AcurFile = tenDir + "/A{:02d}.pkl".format(k)
        AnxtFile = tenDir + "/A{:02d}.pkl".format(k + 1)
        with open(AcurFile, "rb") as f:
            Acur = pkl.load(f)
        with open(AnxtFile, "rb") as f:
            Anxt = pkl.load(f)

        # make sure Anxt and Acur are of same shape
        if Anxt.shape == Acur.shape:
            # save k
            rgsteps.append(k)
            # check isometry correctness
            if scheme == "hotrg3d":
                AoutCheck = hotrg3d.fullContr(Acur, isom[k], comm=comm)[-1]
            elif scheme == "blockHOTRG":
                AoutCheck = bkten3d.fullContr(Acur, isom[k], comm=comm)[-1]
            elif scheme == "efrg":
                AoutCheck = efrg3d.fullContr(Acur, isom[k], comm=comm)[-1]
            AoutCheck = AoutCheck / AoutCheck.norm()
            checkDiff = (AoutCheck - Anxt).norm()
            errMsg = ("isometries have wrong gauge!" +
                      "(Check Difference is {:.2e})".format(checkDiff)
                      )
            assert checkDiff < 1e-10, errMsg
        else:
            if rank == 0:
                print("The shape of the tensor changes in the RG.")
                print("Skip!!")
            # skip this iteration
            continue
        # fix the norm of the tensor (optional)
        Astar = fixMagTen(Acur, Amags[k])

        # find scaling dimensions
        if scheme == "hotrg3d":
            scDims = linRG2x(
                Astar, isom[k], scheme="hotrg3d", ver="base",
                nscaleD=[evenN, oddN], comm=comm)
            # save scaling dimensions
            scDList.append(scDims)
        elif scheme in ["blockHOTRG", "efrg"]:
            scDims000, idenEig = linRG2x(
                Astar, isom[k], scheme=scheme, ver="base",
                nscaleD=[7, 5], comm=comm, refl_c=[0, 0, 0]
            )
            scDims100 = linRG2x(
                Astar, isom[k], scheme=scheme, ver="base",
                nscaleD=[3, 4], comm=comm, refl_c=[1, 0, 0],
                sector=[None, idenEig]
            )
            scDims010 = linRG2x(
                Astar, isom[k], scheme=scheme, ver="base",
                nscaleD=[3, 4], comm=comm, refl_c=[0, 1, 0],
                sector=[None, idenEig]
            )
            scDims001 = linRG2x(
                Astar, isom[k], scheme=scheme, ver="base",
                nscaleD=[3, 4], comm=comm, refl_c=[0, 0, 1],
                sector=[None, idenEig]
            )
            scDims110 = linRG2x(
                Astar, isom[k], scheme=scheme, ver="base",
                nscaleD=[2, 2], comm=comm, refl_c=[1, 1, 0],
                sector=[None, idenEig]
            )
            scDims101 = linRG2x(
                Astar, isom[k], scheme=scheme, ver="base",
                nscaleD=[2, 2], comm=comm, refl_c=[1, 0, 1],
                sector=[None, idenEig]
            )
            scDims011 = linRG2x(
                Astar, isom[k], scheme=scheme, ver="base",
                nscaleD=[2, 2], comm=comm, refl_c=[0, 1, 1],
                sector=[None, idenEig]
            )
            # save scaling dimensions
            scDList.append([scDims000,
                            scDims100, scDims010, scDims001,
                            scDims110, scDims101, scDims011]
                           )

        # print out scaling dimensions and time elapsed
        if rank == 0:
            # end time
            endT = time.time()
            diffT = endT - startT
            print("finished! Time elapsed = {:.2f}".format(diffT))
            if scheme == "hotrg3d":
                print("The scaling dimensions of even operators are:")
                with np.printoptions(precision=5, suppress=True):
                    print(scDims[0])
                print("The Exact values for 3D Ising even sector are")
                with np.printoptions(precision=5, suppress=True):
                    print(scDEvenExt)
                print("----------")
                print("The scaling dimensions of odd operators are:")
                with np.printoptions(precision=5, suppress=True):
                    print(scDims[1])
                print("The Exact values for 3D Ising odd sector are")
                with np.printoptions(precision=5, suppress=True):
                    print(scDOddExt)
            elif scheme in ["blockHOTRG", "efrg"]:
                print("The scaling dimensions of spin-flip-EVEN operators are:")
                with np.printoptions(precision=5, suppress=True):
                    print("    ",
                          "Lattice-reflection (000) sector")
                    print("    ", scDims000[0])
                    print("    Expected values from CFT arguments are")
                    print("    ", [0, 1.413, 3, 3, 3.413, 3.413, 3.413])
                    print("    ---")
                    print("    ",
                          "Lattice-reflection (100), (010) and (001) sector")
                    print("    ", scDims100[0], scDims010[0], scDims001[0])
                    print("    Expected values from CFT arguments are all")
                    print("    ", [2.413, 4, 4])
                    print("    ---")
                    print("    ",
                          "Lattice-reflection (110), (101) and (011) sector")
                    print("    ", scDims110[0], scDims101[0], scDims011[0])
                    print("    Expected values from CFT arguments are all")
                    print("    ", [3, 3.413])
                print("-----")
                print()
                print("The scaling dimensions of spin-flip-ODD operators are:")
                with np.printoptions(precision=5, suppress=True):
                    print("    ",
                          "Lattice-reflection (000) sector")
                    print("    ", scDims000[1])
                    print("    Expected values from CFT arguments are")
                    print("    ", [0.518, 2.518, 2.518, 2.518, 4.518])
                    print("    ---")
                    print("    ",
                          "Lattice-reflection (100), (010) and (001) sector")
                    print("    ", scDims100[1], scDims010[1], scDims001[1])
                    print("    Expected values from CFT arguments are all")
                    print("    ", [1.518, 3.518, 3.518, 3.518])
                    print("    ---")
                    print("    ",
                          "Lattice-reflection (110), (101) and (011) sector")
                    print("    ", scDims110[1], scDims101[1], scDims011[1])
                    print("    Expected values from CFT arguments are all")
                    print("    ", [2.518, 4.518])
            print("\\--------------------/")
            print()

    # save scaling dimensions
    if rank == 0:
        scDFile = tenDir + "/scDimSep.pkl"
        with open(scDFile, "wb") as f:
            pkl.dump([rgsteps, scDList], f)
        # plot scaling dimensions
        if scheme == "hotrg3d":
            plotscaleD(rgsteps, scDList, saveDir, pars)
        elif scheme == "blockHOTRG":
            pass


def linRG2scaleD1rg(scheme="hotrg3d", ver="base", pars={},
                    rgn=3, scaleN=10, outDir="./", comm=None,
                    sectorChoice="even"):
    def fixMagTen(A, Amag):
        # fix the tensor norm (optional)
        return A * (Amag**(-1/7))

    # take care of PARAL CODE
    if comm is None:
        rank = 0
    else:
        rank = comm.Get_rank()

    saveDir = saveDirName(scheme, ver, pars, outDir,
                          comm=comm)

    # take care of the save directory path
    if pars["dataDir"] is None:
        pars["dataDir"] = saveDir
    # read isometry tensors and magnituites of A at rank-0 process
    tenDir = tensorsDir(pars["dataDir"])
    isom, Amags = 0, 0
    if rank == 0:
        scDEvenExt = [0, 1.413, 2.413, 2.413, 2.413,
                      3, 3, 3, 3, 3]
        scDOddExt = [0.518, 1.518, 1.518, 1.518,
                     2.518, 2.518, 2.518, 2.518, 2.518, 2.518]
        tenFile = tenDir + "/tenflows.pkl"
        with open(tenFile, "rb") as f:
            alltens = pkl.load(f)
            isom, Amags = alltens[:2]
    # broadcast isom and Amags
    if comm is not None:
        isom = comm.bcast(isom, root=0)
        Amags = comm.bcast(Amags, root=0)

    # %%%%%%%%%%%%%%%%%%%%%%%%%% \
    # calculate scaling dimensions \
    Acur, Anxt = 0, 0
    if rank == 0:
        # Print out the time when the script is executed
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d. %H:%M:%S")
        print("Running Time =", current_time)
        print("Perform the tensor RG prescription using",
              "{:s}-{:s}".format(scheme, ver))
        print("/--------------------\\")
        print("For {}-th to {}-th RG step...".format(rgn, rgn+1))
        # start time
        startT = time.time()
        # read out the fixed-point tensor
        AcurFile = tenDir + "/A{:02d}.pkl".format(rgn)
        AnxtFile = tenDir + "/A{:02d}.pkl".format(rgn + 1)
        with open(AcurFile, "rb") as f:
            Acur = pkl.load(f)
        with open(AnxtFile, "rb") as f:
            Anxt = pkl.load(f)
    # broadcast Acur and Anxt
    if comm is not None:
        Acur = comm.bcast(Acur, root=0)
        Anxt = comm.bcast(Anxt, root=0)

    # make sure Anxt and Acur are of same shape
    if Anxt.shape == Acur.shape:
        # check isometry correctness
        AoutCheck = hotrg3d.fullContr(Acur, isom[rgn], comm=comm)[-1]
        AoutCheck = AoutCheck / AoutCheck.norm()
        checkDiff = (AoutCheck - Anxt).norm()
        errMsg = ("isometries have wrong gauge!" +
                  "(Check Difference is {:.2e})".format(checkDiff)
                  )
        assert checkDiff < 1e-10, errMsg
    else:
        if rank == 0:
            print("The shape of the tensor changes in the RG.")
            print("Skip!!")
        return None

    # fix the norm of the tensor (optional)
    Astar = fixMagTen(Acur, Amags[rgn])
    # read the base eigenvalue
    if sectorChoice == "even":
        sector = [sectorChoice, None]
    elif sectorChoice == "odd":
        baseEig = 1
        if rank == 0:
            scDFile = tenDir + "/scDim-rg{:02d}-even.pkl".format(rgn)
            with open(scDFile, "rb") as f:
                baseEig = pkl.load(f)[2]
        if comm is not None:
            baseEig = comm.bcast(baseEig, root=0)
        sector = [sectorChoice, baseEig]
    # find scaling dimensions
    scDims, baseEig = linRG2x(
        Astar, isom[rgn], scheme="hotrg3d", ver="base",
        nscaleD=scaleN, comm=comm, sector=sector)

    # print out scaling dimensions and time elapsed
    if rank == 0:
        # end time
        endT = time.time()
        diffT = endT - startT
        print("finished! Time elapsed = {:.2f}".format(diffT))
        if sector[0] == "even":
            print("The scaling dimensions of even operators are:")
            with np.printoptions(precision=5, suppress=True):
                print(scDims)
            print("The Exact values for 3D Ising even sector are")
            with np.printoptions(precision=5, suppress=True):
                print(scDEvenExt)
        if sector[0] == "odd":
            print("The scaling dimensions of odd operators are:")
            with np.printoptions(precision=5, suppress=True):
                print(scDims)
            print("The Exact values for 3D Ising odd sector are")
            with np.printoptions(precision=5, suppress=True):
                print(scDOddExt)
        print("\\--------------------/")
        # save scaling dimensions
        scDFile = tenDir + "/scDim-rg{:02d}-{:s}.pkl".format(rgn, sector[0])
        with open(scDFile, "wb") as f:
            pkl.dump([rgn, scDims, baseEig], f)


def linRG2x(Astar, cgtens, scheme="hotrg3d", ver="base",
            nscaleD=[10, 10], comm=None, sector=[None, None],
            refl_c=[0, 0, 0]):
    """scaling dimensions from linearized RG equation
    Currently only design for 3d HOTRG

    Args:
        Astar (TensorZ2): fixed-point tensor
        cgtens : tensors to coarse grain Astar

    Kwargs:
        scheme (str): tnrg scheme
        ver (str): version
        nscaleD (list): number of scaling dimensions to get
        comm (MPI.COMM_WORLD): for parallelization

    """
    ising3d = tnrg.TensorNetworkRG3D("ising3d")
    if scheme == "hotrg3d":
        linearRGSet, dims_PsiA = hotrg3d.get_linearRG(
            Astar, cgtens, comm=comm)
    elif scheme == "blockHOTRG":
        linearRGSet, dims_PsiA = ising3d.linear_block_hotrg(
            Astar, cgtens, refl_c, comm=comm, isEF=False
        )
    elif scheme == "efrg":
        linearRGSet, dims_PsiA = ising3d.linear_block_hotrg(
            Astar, cgtens, refl_c, comm=comm, isEF=True
        )

    if sector[0] is None:
        # even and odd sector in series
        if (scheme == "hotrg3d") or (refl_c == [0, 0, 0]):
            scDimsEven, baseEig = ising3d.linearRG2scaleD(
                linearRGSet[0], dims_PsiA[0], nscaleD[0], baseEig=None
            )
            scDimsOdd = ising3d.linearRG2scaleD(
                linearRGSet[1], dims_PsiA[1], nscaleD[1], baseEig=baseEig
            )
            scDims = [scDimsEven, scDimsOdd]
            if scheme == "hotrg3d":
                # for usual HOTRG
                return scDims
            else:
                # for reflection-symmetric schemes
                return scDims, baseEig
        else:  # for blockHOTRG and efrg
            scDimsEven = ising3d.linearRG2scaleD(
                linearRGSet[0], dims_PsiA[0], nscaleD[0], baseEig=sector[1]
            )
            scDimsOdd = ising3d.linearRG2scaleD(
                linearRGSet[1], dims_PsiA[1], nscaleD[1], baseEig=sector[1]
            )
            scDims = [scDimsEven, scDimsOdd]
            return scDims
    else:
        # separately (only for hotrg3d now)
        if sector[0] == "even":
            scDimsEven, baseEig = ising3d.linearRG2scaleD(
                linearRGSet[0], dims_PsiA[0], nscaleD, baseEig=None
            )
            return scDimsEven, baseEig
        elif sector[0] == "odd":
            scDimsOdd = ising3d.linearRG2scaleD(
                linearRGSet[1], dims_PsiA[1], nscaleD, baseEig=sector[1]
            )
            return scDimsOdd, sector[1]
        else:
            errMsg = "sector should be even or odd"
            raise ValueError(errMsg)


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


# for plotting
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


def plotscaleD(rgsteps, scDList,
               saveDir, pars):
    # markers
    evenMarker = [".", "o", "+", "+", "+", "s", "s", "s", "s", "s"]
    evenColor = ["k", "k", "b", "b", "b", "k", "k", "k", "k", "k"]
    evenShift = [0, 0, -0.1, 0, 0.1, -0.2, -0.1, 0, 0.1, 0.2]
    oddMarker = ["o", "+", "+", "+", "x", "x", "x", "x", "x", "x"]
    oddColor = ["k", "b", "b", "b", "b", "b", "b", "b", "b", "b"]
    oddShift = [0, -0.1, 0, 0.1, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25]

    plt.figure(figsize=(12, 8))
    ax1 = plt.subplot(211)
    for rgn, scD in zip(rgsteps, scDList):
        for k in range(10):
            ax1.plot(rgn + evenShift[k], scD[0][k],
                     evenColor[k] + evenMarker[k])
    # best-known values
    plt.hlines(1.412625, rgsteps[0]-0.2, rgsteps[-1]+0.2,
               colors="black", linestyles="solid", alpha=0.2)
    plt.hlines(2.412625, rgsteps[0]-0.2, rgsteps[-1]+0.2,
               colors="blue", linestyles="solid", alpha=0.2)
    plt.hlines(3, rgsteps[0]-0.2, rgsteps[-1]+0.2,
               colors="black", linestyles="solid", alpha=0.2)
    # set axis ranges
    plt.xticks(rgsteps)
    if len(rgsteps) == 1:
        plt.xlim([rgsteps[0] - 1, rgsteps[0] + 1])
    plt.ylim([-0.1, 3.4])
    plt.ylabel("Even sector")
    # put explanations on the figure
    plt.text(rgsteps[0] + 0.4, 1.41 - 0.25, r"$\epsilon$", size=14)
    plt.text(rgsteps[0] + 0.4, 3 - 0.25, r"$T_{ij}$", size=14)
    plt.text(rgsteps[0] + 0.38, 2.41 - 0.25, r"$\partial_i \epsilon$",
             size=14, color="blue")
    plt.text(rgsteps[-1] - 1, 0.5,
             r"Bond dimension $\chi$={:d}".format(pars["chi"]),
             fontsize=14)

    ax2 = plt.subplot(212)
    for rgn, scD in zip(rgsteps, scDList):
        for k in range(10):
            ax2.plot(rgn + oddShift[k], scD[1][k],
                     oddColor[k] + oddMarker[k])
    plt.hlines(0.5181489, rgsteps[0]-0.2, rgsteps[-1]+0.2,
               colors="black", linestyles="solid", alpha=0.2)
    plt.hlines(1.5181489, rgsteps[0]-0.2, rgsteps[-1]+0.2,
               colors="blue", linestyles="solid", alpha=0.2)
    plt.hlines(2.5181489, rgsteps[0]-0.2, rgsteps[-1]+0.2,
               colors="blue", linestyles="solid", alpha=0.2)
    plt.xticks(rgsteps)
    plt.ylim([-0.1, 3.0])
    plt.ylabel("Odd sector")
    plt.xlabel("RG step")
    plt.text(rgsteps[0] + 0.4, 0.518 - 0.2, r"$\sigma$", size=14)
    plt.text(rgsteps[0] + 0.35, 1.518 - 0.2, r"$\partial_i\sigma$",
             size=14, color="blue")
    plt.text(rgsteps[0] + 0.3, 2.518 - 0.2,
             r"$\partial_i \partial_j \sigma$", size=14, color="blue")
    if len(rgsteps) == 1:
        plt.xlim([rgsteps[0] - 1, rgsteps[0] + 1])
    plt.savefig(saveDir + "/scDim.png",
                bbox_inches='tight', dpi=300)
