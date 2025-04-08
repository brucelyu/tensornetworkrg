#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : block_rotsym.py
# Author            : Xinliang (Bruce) Lyu <xlyu@ihes.fr>
# Date              : 08.04.2025
# Last Modified Date: 08.04.2025
# Last Modified By  : Xinliang (Bruce) Lyu <xlyu@ihes.fr>
"""
Block-tensor map with the following two lattice symmetry exploited
- lattice reflection
- rotational

Tensor leg order convention is
A[x, y, x', y']
      y
      |
  x'--A--x
      |
      y'

"""

from ncon import ncon


def optProj(A, chi, cg_eps=1e-8):
    """determine the isometric

    Args:
        A (TODO): TODO
        chi (TODO): TODO

    Kwargs:
        cg_eps (TODO): TODO

    Returns: TODO

    """
    pass
