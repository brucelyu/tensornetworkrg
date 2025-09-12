#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : looptnr_rotsym.py
# Author            : Xinliang (Bruce) Lyu <xlyu@ihes.fr>
# Date              : 12.09.2025
# Last Modified Date: 12.09.2025
# Last Modified By  : Xinliang (Bruce) Lyu <xlyu@ihes.fr>
"""
This file contains functions for a modified loop-TNR,
where the lattice reflection and rotational symmetries are exploited.

The method is based on the original scheme proposed in the following paper:
https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.118.110504
Although the above paper has mentioned how to exploit
the lattice rotation symmetry, it seems to us that their ansatz might
contain some additional assumptions about the main tensor being
positive semi-definite in a certain sense.

Our modified scheme is supposed to be more general.
Most of all, the meaning of lattice reflection and rotation symmetry
is more explict and clearer in our scheme.
Our scheme borrows ideas from the following two papers a lot
- Hauru et. al's Gilt:
https://journals.aps.org/prb/abstract/10.1103/PhysRevB.97.045111
- Evenbly's FET:
https://journals.aps.org/prb/abstract/10.1103/PhysRevB.98.085155

Tensor leg order convention is
A[x, y, x', y']
      y
      |
  x'--A--x
      |
      y'
"""
from .block_rotsym import l1Split


# I. Initialize the tensors composing the loop using a local TRG split
def init_trg(A, chi, eps):
    eigv, v, err = l1Split(A, chi, eps)
    # convert the eigvalues array into a
    # diagonal matrix Λ
    Lambda = eigv.diag()
    # Lambda connects with the last leg of v throught its first leg
    # like v @ Lambda
    return v, Lambda, err

















# end of file
