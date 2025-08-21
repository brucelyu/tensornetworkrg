#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : fet2d_reflsym.py
# Author            : Xinliang (Bruce) Lyu <xlyu@ihes.fr>
# Date              : 21.08.2025
# Last Modified Date: 21.08.2025
# Last Modified By  : Xinliang (Bruce) Lyu <xlyu@ihes.fr>
"""
This module implements a Entanglement Filtering;
the lattice reflection symmetry is exploited

The scheme is a combination of
- Evenbly's full environment truncation (FET)
- Graph-Independent Local Truncations (Gilt) by
Hauru, Delcamp, and Mizera

Due to the lattice reflection symmetry,
there are only two filtering matrix sx, sy

The relevant environment is constructed in the file `./env2d_reflsym.py`
Some functions in the file `./env2d_reflsym.py` that only exploits
the lattice reflection are also reused here.
"""
from .. import u1ten
from . import env2d_rotsym, env2d_reflsym
from ncon import ncon

