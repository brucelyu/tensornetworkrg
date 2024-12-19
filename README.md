# Tensor network renormalization group
This is a package for tensor network renormalization group (TNRG) methods.
It implements several 2D and 3D TNRG schemes, as well as useful procedures in TNRG calculations.

## A list of TNRG schemes implemented
For 2D,
- [Tensor Renormalization Group (TRG)](https://arxiv.org/abs/cond-mat/0611687), not in its original form, but following [Evenbly's implementation](https://www.tensors.net/p-trg) closely.
- [Higher-order Tensor Renormalization Group (HOTRG)](https://arxiv.org/abs/1201.1144). Instead of using the higher-order singular value decomposition, we use Evenlby's projective truncations described in [this paper](https://arxiv.org/abs/1509.07484) to formulate HOTRG and implement it.
- Evenly and Vidal's [Tensor Network Renormalization (TNR)](https://arxiv.org/abs/1509.07484), following [Evenbly's implementation](https://www.tensors.net/p-tnr) closely.

The first two schemes are different realizations of the block-tensor map in 2D.
The last one is enhanced by entanglement filtering (or disentanglers) and goes beyond simple block-tensor map.

For 3D,
- [Higher-order Tensor Renormalization Group (HOTRG)](https://arxiv.org/abs/1201.1144). Again, we use Evenlby's projective truncations described in [this paper](https://arxiv.org/abs/1509.07484) to implement it.
- A HOTRG-like block-tensor map. It is similar to the HOTRG, but uses different isometric tensors for inner and outer legs, as is described in [our preprint](https://arxiv.org/abs/2412.13758). It is more suitable than the HOTRG when an entanglement filtering process is incorporated.

## Useful procedures in TNRG calculations

## Description of files
