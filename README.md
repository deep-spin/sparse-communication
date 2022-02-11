# Sparse Communication via Mixed Distributions

Official PyTorch implementation. Follow this procedure to replicate the results reported in our paper [Sparse Communication via Mixed Distributions](https://openreview.net/forum?id=WAid50QschI).

**Abstract**:

_Neural networks and other machine learning models compute continuous representations, while humans communicate mostly through discrete symbols. Reconciling these two forms of communication is desirable for generating human-readable interpretations or learning discrete latent variable models, while maintaining end-to-end differentiability. Some existing approaches (such as the Gumbel-Softmax transformation) build continuous relaxations that are discrete approximations in the zero-temperature limit, while others (such as sparsemax transformations and the Hard Concrete distribution) produce discrete/continuous hybrids. In this paper, we build rigorous theoretical foundations for these hybrids, which we call "mixed random variables.'' Our starting point is a new "direct sum'' base measure defined on the face lattice of the probability simplex. From this measure, we introduce new entropy and Kullback-Leibler divergence functions that subsume the discrete and differential cases and have interpretations in terms of code optimality. Our framework suggests two strategies for representing and sampling mixed random variables, an extrinsic ("sample-and-project'') and an intrinsic one (based on face stratification). We experiment with both approaches on an  emergent communication benchmark and on modeling MNIST and Fashion-MNIST data with variational auto-encoders with mixed latent variables._

## Code Organization

We report experiments in three representation learning tasks: an emergent communication game, a bit-vector VAE modeling Fashion-MNIST images, and a mixed-latent VAE modeling MNIST images. You can find the code for the first two experiments in `communication_and_fmnist` and for the third in `vae_mnist`. Each folder contains an indivual README file describing how to reproduce our experiments. Our code is largely inspired by the structure and implementations found in [sparse-marginalization-lvm](https://github.com/deep-spin/sparse-marginalization-lvm), which was partially built upon [EGG](https://github.com/facebookresearch/EGG).

## Citation

```
@inproceedings{farinhas2022sparse,
title={Sparse Communication via Mixed Distributions},
author={Ant{\'o}nio Farinhas and Wilker Aziz and Vlad Niculae and Andr{\'e} F. T. Martins},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=WAid50QschI}
}
```
