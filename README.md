# Sparse Communication via Mixed Distributions

Official PyTorch implementation. Follow this procedure to replicate the results reported in our paper [Sparse Communication via Mixed Distributions](https://openreview.net/forum?id=WAid50QschI).

**Abstract**:

_Neural networks and other machine learning models compute continuous representations, while humans communicate mostly through discrete symbols. Reconciling these two forms of communication is desirable for generating human-readable interpretations or learning discrete latent variable models, while maintaining end-to-end differentiability. Some existing approaches (such as the Gumbel-Softmax transformation) build continuous relaxations that are discrete approximations in the zero-temperature limit, while others (such as sparsemax transformations and the Hard Concrete distribution) produce discrete/continuous hybrids. In this paper, we build rigorous theoretical foundations for these hybrids, which we call "mixed random variables.'' Our starting point is a new "direct sum'' base measure defined on the face lattice of the probability simplex. From this measure, we introduce new entropy and Kullback-Leibler divergence functions that subsume the discrete and differential cases and have interpretations in terms of code optimality. Our framework suggests two strategies for representing and sampling mixed random variables, an extrinsic ("sample-and-project'') and an intrinsic one (based on face stratification). We experiment with both approaches on an  emergent communication benchmark and on modeling MNIST and Fashion-MNIST data with variational auto-encoders with mixed latent variables._


## Citation

```
@inproceedings{farinhas2022sparse,
title={Sparse Communication via Mixed Distributions},
author={Ant{\'o}nio Farinhas and Wilker Aziz and Vlad Niculae and Andre Martins},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=WAid50QschI}
}
```
