# Sparse Communication via Mixed Distributions

PyTorch implementation of the emergent communication game and the bit-vector VAE experiments. Follow this procedure to replicate the results reported in our paper.


## Python requirements and installation

To install, follow these steps:

1. In a virtual environment, first install Cython: `pip install cython`
2. Clone the [Eigen](https://gitlab.com/libeigen/eigen) repository to your home: `git clone git@gitlab.com:libeigen/eigen.git`
3. Clone the [LP-SparseMAP](https://github.com/deep-spin/lp-sparsemap) repository to your home, and follow the installation instructions found there
4. Install PyTorch: `pip install torch`
5. Install the requirements: `pip install -r requirements.txt`
6. Install the `lvm-helpers` package: `pip install .` (or in editable mode if you want to make changes: `pip install -e .`)

## Datasets

To get the dataset for the emergent communication game, please visit: https://github.com/DianeBouchacourt/SignalingGame. After getting the data, store the `train` and `test` folders under `data/signal-game`. For the bit-vector VAE, FMNIST is downloaded automatically by running the training commands for the first time.

## Running

**Training**:

To train a model for the emergent communication game you should run the corresponding script available in `training_scripts/signal-game`. For the bit-vector VAE experiment, you should replace `signal-game` by `bit_vector`.


**Evaluating**:

For the emergent communication game, to evaluate any trained network on the test set, run:

```
python experiments/signal-game/test.py /path/to/checkpoint/ /path/to/hparams.yaml
```

Replace `signal-game` by `bit_vector-vae` to get test results for the bit-vector VAE experiment. Checkpoints should be found in the appropriate folder inside the automatically generated `checkpoints` directory, and the `yaml` file should be found in the model's automatically generated directory inside `logs`.

The evaluation results should match the paper.

