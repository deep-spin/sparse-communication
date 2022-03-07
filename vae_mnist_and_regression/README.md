VAEs with mixed random variables.

# Dependencies

* Torch and CUDA version: `torch==1.9.1+cu111` or `torch==1.9.1+cpu`

```bash
virtualenv -p python3 ~/envs/mixedrv
source ~/envs/mixed/bin/activate

pip install -r requirements.txt
```

* See `packages.txt` for a list of python packages (from `yolk -l`) in our virtual environment


# Training

```bash
# Gaussian VAE
WANDB_MODE=offline python main.py --cfg cfg/iclr-gaussian.json --batch_size 100 --device cuda:0 --seed 10

# Dirichlet VAE
WANDB_MODE=offline python main.py --cfg cfg/iclr-dirichlet.json --batch_size 100 --device cuda:0 --seed 10

# Mixed Dirichlet VAE
WANDB_MODE=offline python main.py --cfg cfg/iclr-mixed-dirichlet.json --batch_size 100 --device cuda:0 --seed 10

# Gaussian-sparsemax VAE
WANDB_MODE=offline python main.py --cfg cfg/iclr-gaussiansp.json --batch_size 100 --device cuda:0 --seed 10

# Categorical "VAE" (mixture model)
WANDB_MODE=offline python main.py --cfg cfg/iclr-categorical.json --batch_size 100 --device cuda:0 --seed 10

# Gumbel-Softmax Straight-Through (or one-hot-categorical) "VAE" (a mixture model with biased gradient training)
WANDB_MODE=offline python main.py --cfg cfg/iclr-onehotcat.json --batch_size 100 --device cuda:0 --seed 10

```


# Download Trained Models

```bash
wget https://tinyurl.com/mixedrvs -O trained_models.tar
tar -xvf trained_models.tar
```

# Analysis

