import argparse
import distutils
import json
from collections import OrderedDict, namedtuple
import sys


def arg_parser() -> argparse.ArgumentParser:
    """Configure an argparse parser"""
    
    str2bool = lambda x: distutils.util.strtobool(x)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--project', default='neurips21', type=str)
    parser.add_argument('--cfg', default=None, type=str)
    parser.add_argument('--wandb_freq', default=10, type=int)
    parser.add_argument('--output_dir', default='.', type=str)
    parser.add_argument('--batch_size', default=200, type=int)
    parser.add_argument('--data_dir', default=None, type=str)
    parser.add_argument('--height', default=28, type=int)
    parser.add_argument('--width', default=28, type=int)
    parser.add_argument('--seed', default=10, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--z_dim', default=32, type=int)
    parser.add_argument('--prior_z', default='gaussian 0.0 1.0', type=str,
            help="This is a reparameterisable variable. Options: 'gaussian loc scale';  'dirichlet concentration'; 'onehotcat temperature logit'; 'gaussian-sparsemax-max-ent bit-precision'. Note that 'onehotcat' is Gumbel-Softmax-Straight-Through.")
    parser.add_argument('--y_dim', default=0, type=int)
    parser.add_argument('--prior_f', default='gibbs 0.0', type=str,
            help="This is a truly discrete variable. Options: 'gibbs logit'; 'gibbs-max-ent bit-precision'; 'categorical logit'")
    parser.add_argument('--prior_y', default='dirichlet 1.0', type=str,
            help="This is a reparameterisable variable. Options: 'dirichlet (lower-concentration upper-concentration)'; 'identity'")
    parser.add_argument('--hidden_dec_size', default=500, type=int)
    parser.add_argument('--posterior_z', default='gaussian', type=str,
            help="Options: 'gaussian (lower-scale upper-scale (lower-loc upper-loc))'; 'gaussian-sparsemax (lower-scale upper-scale (lower-loc upper-loc))'")
    parser.add_argument('--posterior_f', default='gibbs -10 10', type=str)
    parser.add_argument('--posterior_y', default='dirichlet 1e-3 1e3', type=str)
    #parser.add_argument('--shared_concentrations', default=True, type=str2bool)
    parser.add_argument('--shared_enc_fy', default=True, type=str2bool)
    parser.add_argument('--mean_field', default=True, type=str2bool)
    parser.add_argument('--hidden_enc_size', default=500, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--training_samples', default=1, type=int)
    parser.add_argument('--num_samples', default=100, type=int)
    parser.add_argument('--lr_warmup', default=0, type=int, 
            help="Set to more than 0 to get a linear warmup from 0 to gen_lr (or inf_lr)")
    parser.add_argument('--gen_opt', default='adam', choices=['adam', 'rmsprop'], type=str)
    parser.add_argument('--gen_lr', default=1e-4, type=float)
    parser.add_argument('--gen_l2', default=0., type=float)
    parser.add_argument('--gen_p_drop', default=0., type=float)
    parser.add_argument('--inf_opt', default='adam', choices=['adam', 'rmsprop'], type=str)
    parser.add_argument('--inf_lr', default=1e-4, type=float)
    parser.add_argument('--inf_l2', default=0., type=float)
    parser.add_argument('--inf_p_drop', default=0.1, type=float)
    parser.add_argument('--grad_clip', default=5., type=float)
    parser.add_argument('--load_ckpt', default=None, type=str)
    parser.add_argument('--reset_opt', default=False, type=str2bool)
    parser.add_argument('--exact_marginal', default=False, type=str2bool)
    parser.add_argument('--exact_KL_Y', default=False, type=str2bool)
    parser.add_argument('--use_self_critic', default=False, type=str2bool)
    parser.add_argument('--use_reward_standardisation', default=False, type=str2bool)
    parser.add_argument('--ppo_like_steps', default=0, type=int)
    parser.add_argument('--tqdm', default=False, type=str2bool)
    parser.add_argument('--gsp_cdf_samples', default=100, type=int)
    parser.add_argument('--gsp_KL_samples', default=1, type=int)

    return parser

def default_cfg() -> dict:
    """Return a dictionary of known options and their default values"""
    return vars(arg_parser().parse_args([]))

def load_cfg(path, **kwargs) -> dict:
    """
    Load arguments from a json cfg file and checks them against known arguments. Options not in the json file will be set to a default value.
    Use kwargs to overwrite the json file.
    """
    with open(path, 'r') as f:
        cfg = json.load(open(path), object_hook=OrderedDict)
    known = default_cfg()
    for k, v in known.items():
        if k not in cfg:
            cfg[k] = v
            print(f"Setting {k} to default {v}", file=sys.stdout)
    for k, v in cfg.items():
        if k not in known:
            raise ValueError(f'Unknown cfg: {k}')
    for k, v in kwargs.items():
        if k in known:
            cfg[k] = v
            print(f"Overriding {k} to user choice {v}", file=sys.stdout)
        else:
            raise ValueError(f"Unknown hparam {k}")
    return cfg

def parse_args():
    """
    Parse arguments with argparse possibly loading some settings from a json file (if --cfg is set). 
    Command line arguments overwrites the cfg file.

    Use this if you want to parse from command line.
    """
   
    parser = arg_parser()
    args = parser.parse_args()
    if args.cfg:
        with open(args.cfg, 'r') as f:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(f))
            args = parser.parse_args(namespace=t_args)
    return args

def make_args(cfg: dict):
    return namedtuple("Config", cfg.keys())(*cfg.values())
