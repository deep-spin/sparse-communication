import tempfile
import wandb
import sys
import json
import pathlib
import random
from shutil import copyfile
import torch
import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn
from collections import namedtuple, defaultdict, OrderedDict, deque
from tabulate import tabulate
from tqdm.auto import tqdm
from itertools import chain
from components import GenerativeModel, InferenceModel, VAE
from data import load_mnist
from data import Batcher
import hparams
from scheduler import get_constant_schedule_with_warmup
from typing import Iterable


def get_optimiser(choice: str, params: Iterable[torch.Tensor], lr: float, weight_decay: float, momentum: float = 0.0):    
    """Return an optimizer for the parameters in the argument"""
    if choice == 'adam':
        opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif choice == 'rmsprop':
        opt = torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        raise ValueError("Unknown optimiser")
    return opt


def make_state(args: namedtuple, device: str, ckpt_path: str = None, load_opt=True):
    """
    Parameters:

        args (namedtuple): all hyperparameters
        device (str): name of the device where to load the model 
        ckpt_path (str): use None to get an untrained model
        load_opt (bool): whether to also load the optimizer state

    Return: 
        State namedtuple with the following attributes
            vae (VAE): a VAE object
            p (GenerativeModel): a generative model
            q (InferenceModel): an inference model
            p_opt (torch.optim.Adam or similar): optimizer for generative model
            q_opt (torch.optim.Adam or similar): optimizer for inference model
            stats_tr (dict of lists): training statistics
            stats_val (dict of lists): validation statistics
            args (namedtuple): all hyperparameters
            
    """

    if ckpt_path and not pathlib.Path(ckpt_path).exists():
        raise ValueError(f"I cannot find {ckpt_path}. Use None to get an untrained model.")

    p = GenerativeModel(    
        y_dim=args.y_dim,
        z_dim=args.z_dim, 
        data_dim=args.height * args.width, 
        hidden_dec_size=args.hidden_dec_size,
        p_drop=args.gen_p_drop,
        prior_f=args.prior_f,
        prior_y=args.prior_y,
        prior_z=args.prior_z,
    )
    
    q = InferenceModel(    
        y_dim=args.y_dim,
        z_dim=args.z_dim, 
        data_dim=args.height * args.width, 
        hidden_enc_size=args.hidden_enc_size,
        #shared_concentrations=args.shared_concentrations,
        shared_enc_fy=args.shared_enc_fy,
        p_drop=args.inf_p_drop,
        posterior_f=args.posterior_f,
        posterior_y=args.posterior_y,
        posterior_z=args.posterior_z,
        mean_field=args.mean_field,
        gsp_cdf_samples=args.gsp_cdf_samples,
        gsp_KL_samples=args.gsp_KL_samples,
    )
    
    # Checkpoints
    
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device)
        p.load_state_dict(ckpt['p_state_dict'])            
        p = p.to(torch.device(device))
        q.load_state_dict(ckpt['q_state_dict'])            
        q = q.to(torch.device(device))
        # Get optimisers
        p_opt = get_optimiser(args.gen_opt, p.parameters(), args.gen_lr, args.gen_l2)
        q_opt = get_optimiser(args.inf_opt, q.parameters(), args.inf_lr, args.inf_l2)
        if load_opt:
            p_opt.load_state_dict(ckpt['p_opt_state_dict'])
            q_opt.load_state_dict(ckpt['q_opt_state_dict'])
        stats_tr = ckpt['stats_tr']
        stats_val = ckpt['stats_val']
    else:
        p = p.to(torch.device(device))
        q = q.to(torch.device(device))
        # Get optimisers
        p_opt = get_optimiser(args.gen_opt, p.parameters(), args.gen_lr, args.gen_l2)
        q_opt = get_optimiser(args.inf_opt, q.parameters(), args.inf_lr, args.inf_l2)
        stats_tr = defaultdict(list)
        stats_val = defaultdict(list)

    # Training 
    
    vae = VAE(p, q,
        use_self_critic=args.use_self_critic, 
        use_reward_standardisation=args.use_reward_standardisation
    )

    state = OrderedDict(vae=vae, p=p, q=q, p_opt=p_opt, q_opt=q_opt, stats_tr=stats_tr, stats_val=stats_val, args=args)
    return namedtuple("State", state.keys())(*state.values())


def get_batcher(data_loader, args, binarize=True, num_classes=10, onehot=True):
    """Return """
    batcher = Batcher(
        data_loader, 
        height=args.height, 
        width=args.width, 
        device=torch.device(args.device), 
        binarize=binarize, 
        num_classes=num_classes,
        onehot=onehot
    )
    return batcher


def validate(vae: VAE, batcher: Batcher, num_samples: int, compute_DR=False, exact_KL_Y=False, progressbar=False):
    """
    Return average NLL
        average number of bits per dimension
        and a dictionary with distortion and rate estimates
    """
    with torch.no_grad():
        vae.eval()
        
        nb_obs = 0
        nb_bits = 0.
        ll = 0.
        DR = OrderedDict()
        iterator = tqdm(batcher) if progressbar else iter(batcher)
        for x_obs, y_obs in iterator:
            # [B, H*W]
            x_obs = x_obs.reshape(-1, vae.p.data_dim)     
            # [B]
            ll = ll + vae.estimate_ll(x_obs, num_samples).sum(0)
            nb_bits += np.prod(x_obs.shape)
            nb_obs += x_obs.shape[0]
            # []
            if compute_DR:
                ret = vae.DR(x_obs, exact_KL_Y=exact_KL_Y)
                for k, v in ret.items():
                    if k not in DR:
                        DR[k] = []
                    DR[k].append(v.cpu().numpy())

    nll = - (ll / nb_obs).cpu()
    if compute_DR:
        DR = OrderedDict((k, np.concatenate(v, 0)) for k, v in DR.items())    
        return nll, nll / np.log(2) / vae.p.latent_dim, DR
    else:
        return nll, nll / np.log(2) / vae.p.latent_dim

def save_state(state, path):
    torch.save({
        'p_state_dict': state.p.state_dict(),
        'q_state_dict': state.q.state_dict(),
        'p_opt_state_dict': state.p_opt.state_dict(),
        'q_opt_state_dict': state.q_opt.state_dict(),
        'stats_tr': state.stats_tr,
        'stats_val': state.stats_val,
    }, path)

def main(args: namedtuple):
    
    # Convert to dict (needed for wandb)
    cfg = vars(args)
    
    with wandb.init(config=cfg, project=args.project):
        args = wandb.config
        # Make the wandb name part of the output directory
        cfg['output_dir'] = f"{cfg['output_dir']}/{wandb.run.name}"
        args.update({'output_dir': cfg['output_dir']}, allow_val_change=True)
        output_dir = pathlib.Path(args.output_dir)
        print(f"# Setup\nOutput directory: {output_dir}", file=sys.stdout)
        # Make dirs
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save hparams
        json.dump(cfg, open(f"{output_dir}/cfg.json", "w"), indent=4)
        print(f"Config file: {output_dir}/cfg.json", file=sys.stdout)
        # Reproducibility
        print(f"# Reproducibility\nSetting random seed to {args.seed}", file=sys.stdout)
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)    
        
        # Preparing data
        print("# Preparing MNIST (may take some time the first time)", file=sys.stdout)
        train_loader, valid_loader, test_loader = load_mnist(
            args.batch_size, 
            save_to=args.data_dir if args.data_dir else tempfile.mkdtemp(),
            height=args.height, 
            width=args.width
        )

        print("# Building model", file=sys.stdout)
        state = make_state(
            args, 
            device=args.device, 
            ckpt_path=args.load_ckpt,
            load_opt=not args.reset_opt
        )
        #print('Generative model\n', state.p)
        #print('Inference model\n', state.q)

        # Training 
        
        # Demo 
        print("Example:\n 2 samples from inference model", file=sys.stdout)
        print(state.q.sample(state.p.sample((2,))[-1]), file=sys.stdout)
        print(" their log probability density under q", file=sys.stdout)
        print(state.q.log_prob(
            torch.zeros(args.height * args.width, device=torch.device(args.device)),
            *state.q.sample(state.p.sample((2,))[-1]), 
        ), file=sys.stdout)    
       
        
        print("# Checking initial model", file=sys.stdout)
        val_metrics = validate(state.vae, get_batcher(valid_loader, args), args.num_samples, 
                compute_DR=True, exact_KL_Y=args.exact_KL_Y, progressbar=args.tqdm)
        dr_string = ' '.join(f"{k}={v.mean():.2f}" for k, v in val_metrics[2].items())
        print(f'Validation {0:3d}: nll={val_metrics[0]:.2f} bpd={val_metrics[1]:.2f} {dr_string}', file=sys.stdout)

        best_val_nll = np.min(state.stats_val['val_nll']) if state.stats_val['val_nll'] else np.inf

        if args.lr_warmup > 0:
            p_scheduler = get_constant_schedule_with_warmup(state.p_opt, num_warmup_steps=args.lr_warmup)
            q_scheduler = get_constant_schedule_with_warmup(state.q_opt, num_warmup_steps=args.lr_warmup)
        else:
            p_scheduler, q_scheduler = None, None

        print("# Training", file=sys.stdout)
        for epoch in range(args.epochs):

            if args.tqdm:
                iterator = tqdm(get_batcher(train_loader, args))
            else:
                iterator = iter(get_batcher(train_loader, args))

            for i, (x_obs, c_obs) in enumerate(iterator):
                # [B, H*W]
                x_obs = x_obs.reshape(-1, args.height * args.width)
                ret = None
                for k in range(args.ppo_like_steps + 1):
                    state.vae.train()      
                    #if i % 20 == 0:
                    #    samples, images = dict(), dict()
                    #else:
                    #    samples, images = None, None
                    samples, images = None, None

                    loss, ret = state.vae.loss(x_obs, c_obs, 
                        num_samples=args.training_samples, samples=samples, images=images, 
                        exact_marginal=args.exact_marginal,
                        exact_KL_Y=args.exact_KL_Y,
                    )
                    for k, v in ret.items():
                        state.stats_tr[k].append(v)

                    state.p_opt.zero_grad()
                    state.q_opt.zero_grad()        
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        chain(state.vae.gen_parameters(), state.vae.inf_parameters()), 
                        args.grad_clip
                    )        
                    state.q_opt.step()
               
                state.p_opt.step()

                if p_scheduler:
                    p_scheduler.step()
                if q_scheduler:
                    q_scheduler.step()

                if args.tqdm:
                    iterator.set_description(f'Epoch {epoch+1:3d}/{args.epochs}')
                    iterator.set_postfix(ret)

                if i % args.wandb_freq == 0:
                    wandb.log({f"training.{k}": v for k, v in ret.items()}, commit=False)
                    wandb.log({f"gen.lr{i}": float(param_group['lr']) for i, param_group in enumerate(state.p_opt.param_groups)}, commit=False)
                    wandb.log({f"inf.lr{i}": float(param_group['lr']) for i, param_group in enumerate(state.q_opt.param_groups)}, commit=False)
                    if samples:
                        wandb.log({f"training.{k}": v for k, v in samples.items()}, commit=False)
                    if images:
                        wandb.log({f"training.{k}": wandb.Image(v) for k, v in images.items()})

            val_metrics = validate(state.vae, get_batcher(valid_loader, args), args.num_samples, 
                    compute_DR=True, exact_KL_Y=args.exact_KL_Y, progressbar=args.tqdm)
    

            state.stats_val['val_nll'].append(val_metrics[0].item())
            state.stats_val['val_bpd'].append(val_metrics[1].item())
            for k, v in val_metrics[2].items():
                state.stats_val[f'val_{k}'].append(v.mean().item())
            dr_string = ' '.join(f"{k}={v.mean():.2f}" for k, v in val_metrics[2].items())
            print(f'Validation {epoch+1:3d}: nll={val_metrics[0]:.2f} bpd={val_metrics[1]:.2f} {dr_string}', file=sys.stdout)
            
            
            wandb.log({'val.nll': val_metrics[0], 'val.bpd': val_metrics[1]}, commit=False)
            wandb.log({f"val.{k}": v for k, v in val_metrics[2].items()})
            
            save_state(state, output_dir/"ckpt.last")
            current_val_nll = state.stats_val['val_nll'][-1]
            if current_val_nll < best_val_nll:
                print(f"Saving new best model (val_nll): old={best_val_nll} new={current_val_nll}")
                best_val_nll = current_val_nll 
                save_state(state, output_dir/"ckpt.best")

        
        print("# Final validation run")
        val_nll, val_bpd, val_DR = validate(
            state.vae, get_batcher(valid_loader, args), args.num_samples, 
            compute_DR=True, exact_KL_Y=args.exact_KL_Y, progressbar=args.tqdm)
        rows = [('NLL', val_nll, None), ('BPD', val_bpd, None)]
        for k, v in val_DR.items():
            rows.append((k, v.mean(), v.std()))
        print(tabulate(rows, headers=['metric', 'mean', 'std']))    
        return rows, ['metric', 'mean', 'std']
        

if __name__ == '__main__':
    main(hparams.parse_args())
            
