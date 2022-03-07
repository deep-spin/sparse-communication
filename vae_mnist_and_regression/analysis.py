import torch
import numpy as np
import torch
import torch.distributions as td
import matplotlib.pyplot as plt
import torch.nn as nn
from collections import namedtuple, OrderedDict, defaultdict
from itertools import chain


def bitvec2str(f, as_set=False):
    return ''.join('1' if b else '0' for b in f) if not as_set else '{' + ','.join(f'{i:1d}' for i, b in enumerate(f, 1) if b) + '}'


def probe_corners(vae, batcher, args, num_samples=None): 
    if num_samples is None:
        num_samples = args.num_samples

    with torch.no_grad():
        vae.eval()        
    
        samples = [defaultdict(list) for _ in range(vae.p.y_dim)]

        for i in range(vae.p.y_dim):
            f = torch.zeros(vae.p.y_dim, device=torch.device(args.device))
            f[i] = 1.0
            # [S, K]
            f = f.expand((num_samples, vae.p.y_dim))
            y = vae.p.Y(f).sample()
            # [B, H]
            z = vae.p.Z().expand((num_samples,)).sample()
            x = vae.p.X(z=z, y=y).sample()
                        
            # [B, K]
            samples[i]['y'] = y.cpu().numpy()
            samples[i]['z'] = z.cpu().numpy()
            samples[i]['x'] = x.cpu().numpy()

    return samples


def probe_prior(vae, num_batches, batch_size, data_sample=False): 

    with torch.no_grad():
        vae.eval()        
    
        prior = defaultdict(list)

        # Some visualisations
        for _ in range(num_batches):
            
            
            B, H, K, D = batch_size, vae.p.z_dim, vae.p.y_dim, vae.p.data_dim            
                    

            # [B, K]
            f = vae.p.F().expand((B,)).sample()
            y = vae.p.Y(f).sample()
            # [B, H]
            z = vae.p.Z().expand((B,)).sample()

            # [B, K]
            prior['f'].append(f.cpu().numpy())
            # [B, K]
            prior['y'].append(y.cpu().numpy())
            # [B, K]
            prior['z'].append(z.cpu().numpy())

            if data_sample:
                x = vae.p.X(z=z, y=y).sample()
                prior['x'].append(x.cpu().numpy())

    return prior


def collect_samples(vae, batcher, args, exact_marginals=False, num_samples=None, from_prior=True, from_posterior=True, data_sample=False): 
    if num_samples is None:
        num_samples = args.num_samples

    with torch.no_grad():
        vae.eval()        
    
        prior = defaultdict(list)
        posterior = defaultdict(list)
        num_obs = 0

        # Some visualisations
        for x_obs, y_obs in batcher:
            
            # [B, H*W]
            x_obs = x_obs.reshape(-1, args.height * args.width)
            num_obs += x_obs.shape[0]

            # [B, 10]
            context = None
            
            B, H, K, D = x_obs.shape[0], vae.p.z_dim, vae.p.y_dim, vae.p.data_dim            
                    
            if from_prior:
                # [B, K]
                f = vae.p.F().expand((B,)).sample()
                y = vae.p.Y(f).sample()
                # [B, H]
                z = vae.p.Z().expand((B,)).sample()

                # [B, K]
                prior['f'].append(f.cpu().numpy())
                # [B, K]
                prior['y'].append(y.cpu().numpy())
                # [B, K]
                prior['z'].append(z.cpu().numpy())
                            #
                if data_sample:
                    x = vae.p.X(z=z, y=y).sample()
                    prior['x'].append(x.cpu().numpy())
            
            if from_posterior:
                # [B, K], [B, K], [B, H]
                f, y, z = vae.q.sample(x_obs)            
                # [B, K]
                posterior['f'].append(f.cpu().numpy())
                # [B, K]
                posterior['y'].append(y.cpu().numpy())
                # [B, H]
                posterior['z'].append(z.cpu().numpy())
                if data_sample:
                    x = vae.p.X(z=z, y=y).sample()
                    posterior['x'].append(x.cpu().numpy())

    return prior, posterior


def compare_marginals(vae, batcher, args, cols=5, exact_marginals=False, num_samples=None): 
    if num_samples is None:
        num_samples = args.num_samples

    with torch.no_grad():
        vae.eval()        
    
        prior = defaultdict(list)
        posterior = defaultdict(list)
        other = defaultdict(list)
        num_obs = 0

        # Some visualisations
        for x_obs, y_obs in batcher:
            
            # [B, H*W]
            x_obs = x_obs.reshape(-1, args.height * args.width)
            num_obs += x_obs.shape[0]

            # [B, 10]
            context = None
            
            B, H, K, D = x_obs.shape[0], vae.p.z_dim, vae.p.y_dim, vae.p.data_dim            
                        
            # [B, K]
            f = vae.p.F().expand((B,)).sample()
            y = vae.p.Y(f).sample()
            # [B, H]
            z = vae.p.Z().expand((B,)).sample()
            #x = vae.p.X(z=z, y=y).sample()
                        
            # [B, K]
            prior['f'].append(f.cpu().numpy())
            # [B]
            prior['dim'].append(f.sum(-1).cpu().numpy())
            # [B, K]
            prior['y'].append(y.cpu().numpy())
            # [B, K]
            prior['z'].append(z.cpu().numpy())
            
            # [B, K], [B, K], [B, H]
            f, y, z = vae.q.sample(x_obs)            
            # [B, K]
            posterior['f'].append(f.cpu().numpy())
            # [B]
            posterior['dim'].append(f.sum(-1).cpu().numpy())
            # [B, K]
            posterior['y'].append(y.cpu().numpy())
            # [B, H]
            posterior['z'].append(z.cpu().numpy())
            
            # [B]            
            other['KL_F'].append(td.kl_divergence(vae.q.F(x_obs), vae.p.F().expand((B,))).cpu().numpy())
            other['KL_Y'].append(td.kl_divergence(vae.q.Y(x=x_obs, f=f), vae.p.Y(f)).cpu().numpy())
            other['KL_Z'].append(td.kl_divergence(vae.q.Z(x=x_obs, y=y), vae.p.Z().expand((B,))).cpu().numpy())
            
        # KLs
        print("For a trained VAE: ")
        print(" 1. We want to see that KL(Z|x || Z), KL(F|x || F), and KL(Y|f,x || Y|f) are generally > 0 for any x ~ D.")
        
        if vae.p.z_dim:
            _ = plt.hist(np.concatenate(other['KL_Z'], 0), bins=20)
            _ = plt.xlabel(r'$KL( Z|x,\lambda || Z| \theta )$')
            plt.show()

        if vae.p.y_dim:
            _ = plt.hist(np.concatenate(other['KL_F'], 0), bins=20)
            _ = plt.xlabel(r'$KL( F|x,\lambda || F| \theta )$')
            plt.show()

            _ = plt.hist(np.concatenate(other['KL_Y'], 0), bins=20)
            _ = plt.xlabel(r'$KL( Y|f,x,\lambda || Y|f, \theta )$')
            plt.show()
            
        
        print(" 2. But, marginally, we expect E_X[Z|X] ~ Z E_X[F|X] ~ F and E_FX[Y|F,X] ~ E_F[Y|F].")
        
        if vae.p.z_dim:
            _ = plt.hist(np.concatenate(prior['z'], 0).flatten(), bins=30, density=True, alpha=0.3, label='Z')
            _ = plt.hist(np.concatenate(posterior['z'], 0).flatten(), bins=30, density=True, alpha=0.3, label='E[Z|X,Y]')
            _ = plt.xlabel(r'$Z_d$')
            _ = plt.legend()
            plt.show()
            
            fig, axs = plt.subplots(
                vae.p.z_dim//cols + vae.p.z_dim%cols, cols, 
                sharex=True, sharey=True,
                gridspec_kw={'hspace': 0, 'wspace': 0})
            if len(axs.shape) == 1:
                axs = axs[None,:]
            for c in range(vae.p.z_dim):
                axs[c // cols, c % cols].hist(np.concatenate(prior['z'], 0)[:,c], bins=30, density=True, alpha=0.3, label='Z')
                axs[c // cols, c % cols].hist(np.concatenate(posterior['z'], 0)[:,c], bins=30, density=True, alpha=0.3, label='E[Z|F,X]')
                #axs[c // 5, c % 5].set_title(f"X'|X={c}")
            #for ax in axs.flat:
            #    ax.set_xticks([])
            #    ax.set_yticks([])
            # Hide x labels and tick labels for top plots and y ticks for right plots.
            #for ax in axs.flat:
            #    ax.label_outer()
            #_ = fig.suptitle(r'$Y \overset{?}{\sim} E[Y|F,X_{obs}, \lambda, \theta]$')    
            plt.legend(bbox_to_anchor=(1, 0.85), loc='upper left')
            plt.show()

        if vae.p.y_dim:
            _ = plt.hist(np.concatenate(prior['y'], 0).flatten(), bins=30, density=True, alpha=0.3, label='Y')
            _ = plt.hist(np.concatenate(posterior['y'], 0).flatten(), bins=30, density=True, alpha=0.3, label='E[Y|F,X]')
            _ = plt.xlabel(r'$Y_k$')
            _ = plt.legend()
            plt.show()
            
            fig, axs = plt.subplots(
                vae.p.y_dim//cols + vae.p.y_dim%cols, cols, 
                sharex=True, sharey=True,
                gridspec_kw={'hspace': 0, 'wspace': 0})            
            if len(axs.shape) == 1:
                axs = axs[None,:]
            for c in range(vae.p.y_dim):
                axs[c // cols, c % cols].hist(np.concatenate(prior['y'], 0)[:,c], bins=30, density=True, alpha=0.3, label='Y')
                axs[c // cols, c % cols].hist(np.concatenate(posterior['y'], 0)[:,c], bins=30, density=True, alpha=0.3, label='E[Y|F,X]')
                #axs[c // 5, c % 5].set_title(f"X'|X={c}")
            #for ax in axs.flat:
            #    ax.set_xticks([])
            #    ax.set_yticks([])
            # Hide x labels and tick labels for top plots and y ticks for right plots.
            #for ax in axs.flat:
            #    ax.label_outer()
            #_ = fig.suptitle(r'$Y \overset{?}{\sim} E[Y|F,X_{obs}, \lambda, \theta]$')    
            plt.legend(bbox_to_anchor=(1, 0.85), loc='upper left')
            plt.show()
            
            # Pr(F_k = 1) compared to E_X[ I[F_k = 1] ]
            if exact_marginals:
                _ = plt.imshow(
                    np.stack([vae.p.F().marginals().cpu().numpy(), np.concatenate(prior['f'], 0).mean(0), np.concatenate(posterior['f'], 0).mean(0)]),     
                    interpolation='nearest',
                )
                _ = plt.colorbar()
                _ = plt.xlabel('k')
                _ = plt.yticks([0, 1, 2], ['F', 'E[F]', 'E[F|X]'])
                _ = plt.title(r'Marginal probability that $F_k = 1$')
                plt.show()

            else:
                _ = plt.imshow(
                    np.stack([np.concatenate(prior['f'], 0).mean(0), np.concatenate(posterior['f'], 0).mean(0)]), 
                    interpolation='nearest',
                )
                _ = plt.colorbar()
                _ = plt.xlabel('k')
                _ = plt.yticks([0, 1], ['E[F]', 'E[F|X]'])
                _ = plt.title(r'Marginal probability that $F_k = 1$')
                plt.show()

            # Y_k compared to E_X[Y_k]

            _ = plt.imshow(
                np.stack([np.concatenate(prior['y'], 0).mean(0), np.concatenate(posterior['y'], 0).mean(0)]), 
                interpolation='nearest'
            )
            _ = plt.colorbar()
            _ = plt.xlabel('k')
            _ = plt.yticks([0, 1], ['E[Y|F]', 'E[Y|F,X]'])
            _ = plt.title(r'Average $Y_k$')
            plt.show()

            _ = plt.hist(
                np.concatenate(prior['dim'], 0), 
                alpha=0.3, label='dim(F)', bins=np.arange(0, 11))
            _ = plt.hist(
                np.concatenate(posterior['dim'], 0), 
                alpha=0.3, label='E[dim(F)|X]', bins=np.arange(0, 11))
            _ = plt.ylabel(f'Count')
            _ = plt.xlabel('dim(f)+1')
            _ = plt.xticks(np.arange(1,11), np.arange(1,11))
            _ = plt.title(f'Distribution of dim(f)')
            _ = plt.legend()
            plt.show()

def compare_samples(vae, batcher, args, N=4, num_figs=1, num_samples=None, prior_samples=False, filename=None): 

    assert N <= args.batch_size, "N should be no bigger than a batch"
    if num_samples is None:
        num_samples = args.num_samples

    with torch.no_grad():
        vae.p.eval()        
        vae.q.eval()
            
        # Some visualisations
        for r, (x_obs, y_obs) in enumerate(batcher, 1):
            #plt.figure(figsize=(2*N, 2*N))
            #plt.subplots_adjust(wspace=0.5, hspace=0.5)        
        
            
            # [B, H*W]
            x_obs = x_obs.reshape(-1, args.height * args.width)
            x_obs = x_obs[:N]
            # [B, 10]
            context = None
            
            B, H, K, D = x_obs.shape[0], vae.p.z_dim, vae.p.y_dim, vae.p.data_dim            
            # marginal probability
            prob = vae.estimate_ll_per_bit(x_obs, num_samples).exp()            
            # posterior samples
            f, y, z = vae.q.sample(x_obs)
            x = vae.p.X(y=y, z=z).sample()
            # prior samples
            f_, y_, z_, x_ = vae.p.sample((N,))

            fig, axs = plt.subplots(3 + int(prior_samples), N, figsize=(2*N, 2*N))
            
            for i in range(N):
                #plt.subplot(4, N, fig0*N + i + 1)
                axs[0, i].imshow(x_obs[i].reshape(args.height, args.width).cpu(), cmap='Greys')
                axs[0, i].set_title("$x^{(%d)}$" % (i+1))

                #plt.subplot(4, N, 1*N + i + 1)
                axs[1, i].imshow(x[i].reshape(args.height, args.width).cpu(), cmap='Greys')
                axs[1, i].set_title("$p(x^{(%d)})$" % (i+1))
                
                #plt.subplot(4, N, 2*N + i + 1)                
                #plt.axhline(y=args.height//2, c='red', linewidth=1, ls='--')
                axs[2, i].imshow(x[i].reshape(args.height, args.width).cpu(), cmap='Greys')
                axs[2, i].set_title("X,Y,F|$x^{(%d)}$" % (i+1))
                #if 0 < vae.p.y_dim <= 10:
                #    plt.xlabel(f'f={bitvec2str(f[i])}')
                
                if prior_samples:
                    #plt.subplot(4, N, 3*N + i + 1)
                    axs[3, i].imshow(x_[i].reshape(args.height, args.width).cpu(), cmap='Greys')
                    axs[3, i].set_title("X,Y,F")
                    #if 0 < vae.p.y_dim <= 10:
                    #    plt.xlabel(f'f={bitvec2str(f[i])}')
            
            #plt.show()
            if filename is not None:
                fig.savefig(f'{filename}-{r}.pdf')
                
            if r == num_figs:
                break 
                
                
def samples_per_digit(vae, batcher, args, return_marginal=False): 

    with torch.no_grad():
        vae.p.eval()        
        vae.q.eval()
            
        groups_f = defaultdict(list)
        groups_y = defaultdict(list)
        groups_z = defaultdict(list)
        groups_x = defaultdict(list)
        groups_marginal_f = defaultdict(list)
        groups_scores = defaultdict(list)
        groups_concs = defaultdict(list)
        # Some visualisations
        for r, (x_obs, c_obs) in enumerate(batcher, 1):
            
            # [B, H*W]
            x_obs = x_obs.reshape(-1, args.height * args.width)
            
            B, H, K, D = x_obs.shape[0], vae.p.z_dim, vae.p.y_dim, vae.p.data_dim            
            # posterior samples
            f, y, z = vae.q.sample(x_obs)
            x = vae.p.X(y=y, z=z).sample()
            
            if return_marginal:
                # [sample_shape, B, K]
                F = vae.q.F(x)
                marginal_f = F.marginals()
                scores = F.scores
                # [sample_shape, K]
                Y = vae.q.Y(f=f, x=x)
                concs = Y.concentration
            else:
                marginal_f = torch.zeros_like(f)
                scores = torch.zeros_like(f)
                concs = torch.zeros_like(f)

            for n in range(x_obs.shape[0]):
                c = c_obs[n].item()
                groups_f[c].append(f[n].cpu().numpy())
                groups_y[c].append(y[n].cpu().numpy())
                groups_z[c].append(z[n].cpu().numpy())
                groups_x[c].append(x[n].cpu().numpy())
                groups_marginal_f[c].append(marginal_f[n].cpu().numpy())
                groups_scores[c].append(scores[n].cpu().numpy())
                groups_concs[c].append(concs[n].cpu().numpy())
      
    def trim(samples: dict):
        samples = [np.stack(vecs) for c, vecs in sorted(samples.items(), key=lambda pair: pair[0])]
        size = min(v.shape[0] for v in samples)
        return np.array([v[np.random.choice(v.shape[0], size, replace=False)] for v in samples])
    
    return trim(groups_f), trim(groups_y), trim(groups_z), trim(groups_x), trim(groups_marginal_f), trim(groups_scores), trim(groups_concs)
        
                
