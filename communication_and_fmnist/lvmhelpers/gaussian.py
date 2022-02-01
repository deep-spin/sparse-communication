import torch
import torch.nn as nn
import torch.distributions as td
from torch.distributions import Normal

from entmax import sparsemax

import math


class BitVectorGaussianSparsemaxWrapper(nn.Module):
    """
    Gaussian-Sparsemax Wrapper for a network that parameterizes independent Bernoulli distributions.
    Assumes that during the forward pass, the network returns scores for the Bernoulli parameters.
    The wrapper transforms them into a sample from the Gaussian-Sparsemax distribution.
    """
    def __init__(self,
                 agent,
                 temperature=1.0,
                 trainable_temperature=False,
                 straight_through=False):
        """
        Arguments:
            agent -- The agent to be wrapped. agent.forward() has to output
                scores for each Bernoulli

        Keyword Arguments:
            None of the arguments (temperature, trainable_temperature, and straight_through) are needed for this distribution.
        """

        super(BitVectorGaussianSparsemaxWrapper, self).__init__()
        self.agent = agent
        self.distr_type = BitGaussianSparsemax
        self.count = False # Set to True to check sparsity
        if self.count ==True:
            self.non_sparsity = 0
            self.counting = 0
            print('\n self.non_sparsity:', self.non_sparsity)

    def forward(self, *args, **kwargs):
        """Forward pass.

        Returns:
            sample {torch.Tensor} -- Gaussian-Sparsemax relaxed sample.
                Size: [batch_size, n_bits]
            scores {torch.Tensor} -- the output of the network.
                Can be useful for logging purposes.
                Size: [batch_size, n_bits]
            entropy {torch.Tensor} -- the entropy of the distribution.
                Size: [batch_size]
        """
        scores = self.agent(*args, **kwargs)
        
        distr = self.distr_type(logits=scores) # could have passed also std: self.distr_type(logits=scores, std=...); default std=1
        sample = distr.rsample() # get one sample
        if self.count == True:
            non_zero = torch.count_nonzero(sample, dim=1)
            non_one = torch.count_nonzero(1.-sample, dim=1)
            non_sparse = non_zero + non_one - sample.size(1)
            self.counting = self.counting + 1
            self.non_sparsity = self.non_sparsity + (non_sparse/sample.size(1)).mean()
            print('\n self.counting:', self.counting)
            print('\n self.non_sparsity:', self.non_sparsity)
            print('\n non_sparsity_mean:', self.non_sparsity/self.counting)

        exact_entropy = True # Set to False to MC estimate the entropy
        if exact_entropy == True:
            entropy = distr.entropy().sum(dim=-1) # this should be mixed entropy
        else:
            entropy = mc_entropy(distr).sum(dim=-1) # MC estimate entropy 

        return sample, scores, entropy


# For simplicty, we can use BitVectorGumbel for this distribution




def gaussian_sparsemax_bit_vector_sample(
        logits: torch.Tensor,
        std: float = 1.0,
        n_samples: int = 1,
        straight_through: bool = False):

    """Samples from a Gaussian-Sparsemax of independent distributions.
    Arguments:
        logits {torch.Tensor} -- tensor of logits, the output of an inference network.
            Size: [batch_size, n_bits]
    Keyword Arguments:
        straight_through {bool} -- Whether to use the straight-through estimator.
            (default: {False})
    Returns:
        torch.Tensor -- the relaxed sample.
            Size: [batch_size, n_bits]
    """

    # does not work with straight_through yet
    # std not squared!
    # convert logits to required shape
    l_ll = torch.stack((logits.reshape(-1),(1-logits).reshape(-1)))
    z = torch.zeros(logits.size(0)*logits.size(1), 2).to(logits.device)
    z[:,0], z[:,1]=l_ll[0], l_ll[1] # [batch*D, 2]
    # z_reshape = z.reshape(logits.size(0),logits.size(1),2) # to recover the original shape 

    if n_samples > 1:
        z = z.expand(n_samples , z.size(0), z.size(1)) # expanding this s.t. shape = [n_samples x (...)]

    gaussians = Normal(torch.tensor([0.0]), torch.tensor([1.0])).rsample(z.size()).squeeze(-1).to(logits.device)  # [batch*D,2]

    y_sparse = sparsemax(z + std*gaussians, dim=-1) # sparsemax(z + std*N)
    if n_samples == 1 :
        y_sparse = y_sparse[:,0].reshape(logits.size()) # to recover the correct size
    else:
        y_sparse = y_sparse[:,:,0].reshape(n_samples, logits.size(0), logits.size(1)) # to recover the correct size
    return y_sparse


def gaussian_1D(x, mu, sigma_sq):
    return(1/math.sqrt(2*math.pi*sigma_sq) *
                torch.exp(-.5*(x-mu)**2/sigma_sq))


def gaussian_sparsemax_mixed_entropy(logits, std):
    # logits here correspond to "z" in the paper
    P_0 = (1-torch.erf(logits/(math.sqrt(2)*std)))/2
    P_1 = (1+torch.erf((logits-1)/(math.sqrt(2)*std)))/2
    # For high values of logits (>5.tal) P_0 can be zero -> log(0) -> nan
    P_0,P_1 = P_0+1e-12, P_1+1e-12 # to avoid numerical issues
    discrete_part = - (P_0*torch.log(P_0)) - (P_1*torch.log(P_1))
    continuous_part = (1 - P_0 - P_1) * math.log(math.sqrt(2*math.pi*(std**2))) \
                    + (std/2) * ( \
                        0.5* (torch.erf((1-logits)/math.sqrt(2*std*std)) - torch.erf(-(logits/math.sqrt(2*std*std)))) \
                        - ((1-logits)/std) * gaussian_1D((1-logits)/std, 0, 1) \
                        - (logits/std) * gaussian_1D(-logits/std, 0, 1) \
                        )
    #direct_sum_entropy = discrete_part + continuous_part
    return discrete_part + continuous_part


def gaussian_sparsemax_log_prob(logits, std, values):

    """
    Compute the log of the probability density/mass function evaluated at a given sample value
    """
    P_0 = (1-torch.erf(logits/(math.sqrt(2)*std)))/2
    P_1 = (1+torch.erf((logits-1)/(math.sqrt(2)*std)))/2
    gaussian = gaussian_1D(values, logits, std**2)
    prob = gaussian

    prob = torch.where(values==0.0, P_0 , prob)
    prob = torch.where(values==1.0, P_1 , prob)
    return torch.log(prob), prob


class BitGaussianSparsemax(td.Distribution):
    def __init__(self, logits, std=math.sqrt(1.0), validate_args=None): # It is possible to change std here
        self.logits = logits
        self.std = std
        super(BitGaussianSparsemax, self).__init__(validate_args=False)

    def rsample(self, n_samples=1):
        '''
        Sample from a Bit Gaussian Sparsemax distribution;
        n_samples: number of samples. default: 1
        '''
        sample = gaussian_sparsemax_bit_vector_sample(self.logits, self.std, int(torch.tensor(n_samples)))
        return sample

    def entropy(self):
        '''
        Compute mixed entropy in closed-form
        '''
        entropy = gaussian_sparsemax_mixed_entropy(self.logits, self.std)
        return entropy

    def log_prob(self, value):
        '''
        log_prob and prob in closed-form
        '''
        log_prob, prob = gaussian_sparsemax_log_prob(self.logits, self.std, value)
        # not using prob for now
        return log_prob



def mc_entropy(p, n_samples=1):
    x = p.rsample(n_samples=torch.Size([n_samples]))
    return - p.log_prob(x)
