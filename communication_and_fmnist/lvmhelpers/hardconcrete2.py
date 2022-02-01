import torch
import torch.nn as nn
import torch.distributions as td
import math
from lvmhelpers.d01c01 import Rectified01
from lvmhelpers.gumbel import GumbelSoftmaxWrapper 
EPS = 1e-4

def stretched_distribution(distribution, lower=-0.1, upper=1.1):
    assert lower < upper
    return torch.distributions.TransformedDistribution(distribution,
        torch.distributions.AffineTransform(loc=lower, scale=upper - lower))
    
# Create BinaryConcrete with cdf (needed for HardConcrete log_prob)
class BinaryConcrete(torch.distributions.relaxed_bernoulli.RelaxedBernoulli):
    
    def __init__(self, temperature, probs=None, logits=None, validate_args=False):
        super(BinaryConcrete, self).__init__(temperature, probs=probs, logits=logits, validate_args=validate_args)
        
    def cdf(self, value):
        return torch.sigmoid((torch.log(value + EPS) - torch.log(1. - value + EPS)) * self.temperature - self.logits)
    def icdf(self, value):
        return torch.sigmoid((torch.log(value + EPS) - torch.log(1. - value + EPS) + self.logits) / self.temperature)

    def rsample_truncated(self, k0, k1, sample_shape=torch.Size()):        
        shape = self._extended_shape(sample_shape)
        probs = torch.distributions.utils.clamp_probs(self.probs.expand(shape))
        uniforms = Uniform(self.cdf(torch.full_like(self.logits, k0)), 
                           self.cdf(torch.full_like(self.logits, k1))).rsample(sample_shape)
        x = (uniforms.log() - (-uniforms).log1p() + probs.log() - (-probs).log1p()) / self.temperature
        return torch.sigmoid(x)


def mc_entropy(p, n_samples=1):
    x = p.rsample(sample_shape=torch.Size([n_samples]))
    return - p.log_prob(x).mean(0)



class HardConcreteWrapper(GumbelSoftmaxWrapper):
    """
    HardConcrete Wrapper for a network that parameterizes
    independent Bernoulli distributions.
    Assumes that during the forward pass,
    the network returns scores for the Bernoulli parameters.
    The wrapper transforms them into a sample from the HardConcrete distribution.
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
            temperature {float} -- The temperature of the Concrete distribution
                (default: {1.0})
            trainable_temperature {bool} -- If set to True, the temperature becomes
                a trainable parameter of the model (default: {False})
        """
        super(HardConcreteWrapper, self).__init__(
            agent,
            temperature,
            trainable_temperature,
            straight_through)
        self.distr_type = Rectified01

        self.count = False # Set to True to check sparsity
        if self.count == True:
            self.non_sparsity = 0
            self.counting = 0
            print('\n self.non_sparsity:', self.non_sparsity)

    def forward(self, *args, **kwargs):
        """Forward pass.

        Returns:
            sample {torch.Tensor} -- HardConcrete relaxed sample.
                Size: [batch_size, n_bits]
            scores {torch.Tensor} -- the output of the network.
                Can be useful for logging purposes.
                Size: [batch_size, n_bits]
            entropy {torch.Tensor} -- the entropy of the distribution.
                Size: [batch_size]
        """
        scores = self.agent(*args, **kwargs)
        scores = torch.clamp(scores, min = -85.0, max=85.0)
        if torch.any(scores.isnan()) == True:
            print('\nscores: ', scores)

        stretched = stretched_distribution(BinaryConcrete(temperature=torch.tensor(self.temperature), logits=scores), lower=-0.1, upper=1.1) # stretched
        distr = self.distr_type(stretched) # rectified
        sample = distr.rsample()
        if torch.any(sample.isnan()) == True:
            print('\nsample: ', sample)
        if self.count == True:
            non_zero = torch.count_nonzero(sample, dim=1)
            non_one = torch.count_nonzero(1.-sample, dim=1)
            non_sparse = non_zero + non_one - sample.size(1)
            self.counting = self.counting + 1
            self.non_sparsity = self.non_sparsity + (non_sparse/sample.size(1)).mean()
            print('\n self.counting:', self.counting)
            print('\n self.non_sparsity:', self.non_sparsity)
            print('\n non_sparsity_mean:', self.non_sparsity/self.counting)
        entropy = mc_entropy(distr, n_samples=2).sum(dim=-1) # MC estimate entropy (there is no closed-form expression)
        return sample, scores, entropy


class BitVectorHardConcrete(torch.nn.Module):
    """
    The training loop for the HardConcrete method to train a
    bit-vector of independent latent variables.
    Encoder needs to be HardConcreteWrapper.
    Decoder needs to be utils.DeterministicWrapper.
    """
    def __init__(
            self,
            encoder,
            decoder,
            loss_fun,
            encoder_entropy_coeff=0.0,
            decoder_entropy_coeff=0.0):
        super(BitVectorHardConcrete, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss = loss_fun
        self.encoder_entropy_coeff = encoder_entropy_coeff
        self.decoder_entropy_coeff = decoder_entropy_coeff

    def forward(self, encoder_input, decoder_input, labels):
        discrete_latent_z, encoder_scores, encoder_entropy = self.encoder(encoder_input)
        decoder_output = self.decoder(discrete_latent_z, decoder_input)

        # entropy component of the final loss, we can
        # compute already but will only use it later on
        entropy_loss = -(encoder_entropy.mean() * self.encoder_entropy_coeff)

        argmax = (encoder_scores > 0).to(torch.float)

        loss, logs = self.loss(
            encoder_input,
            argmax,
            decoder_input,
            decoder_output,
            labels)

        full_loss = loss.mean() + entropy_loss

        for k, v in logs.items():
            if hasattr(v, 'mean'):
                logs[k] = v.mean()

        logs['loss'] = loss.mean()
        logs['encoder_entropy'] = encoder_entropy.mean()
        logs['distr'] = self.encoder.distr_type(stretched_distribution(BinaryConcrete(temperature=torch.tensor(self.encoder.temperature), logits=encoder_scores), lower=-0.1, upper=1.1))
        return {'loss': full_loss, 'log': logs}
