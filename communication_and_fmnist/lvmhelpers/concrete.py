import torch
import torch.nn as nn
from torch.distributions import RelaxedOneHotCategorical, RelaxedBernoulli
import math

from lvmhelpers.gumbel import GumbelSoftmaxWrapper 

class BinaryConcreteWrapper(GumbelSoftmaxWrapper):
    """
    Binary Concrete Wrapper for a network that parameterizes independent Bernoulli distributions.
    Assumes that during the forward pass, the network returns scores for the Bernoulli parameters.
    The wrapper transforms them into a sample from the Binary Concrete distribution.
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
        """
        super(BinaryConcreteWrapper, self).__init__(
            agent,
            temperature,
            trainable_temperature,
            straight_through)
        self.distr_type = RelaxedBernoulli # BinaryConcrete distribution

    def forward(self, *args, **kwargs):
        """Forward pass.

        Returns:
            sample {torch.Tensor} -- Concrete relaxed sample.
                Size: [batch_size, n_bits]
            scores {torch.Tensor} -- the output of the network.
                Can be useful for logging purposes.
                Size: [batch_size, n_bits]
            entropy {torch.Tensor} -- the entropy of the distribution.
        """
        scores = self.agent(*args, **kwargs)
        scores = torch.clamp(scores, min = -85.0, max=85.0)
        if torch.any(scores.isnan()) == True:
            print('\nscores: ', scores)

        distr = self.distr_type(logits=scores, temperature=torch.tensor(self.temperature)) # BinaryConcrete distribution
        sample = distr.rsample()
        entropy = mc_entropy(distr).sum(dim=-1) # MC estimate entropy (there is no closed-form expression)

        count = False # Set to True to check sparsity
        if count == True:
            non_zero = torch.count_nonzero(sample, dim=1)
            non_one = torch.count_nonzero(1.-sample, dim=1)
            non_sparse = non_zero + non_one - sample.size(1)
            print('\n non_sparse: ', non_sparse)
            print('\n sample: ', sample)
            print('\n avg non_sparse: ',(non_sparse/sample.size(1)).mean())
        return sample, scores, entropy


def mc_entropy(p, n_samples=1):
    x = p.rsample(sample_shape=torch.Size([n_samples]))
    return - p.log_prob(x).mean(0)




class BitVectorConcrete(torch.nn.Module):
    """
    The training loop for the Binary Concrete method to train a
    bit-vector of independent latent variables.
    Encoder needs to be BinaryConcreteWrapper.
    Decoder needs to be utils.DeterministicWrapper.
    """
    def __init__(
            self,
            encoder,
            decoder,
            loss_fun,
            encoder_entropy_coeff=0.0,
            decoder_entropy_coeff=0.0):
        super(BitVectorConcrete, self).__init__()
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
        logs['distr'] = self.encoder.distr_type(logits=encoder_scores, temperature=torch.tensor(self.encoder.temperature))
        return {'loss': full_loss, 'log': logs}

