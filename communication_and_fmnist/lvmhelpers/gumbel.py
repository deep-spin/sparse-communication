import torch
import torch.nn as nn
from torch.distributions import RelaxedOneHotCategorical, RelaxedBernoulli
from torch.distributions import Categorical, Bernoulli
from torch.distributions import MultivariateNormal, Normal
from entmax import sparsemax
import math

def gumbel_softmax_sample(
        logits: torch.Tensor,
        temperature: float = 1.0,
        straight_through: bool = False):
    """Samples from a Gumbel-Sotmax/Concrete of a Categorical distribution.

    Arguments:
        logits {torch.Tensor} -- tensor of logits, the output of an inference network.
            Size: [batch_size, n_categories]

    Keyword Arguments:
        temperature {float} -- temperature of the softmax relaxation. The lower the
            temperature (-->0), the closer the sample is to a discrete sample.
            (default: {1.0})
        straight_through {bool} -- Whether to use the straight-through estimator.
            (default: {False})

    Returns:
        torch.Tensor -- the relaxed sample.
            Size: [batch_size, n_categories]
    """

    sample = RelaxedOneHotCategorical(
        logits=logits, temperature=temperature).rsample()

    if straight_through:
        size = sample.size()
        indexes = sample.argmax(dim=-1)
        hard_sample = torch.zeros_like(sample).view(-1, size[-1])
        hard_sample.scatter_(1, indexes.view(-1, 1), 1)
        hard_sample = hard_sample.view(*size)

        sample = sample + (hard_sample - sample).detach()
    return sample


def gumbel_softmax_bit_vector_sample(
        logits: torch.Tensor,
        temperature: float = 1.0,
        straight_through: bool = False):
    """Samples from a Gumbel-Sotmax/Concrete of independent Bernoulli distributions.

    Arguments:
        logits {torch.Tensor} -- tensor of logits, the output of an inference network.
            Size: [batch_size, n_bits]

    Keyword Arguments:
        temperature {float} -- temperature of the softmax relaxation. The lower the
            temperature (-->0), the closer the sample is to discrete samples.
            (default: {1.0})
        straight_through {bool} -- Whether to use the straight-through estimator.
            (default: {False})

    Returns:
        torch.Tensor -- the relaxed sample.
            Size: [batch_size, n_bits]
    """

    sample = RelaxedBernoulli(
        logits=logits, temperature=temperature).rsample()

    if straight_through:
        hard_sample = (logits > 0).to(torch.float)
        sample = sample + (hard_sample - sample).detach()

    return sample


class GumbelSoftmaxWrapper(nn.Module):
    """
    Gumbel-Softmax Wrapper for a network that parameterizes a Categorical distribution.
    Assumes that during the forward pass,
    the network returns scores over the potential output categories.
    The wrapper transforms them into a sample from the Gumbel-Softmax (GS) distribution.
    """

    def __init__(self,
                 agent,
                 temperature=1.0,
                 trainable_temperature=False,
                 straight_through=False):
        """
        Arguments:
            agent -- The agent to be wrapped. agent.forward() has to output
                scores over the categories

        Keyword Arguments:
            temperature {float} -- The temperature of the Gumbel-Softmax distribution
                (default: {1.0})
            trainable_temperature {bool} -- If set to True, the temperature becomes
                a trainable parameter of the model (default: {False})
            straight_through {bool} -- Whether straigh-through Gumbel-Softmax is used
                (default: {False})
        """
        super(GumbelSoftmaxWrapper, self).__init__()
        self.agent = agent
        self.straight_through = straight_through
        if not trainable_temperature:
            self.temperature = temperature
        else:
            self.temperature = torch.nn.Parameter(
                torch.tensor([temperature]), requires_grad=True)

        self.distr_type = Categorical

    def forward(self, *args, **kwargs):
        """Forward pass.

        Returns:
            sample {torch.Tensor} -- Gumbel-Softmax relaxed sample.
                Size: [batch_size, n_categories]
            scores {torch.Tensor} -- the output of the network.
                Can be useful for logging purposes.
                Size: [batch_size, n_categories]
            entropy {torch.Tensor} -- the entropy of the distribution.
                We assume a Categorical to compute this, which is common-practice,
                but may not be ideal.
                Size: [batch_size]
        """
        scores = self.agent(*args, **kwargs)
        sample = gumbel_softmax_sample(
            scores, self.temperature, self.straight_through)
        distr = self.distr_type(logits=scores)
        entropy = distr.entropy()
        return sample, scores, entropy

    def update_temperature(
            self,
            current_step: int,
            temperature_update_freq: int,
            temperature_decay: float):
        """use this at the end of each training step to anneal the temperature according
        to max(0.5, exp(-rt)) with r and t being the decay rate and training step,
        respectively.

        Arguments:
            current_step {int} -- current global step in the training process
            temperature_update_freq {int} -- how often to update the temperature
            temperature_decay {float} -- decay rate r
        """
        if current_step % temperature_update_freq == 0:
            rt = temperature_decay * torch.tensor(current_step)
            self.temperature = torch.max(
                torch.tensor(0.5), torch.exp(-rt))


class Gumbel(torch.nn.Module):
    """
    The training loop for the Gumbel-Softmax method to train discrete latent variables.
    Encoder needs to be GumbelSoftmaxWrapper.
    Decoder needs to be utils.DeterministicWrapper.
    """
    def __init__(
            self,
            encoder,
            decoder,
            loss_fun,
            encoder_entropy_coeff=0.0,
            decoder_entropy_coeff=0.0):
        super(Gumbel, self).__init__()
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

        argmax = encoder_scores.argmax(dim=-1)

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
        logs['distr'] = self.encoder.distr_type(logits=encoder_scores)
        return {'loss': full_loss, 'log': logs}


class BitVectorGumbelSoftmaxWrapper(GumbelSoftmaxWrapper):
    """
    Gumbel-Softmax Wrapper for a network that parameterizes
    independent Bernoulli distributions.
    Assumes that during the forward pass,
    the network returns scores for the Bernoulli parameters.
    The wrapper transforms them into a sample from the Gumbel-Softmax (GS) distribution.
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
            temperature {float} -- The temperature of the Gumbel-Softmax distribution
                (default: {1.0})
            trainable_temperature {bool} -- If set to True, the temperature becomes
                a trainable parameter of the model (default: {False})
            straight_through {bool} -- Whether straigh-through Gumbel-Softmax is used
                (default: {False})
        """
        super(BitVectorGumbelSoftmaxWrapper, self).__init__(
            agent,
            temperature,
            trainable_temperature,
            straight_through)
        self.distr_type = Bernoulli

    def forward(self, *args, **kwargs):
        """Forward pass.

        Returns:
            sample {torch.Tensor} -- Gumbel-Softmax relaxed sample.
                Size: [batch_size, n_bits]
            scores {torch.Tensor} -- the output of the network.
                Can be useful for logging purposes.
                Size: [batch_size, n_bits]
            entropy {torch.Tensor} -- the entropy of the distribution.
                We assume independent Bernoulli to compute this, which is common-practice,
                but may not be ideal.
                Size: [batch_size]
        """
        scores = self.agent(*args, **kwargs)
        sample = gumbel_softmax_bit_vector_sample(
            scores, self.temperature, self.straight_through)
        distr = self.distr_type(logits=scores)
        entropy = distr.entropy().sum(dim=-1)
        count = False # Set to True to check sparsity
        if count == True:
            non_zero = torch.count_nonzero(sample, dim=1)
            non_one = torch.count_nonzero(1.-sample, dim=1)
            non_sparse = non_zero + non_one - sample.size(1)
            print('\n non_sparse: ', non_sparse)
            print('\n sample: ', sample)
            print('\n avg non_sparse: ',(non_sparse/sample.size(1)).mean())
        return sample, scores, entropy


class BitVectorGumbel(torch.nn.Module):
    """
    The training loop for the Gumbel-Softmax method to train a
    bit-vector of independent latent variables.
    Encoder needs to be BitVectorGumbelSoftmaxWrapper.
    Decoder needs to be utils.DeterministicWrapper.
    """
    def __init__(
            self,
            encoder,
            decoder,
            loss_fun,
            encoder_entropy_coeff=0.0,
            decoder_entropy_coeff=0.0):
        super(BitVectorGumbel, self).__init__()
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
        logs['distr'] = self.encoder.distr_type(logits=encoder_scores)
        return {'loss': full_loss, 'log': logs}



def gumbel_softmax_sparsemax_sample(
        logits: torch.Tensor,
        temperature: float = 1.0,
        straight_through: bool = False,
        lbd: float = 1.2):

    """
    Multivariate hard concrete sample
    1. Get gumbel softmax sample
    2. Stretch 
    3. Apply sparsemax
    """
    # does not work with straight_through yet
    sample = gumbel_softmax_sample(logits, temperature, straight_through)
    sample = sparsemax(lbd*sample,dim=1)
    return sample


class GumbelSoftmaxSparsemaxWrapper(nn.Module):
    """
    K-D Hard Concrete Wrapper for a network that parameterizes a Categorical distribution.
    Assumes that during the forward pass, the network returns scores over the potential output categories.
    The wrapper transforms them into a sample from the K-D Hard Concrete distribution.
    """

    def __init__(self,
                 agent,
                 temperature=1.0,
                 trainable_temperature=False,
                 straight_through=False):
        """
        Arguments:
            agent -- The agent to be wrapped. agent.forward() has to output
                scores over the categories

        Keyword Arguments:
            temperature {float} -- The temperature of the Gumbel-Softmax distribution
                (default: {1.0})
            trainable_temperature {bool} -- If set to True, the temperature becomes
                a trainable parameter of the model (default: {False})
            straight_through {bool} -- Whether straigh-through Gumbel-Softmax is used
                (default: {False})
        """
        super(GumbelSoftmaxSparsemaxWrapper, self).__init__()
        self.agent = agent
        self.lbd = 1.1 # It is possible to change this value
        self.straight_through = straight_through
        if not trainable_temperature:
            self.temperature = temperature
        else:
            self.temperature = torch.nn.Parameter(
                torch.tensor([temperature]), requires_grad=True)

        self.distr_type = Categorical
        self.count = False # Set to True to check sparsity
        if self.count == True:
            self.counting = 0
            self.non_sparsity = 0

    def forward(self, *args, **kwargs):
        """Forward pass.

        Returns:
            sample {torch.Tensor} -- K-D Hard Concrete relaxed sample.
                Size: [batch_size, n_categories]
            scores {torch.Tensor} -- the output of the network.
                Can be useful for logging purposes.
                Size: [batch_size, n_categories]
            entropy {torch.Tensor} -- the entropy of the distribution.
                Size: [batch_size]
        """
        scores = self.agent(*args, **kwargs)
        sample = gumbel_softmax_sparsemax_sample(
            scores, self.temperature, self.straight_through, self.lbd)
        if self.count == True:
            print('\n sample.size: ', sample.size())
            self.counting = self.counting + 1 
            self.non_sparsity = self.non_sparsity + (torch.count_nonzero(sample, dim=1)).float().mean()
            print('\n non_sparsity: ', torch.count_nonzero(sample, dim=1).float().mean())
            print('\n non_sparsity_mean:', self.non_sparsity/self.counting)
        distr = self.distr_type(logits=scores)
        entropy = distr.entropy()
        return sample, scores, entropy

    def update_temperature(
            self,
            current_step: int,
            temperature_update_freq: int,
            temperature_decay: float):
        """use this at the end of each training step to anneal the temperature according
        to max(0.5, exp(-rt)) with r and t being the decay rate and training step,
        respectively.

        Arguments:
            current_step {int} -- current global step in the training process
            temperature_update_freq {int} -- how often to update the temperature
            temperature_decay {float} -- decay rate r
        """
        if current_step % temperature_update_freq == 0:
            rt = temperature_decay * torch.tensor(current_step)
            self.temperature = torch.max(
                torch.tensor(0.5), torch.exp(-rt))


def gumbel_sparsemax_sample(
        logits: torch.Tensor,
        temperature: float = 1.0,
        straight_through: bool = False):
    # does not work with straight_through yet
    const = 0.01
    gumbels = (-torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log())  # ~Gumbel(0,1)
    gumbels = gumbels*const
    gumbels = (logits + gumbels) / temperature  # ~Gumbel(logits,temperature)
    y_sparse = sparsemax(gumbels, dim=-1)
    return y_sparse


class GumbelSparsemaxWrapper(nn.Module):
    """
    Gumbel-Sparsemax Wrapper for a network that parameterizes a Categorical distribution.
    Assumes that during the forward pass,
    the network returns scores over the potential output categories.
    The wrapper transforms them into a sample from the Gumbel-Sparsemax distribution.
    """

    def __init__(self,
                 agent,
                 temperature=1.0,
                 trainable_temperature=False,
                 straight_through=False):
        """
        Arguments:
            agent -- The agent to be wrapped. agent.forward() has to output
                scores over the categories

        Keyword Arguments:
            temperature {float} -- The temperature of the Gumbel-Sparsemax distribution
                (default: {1.0})
            trainable_temperature {bool} -- If set to True, the temperature becomes
                a trainable parameter of the model (default: {False})
            straight_through {bool} -- Whether straigh-through Gumbel-Sparsemax is used
                (default: {False})
        """
        super(GumbelSparsemaxWrapper, self).__init__()
        self.agent = agent
        self.straight_through = straight_through
        if not trainable_temperature:
            self.temperature = temperature
        else:
            self.temperature = torch.nn.Parameter(
                torch.tensor([temperature]), requires_grad=True)

        self.distr_type = Categorical

    def forward(self, *args, **kwargs):
        """Forward pass.

        Returns:
            sample {torch.Tensor} -- Gumbel-Sparsemax relaxed sample.
                Size: [batch_size, n_categories]
            scores {torch.Tensor} -- the output of the network.
                Can be useful for logging purposes.
                Size: [batch_size, n_categories]
            entropy {torch.Tensor} -- the entropy of the distribution.
                Size: [batch_size]
        """
        scores = self.agent(*args, **kwargs)
        sample = gumbel_sparsemax_sample(
            scores, self.temperature, self.straight_through)
        distr = self.distr_type(logits=scores)
        entropy = distr.entropy()
        return sample, scores, entropy

    def update_temperature(
            self,
            current_step: int,
            temperature_update_freq: int,
            temperature_decay: float):
        """use this at the end of each training step to anneal the temperature according
        to max(0.5, exp(-rt)) with r and t being the decay rate and training step,
        respectively.

        Arguments:
            current_step {int} -- current global step in the training process
            temperature_update_freq {int} -- how often to update the temperature
            temperature_decay {float} -- decay rate r
        """
        if current_step % temperature_update_freq == 0:
            rt = temperature_decay * torch.tensor(current_step)
            self.temperature = torch.max(
                torch.tensor(0.5), torch.exp(-rt))


def gaussian_sparsemax_sample(
        logits: torch.Tensor,
        temperature: float = 1.0,
        straight_through: bool = False):

    # does not work with straight_through yet
    
    # should use scale_tril instead of covariance matrix to make it faster
    gaussians = MultivariateNormal(torch.zeros(logits.size()).to(logits.device), scale_tril = torch.eye(logits.size(1)).to(logits.device)).rsample() # (batch, n_cat)
    y_sparse = sparsemax(logits + gaussians, dim=-1)
    return y_sparse


class GaussianSparsemaxWrapper(nn.Module):
    """
    Gaussian-Sparsemax Wrapper for a network that parameterizes a Categorical distribution.
    Assumes that during the forward pass,
    the network returns scores over the potential output categories.
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
                scores over the categories

        Keyword Arguments:
            None of the arguments (temperature, trainable_temperature, and straight_through) are needed for this distribution.
        """
        super(GaussianSparsemaxWrapper, self).__init__()
        self.agent = agent
        self.straight_through = straight_through
        if not trainable_temperature:
            self.temperature = temperature
        else:
            self.temperature = torch.nn.Parameter(
                torch.tensor([temperature]), requires_grad=True)

        self.distr_type = Categorical
        self.count = False # Set to True to check sparsity
        if self.count == True:
            self.counting = 0
            self.non_sparsity = 0

    def forward(self, *args, **kwargs):
        """Forward pass.

        Returns:
            sample {torch.Tensor} -- Gaussian-Sparsemax relaxed sample.
                Size: [batch_size, n_categories]
            scores {torch.Tensor} -- the output of the network.
                Can be useful for logging purposes.
                Size: [batch_size, n_categories]
            entropy {torch.Tensor} -- the entropy of the distribution.
                Size: [batch_size]
        """
        scores = self.agent(*args, **kwargs)
        sample = gaussian_sparsemax_sample(
            scores, self.temperature, self.straight_through)
        if self.count == True:
            print('\n sample.size: ', sample.size())
            self.counting = self.counting + 1 
            self.non_sparsity = self.non_sparsity + (torch.count_nonzero(sample, dim=1)).float().mean()
            print('\n non_sparsity: ', torch.count_nonzero(sample, dim=1).float().mean())
            print('\n non_sparsity_mean:', self.non_sparsity/self.counting)
        distr = self.distr_type(logits=scores)
        entropy = distr.entropy()
        return sample, scores, entropy

    def update_temperature(
            self,
            current_step: int,
            temperature_update_freq: int,
            temperature_decay: float):

        # we don't need this for Gaussian-Sparsemax
        """use this at the end of each training step to anneal the temperature according
        to max(0.5, exp(-rt)) with r and t being the decay rate and training step,
        respectively.

        Arguments:
            current_step {int} -- current global step in the training process
            temperature_update_freq {int} -- how often to update the temperature
            temperature_decay {float} -- decay rate r
        """
        if current_step % temperature_update_freq == 0:
            rt = temperature_decay * torch.tensor(current_step)
            self.temperature = torch.max(
                torch.tensor(0.5), torch.exp(-rt))
