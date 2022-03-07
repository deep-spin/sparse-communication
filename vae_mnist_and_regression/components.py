import torch
import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn
from collections import OrderedDict, deque
from mixedrvs.gaussiansp import GaussianSparsemax, GaussianSparsemaxPrior
from mixedrvs.concrete import RelaxedOneHotCategoricalStraightThrough
from mixedrvs.bitvector import MaxEntropyFaces, NonEmptyBitVector
from mixedrvs.dirichlet import MaskedDirichlet
from mixedrvs.deterministic import Delta


@td.register_kl(td.RelaxedOneHotCategorical, td.Dirichlet)
def _kl_concrete_dirichlet(p, q):
    z = p.rsample()
    return p.log_prob(z) - q.log_prob(z)


@td.register_kl(td.RelaxedOneHotCategorical, td.RelaxedOneHotCategorical)
def _kl_concrete_concrete(p, q):
    z = p.rsample()
    return p.log_prob(z) - q.log_prob(z)


def assert_shape(t, shape, message):
    assert t.shape == shape, f"{message} has the wrong shape: got {t.shape}, expected {shape}"


def parse_prior_name(string):
    parts = string.split(' ')
    dist, params = parts[0], [float(x) for x in parts[1:]]
    if dist == 'gaussian' and len(params) != 2:
        raise ValueError("A Gaussian prior takes two parameters (loc, scale)")
    if dist == 'dirichlet' and len(params) != 1:
        raise ValueError("A Dirichlet prior takes 1 parameter (concentration)")
    if dist == 'gibbs' and len(params) != 1:
        raise ValueError("A Gibbs prior takes one parameter (score)")
    if dist == 'gibbs-max-ent' and len(params) != 1:
        raise ValueError("A Gibbs prior takes one parameter (precision)")
    if dist == 'categorical' and len(params) != 1:
        raise ValueError("A Categorical prior takes one parameter (logit)")
    if dist == 'concrete' and len(params) != 2:
        raise ValueError("A Concrete prior takes two parameters (temperature, logit)")
    if dist == 'onehotcat' and len(params) != 2:
        raise ValueError("A OneHotCategorical prior takes two parameters (temperature, logit)")
    if dist == 'gaussian-sparsemax-max-ent' and len(params) != 1:
        raise ValueError("A Gaussian-Sparsemax-Max-Ent prior takes one parameter (precision)")
    return dist, params


def parse_posterior_name(string):
    parts = string.split(' ')
    dist, params = parts[0], [float(x) for x in parts[1:]]
    if dist == 'concrete' and len(params) == 0:
        raise ValueError("A Concrete posterior takes at least one parameter (temperature)")
    if dist == 'onehotcat' and len(params) == 0:
        raise ValueError("A OneHotCategorical posterior takes at least one parameter (temperature)")
    return dist, params


class GenerativeModel(nn.Module):
    """
    A joint distribution over

        Y \in \Delta_{K-1}
            a sparse probability vector
        Z \in R^H
            a latent embedding
        X in {0,1}^D
            an MNIST digit

    The prior over Y is prescribed hierarchically (as a fine mixture):
        * Let F take on one of the faces of the simplex. An outcome f in {0,1}^K
          is a bit vector where f_k indicates whether vertex e_k is in the face.
        * Y|F=f is distributed over the dim(f)-dimensional simplex.
        * The probability of y is given by
            p(Y=y) = \sum_f p(F=f)p(Y=y|F=f)
    """

    def __init__(self, y_dim, z_dim, data_dim, hidden_dec_size,
                 p_drop=0.0,
                 prior_f='gibbs 0.0',
                 prior_y='dirichlet 1.0',
                 prior_z='gaussian 0.0 1.0'
            ):
        """

        Parameters:
            y_dim: dimensionality (K) of the mixed rv
            z_dim: dimensionality (H) of the Gaussian rv (use 0 to disable it)
            data_dim: dimensionality (D) of the observation
            hidden_dec_size: hidden size of the decoder that parameterises X|Z=z, Y=y
            p_drop: dropout probability
            prior_f: available options are
                gibbs score
                gibbs-max-ent bit-precision
                categorical logit
            prior_y: available options are
                dirichlet concentration
                identity
            prior_z: available options are
                gaussian loc scale
                dirichlet concentration
                concrete temperature logit
                onehotcat temperature logit  -- this is GumbelSoftmax-ST
                gaussian-sparsemax-max-ent bit-precision
        """
        super().__init__()
        self._y_dim = y_dim
        self._z_dim = z_dim
        self._data_dim = data_dim
        self._f_dist, self._f_params = parse_prior_name(prior_f)
        self._y_dist, self._y_params = parse_prior_name(prior_y)
        self._z_dist, self._z_params = parse_prior_name(prior_z)

        assert z_dim + y_dim > 0
        assert self._z_dist in ['gaussian', 'dirichlet', 'concrete', 'onehotcat', 'gaussian-sparsemax-max-ent'], f"Unknown choice of distribution for Z: {self._z_dist}"
        assert self._f_dist in ['gibbs', 'gibbs-max-ent', 'categorical'], f"Unknown choice of distribution for F: {self._f_dist}"
        assert self._y_dist in ['dirichlet', 'identity'], f"Unknown choice of distribution for Y|f: {self._y_dist}"

        # TODO: support TransposedCNN?
        self._decoder = nn.Sequential(
            nn.Dropout(p_drop),
            nn.Linear(z_dim + y_dim, hidden_dec_size),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden_dec_size, hidden_dec_size),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden_dec_size, data_dim),
        )
        if z_dim > 0:
            if self._z_dist == 'gaussian':
                self.register_buffer("_prior_z_location", (torch.zeros(z_dim, requires_grad=False) + self._z_params[0]).detach())
                if self._z_params[1] <= 0:
                    raise ValueError("The Gaussian scale for Z must be strictly positive")
                self.register_buffer("_prior_z_scale", (torch.zeros(z_dim, requires_grad=False) + self._z_params[1]).detach())
            elif self._z_dist == 'dirichlet':
                if self._z_params[0] <= 0:
                    raise ValueError("The Dirichlet concentration for Z must be strictly positive")
                self.register_buffer("_prior_z_concentration", (torch.zeros(z_dim, requires_grad=False) + self._z_params[0]).detach())
            elif self._z_dist in ['concrete', 'onehotcat']:
                if self._z_params[0] <= 0:
                    raise ValueError("The Concrete teperature for Z must be strictly positive")
                self.register_buffer("_prior_z_temperature", (torch.zeros(1, requires_grad=False) + self._z_params[0]).detach())
                self.register_buffer("_prior_z_logits", (torch.zeros(z_dim, requires_grad=False) + self._z_params[1]).detach())
            elif self._z_dist == 'gaussian-sparsemax-max-ent':
                self.register_buffer("_prior_z_pmf_n", MaxEntropyFaces.pmf_n(z_dim, self._z_params[0]).detach())
        else:
            self.register_buffer("_prior_z_location", (torch.zeros(0, requires_grad=False)).detach())

        if y_dim > 0:
            if self._f_dist == 'gibbs':
                self.register_buffer("_prior_f_scores", (torch.zeros(y_dim, requires_grad=False) + self._f_params[0]).detach())
            elif self._f_dist == 'gibbs-max-ent':
                self.register_buffer("_prior_f_pmf_n", MaxEntropyFaces.pmf_n(y_dim, self._f_params[0]).detach())
                p = MaxEntropyFaces(self._prior_f_pmf_n)
            elif self._f_dist == 'categorical':
                self.register_buffer("_prior_f_logits", (torch.zeros(y_dim, requires_grad=False) + self._f_params[0]).detach())
            if self._y_dist == 'dirichlet':
                if self._y_params[0] <= 0:
                    raise ValueError("The Dirichlet concentration for Y|F=f must be strictly positive")
                self.register_buffer("_prior_y_concentration", (torch.zeros(1, requires_grad=False) + self._y_params[0]).detach())

        else:
            self.register_buffer("_prior_f_location", (torch.zeros(0, requires_grad=False)).detach())
            self.register_buffer("_prior_y_location", (torch.zeros(0, requires_grad=False)).detach())

    @property
    def data_dim(self):
        return self._data_dim

    @property
    def latent_dim(self):
        return self._z_dim + self._y_dim

    @property
    def z_dim(self):
        return self._z_dim

    @property
    def y_dim(self):
        return self._y_dim

    def Z(self):
        """Return a Normal distribution over latent space"""
        if self._z_dim:
            if self._z_dist == 'gaussian':
                Z = td.Independent(
                        td.Normal(loc=self._prior_z_location, scale=self._prior_z_scale),
                        1
                )
            elif self._z_dist == 'dirichlet':
                Z = td.Dirichlet(self._prior_z_concentration)
            elif self._z_dist == 'concrete':
                Z = td.RelaxedOneHotCategorical(self._prior_z_temperature, logits=self._prior_z_logits)
            elif self._z_dist == 'onehotcat':
                Z = RelaxedOneHotCategoricalStraightThrough(self._prior_z_temperature, logits=self._prior_z_logits)
            elif self._z_dist == 'gaussian-sparsemax-max-ent':
                Z = GaussianSparsemaxPrior(MaxEntropyFaces(pmf_n=self._prior_z_pmf_n), torch.ones_like)
            else:
                raise ValueError(f"Unknown choice of distribution for Z: {self._z_dist}")
        else:
            Z = td.Independent(Delta(self._prior_z_location), 1)
        return Z

    def F(self, state=dict()):
        """
        Return a distribution over the non-empty faces of the simplex

        Parameters:
            state: save computations that can be used downstream by the hierarchical model
        """
        if self._y_dim:
            if self._f_dist == 'gibbs':
                return NonEmptyBitVector(scores=self._prior_f_scores)
            elif self._f_dist == 'gibbs-max-ent':
                return MaxEntropyFaces(pmf_n=self._prior_f_pmf_n)
            elif self._f_dist == 'categorical':
                return td.OneHotCategorical(logits=self._prior_f_logits)
        else:
            return td.Independent(Delta(self._prior_f_location), 1)

    def Y(self, f, state=dict()):
        """
        Return a batch of masked Dirichlet distributions Y|F=f

        Parameters:
            f: face-encoding [batch_size, K]
            state: used to save computations relevant to the hierarchical model
        """
        if self._y_dim:
            if self._y_dist == 'dirichlet':
                return MaskedDirichlet(f.bool(), self._prior_y_concentration * torch.ones_like(f))
            elif self._y_dist == 'identity':
                return td.Independent(Delta(f), 1)
        else:
            return td.Independent(Delta(torch.zeros_like(f)), 1)

    def X(self, y, z):
        """Return a product of D Bernoulli distributions"""
        if z.shape[:-1] != y.shape[:-1]:
            raise ValueError("z and y must have the same batch_shape")
        inputs = torch.cat([y, z], -1)
        logits = self._decoder(inputs)
        return td.Independent(td.Bernoulli(logits=logits), 1)

    def sample(self, sample_shape=torch.Size([])):
        """Return (f, y, z, x)"""
        # [sample_shape, K]
        f = self.F().sample(sample_shape)
        # [sample_shape, K]
        y = self.Y(f=f).sample()
        # [sample_shape, H]
        z = self.Z().sample(sample_shape)
        # [sample_shape, D]
        x = self.X(z=z, y=y).sample()
        return f, y, z, x

    def log_prob(self, f, y, z, x, per_bit=False, reduce=True):
        """
        Return the log probability of each one of the variables in order
        or the sum of their log probabilities.
        """
        if not reduce:
            if per_bit:
                return self.F().log_prob(f), self.Y(f).log_prob(y), self.Z().log_prob(z), self.X(z=z, y=y).base_dist.log_prob(x)
            else:
                return self.F().log_prob(f), self.Y(f).log_prob(y), self.Z().log_prob(z), self.X(z=z, y=y).log_prob(x)
        if reduce:
            if per_bit:
                return self.F().log_prob(f).unsqueeze(-1) + self.Y(f).log_prob(y).unsqueeze(-1) + self.Z().log_prob(z).unsqueeze(-1) + self.X(z=z, y=y).base_dist.log_prob(x)
            else:
                return self.F().log_prob(f) + self.Y(f).log_prob(y) + self.Z().log_prob(z) + self.X(z=z, y=y).log_prob(x)


class InferenceModel(nn.Module):
    """
    A joint distribution over

        Y \in \Delta_{K-1}
            a sparse probability vector
        Z \in R^H
            a latent embedding


    q(Y=y,Z=z|X=x) = q(Y=y|X=x)q(Z=z|Y=y, X=x)
    and
    q(Y=y|X=x) = \sum_f q(F=f|X=x)q(Y=y|F=f,X=x)

    Optionally we make a mean field assumption, then q(Z=z|Y=y, X=x)=q(Z=z|X=x),
    and optionally we predict a shared set of concentrations for all faces given x.

    """

    def __init__(self, y_dim, z_dim, data_dim, hidden_enc_size,
                 p_drop=0.0,
                 posterior_z='gaussian',
                 posterior_f='gibbs -10 10',
                 posterior_y='dirichlet 1e-3 1e3',
                 mean_field=True,
                 shared_enc_fy=True,
                 gsp_cdf_samples=None,
                 gsp_KL_samples=None,
                 ):
        """
        Parameters:
            y_dim (int): use more than 0 to introduce a mixed-rv
            z_dim (int): use more than 0 to introduce a fully reparameterisable rv
            data_dim (int):
            hidden_enc_size (int): size of the hidden layer for encoders (from x to parameters)
            p_drop (float): dropout probability (defaults to 0.0 as dropout in inference nets is not super justified)
            posterior_z (str): available options are
                gaussian (lower-loc upper-loc)
                dirichlet (lower-concentration upper-concentration)
                concrete (lower-logit upper-logit)
                onehotcat (lower-logit upper-logit)
                gaussian-sparsemax (lower-loc upper-loc)
            posterior_f (str): available options are
                gibbs (lower-score upper-score)
                categorical (lower-score upper-score)
            posterior_y (str):
                dirichlet (lower-concentration upper-concentration)
                identity
            mean_field (bool): whether to model q(y|x)q(z|x) versus q(y|x)q(z|x,y)
            shared_enc_fy (bool):
        """
        assert z_dim + y_dim > 0
        assert parse_posterior_name(posterior_z)[0] in ['gaussian', 'dirichlet', 'concrete', 'onehotcat', 'gaussian-sparsemax'], "Unknown choice of distribution for Z|x"
        assert parse_posterior_name(posterior_f)[0] in ['gibbs', 'categorical'], "Unknown choice of distribution for F|x"
        assert parse_posterior_name(posterior_y)[0] in ['dirichlet', 'identity'], "Unknown choice of distribution for Y|f,x"

        super().__init__()

        self._y_dim = y_dim
        self._z_dim = z_dim

        self._f_dist, self._f_params = parse_posterior_name(posterior_f)
        self._y_dist, self._y_params = parse_posterior_name(posterior_y)
        self._z_dist, self._z_params = parse_posterior_name(posterior_z)

        self._gsp_cdf_samples = gsp_cdf_samples
        self._gsp_KL_samples = gsp_KL_samples

        self._mean_field = mean_field
        self._shared_enc_fy = shared_enc_fy

        # Y|X=x
        if y_dim:
            if shared_enc_fy:
                self._enc_for_fy = nn.Sequential(
                    nn.Dropout(p_drop),
                    nn.Linear(data_dim, hidden_enc_size),
                    nn.ReLU(),
                    nn.Dropout(p_drop),
                    nn.Linear(hidden_enc_size, hidden_enc_size),
                    nn.ReLU(),
                )
            else:
                self._enc_for_f = nn.Sequential(
                    nn.Dropout(p_drop),
                    nn.Linear(data_dim, hidden_enc_size),
                    nn.ReLU(),
                    nn.Dropout(p_drop),
                    nn.Linear(hidden_enc_size, hidden_enc_size),
                    nn.ReLU(),
                )
                self._enc_for_y = nn.Sequential(
                    nn.Dropout(p_drop),
                    nn.Linear(data_dim + y_dim, hidden_enc_size),
                    nn.ReLU(),
                    nn.Dropout(p_drop),
                    nn.Linear(hidden_enc_size, hidden_enc_size),
                    nn.ReLU(),
                )
            self._f_scores = nn.Sequential(
                nn.Dropout(p_drop),
                nn.Linear(hidden_enc_size, y_dim)
            )
            self._y_concentrations = nn.Sequential(
                nn.Dropout(p_drop),
                nn.Linear(hidden_enc_size, y_dim),
                nn.Softplus()
            )

        # Z|X=x, Y=y
        if z_dim:
            z_num_params = 2 * z_dim if self._z_dist in ['gaussian', 'gaussian-sparsemax'] else z_dim
            self._zparams_net = nn.Sequential(
                nn.Dropout(p_drop),
                nn.Linear(data_dim if mean_field else y_dim + data_dim, hidden_enc_size),
                nn.ReLU(),
                nn.Dropout(p_drop),
                nn.Linear(hidden_enc_size, hidden_enc_size),
                nn.ReLU(),
                nn.Linear(hidden_enc_size, z_num_params)
            )


    @property
    def mean_field(self):
        return self._mean_field

    def _match_sample_shape(self, x, y):
        if len(x.shape) == len(y.shape):
            return x, y
        if len(y.shape) > len(x.shape):
            sample_dims = len(y.shape) - len(x.shape)
            sample_shape = y.shape[:sample_dims]
            x = x.view((1,) * sample_dims + x.shape).expand(sample_shape + (-1,) * len(x.shape))
        else:
            y, x  = self._match_sample_shape(y, x)
        return x, y

    def Z(self, x, y):
        x, y = self._match_sample_shape(x, y)
        if self._z_dim:
            inputs = x if self._mean_field else torch.cat([y, x], -1)
            params = self._zparams_net(inputs)
            if self._z_dist == 'gaussian':
                loc, scale = params[..., :self._z_dim], nn.functional.softplus(params[..., self._z_dim:])
                if len(self._z_params) >= 2:  # we are clamping scales
                    scale = torch.clamp(scales, self._z_params[0], self._z_params[1])
                if len(self._z_params) == 4:  # we are clamping loc
                    loc = torch.clamp(loc, self._z_params[2], self._z_params[3])
                Z = td.Independent(td.Normal(loc=loc, scale=scale), 1)
            elif self._z_dist == 'gaussian-sparsemax':
                loc, scale = params[..., :self._z_dim], nn.functional.softplus(params[..., self._z_dim:])
                if len(self._z_params) >= 2:  # we are clamping scales
                    scale = torch.clamp(scale, self._z_params[0], self._z_params[1])
                if len(self._z_params) == 4:  # we are clamping loc
                    loc = torch.clamp(loc, self._z_params[2], self._z_params[3])
                Z = GaussianSparsemax(loc=loc, scale=scale, cdf_samples=self._gsp_cdf_samples, KL_samples=self._gsp_KL_samples)
            elif self._z_dist == 'dirichlet':
                conc = nn.functional.softplus(params)
                if len(self._z_params) == 2: # we are clamping concentrations
                    conc = torch.clamp(conc, self._z_params[0], self._z_params[1])
                Z = td.Dirichlet(conc)
            elif self._z_dist in ['concrete', 'onehotcat']:
                logits = params
                if len(self._z_params) == 3: # we are clamping logits
                    logits = torch.clamp(logits, self._z_params[1], self._z_params[2])
                if self._z_dist == 'concrete':
                    Z = td.RelaxedOneHotCategorical(torch.zeros(1, device=params.device) + self._z_params[0], logits=logits)
                else:
                    Z = RelaxedOneHotCategoricalStraightThrough(torch.zeros(1, device=params.device) + self._z_params[0], logits=logits)
            else:
                raise ValueError(f"Unknown distribution for Z|x,y: {self._z_dist}")

        else:
            Z = td.Independent(Delta(torch.zeros(x.shape[:-1] + (0,), device=x.device)), 1)
        return Z


    def F(self, x, state=dict()):
        if not self._y_dim:
            F = td.Independent(Delta(torch.zeros(x.shape[:-1] + (0,), device=x.device)), 1)
        else:
            if self._shared_enc_fy:
                h = self._enc_for_fy(x)
                state['encoded-x'] = h
            else:
                h = self._enc_for_f(x)
            scores = self._f_scores(h)
            state['scores'] = scores
            if len(self._f_params) == 2: # we are clamping scores
                scores = torch.clamp(scores, self._f_params[0], self._f_params[1])

            if self._f_dist == 'gibbs':
                F = NonEmptyBitVector(scores)
            elif self._f_dist == 'categorical':
                F = td.OneHotCategorical(logits=scores)
        return F

    def Y(self, x, f, state=dict()):
        x, f = self._match_sample_shape(x, f)
        if not self._y_dim:
            Y = td.Independent(Delta(torch.zeros_like(f)), 1)
        else:
            if self._y_dist == 'identity':
                Y = td.Independent(Delta(f), 1)
            else:
                if self._shared_enc_fy:
                    h = state['encoded-x'] if 'encoded-x' in state else self._enc_for_fy(x)
                    h, f = self._match_sample_shape(h, f)
                else:
                    h = self._enc_for_y(torch.cat([f, x], -1))
                concentrations = self._y_concentrations(h)
                if len(self._y_params) == 2:  # we are clamping concentrations
                    concentrations = torch.clamp(concentrations, self._y_params[0], self._y_params[1])
                Y = MaskedDirichlet(f.bool(), concentrations)
        return Y


    def _F(self, x, predictors=None):
        if not self._y_dim:
            return td.Independent(Delta(torch.zeros(x.shape[:-1] + (0,), device=x.device)), 1)
        # [B, K]
        if self._shared_concentrations and self._share_fy_net:
            scores = self._scores_and_concs_net(x)[...,:self._y_dim]
        else:
            scores = self._scores_net(x)
        if len(self._f_params) == 2: # we are clamping scores
            scores = torch.clamp(scores, self._f_params[0], self._f_params[1])
            #scores = torch.tanh(scores) * 10
        if self._f_dist == 'gibbs':
            return NonEmptyBitVector(scores)
        elif self._f_dist == 'categorical':
            return td.OneHotCategorical(logits=scores)

    def _Y(self, x, f, predictors=None):
        x, f = self._match_sample_shape(x, f)
        if not self._y_dim:
            return td.Independent(Delta(torch.zeros_like(f)), 1)
        if self._y_dist == 'identity':
            return td.Independent(Delta(f), 1)

        if self._shared_concentrations:
            inputs = x  # [...,D]
            if self._share_fy_net:
                concentration = self._scores_and_concs_net(x)[...,self._y_dim:]
            else:
                concentration = self._concentrations_net(inputs)
        else:
            inputs = torch.cat([f, x], -1)  # [...,K+D]
            # [...,K]
            concentration = self._concentrations_net(inputs)
        if len(self._y_params) == 2:  # we are clamping concentrations
            concentration = torch.clamp(concentration, self._y_params[0], self._y_params[1])
            #concentration = torch.sigmoid(concentration) * 100.0
        return MaskedDirichlet(f.bool(), concentration)


    def sample(self, x, sample_shape=torch.Size([])):
        """Return (f, y, z), No gradients through this."""
        with torch.no_grad():
            # [sample_shape, B, D]
            x = x.expand(sample_shape + x.shape)
            state = dict()
            # [sample_shape, B, K]
            f = self.F(x, state=state).sample()
            # [sample_shape, B, K]
            y = self.Y(f=f, x=x, state=state).sample()
            # [sample_shape, B, H]
            z = self.Z(x=x, y=y).sample()
            return f, y, z

    def log_prob(self, x, f, y, z, reduce=True):
        """log q(f|x), log q(y|f, x), log q(z|y, x)"""
        state = dict()
        log_prob_f = self.F(x, state=state).log_prob(f)
        log_prob_y = self.Y(x=x, f=f, state=state).log_prob(y)
        log_prob_z = self.Z(x=x, y=y).log_prob(z)
        if reduce:
            return log_prob_f + log_prob_y + log_prob_z
        else:
            return log_prob_f, log_prob_y, log_prob_z


class VAE:
    """
    Helper class to compute quantities related to VI.
    """

    def __init__(self, p: GenerativeModel, q: InferenceModel,
                 use_self_critic=False, use_reward_standardisation=True):
        self.p = p
        self.q = q
        self.use_self_critic = use_self_critic
        self.use_reward_standardisation = use_reward_standardisation
        self._rewards = deque([])

    def train(self):
        self.p.train()
        self.q.train()

    def eval(self):
        self.p.eval()
        self.q.eval()

    def gen_parameters(self):
        return self.p.parameters()

    def inf_parameters(self):
        return self.q.parameters()

    def critic(self, x_obs, z, q_F, state, exact_KL_Y=False):
        """This estimates reward (w.r.t sampling of F) on a single sample for variance reduction"""
        S, B, H, K, D = x_obs.shape[0], x_obs.shape[1], self.p.z_dim, self.p.y_dim, self.p.data_dim
        with torch.no_grad():
            # Approximate posteriors and samples
            # [S, B, K]
            f = q_F.sample((S,))  # we resample f
            assert_shape(f, (S, B, K), "f ~ F|X=x, \lambda")
            q_Y = self.q.Y(x=x_obs, f=f, state=state)  # and thus also resample y
            #if q_Y.batch_shape != (S, B):
            #    q_Y = q_Y.expand((S, B))
            # [S, B, K]
            y = q_Y.sample()
            assert_shape(y, (S, B, K), "y ~ Y|X=x, F=f, \lambda")
            if not self.q.mean_field:
                q_Z = self.q.Z(x=x_obs, y=y)
                #if q_Z.batch_shape != (S,B):
                #    q_Z = q_Z.expand((S, B))
                # [S, B, H]
                z = q_Z.sample()  # and thus also resample z
                assert_shape(z, (S, B, H), "z ~ Z|X=x, Y=y, \lambda")
            else:
                q_Z = None

            # Priors
            p_F = self.p.F()
            if p_F.batch_shape != (S,B):
                p_F = p_F.expand((S,B))

            p_Y = self.p.Y(f)  # we condition on f ~ q_F
            if p_Y.batch_shape != (S,B):
                p_Y = p_Y.expand((S,B))

            # Sampling distribution
            p_X = self.p.X(y=y, z=z)  # we condition on y ~ q_Y and z ~ q_Z

            # [S, B]
            ll = p_X.log_prob(x_obs)
            # [S, B]
            kl_Y_f = td.kl_divergence(q_Y, p_Y)
            # [S, B]
            if exact_KL_Y:
                critic = ll
            else:
                critic = ll - kl_Y_f

            if not self.q.mean_field:
                p_Z = self.p.Z()
                if p_Z.batch_shape != (S, B):
                    p_Z = p_Z.expand((S,B))
                # [S, B]
                kl_Z = td.kl_divergence(q_Z, p_Z)
                # [S, B]
                critic -= kl_Z

            return critic

    def update_reward_stats(self, reward, dim=0):
        """Return the current statistics and update the vector"""
        if len(self._rewards) > 1:
            avg = np.mean(self._rewards)
            std = np.std(self._rewards)
        else:
            avg = 0.0
            std = 1.0
        if len(self._rewards) == 100:
            self._rewards.popleft()
        self._rewards.append(reward.mean(dim).item())
        return avg, std

    def DR(self, x_obs, exact_KL_Y=False):
        with torch.no_grad():
            B, H, K, D = x_obs.shape[0], self.p.z_dim, self.p.y_dim, self.p.data_dim

            fy_state = dict()
            # Posterior approximations and samples
            q_F = self.q.F(x_obs, state=fy_state)
            # [B, K]
            f = q_F.sample() # not rsample
            assert_shape(f, (B, K), "f ~ F|X=x, \lambda")

            q_Y = self.q.Y(x=x_obs, f=f, state=fy_state)
            y = q_Y.rsample()
            assert_shape(y, (B, K), "y ~ Y|X=x, F=f, \lambda")

            q_Z = self.q.Z(x=x_obs, y=y)
            # [B, H]
            z = q_Z.rsample()
            assert_shape(z, (B, H), "z ~ Z|X=x, Y=y, \lambda")

            # Priors
            p_F = self.p.F()
            if p_F.batch_shape != x_obs.shape[:1]:
                p_F = p_F.expand(x_obs.shape[:1] + p_F.batch_shape)

            p_Y = self.p.Y(f)  # we condition on f ~ q_F thus batch_shape is already correct

            p_Z = self.p.Z()
            if p_Z.batch_shape != x_obs.shape[:1]:
                p_Z = p_Z.expand(x_obs.shape[:1] + p_Z.batch_shape)

            # Sampling distribution
            p_X = self.p.X(y=y, z=z)  # we condition on y ~ q_Y

            # Return type
            ret = OrderedDict(
                ELBO=0.,
                D=0.,
                R=0.,
            )

            # ELBO: the first term is an MC estimate (we sampled (f,y))
            # the second term is exact
            # the third tuse_self_criticis an MC estimate (we sampled f)
            D = -p_X.log_prob(x_obs)
            kl_Y_f = td.kl_divergence(q_Y, p_Y)
            kl_F = td.kl_divergence(q_F, p_F)
            kl_Z = td.kl_divergence(q_Z, p_Z)

            # [B]
            if exact_KL_Y:
                if type(q_Y) is not MaskedDirichlet:
                    raise ValueError("I can only compute exact KL Y if you use MaskedDirichlet posteriors")
                kl_Y = self.KL_Y_Dir1(q_F, q_Y._concentration)
            else:
                kl_Y = torch.zeros_like(kl_F)

            if exact_KL_Y:
                ret['ELBO'] = -D - (kl_F + kl_Y + kl_Z)
                ret['D'] = D
                ret['R'] = kl_F + kl_Y + kl_Z
            else:
                ret['ELBO'] = -D - (kl_F + kl_Y_f + kl_Z)
                ret['D'] = D
                ret['R'] = kl_F + kl_Y_f + kl_Z
            if self.p.y_dim:
                ret['R_F'] = kl_F
                ret['R_Y|f'] = kl_Y_f
                if exact_KL_Y:
                    ret['R_Y'] = kl_Y
            if self.p.z_dim:
                ret['R_Z'] = kl_Z
        return ret

    def KL_Y_Dir1(self, q_F, alphas):
        # [S, B, K]
        all_f = q_F.enumerate_support()
        assert alphas.shape == all_f.shape[1:], f"I need shape: {all_f.shape[1:]}"
        all_q_f = q_F.log_prob(all_f).exp()
        all_q_y = MaskedDirichlet(all_f.bool(), alphas.expand(all_f.shape[:1] + alphas.shape))
        all_p_y = MaskedDirichlet(all_f.bool(), torch.ones_like(all_f))
        # [S, B]
        all_KL_Y = all_q_f * ( - all_q_y.entropy() - all_p_y.log_prob(all_p_y.sample()))  # TODO: this works because the prior Dirichlet is Dir(1.0)
        # [B]
        all_KL_Y = all_KL_Y.sum(0)
        return all_KL_Y
    
    def density_estimation_z(self, x_obs, z, reduce='mean'):
        N, B, H, K, D = z.shape[0], x_obs.shape[0], self.p.z_dim, self.p.y_dim, self.p.data_dim  # S > 1

        # [N, B, K]
        z = z.unsqueeze(1).repeat((1, B, 1))
        
        # [N, B, K]
        x_obs = x_obs.expand((N, B, D))
        
        # Approximate posteriors and samples
        fy_state = dict()
        q_F = self.q.F(x_obs, state=fy_state)
        f = q_F.sample() # not rsample
        q_Y = self.q.Y(x=x_obs, f=f, state=fy_state)
        y = q_Y.rsample()  # with reparameterisation! (important)
        
        q_Z = self.q.Z(x=x_obs, y=y)
        # [S, B, H]
        z = q_Z.rsample()  # with reparameterisation

        # [N, B]
        log_prob_q = q_F.log_prob(f) 
        log_prob_q += q_Y.log_prob(y)
        log_prob_q += q_Z.log_prob(z)
        
        # Priors
        p_F = self.p.F()
        if p_F.batch_shape != (N, B,):
            p_F = p_F.expand((N, B,))

        p_Y = self.p.Y(f)  # we condition on f ~ q_F  thus batch_shape is already correct
        
        p_Z = self.p.Z()
        if p_Z.batch_shape != (N, B):
            p_Z = p_Z.expand((N,B))

        # [N, B]
        log_prob_p = p_F.log_prob(f) 
        log_prob_p += p_Y.log_prob(y)
        log_prob_p += p_Z.log_prob(z)
        
        if reduce == 'mean':
            # [N], [N]
            return log_prob_q.mean(1), log_prob_p.mean(1)
        elif reduce == 'sum':
            # [N], [N]
            return log_prob_q.sum(1), log_prob_p.sum(1)
        else:
            # [N, B], [N, B]
            return log_prob_q, log_prob_p
    
    def density_estimation_y(self, x_obs, y, reduce='mean'):
        N, B, H, K, D = y.shape[0], x_obs.shape[0], self.p.z_dim, self.p.y_dim, self.p.data_dim  # S > 1

        # [N, B, K]
        y = y.unsqueeze(1).repeat((1, B, 1))
        f = (y > 0).float()   
        
        # [N, B, K]
        x_obs = x_obs.expand((N, B, D))
        
        # Approximate posteriors and samples
        fy_state = dict()
        q_F = self.q.F(x_obs, state=fy_state)

        q_Y = self.q.Y(x=x_obs, f=f, state=fy_state)

        # [N, B]
        log_prob_q = q_F.log_prob(f) 
        log_prob_q += q_Y.log_prob(y)
        
        # Priors
        p_F = self.p.F()
        if p_F.batch_shape != (B,):
            p_F = p_F.expand((B,))

        p_Y = self.p.Y(f)  # we condition on f ~ q_F  thus batch_shape is already correct
        # [N, B]
        log_prob_p = p_F.log_prob(f) 
        log_prob_p += p_Y.log_prob(y)
        
        if reduce == 'mean':
            # [N], [N]
            return log_prob_q.mean(1), log_prob_p.mean(1)
        elif reduce == 'sum':
            # [N], [N]
            return log_prob_q.sum(1), log_prob_p.sum(1)
        else:
            # [N, B], [N, B]
            return log_prob_q, log_prob_p
    
    def loss(self, x_obs, c_obs=None, num_samples=1, samples=None, images=None, exact_marginal=False, exact_KL_Y=False):
        """
        :param x_obs: [B, D]
        """
        if num_samples < 1:
            raise ValueError("I cannot compute an estimate without sampling")

        S, B, H, K, D = num_samples, x_obs.shape[0], self.p.z_dim, self.p.y_dim, self.p.data_dim  # S > 1

        # Approximate posteriors and samples
        fy_state = dict()
        q_F = self.q.F(x_obs, state=fy_state)

        if exact_marginal:
            # [S, B, K]
            f = q_F.enumerate_support()
            S = f.shape[0]  # rewrite S
        else:
            # [S, B, K]
            f = q_F.sample((S,)) # not rsample

        assert_shape(f, (S, B, K), "f ~ F|X=x, \lambda")

        # [S, B, D]
        x_obs = x_obs.expand((S, B, D))

        q_Y = self.q.Y(x=x_obs, f=f, state=fy_state)
        # [S, B, K]
        y = q_Y.rsample()  # with reparameterisation! (important)
        assert_shape(y, (S, B, K), "y ~ Y|F=f, \lambda")

        q_Z = self.q.Z(x=x_obs, y=y)
        # [S, B, H]
        z = q_Z.rsample()  # with reparameterisation
        assert_shape(z, (S, B, H), "z ~ Z|X=x, \lambda")

        # Priors
        p_F = self.p.F()
        if p_F.batch_shape != (B,):
            p_F = p_F.expand((B,))

        p_Y = self.p.Y(f)  # we condition on f ~ q_F  thus batch_shape is already correct

        p_Z = self.p.Z()
        if p_Z.batch_shape != (S,B):
            p_Z = p_Z.expand((S,B))

        # Sampling distribution
        p_X = self.p.X(y=y, z=z)  # we condition on y ~ q_Y

        # Return type
        ret = OrderedDict(
            loss=0.,
        )

        # E_F E_Y E_Z [log p(x|y)] - KL_F - E_F[KL_Y] - KL_Z
        # sum_f q(f) E_Y E_Z [log p(x|y)] - KL_F - \sum_f KL_Y - KL_Z
        # sum_f q(f) log p(x|y)
        # - KL_F
        # - sum_f q(f) KL(Y|f,x || Y|f)
        # - KL_Z

        # ELBO: the first term is an MC estimate (we sampled (f,y))
        # the second term is exact
        # the third tuse_self_criticis an MC estimate (we sampled f)
        # [S, B]
        ll = p_X.log_prob(x_obs)
        kl_Y_f = td.kl_divergence(q_Y, p_Y)

        # [1, B]
        kl_F = td.kl_divergence(q_F, p_F).unsqueeze(0)
        # [S, B]
        kl_Z = td.kl_divergence(q_Z, p_Z)

        if exact_marginal:
            # [S, B]
            prob_f = q_F.log_prob(f).exp()
            # [1, B]
            ll = (prob_f * ll).sum(0, keepdims=True)
            kl_Y = (prob_f * kl_Y_f).sum(0, keepdims=True)
        elif exact_KL_Y:  # TODO: this can be optimised
            if type(q_Y) is not MaskedDirichlet:
                raise ValueError("I can only compute exact KL Y if you use MaskedDirichlet posteriors")
            # [1, B]
            kl_Y = self.KL_Y_Dir1(q_F, q_Y._concentration.squeeze(0)).unsqueeze(0)
        else:
            # [1, B]
            kl_Y = torch.zeros_like(kl_F)

        # Logging ELBO terms
        # []
        reduce_dims = (0,1)
        ret['D'] = -ll.mean((0,1)).item()
        if exact_marginal or exact_KL_Y:
            ret['R'] = (kl_F + kl_Y + kl_Z).mean((0,1)).item()
            ret['ELBO'] = (ll - kl_Y - kl_Z - kl_F).mean((0,1)).item()
        else:
            ret['R'] = (kl_F + kl_Y_f + kl_Z).mean((0,1)).item()
            ret['ELBO'] = (ll - kl_Y_f - kl_Z - kl_F).mean((0,1)).item()
        if self.p.y_dim:
            ret['R_F'] = kl_F.mean((0,1)).item()
            ret['R_Y|f'] = kl_Y_f.mean((0,1)).item()
            if exact_marginal or exact_KL_Y:
                ret['R_Y'] = kl_Y.mean((0,1)).item()
        if self.p.z_dim:
            ret['R_Z'] = kl_Z.mean((0,1)).item()

        # Gradient surrogates and loss

        # i) reparameterised gradient (g_rep)
        # [S,B]
        if exact_marginal or exact_KL_Y:
            grep_surrogate = ll - kl_Z - kl_F - kl_Y
        else:
            grep_surrogate = ll - kl_Z - kl_F - kl_Y_f

        # ii) score function estimator (g_SFE)
        if self.p.y_dim and not exact_marginal:
            # E_ZFY[ log p(x|z,f,y)] - -KL(Z) - KL(F) - E_F[ KL(Y) ]
            # E_F[ E_Y[ E_Z[ log p(x|z,f,y) ] - KL(Y) ] ] -KL(Z) - KL(F)
            # E_F[ r(F) ] for r(f) = log p(x|z,f,y)
            # r(f).detach() * log q(f)
            # [S, B]
            if exact_KL_Y:
                reward = ll.detach() if self.q.mean_field else (ll - kl_Z).detach()
            else:
                reward = (ll - kl_Y_f).detach() if self.q.mean_field else (ll - kl_Y_f - kl_Z).detach()
            # Variance reduction tricks
            if self.use_self_critic:
                if num_samples > 1:
                    # [S, B]
                    sum_others = reward.sum(0, keepdims=True) - reward
                    critic = sum_others / (num_samples - 1)
                    # [S, B]
                    critic = (sum_others + critic) / num_samples
                    criticised_reward = reward - critic.detach()
                else:
                    # [S, B]
                    criticised_reward = reward - self.critic(x_obs, z=z, q_F=q_F, state=fy_state, exact_KL_Y=exact_KL_Y).detach()
            else:
                criticised_reward = reward
            if self.use_reward_standardisation:
                reward_avg, reward_std = self.update_reward_stats(criticised_reward, dim=reduce_dims)
                standardised_reward = (criticised_reward - reward_avg) / np.maximum(reward_std, 1.0)
            else:
                standardised_reward = criticised_reward

            # [S, B]
            sfe_surrogate = standardised_reward * q_F.log_prob(f)

            # Loggin SFE variants
            ret['SFE_reward'] = reward.mean(reduce_dims).item()
            if self.use_self_critic:
                ret['SFE_criticised_reward'] = criticised_reward.mean(reduce_dims).item()
            if self.use_reward_standardisation:
                ret['SFE_standardised_reward'] = standardised_reward.mean(reduce_dims).item()
        else:
            sfe_surrogate = torch.zeros_like(grep_surrogate)

        # []
        loss = -(grep_surrogate + sfe_surrogate).mean((0,1))
        ret['loss'] = loss.item()

        if samples is not None:
            if self.p.y_dim:
                samples['f'] = f.detach().cpu()
                samples['y'] = y.detach().cpu()
            if self.p.z_dim:
                samples['z'] = z.detach().cpu()
        if images is not None:
            if self.p.y_dim:
                images['f'] = f.mean(0, keepdims=True).detach().cpu()
                images['y'] = y.mean(0, keepdims=True).detach().cpu()
                if c_obs is not None:
                    # [B, K] -> [B, K, K] -> [K, K]
                    images['f_given_c'] = ((f.unsqueeze(1) * c_obs.unsqueeze(-1)).sum(0) / c_obs.sum(0).unsqueeze(-1)).detach().cpu()
                    images['y_given_c'] = ((y.unsqueeze(1) * c_obs.unsqueeze(-1)).sum(0) / c_obs.sum(0).unsqueeze(-1)).detach().cpu()
            #if self.p.z_dim:
            #    images['z'] = f.mean(0, keepdims=True).detach().cpu()
            #    if c_obs is not None:
            #        # [B, H] -> [B, K, H] -> [K, H]
            #        images['z_given_c'] = ((z.unsqueeze(1) * c_obs.unsqueeze(-1)).sum(0) / c_obs.sum(0).unsqueeze(-1)).detach().cpu()
        return loss, ret

    def _loss(self, x_obs, c_obs=None, samples=None, images=None):
        """
        :param x_obs: [B, D]
        """
        B, H, K, D = x_obs.shape[0], self.p.z_dim, self.p.y_dim, self.p.data_dim

        # Approximate posteriors and samples
        q_F = self.q.F(x_obs)
        # [B, K]
        f = q_F.sample() # not rsample
        assert_shape(f, (B, K), "f ~ F|X=x, \lambda")

        q_Y = self.q.Y(x=x_obs, f=f)
        y = q_Y.rsample()  # with reparameterisation! (important)
        assert_shape(y, (B, K), "y ~ Y|F=f, \lambda")

        q_Z = self.q.Z(x=x_obs, y=y)
        # [B, H]
        z = q_Z.rsample()  # with reparameterisation
        assert_shape(z, (B, H), "z ~ Z|X=x, \lambda")

        # Priors
        p_F = self.p.F()
        if p_F.batch_shape != x_obs.shape[:1]:
            p_F = p_F.expand(x_obs.shape[:1] + p_F.batch_shape)

        p_Y = self.p.Y(f)  # we condition on f ~ q_F  thus batch_shape is already correct

        p_Z = self.p.Z()
        if p_Z.batch_shape != x_obs.shape[:1]:
            p_Z = p_Z.expand(x_obs.shape[:1] + p_Z.batch_shape)

        # Sampling distribution
        p_X = self.p.X(y=y, z=z)  # we condition on y ~ q_Y

        # Return type
        ret = OrderedDict(
            loss=0.,
        )

        # ELBO: the first term is an MC estimate (we sampled (f,y))
        # the second term is exact
        # the third tuse_self_criticis an MC estimate (we sampled f)
        ll = p_X.log_prob(x_obs)
        kl_Z = td.kl_divergence(q_Z, p_Z)
        kl_Y = td.kl_divergence(q_Y, p_Y)
        kl_F = td.kl_divergence(q_F, p_F)

        # Logging ELBO terms
        ret['D'] = -ll.mean(0).item()
        ret['R'] = (kl_F + kl_Y + kl_Z).mean(0).item()
        ret['ELBO'] = (ll - kl_F - kl_Y - kl_Z).mean(0).item()
        if self.p.y_dim:
            ret['R_F'] = kl_F.mean(0).item()
            ret['R_Y'] = kl_Y.mean(0).item()
        if self.p.z_dim:
            ret['R_Z'] = kl_Z.mean(0).item()

        # Gradient surrogates and loss

        # i) reparameterised gradient (g_rep)
        grep_surrogate = ll - kl_Z - kl_F - kl_Y

        # ii) score function estimator (g_SFE)
        if self.p.y_dim:
            # E_ZFY[ log p(x|z,f,y)] - -KL(Z) - KL(F) - E_F[ KL(Y) ]
            # E_F[ E_Y[ E_Z[ log p(x|z,f,y) ] - KL(Y) ] ] -KL(Z) - KL(F)
            # E_F[ r(F) ] for r(f) = log p(x|z,f,y)
            # r(f).detach() * log q(f)
            reward = (ll - kl_Y).detach() if self.q.mean_field else (ll - kl_Y - kl_Z).detach()
            # Variance reduction tricks
            if self.use_self_critic:
                criticised_reward = reward - self.critic(x_obs, z=z, q_F=q_F).detach()
            else:
                criticised_reward = reward
            if self.use_reward_standardisation:
                reward_avg, reward_std = self.update_reward_stats(criticised_reward)
                standardised_reward = (criticised_reward - reward_avg) / np.minimum(reward_std, 1.0)
            else:
                standardised_reward = criticised_reward

            sfe_surrogate = standardised_reward * q_F.log_prob(f)

            # Loggin SFE variants
            ret['SFE_reward'] = reward.mean(0).item()
            if self.use_self_critic:
                ret['SFE_criticised_reward'] = criticised_reward.mean(0).item()
            if self.use_reward_standardisation:
                ret['SFE_standardised_reward'] = standardised_reward.mean(0).item()
        else:
            sfe_surrogate = torch.zeros_like(grep_surrogate)

        # []
        loss = -(grep_surrogate + sfe_surrogate).mean(0)
        ret['loss'] = loss.item()

        if samples is not None:
            if self.p.y_dim:
                samples['f'] = f.detach().cpu()
                samples['y'] = y.detach().cpu()
            if self.p.z_dim:
                samples['z'] = z.detach().cpu()
        if images is not None:
            if self.p.y_dim:
                images['f'] = f.mean(0, keepdims=True).detach().cpu()
                images['y'] = y.mean(0, keepdims=True).detach().cpu()
                if c_obs is not None:
                    # [B, K] -> [B, K, K] -> [K, K]
                    images['f_given_c'] = ((f.unsqueeze(1) * c_obs.unsqueeze(-1)).sum(0) / c_obs.sum(0).unsqueeze(-1)).detach().cpu()
                    images['y_given_c'] = ((y.unsqueeze(1) * c_obs.unsqueeze(-1)).sum(0) / c_obs.sum(0).unsqueeze(-1)).detach().cpu()
            #if self.p.z_dim:
            #    images['z'] = f.mean(0, keepdims=True).detach().cpu()
            #    if c_obs is not None:
            #        # [B, H] -> [B, K, H] -> [K, H]
            #        images['z_given_c'] = ((z.unsqueeze(1) * c_obs.unsqueeze(-1)).sum(0) / c_obs.sum(0).unsqueeze(-1)).detach().cpu()
        return loss, ret

    def estimate_ll(self, x_obs, num_samples):
        with torch.no_grad():
            self.eval()
            # log 1/N \sum_{n} p(x, z_n)/q(z_n|x)
            # [N, B, K], [N, B, K], [N, B, H]
            f, y, z = self.q.sample(x_obs, (num_samples,))
            # Here I compute: log p(f) + log p(y|f) + log p(z) + log p(x|y,z)
            # [N, B]
            log_p = self.p.log_prob(f=f, y=y, z=z, x=x_obs)
            # Here I compute: log q(f|x) + log q(y|x,f) + log q(z|x,y)
            # [N, B]
            log_q = self.q.log_prob(x=x_obs, f=f, y=y, z=z)
            # [B]
            ll = torch.logsumexp(log_p - log_q, 0) - np.log(num_samples)
        return ll

    def estimate_ll_per_bit(self, x_obs, num_samples):
        with torch.no_grad():
            # log 1/N \sum_{n} p(x, z_n)/q(z_n|x)
            # [N, B, K], [N, B, K], [N, B, H]
            f, y, z = self.q.sample(x_obs, (num_samples,))
            # [N, B, D]
            log_p = self.p.log_prob(f=f, y=y, z=z, x=x_obs, per_bit=True)
            # [N, B]
            log_q = self.q.log_prob(z=z, f=f, y=y, x=x_obs)
            # [B, D]
            ll = torch.logsumexp(log_p - log_q.unsqueeze(-1), 0) - np.log(num_samples)
        return ll

