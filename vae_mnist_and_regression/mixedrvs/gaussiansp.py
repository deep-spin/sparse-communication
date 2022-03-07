import sys
from itertools import product

import numpy as np
from scipy.special import ndtri

import torch
import torch.distributions as td
from torch.distributions import constraints
from torch.distributions.utils import _standard_normal, broadcast_all

from mixedrvs.dirichlet import MaskedDirichlet
from entmax import sparsemax

# numeric magic, VERY IMPORTANT
from mixedrvs.torch_log_ndtr import log_ndtr


class GaussianSparsemax(td.Distribution):

    arg_constraints = {
        'loc': constraints.real,
        'scale': constraints.positive
    }
    support = td.constraints.simplex
    has_rsample = True

    @classmethod
    def all_faces(K):
        """Generate a list of 2**K - 1 bit vectors indicating all possible faces of a K-dimensional simplex."""
        return list(product([0, 1], repeat=K))[1:]

    def __init__(self, loc, scale, cdf_samples=100, KL_samples=1, validate_args=None):
        self.loc, self.scale = broadcast_all(loc, scale)
        self._cdf_samples = cdf_samples
        self._KL_samples = KL_samples
        batch_shape, event_shape = self.loc.shape[:-1], self.loc.shape[-1:]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(GaussianSparsemax, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape + self.event_shape)
        new.scale = self.scale.expand(batch_shape + self.event_shape)
        new._cdf_samples = self._cdf_samples
        new._KL_samples = self._KL_samples
        super(GaussianSparsemax, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape=torch.Size()):
        # sample_shape + batch_shape + (K,)
        z = td.Normal(loc=self.loc, scale=self.scale).rsample(sample_shape)
        return sparsemax(z, dim=-1)

    def IS(self, loc, cov, diag, sample_size):
        q = td.Normal(loc=loc, scale=diag)
        p = td.MultivariateNormal(loc, cov, validate_args=False)
        # [S, B, K]
        x = q.sample((sample_size,))
        x = torch.where(x > 0, -x, x)
        # [S, B]
        log_prob = (p.log_prob(x) - 2.0*q.log_prob(x).sum(-1)).logsumexp(0) - np.log(sample_size)
        return log_prob

    def _log_cdf_integrate(self, y, loc, scale, var, face, zero, n_samples):

        if not hasattr(self, '_inv_cdf_us'):
            us = np.linspace(0, 1, n_samples + 2)[1:-1]
            self.log_du = np.log(us[1] - us[0])
            inv_cdf_us = ndtri(us)
            inv_cdf_us = torch.tensor(inv_cdf_us, device=y.device, dtype=y.dtype)
            self._inv_cdf_us = inv_cdf_us
        else:
            inv_cdf_us = self._inv_cdf_us

        inv_vars = var.reciprocal()

        # [B]
        inv_sum_inv_vars = torch.where(face, inv_vars, zero).sum(dim=-1).reciprocal()

        # [B, K]
        term_a_coef = torch.sqrt(inv_sum_inv_vars).unsqueeze(dim=-1) / scale
        # [B, K, S]

        # [1, 1, S]
        inv_cdf_us = inv_cdf_us.reshape(tuple(1 for _ in term_a_coef.shape)
                                        + inv_cdf_us.shape)
        # [B, K, 1]
        term_a_coef = term_a_coef.unsqueeze(dim=-1)

        # [B, K, S]
        term_a = term_a_coef * inv_cdf_us

        # [B, Ki, Kj]
        pairwise_diff = (y - loc).unsqueeze(-1) + loc.unsqueeze(-2)
        pairwise_diff = pairwise_diff / var.unsqueeze(-1)

        # [B, Kj]
        num = torch.where(face.unsqueeze(-1), pairwise_diff, zero).sum(dim=-2)

        # [B, Kj] also
        idenom = inv_sum_inv_vars.unsqueeze(dim=-1) / scale

        term_b = num * idenom

        # [B, K, S]
        t = term_a - term_b.unsqueeze(-1)
        log_phi_t = log_ndtr(t)

        # add up over zeros
        log_integrand = torch.where(face.unsqueeze(-1),
                                    zero,
                                    log_phi_t).sum(dim=-2)

        # do trapezoidal rule: average the two function evals in log space.
        log_centers = torch.logaddexp(log_integrand[..., :-1],
                                      log_integrand[..., 1:])
        log_centers -= np.log(2)  # for trapezoidal
        log_centers += self.log_du  #  interval width

        log_cdf = torch.logsumexp(log_centers, dim=-1)

        return log_cdf


    def _log_cdf_mc(self, y, loc, scale, var, face, zero, n_samples):
        # Joint log prob for the zeros

        # [B, K]
        inv_vars = var.reciprocal()
        # [B]
        inv_sum_inv_vars = (torch.where(face, inv_vars, zero)
                            .sum(dim=-1).reciprocal())
        zscore = (y - loc) / var
        # [B]
        zsc_nz = torch.where(face, zscore, zero).sum(dim=-1)

        # monte carlo sample (possible optimisation: share noise across instances in batch, it should be safe from an MC point of view)
        # [B, S]
        t0 = torch.randn(tuple(1 for _ in loc.shape[:-1]) + (n_samples,), dtype=y.dtype, device=y.device)

        # [B]
        adj_loc = (-inv_sum_inv_vars * zsc_nz)
        adj_scale = torch.sqrt(inv_sum_inv_vars)

        # [B, S]
        T = adj_loc.unsqueeze(dim=-1) + adj_scale.unsqueeze(dim=-1) * t0
        # [B, 1, S]
        T = T.unsqueeze(dim=-2)

        # [B, K, 1]
        loc_ = loc.unsqueeze(dim=-1)
        scale_ = scale.unsqueeze(dim=-1)

        # compute univariate standard normal CDF, elementwise.
        # This crucially requires careful numerical code.
        # [B, K, S]
        T_zsc = (T - loc_) / scale_
        log_cdf_1d = log_ndtr(T_zsc)

        # [B, S]
        # we mask and reduce_sum over k in [K] (the zeros)
        log_cdf_1d = (torch.where(face.unsqueeze(dim=-1), zero, log_cdf_1d)
                      .sum(dim=-2))
        log_cdf_1d -= np.log(n_samples)

        # [B]
        # reduce the sample dimension
        log_cdf = torch.logsumexp(log_cdf_1d, dim=-1)
        return log_cdf


    def log_prob(self, y, pivot_alg='first', tiny=1e-9, huge=1e9,
            n_samples=None):

        if n_samples is None:
            n_samples = self._cdf_samples

        K = y.shape[-1]

        # create a single zero with the same dtype and device sa y
        zero = y.new_zeros(1)

        # [B, K]
        loc = self.loc
        scale = self.scale
        var = scale ** 2

        # The face contains the set of coordinates greater than zero
        # [B, K]
        face = y > 0

        # Chose a pivot coordinate (a non-zero coordinate)
        # [B]
        if pivot_alg == 'first':
            ind_pivot = torch.argmax((face > 0).float(), -1)
        elif pivot_alg == 'random':
            ind_pivot = td.Categorical(
                probs=face.float()/(face.float().sum(-1, keepdims=True))
            ).sample()

        # Select a batch of pivots
        # [B, K]
        pivot_indicator = torch.nn.functional.one_hot(ind_pivot, K).bool()

        # All non-zero coordinates but the pivot
        # [B, K]
        others = torch.logical_xor(face, pivot_indicator)

        # The value of the pivot coordinate
        # [B]
        t = (y * pivot_indicator.float()).sum(-1)

        # Pivot mean and variance
        # [B]
        t_mean = torch.where(pivot_indicator, loc, zero).sum(-1)
        t_scale = torch.where(pivot_indicator, scale, zero).sum(-1)
        #t_var = torch.where(pivot_indicator, var, zero).sum(-1)

        # Difference with respect to the pivot
        # [B, K]
        y_diff = torch.where(others, y - t.unsqueeze(-1), zero)

        # [B, K]
        mean_diff = torch.where(others, loc - t_mean.unsqueeze(-1), zero)

        # Joint log pdf for the non-zeros
        # [B, K, K]
        #diag = torch.diag_embed(torch.where(others, var, var.new_ones(1)))
        #offset = t_var.unsqueeze(-1).unsqueeze(-1)

        # A = diag(d)
        # u = t_var * 1
        # v = 1
        # inv(A) = diag(1/d)

        # pdf:
        # 2pi ^ {-k/2} det(cov)^{-1/2} exp(-1/2*(x-mu)Tcov^{-1} (x-mu))

        # [B, K]
        cov_diag = torch.where(others, var, var.new_ones(1))
        # [B, K, rank=1]
        # others: [B, K], t_scale: [B], the final unsqueeze creates the rank=1 dimension
        cov_factor = torch.where(others, t_scale.unsqueeze(-1), zero).unsqueeze(-1)

        # We need a multivariate normal for the non-zero coordinates in `other`
        # but to batch mvns we will need to use K-by-K covariances
        # we can do so by embedding the lower-dimensional mvn in a higher dimensional mvn
        # with cov=I.
        # [B, K, K]
        #cov_mask = others.unsqueeze(-1) * others.unsqueeze(-2)
        #cov = torch.where(cov_mask, diag + offset, diag)

        # This computes log prob of y[other] under  the lower dimensional mvn
        # times log N(0|0,1) for the other dimensions
        # [B]
        #log_prob = td.MultivariateNormal(mean_diff, cov).log_prob(y_diff)
        # [B]
        log_prob = td.LowRankMultivariateNormal(mean_diff, cov_diag=cov_diag, cov_factor=cov_factor).log_prob(y_diff)
        #log_prob = torch.zeros(loc.shape[:-1], device=loc.device)
        # so we discount the contribution from the masked coordinates
        # [B, K]
        log_prob0 = td.Normal(torch.zeros_like(mean_diff), torch.ones_like(mean_diff)).log_prob(torch.zeros_like(y_diff))
        # [B]
        log_prob = log_prob - torch.where(others, zero, log_prob0).sum(-1)

        # log_cdf = self._log_cdf_mc(y, loc, scale, var, face, zero, n_samples)
        log_cdf = self._log_cdf_integrate(y, loc, scale, var, face, zero, n_samples)


        # [B]
        log_det = face.float().sum(-1).log()

        # [B]
        return log_prob + log_cdf + log_det

    def log_prob_IS(self, y, n_samples=None):

        if n_samples is None:
            n_samples = self._cdf_samples

        assert y.shape[:-1] == self.batch_shape, f"For now I need the same batch_shape: {y.shape} got {y.shape[:-1]} instead of {self.batch_shape}"

        batch_shape = y.shape[:-1]
        K = self.loc.shape[-1]

        # [B]
        dtau = td.Normal(
            loc=torch.zeros(batch_shape, device=self.loc.device),
            scale=torch.ones(batch_shape, device=self.scale.device)
        )
        # [B, K]
        dnu = td.Exponential(torch.ones(batch_shape + (K,), device=self.loc.device),
        )


        # [B]
        n = (y > 0).float().sum(-1)

        # [S, B, K]
        y = y.expand((n_samples,) + batch_shape + (K,))

        # [S, B, K]
        antisupp = y == 0
        # [S, B]
        n_antisupp = antisupp.float().sum(-1)
        # [S, B]
        taus = dtau.sample(sample_shape=(n_samples,))
        # [S, B, K]
        nus = dnu.sample(sample_shape=(n_samples,))


        # [S, B, K]
        X = y + taus.unsqueeze(-1)
        X = X - torch.where(antisupp, nus, torch.zeros_like(nus))

        # this just asserts we are sampling over the correct set
        # print("Error: ", torch.sum((sparsemax(X, 1) - y) ** 2))

        # probability of each of these Xs
        # [S, B]
        logp_X = td.Normal(loc=self.loc, scale=self.scale).log_prob(X).sum(-1)

        # probability of our importance distribution (iid over tau and nu)
        # [S, B]
        logq_X = dtau.log_prob(taus)
        # [S, B]
        logq_X += torch.where(antisupp, dnu.log_prob(nus), torch.zeros_like(nus)).sum(-1)

        # \int_s p(x) dx = E_p [1_S] = E_q[1_S * p/q]
        # 1_S is 1 by construction

        ## this multiplication by n here gives the expected result
        ##  - in the 2d case
        ##  - in the 3d case when looking at dimension 1- and 2- faces.
        ## WHERE DOES IT COME FROM? I discovered it just randomly.
        ## Can we check if this gives correct values on the interior on the 3-simplex?
        # [B]
        log_prob = (logp_X - logq_X).logsumexp(0) - np.log(n_samples)
        log_prob += n.log()

        return log_prob


@td.register_kl(GaussianSparsemax, GaussianSparsemax)
def _kl_gaussiansparsemax_gaussiansparsemax(p, q):
    # [S, ...]
    x = p.rsample((p._KL_samples,))
    return (p.log_prob(x) - q.log_prob(x)).mean(dim=0)


class GaussianSparsemaxPrior(td.Distribution):

    def __init__(self, pF, alpha_net, validate_args=False):
        self.pF = pF
        self.alpha_net = alpha_net
        batch_shape, event_shape = pF.batch_shape, pF.event_shape
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(GaussianSparsemaxPrior, _instance)
        new.pF = self.pF.expand(batch_shape)
        new.alpha_net = self.alpha_net
        super(GaussianSparsemaxPrior, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=torch.Size()):
        f = self.pF.sample(sample_shape)
        Y = MaskedDirichlet(f.bool(), self.alpha_net(f))
        return Y.sample()

    def log_prob(self, value):
        f = (value > 0).float()
        Y = MaskedDirichlet(f.bool(), self.alpha_net(f))
        return self.pF.log_prob(f) + Y.log_prob(value)


@td.register_kl(GaussianSparsemax, GaussianSparsemaxPrior)
def _kl_gaussiansparsemax_gaussiansparsemaxprior(p, q):
    # [S, ...]
    x = p.rsample((p._KL_samples,))
    return (p.log_prob(x) - q.log_prob(x)).mean(dim=0)


