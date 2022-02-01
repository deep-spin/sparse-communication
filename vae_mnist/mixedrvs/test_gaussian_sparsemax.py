import torch
import torch.distributions as td
from scipy.stats import multivariate_normal

from gaussiansp import GaussianSparsemax

def _scipy_log_prob(y, loc, scale, pivot_alg='first', tiny=1e-12, huge=1e12):
    K = y.shape[-1]
    # [B, K]
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
    t_mean = torch.where(pivot_indicator, loc, torch.zeros_like(loc)).sum(-1)
    t_var = torch.where(pivot_indicator, var, torch.zeros_like(var)).sum(-1)

    # Difference with respect to the pivot
    # [B, K]
    y_diff = torch.where(others, y - t.unsqueeze(-1), torch.zeros_like(y))
    # [B, K]
    mean_diff = torch.where(
        others,
        loc - t_mean.unsqueeze(-1),
        torch.zeros_like(loc)
    )

    # Joint log pdf for the non-zeros
    # [B, K, K]
    diag = torch.diag_embed(torch.where(others, var, torch.ones_like(var)))
    offset = t_var.unsqueeze(-1).unsqueeze(-1)
    # We need a multivariate normal for the non-zero coordinates in `other`
    # but to batch mvns we will need to use K-by-K covariances
    # we can do so by embedding the lower-dimensional mvn in a higher dimensional mvn
    # with cov=I.
    # [B, K, K]
    cov_mask = others.unsqueeze(-1) * others.unsqueeze(-2)
    cov = torch.where(cov_mask, diag + offset, diag)
    # This computes log prob of y[other] under  the lower dimensional mvn
    # times log N(0|0,1) for the other dimensions
    # [B]
    log_prob = td.MultivariateNormal(mean_diff, cov).log_prob(y_diff)
    # so we discount the contribution from the masked coordinates
    # [B, K]
    log_prob0 = td.Normal(torch.zeros_like(mean_diff), torch.ones_like(mean_diff)).log_prob(torch.zeros_like(y_diff))
    log_prob = log_prob - torch.where(others, torch.zeros_like(log_prob0), log_prob0).sum(-1)

    # Joint log prob for the zeros (needs the cdf)
    # [B]
    constant_term = 1. / torch.where(face, 1./var, torch.zeros_like(var)).sum(-1)

    B = y.shape[0]
    log_cdf = y.new_zeros(B)

    zsc = (y - loc) / var


    for i in range(B):
        if (~face[i]).sum() == 0:
            continue

        # The non-zeros get a tiny variance
        # [K, K]
        diag_ = torch.diag(var[0][~face[i]])
        # [1]
        offset_ = constant_term[i]  # .unsqueeze(-1).unsqueeze(-1)
        cov_ = diag_ + offset_

        # [B]
        mean_constant_term = constant_term[i] * zsc[i][face[i]].sum()
        mean_ = loc[0][~face[i]] + mean_constant_term

        # [B]

        # invoke scipy here
        log_cdf[i] = multivariate_normal.logcdf(torch.zeros_like(mean_).numpy(),
                                                mean=mean_.numpy(),
                                                cov=cov_.numpy(),
                                                allow_singular=True,
                                                maxpts=100000000,
                                                abseps=1e-12,
                                                releps=1e-12)

    # [B]
    log_det = face.float().sum(-1).log()

    # [B]
    return log_prob + log_cdf + log_det


def test_dtypes():

    for dtype in (torch.float32, torch.float64):
        mu = torch.tensor([.6, .2, .2], dtype=dtype)
        std = torch.tensor([.2, .1, .3], dtype=dtype)
        gs = GaussianSparsemax(loc=mu, scale=std)
        Y = gs.sample(sample_shape=(1,))
        lp = gs.log_prob(Y)
        print(Y.dtype, lp.dtype)


def main():
    torch.set_default_dtype(torch.double)
    torch.set_default_tensor_type(torch.DoubleTensor)

    mu = torch.tensor([.6, .2, .2]).unsqueeze(0)
    std = 3 * torch.tensor([.2, .1, .3]).unsqueeze(0)

    gs = GaussianSparsemax(loc=mu, scale=std)

    gs = gs.expand((5,))
    Y = gs.sample()
    print(Y)

    print("scipy")
    print(_scipy_log_prob(Y, mu, std))

    Y = torch.stack([Y, Y])

    print("our log_prob (1-D 1000 samples)")
    print(gs.log_prob(Y, n_samples=1000))

    # lp = torch.zeros(5)
    # print("our log_prob (1-D MC very many samples)")

    # N = 1000
    # for _ in range(N):
        # lp += gs.log_prob(Y, n_samples=1000)

    # print(lp / N)



if __name__ == '__main__':
    #test_dtypes()
    main()
