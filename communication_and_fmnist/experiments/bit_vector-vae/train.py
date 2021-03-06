import argparse
import pathlib

import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli
from torchvision import datasets
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from lvmhelpers.structmarg import \
    TopKSparsemaxWrapper, TopKSparsemaxMarg, SparseMAPWrapper, SparseMAPMarg
from lvmhelpers.sfe import \
    BitVectorSFEWrapper, SFEDeterministicWrapper, \
    BitVectorScoreFunctionEstimator
from lvmhelpers.gumbel import \
    BitVectorGumbelSoftmaxWrapper, BitVectorGumbel
from lvmhelpers.nvil import \
    BitVectorNVILWrapper, BitVectorNVIL
from lvmhelpers.utils import DeterministicWrapper, populate_common_params
from lvmhelpers.gaussian import BitVectorGaussianSparsemaxWrapper
from lvmhelpers.concrete import BinaryConcreteWrapper, BitVectorConcrete
from lvmhelpers.hardconcrete2 import HardConcreteWrapper, BitVectorHardConcrete

from data import transform
from archs import MLP, CategoricalGenerator
from opts import populate_experiment_params


class VAE(pl.LightningModule):
    def __init__(
            self,
            n_features,
            hidden_size,
            out_rank,
            out_classes,
            budget,
            init,
            mode,
            entropy_coeff,
            latent_size,
            normalizer,
            topksparse,
            gs_tau,
            temperature_decay,
            temperature_update_freq,
            straight_through,
            baseline_type,
            topk,
            random_seed,
            batch_size,
            lr,
            weight_decay,
            optimizer):
        super(VAE, self).__init__()

        self.save_hyperparameters()

        inf = MLP(
            dim_in=n_features,
            dim_hid=self.hparams.hidden_size,
            dim_out=self.hparams.latent_size,
            n_layers=0)
        gen = MLP(
            dim_in=self.hparams.latent_size,
            dim_hid=self.hparams.hidden_size,
            dim_out=self.hparams.n_features * self.hparams.out_rank,
            n_layers=0)
        gen = CategoricalGenerator(
            gen,
            n_features=self.hparams.n_features,
            out_rank=self.hparams.out_rank,
            n_classes=256)

        loss_fun = reconstruction_loss

        if self.hparams.mode == 'sfe':
            inf = BitVectorSFEWrapper(
                inf, baseline_type=self.hparams.baseline_type)
            gen = SFEDeterministicWrapper(gen)
            lvm_method = BitVectorScoreFunctionEstimator
        elif self.hparams.mode == 'nvil':
            inf = BitVectorNVILWrapper(inf, input_size=n_features)
            gen = DeterministicWrapper(gen)
            lvm_method = BitVectorNVIL
        elif self.hparams.mode == 'gs':
            inf = BitVectorGumbelSoftmaxWrapper(
                inf,
                temperature=self.hparams.gs_tau,
                straight_through=self.hparams.straight_through)
            gen = DeterministicWrapper(gen)
            lvm_method = BitVectorGumbel
        elif self.hparams.mode == 'gaussian-sparsemax':
            inf = BitVectorGaussianSparsemaxWrapper(
                inf,
                temperature=self.hparams.gs_tau,
                straight_through=self.hparams.straight_through)
            gen = DeterministicWrapper(gen)
            lvm_method = BitVectorGumbel
        elif self.hparams.mode == 'topksparse':
            inf = TopKSparsemaxWrapper(inf, k=self.hparams.topksparse)
            gen = DeterministicWrapper(gen)
            lvm_method = TopKSparsemaxMarg
        elif self.hparams.mode == 'sparsemap':
            inf = SparseMAPWrapper(
                inf, budget=self.hparams.budget, init=self.hparams.init)
            gen = DeterministicWrapper(gen)
            lvm_method = SparseMAPMarg
        elif self.hparams.mode == 'binary-concrete':
            inf = BinaryConcreteWrapper(
                inf,
                temperature=self.hparams.gs_tau,
                straight_through=self.hparams.straight_through)
            gen = DeterministicWrapper(gen)
            lvm_method = BitVectorConcrete
        elif self.hparams.mode == 'hard-concrete':
            inf = HardConcreteWrapper(
                inf,
                temperature=self.hparams.gs_tau,
                straight_through=self.hparams.straight_through)
            gen = DeterministicWrapper(gen)
            lvm_method = BitVectorHardConcrete
        else:
            raise RuntimeError(f"Unknown training mode: {self.hparams.mode}")

        self.lvm_method = lvm_method(
            inf,
            gen,
            loss_fun,
            encoder_entropy_coeff=1.0)

    def forward(self, _inf_input, inf_input):
        return self.lvm_method(_inf_input, torch.zeros_like(_inf_input), inf_input)

    def training_step(self, batch, batch_nb):
        inf_input, _ = batch
        _inf_input = inf_input.to(dtype=torch.float) / 255
        training_result = self(_inf_input, inf_input)
        loss = training_result['loss']
        #print('\nloss:', loss)

        result = pl.TrainResult(minimize=loss)
        if (self.hparams.mode == 'gaussian-sparsemax')|(self.hparams.mode == 'hard-concrete'):
            elbo = \
                - training_result['log']['loss'] + \
                training_result['log']['encoder_entropy'] - \
                self.hparams.latent_size * torch.log(torch.tensor(3.0))        
        elif self.hparams.mode == 'binary-concrete':
            elbo = \
                - training_result['log']['loss'] + \
                training_result['log']['encoder_entropy']      
        else:
            elbo = \
                - training_result['log']['loss'] + \
                training_result['log']['encoder_entropy'] + \
                self.hparams.latent_size * torch.log(torch.tensor(0.5))

        result.log(
            '-train_elbo',
            -elbo,
            prog_bar=True, logger=True)

        if 'support' in training_result['log'].keys():
            result.log(
                'train_support_median', training_result['log']['support'],
                reduce_fx=torch.median, on_epoch=True, on_step=False)
            result.log(
                'train_support_mean', torch.mean(training_result['log']['support']),
                prog_bar=True, reduce_fx=torch.mean, on_epoch=True, on_step=False)

        # Update temperature if Gumbel
        if self.hparams.mode == 'gs':
            self.lvm_method.encoder.update_temperature(
                self.global_step,
                self.hparams.temperature_update_freq,
                self.hparams.temperature_decay)
            result.log(
                'temperature', self.lvm_method.encoder.temperature)
        if (self.hparams.mode == 'binary-concrete')|(self.hparams.mode == 'hard-concrete'):
            self.lvm_method.encoder.update_temperature(
                self.global_step,
                self.hparams.temperature_update_freq,
                self.hparams.temperature_decay)
            result.log(
                'temperature', self.lvm_method.encoder.temperature)        

        return result

    def validation_step(self, batch, batch_nb):
        inf_input, _ = batch
        _inf_input = inf_input.to(dtype=torch.float) / 255
        validation_result = self(_inf_input, inf_input)

        if (self.hparams.mode == 'gaussian-sparsemax')|(self.hparams.mode == 'hard-concrete'):
            elbo = \
                - validation_result['log']['loss'] + \
                validation_result['log']['encoder_entropy'] - \
                self.hparams.latent_size * torch.log(torch.tensor(3.0))
        elif self.hparams.mode == 'binary-concrete':
            elbo = \
                - validation_result['log']['loss'] + \
                validation_result['log']['encoder_entropy']     
        else:
            elbo = \
                - validation_result['log']['loss'] + \
                validation_result['log']['encoder_entropy'] + \
                self.hparams.latent_size * torch.log(torch.tensor(0.5))

        if (self.current_epoch + 1) % 20 == 0:
            n_importance_samples = 64
        else:
            n_importance_samples = 16

        logp_x_bits, logp_x_nats, non_supp_influence = \
            self.compute_importance_sampling(
                validation_result['log'],
                inf_input,
                n_importance_samples)

        result = pl.EvalResult(checkpoint_on=-elbo)
        result.log('-val_elbo', -elbo, prog_bar=True)
        result.log('val_logp_x_bits', logp_x_bits, prog_bar=True)
        result.log('val_non_supp_influence', non_supp_influence)

        if 'support' in validation_result['log'].keys():
            result.log(
                'val_support_median', validation_result['log']['support'],
                reduce_fx=torch.median)
            result.log(
                'val_support_mean', torch.mean(validation_result['log']['support']),
                reduce_fx=torch.mean, on_epoch=True)

        return result

    def test_step(self, batch, batch_nb):
        inf_input, _ = batch
        _inf_input = inf_input.to(dtype=torch.float) / 255
        test_result = self(_inf_input, inf_input)

        if (self.hparams.mode == 'gaussian-sparsemax')|(self.hparams.mode == 'hard-concrete'):
            elbo = \
                - test_result['log']['loss'] + \
                test_result['log']['encoder_entropy'] - \
                self.hparams.latent_size * torch.log(torch.tensor(3.0))
        elif self.hparams.mode == 'binary-concrete':
            elbo = \
                - test_result['log']['loss'] + \
                test_result['log']['encoder_entropy'] 
        else:
            elbo = \
                - test_result['log']['loss'] + \
                test_result['log']['encoder_entropy'] + \
                self.hparams.latent_size * torch.log(torch.tensor(0.5))

        logp_x_bits, logp_x_nats, non_supp_influence = \
            self.compute_importance_sampling(
                test_result['log'],
                inf_input,
                1024)

        result = pl.EvalResult(checkpoint_on=-elbo)
        result.log('-test_elbo', -elbo, prog_bar=True)
        result.log('test_logp_x_bits', logp_x_bits, prog_bar=True)
        result.log('test_non_supp_influence', non_supp_influence)
        result.log('test_distortion', test_result['log']['loss'])
        if (self.hparams.mode == 'gaussian-sparsemax')|(self.hparams.mode == 'hard-concrete'):
            result.log(
                'test_rate',
                + self.hparams.latent_size * torch.log(torch.tensor(3.0))
                - test_result['log']['encoder_entropy'])
        elif self.hparams.mode == 'binary-concrete':
            result.log(
                'test_rate',
                - test_result['log']['encoder_entropy'])         
        else:        
            result.log(
                'test_rate',
                - self.hparams.latent_size * torch.log(torch.tensor(0.5))
                - test_result['log']['encoder_entropy'])

        if 'support' in test_result['log'].keys():
            result.log(
                'test_support_step', test_result['log']['support'],
                reduce_fx=torch.median)
            result.log(
                'test_support_mean', torch.mean(test_result['log']['support']),
                reduce_fx=torch.mean, on_epoch=True)

        return result

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            torch.utils.data.Subset(
                datasets.FashionMNIST(
                    'data/bit_vector-vae/fmnist_data/',
                    train=True,
                    download=True,
                    transform=transform),
                indices=range(55000)),
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            torch.utils.data.Subset(
                datasets.FashionMNIST(
                    'data/bit_vector-vae/fmnist_data/',
                    train=True,
                    download=True,
                    transform=transform),
                indices=range(55000, 60000)),
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            datasets.FashionMNIST(
                'data/bit_vector-vae/fmnist_data/',
                train=False,
                download=True,
                transform=transform),
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True)

    def compute_importance_sampling(self, util_dict, inf_input, n_samples):

        distr = util_dict['distr']
        if self.hparams.mode == 'sparsemap' or self.hparams.mode == 'topksparse':
            sampling_distr = Bernoulli(
                probs=torch.full(
                    (inf_input.size(0), self.hparams.latent_size),
                    0.5).to(inf_input.device))
        else:
            sampling_distr = distr

        # importance_samples: [n_samples, batch_size, nlatents]
        importance_samples = sampling_distr.sample((n_samples,))
        # logq_z_given_x_importance: [n_samples, batch_size]
        logq_z_given_x_importance = \
            sampling_distr.log_prob(importance_samples).sum(dim=-1)

        batch_n_samples = 16
        logp_x_given_z_importance = []
        for importance_sample_batch in importance_samples.split(batch_n_samples):
            # Xhat_importance: [batch_n_samples * batch_size, n_features]
            if self.hparams.mode == 'sfe':
                Xhat_importance, _, _ = self.lvm_method.decoder(importance_sample_batch)
            else:
                Xhat_importance = self.lvm_method.decoder(importance_sample_batch)
            # inf_input_repeat: [batch_n_samples * batch_size, n_features]
            inf_input_repeat = inf_input.repeat(
                batch_n_samples, 1, 1).view(-1, inf_input.size(-1))
            # logp_x_given_z_importance: [batch_n_samples, batch_size]
            logp_x_given_z_importance.append(
                reconstruction_loss(
                    inf_input_repeat,
                    importance_samples,
                    inf_input_repeat,
                    Xhat_importance,
                    inf_input_repeat)[0].view(
                        batch_n_samples, inf_input.size(0)))
        # logp_x_given_z_importance: [n_samples, batch_size]
        logp_x_given_z_importance = -torch.cat(logp_x_given_z_importance, dim=0)
        # logp_z: []
        if (self.hparams.mode == 'gaussian-sparsemax')|(self.hparams.mode == 'hard-concrete'):
            logp_z = importance_samples.shape[-1] * torch.log(torch.tensor(1/3))
        elif self.hparams.mode == 'binary-concrete':
            logp_z = importance_samples.shape[-1] * torch.log(torch.tensor(1.0))       
        else:
            logp_z = importance_samples.shape[-1] * torch.log(torch.tensor(0.5))
        # aux will be the log of p(x,z)/q(z|x)
        # samples are taken from q(z|x) and then we assess this value
        # and average over all samples
        # aux: [n_samples, batch_size]
        aux = (
            logp_x_given_z_importance
            + logp_z
            - logq_z_given_x_importance
        )
        # logp_x_importance: [batch_size]
        logp_x_importance = torch.logsumexp(aux, dim=0) - torch.log(
            torch.tensor(float(n_samples))
        )

        non_supp_influence = logp_x_importance.mean(dim=0)

        if self.hparams.mode == 'sparsemap':
            # logp_x_deterministic_term: [batch_size]
            logp_x_deterministic_term = []
            logp_x_given_z = - util_dict['loss_output']
            idxs = util_dict['idxs']
            for k in range(inf_input.size(0)):
                logp_x_deterministic_term.append(
                    torch.logsumexp(
                        logp_x_given_z[torch.tensor(idxs) == k] + logp_z,
                        dim=0))
            logp_x_deterministic_term = torch.stack(logp_x_deterministic_term)

            # logp_x_importance: [batch_size]
            logp_x_importance = torch.logsumexp(
                torch.stack([logp_x_deterministic_term, logp_x_importance]), dim=0)
        elif self.hparams.mode == 'topksparse':
            # need to 'reconstruct' logp_x_given_z since
            # we were just dealing with nonzeros
            # logp_x_deterministic_term: [batch_size]
            mask = distr.view(-1) > 0
            logp_x_given_z = -torch.ones_like(mask).to(
                torch.float32
            ) * float("inf")
            logp_x_given_z = logp_x_given_z.masked_scatter(
                mask, -util_dict['loss_output']).view(distr.shape)
            logp_x_deterministic_term = torch.logsumexp(
                logp_x_given_z + logp_z, dim=-1
            )
            # logp_x_importance: [batch_size]
            logp_x_importance = torch.logsumexp(
                torch.stack([logp_x_deterministic_term, logp_x_importance]), dim=0)
        else:
            non_supp_influence = torch.tensor(0.0)

        logp_x_bits = logp_x_importance.mean(dim=0) / torch.log(torch.tensor(2.0))
        logp_x_bits = - logp_x_bits / self.hparams.n_features
        logp_x_nats = logp_x_importance.mean(dim=0)

        return logp_x_bits, logp_x_nats, non_supp_influence


def reconstruction_loss(
        inf_input,
        discrete_latent_z,
        _gen_input,
        gen_output,
        true_labels):
    Xhat_logits = gen_output.permute(0, 2, 1)
    lv = F.cross_entropy(
        Xhat_logits, true_labels.to(dtype=torch.long), reduction="none"
    )
    return lv.sum(dim=1), {}


def get_model(opt):
    n_features = 28 * 28
    hidden_size = 128
    out_rank = 5
    out_classes = 256
    model = VAE(
        n_features=n_features,
        hidden_size=hidden_size,
        out_rank=out_rank,
        out_classes=out_classes,
        budget=opt.budget,
        init=not opt.noinit,
        mode=opt.mode,
        entropy_coeff=opt.entropy_coeff,
        latent_size=opt.latent_size,
        normalizer=opt.normalizer,
        topksparse=opt.topksparse,
        gs_tau=opt.gs_tau,
        temperature_decay=opt.temperature_decay,
        temperature_update_freq=opt.temperature_update_freq,
        straight_through=opt.straight_through,
        baseline_type=opt.baseline_type,
        topk=opt.topk,
        random_seed=opt.random_seed,
        batch_size=opt.batch_size,
        lr=opt.lr,
        weight_decay=opt.weight_decay,
        optimizer=opt.optimizer)

    return model


def main(params):

    arg_parser = argparse.ArgumentParser()
    arg_parser = populate_experiment_params(arg_parser)
    arg_parser = populate_common_params(arg_parser)
    opts = arg_parser.parse_args(params)

    # fix seed
    pl.seed_everything(opts.random_seed)

    pathlib.Path(
            'data/bit_vector-vae/fmnist_data/').mkdir(
                parents=True, exist_ok=True)

    bit_vector_vae = get_model(opts)

    experiment_name = 'bit-vector'
    model_name = '%s/%s' % (experiment_name, opts.mode)
    other_info = [
        "lr-{}".format(opts.lr),
        "latent_size-{}".format(opts.latent_size),
    ]
    if opts.mode == "sparsemap":
        if opts.budget > 0:
            other_info.append(f"b{opts.budget}")
        if opts.noinit:
            other_info.append("noinit")
    elif opts.mode == "gs":
        if opts.straight_through:
            other_info.append("straight_through")
        other_info.append("decay-{}".format(opts.temperature_decay))
        other_info.append("updatefreq-{}".format(opts.temperature_update_freq))
    elif opts.mode == 'sfe':
        other_info.append("baseline-{}".format(opts.baseline_type))
    elif opts.mode == "topksparse":
        other_info.append("k-{}".format(opts.topksparse))

    model_name = '%s/%s' % (model_name, '_'.join(other_info))

    tb_logger = pl_loggers.TensorBoardLogger(
        'logs/',
        name=model_name)

    tb_logger.log_hyperparams(opts, metrics=None)

    trainer = pl.Trainer(
        progress_bar_refresh_rate=20,
        logger=tb_logger,
        max_epochs=opts.n_epochs,
        weights_save_path='checkpoints/',
        weights_summary='full',
        gpus=1 if torch.cuda.is_available() else 0,
        resume_from_checkpoint=opts.load_from_checkpoint,
        deterministic=True)

    trainer.fit(bit_vector_vae)


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
