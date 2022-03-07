#!/usr/bin/env python
# coding: utf-8

import sys
import os

sys.path.append(os.path.abspath('../../'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.distributions as td
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm.auto import tqdm
from collections import OrderedDict, defaultdict

from mixedrvs.mixed import MixedDirichlet
from cc import train_and_validate as train_and_validate_cc

import random
random.seed(0)
import numpy as np
np.random.seed(0)

torch.manual_seed(0)
device = torch.device('cuda:0')
print("Device", device)

# # Baseline: GLM with CC Likelihood
# 
# 
# [The Continuous Categorical: a Novel Simplex-Valued Exponential Family](https://proceedings.icml.cc/static/paper_files/icml/2020/4730-Paper.pdf)
# 
# 
# GLM with the CC distribution, trained by gradient descent, and compare to their Dirichlet counterparts. 



cc_results = train_and_validate_cc()



# # GLM with Mixed Dirichlet Likelihood


# Global parameters
epochs = 400 # fewer epochs run faster, but at the chance of not converging
# the number of political parties in our output variable, including remainder
num_classes = 5 # a higher number models more parties, with a smaller remainder



def create_Y(num_parties=num_classes):
    """Creates a data frame with our target data.

    Each observation (row) represents a constituency. Each major party 
    corresponds to one column, with an additional column for the remainder class
    summing up all smaller parties and independents. Each entry shows the share 
    of the vote for the corresponding party in the corresponding constituency.
    Thus, rows add to 1, i.e. data is compositional.

    Args:
        num_parties (int): the number of parties to be used in our composition.
            Equivalent to the dimensionality of the response.
    Returns:
        Y (DataFrame): response matrix to be fed as a target to our model.
    """
    
    # Load the raw data directly from the UK government
    url = 'https://data.parliament.uk/resources/constituencystatistics/general-election-results-2019.xlsx'
    raw_data = pd.read_excel(url, sheet_name='voting-full')

    # Add the total share of the vote share by political party to find which are 
    # the largest parties, which will form the basis of our composition.
    # In practice could also go ahead and pick the important parties directly
    # (i.e. Tories, Labour, Lib Dems, SNP...)
    share_by_party = raw_data.pivot_table(index='party_name', values='share', aggfunc=sum)
    share_by_party = share_by_party.sort_values(by='share', ascending=False)
    largest_parties = share_by_party.index[0:num_parties - 1].to_list()

    # Separate the main parties from the rest of the parties 
    # (to be aggregated into a 'remainder' class)
    split_idx = raw_data['party_name'].isin(largest_parties)
    majority_frame = raw_data[split_idx]
    remainder_frame = raw_data[~split_idx]
    remainder_frame = remainder_frame.pivot_table(index='constituency_name', values='share', aggfunc=sum)
    remainder_frame = remainder_frame.rename({'share': 'remainder'}, axis=1)

    # Merge the main parties with the remainder into one frame
    Y = majority_frame.pivot_table(index='constituency_name', columns='party_name', values='share')
    # Outer join makes sure we pick up all constituencies, including those with 
    # no remainder and those with no main parties (e.g. Northern Ireland)
    Y = pd.merge(Y, remainder_frame, left_index=True, right_index=True, how='outer') 
    Y = Y.fillna(0.0) # NA appeared from parties with no votes (e.g. SNP outside of Scotland)

    # Sort by constituencies
    Y = Y.sort_index()
    
    return Y

def create_X():
    """Creates a data frame with the predictors for each constituency.

    Note that our modeling exercise is intended to illustrate the
    CC vs the Dirichlet; therefore the features are chosen for simplicity and 
    reproducibility, directly from the same webpage as the voting data
    (different tab from the same excel file from the UK gov)

    Returns:
        X (DataFrame): design matrix to be fed as an input to our model.
    """
    
    # Load the raw data directly from the UK government
    url = 'https://data.parliament.uk/resources/constituencystatistics/general-election-results-2019.xlsx'
    raw_data = pd.read_excel(url, sheet_name='voting-summary')

    # Pick the predictors that are readily available in this table
    X_vars = ['constituency_name', 'country_name', 'constituency_type', 
              'electorate', 'turnout_2017']
    X = raw_data[X_vars].drop_duplicates()
    # Convert our categorical variables to dummy indicator variables
    dummy1 = pd.get_dummies(X['constituency_type'])
    dummy2 = pd.get_dummies(X['country_name'])
    X = X.drop(['country_name', 'constituency_type'], axis=1)
    X = pd.concat([X, dummy1, dummy2], axis=1)
    # Drop the baseline classes from our dataframe to avoid collinearity
    X = X.drop(['County', 'England'], axis=1)

    # Set index to constituency
    X = X.set_index('constituency_name')
    X = X.sort_index()

    return X

def create_model_data(device, open_simplex=False, backend='torch'):
    """Creates input and output data tensors, splits for training and validation

    Returns:
        X_train, Y_train, X_val, Y_val: Tensors of model inputs and outputs
    """
    Y = create_Y()
    X = create_X()

    # Make sure they match up (this check can pick up changes in the source)
    if np.any(Y.index != X.index):
        raise ValueError('Source data issue - check input URL')
    
    # Set up training/validation split
    n, K = X.shape
    np.random.seed(0) # Fixes a training/validation split
    train_id = np.random.binomial(1, 0.2, n) == 1
    
    # Normalize the electorate and 2017 turnout (our continuous input variables)
    for var in ['electorate', 'turnout_2017']:
        m = np.mean(X[var][train_id])
        s = np.std(X[var][train_id])
        X[var] = (X[var] - m) / s
      
    # Move Y away from zero for the benefit of the Dirichlet distribution
    if open_simplex:
        Y = Y + 1e-3 / (1 + 1e-3 * K)

    # Convert to tf Tensors and return    
    if backend == 'tensorflow':    
        X_train = tf.convert_to_tensor(X[train_id].to_numpy(), dtype='float32')
        X_val = tf.convert_to_tensor(X[~train_id].to_numpy(), dtype='float32')
        Y_train = tf.convert_to_tensor(Y[train_id].to_numpy(), dtype='float32')
        Y_val = tf.convert_to_tensor(Y[~train_id].to_numpy(), dtype='float32')
    elif backend == 'torch':
        X_train = torch.tensor(X[train_id].to_numpy(), dtype=torch.float32, device=device)
        X_val = torch.tensor(X[~train_id].to_numpy(), dtype=torch.float32, device=device)
        Y_train = torch.tensor(Y[train_id].to_numpy(), dtype=torch.float32, device=device)
        Y_val = torch.tensor(Y[~train_id].to_numpy(), dtype=torch.float32, device=device)
    else:
        raise ValueError(f"Unknown backend {backend}")
    return X_train, Y_train, X_val, Y_val


X_train, Y_train, X_val, Y_val = create_model_data(device)
print("Data", X_train.shape, Y_train.shape, X_val.shape, Y_val.shape)


class GLM(torch.nn.Module):

    def __init__(self, input_size, output_size, p_drop=0.0):
        super().__init__()
        self.linear1 = torch.nn.Sequential(
            torch.nn.Dropout(p_drop),
            torch.nn.Linear(input_size, output_size, bias=True)
        )
        self.linear2 = torch.nn.Sequential(
            torch.nn.Dropout(p_drop),
            torch.nn.Linear(input_size, output_size, bias=True)
        )

    def forward(self, x):
        scores = torch.clamp(self.linear1(x), min=-10, max=10)        
        alphas = torch.exp(torch.clamp(self.linear2(x), min=-10, max=10)) + 1e-6
        return MixedDirichlet(concentration=alphas, scores=scores)


# ## Training 

class VotingData(Dataset):

    def __init__(self, x, y):
        super().__init__()
        if not torch.is_tensor(x):            
            x = torch.from_numpy(x)
        if not torch.is_tensor(y):
            y = torch.from_numpy(y)
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def validate(dl, model, num_samples=100, viterbi=False):
    mse = []
    mae = []
    r = defaultdict(list)
    with torch.no_grad():
        for x, y in dl:
            p = model(x)
            mean = p.sample((num_samples,)).mean(0)  
            
            all_f = p.faces.enumerate_support()
            # [T, B, K]
            idx = torch.argmax(p.faces.log_prob(all_f), 0)
            f_ = torch.stack([all_f[k, i] for i, k in enumerate(idx)], 0) 
            y_ = p.Y(f_).mean
            r['e-acc'].append(((y > 0) == f_).float().mean(-1).cpu().numpy())
            r['f-exact'].append((torch.all((y > 0) == f_)).float().cpu().numpy())
            
            
            r['mean_RMSE'].append(torch.sqrt(((y - mean) ** 2).mean(-1)).cpu().numpy())
            r['mean_MAE'].append(torch.abs(y - mean).mean(-1).cpu().numpy())

            r['viterbi_RMSE'].append(torch.sqrt(((y - y_) ** 2).mean(-1)).cpu().numpy())
            r['viterbi_MAE'].append(torch.abs(y - y_).mean(-1).cpu().numpy())

    return {k: np.array(v).flatten() for k, v in r.items()}


# I am using exactly the same hyperparameters as the CC paper
num_epochs = 400
training_dl = DataLoader(VotingData(X_train, Y_train), batch_size=len(X_train), shuffle=True)
val_dl = DataLoader(VotingData(X_val, Y_val), batch_size=len(X_val), shuffle=False)
model = GLM(X_train.shape[-1], Y_train.shape[-1], p_drop=0.0).to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.1)

log = defaultdict(list)
with tqdm(total=len(training_dl) * num_epochs) as pbar:
    for e in range(num_epochs):
        for x, y in training_dl:
            # preprocess to push into the open simplex
            # y = (y + 1e-3) / (y + 1e-3).sum(-1, keepdims=True)
            f = (y > 0)

            model.train()        
            opt.zero_grad()

            p = model(x)                         
            loss = - p.log_prob(y).mean(0)        
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                1.
            )       
            opt.step()

            pbar.set_description(f'Epoch {e+1:3d}/{num_epochs}')
            pbar.set_postfix(OrderedDict([('loss', loss.item())]))
            pbar.update(1)
            log['loss'].append(loss.item())
        r = validate(val_dl, model)
        log['mean_RMSE'].append(r['mean_RMSE'].mean())
        log['mean_MAE'].append(r['mean_MAE'].mean())
        log['viterbi_RMSE'].append(r['viterbi_RMSE'].mean())
        log['viterbi_MAE'].append(r['viterbi_MAE'].mean())


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


#curve = moving_average(log['loss'], 100)
#_ = plt.plot(np.arange(len(curve)), curve, '.')

#_ = plt.plot(np.arange(len(log['mean_RMSE'])), log['mean_RMSE'], '.', label='RMSE (Y: sample mean)')
#_ = plt.plot(np.arange(len(log['viterbi_RMSE'])), log['viterbi_RMSE'], '.', label='RMSE (F: argmax, Y|f: mean)')
#_ = plt.legend()
#_ = plt.plot(np.arange(len(log['mean_MAE'])), log['mean_MAE'], '.', label='MAE (Y: sample mean)')
#_ = plt.plot(np.arange(len(log['viterbi_MAE'])), log['viterbi_MAE'], '.', label='MAE (F: argmax, Y|f: mean)')
#_ = plt.legend()

r = validate(val_dl, model, 1000)


def plot_results(md_results, cc_results):
    """Plots training curves following the format of Figure 3 from the CC paper:
    https://proceedings.icml.cc/static/paper_files/icml/2020/4730-Paper.pdf

    Args:
        cc_model: CC model object with stored validation scores.
        dirichlet_model: Dirichlet model object with stored validation scores.
    """

    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12, 5))
    color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
        '#f781bf', '#a65628', '#984ea3',
        '#999999', '#e41a1c', '#dede00']

    
    ax[0].plot(cc_results['dirichlet']['RMSE'], color=color_cycle[0], linestyle='dotted', label='Dir')
    ax[1].plot(cc_results['dirichlet']['MAE'], color=color_cycle[0], linestyle='dotted')

    ax[0].plot(cc_results['cc']['RMSE'], color=color_cycle[1], linestyle='dashed', label='CC')
    ax[1].plot(cc_results['cc']['MAE'], color=color_cycle[1], linestyle='dashed')
        
    ax[0].plot(np.arange(len(md_results['mean_RMSE'])), md_results['mean_RMSE'], 
             color=color_cycle[2], label='Mixed Dir (sample mean)')
    ax[1].plot(np.arange(len(md_results['mean_MAE'])), md_results['mean_MAE'], 
             color=color_cycle[2])
    
    ax[0].plot(np.arange(len(md_results['viterbi_RMSE'])), md_results['viterbi_RMSE'], 
             color=color_cycle[3], linestyle='-.', label='Mixed Dir (most probable mean)')
    ax[1].plot(np.arange(len(md_results['viterbi_MAE'])), md_results['viterbi_MAE'], 
             color=color_cycle[3], linestyle='-.', )
    
    #plt.plot(np.arange(len(log['viterbi_RMSE'])), log['viterbi_RMSE'], '.', label='RMSE (F: argmax, Y|f: mean)')
    # plt.plot(np.arange(len(log['viterbi_MAE'])), log['viterbi_MAE'], '.', label='MAE (F: argmax, Y|f: mean)')

    
    ax[0].set_xlabel('Training step', fontsize=14)
    ax[1].set_xlabel('Training step', fontsize=14)
    ax[0].set_ylabel('RMSE', fontsize=14)
    ax[1].set_ylabel('MAE', fontsize=14)
    fig.legend(fontsize=14)
    fig.tight_layout(w_pad=2, h_pad=2)
    return fig


fig = plot_results(log, cc_results)
fig.savefig('comparison-cc.pdf')

def predict(dl, model, num_samples=100):
    stochastic = []
    deterministic = []
    gold = []
    
    with torch.no_grad():
        for x, y in dl:
            p = model(x)
            mean = p.sample((num_samples,)).mean(0)  
            
            all_f = p.faces.enumerate_support()
            # [T, B, K]
            idx = torch.argmax(p.faces.log_prob(all_f), 0)
            f_ = torch.stack([all_f[k, i] for i, k in enumerate(idx)], 0) 
            y_ = p.Y(f_).mean

            stochastic.append(mean.cpu().numpy())
            deterministic.append(y_.cpu().numpy())   
            gold.append(y.cpu().numpy())         

    return np.array(stochastic), np.array(deterministic), np.array(gold)

st, det, gold = predict(val_dl, model)
st = st.reshape(-1, 5)
det = det.reshape(-1, 5)
gold = gold.reshape(-1, 5)

fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(14, 4))
_ = ax[0].boxplot(Y_val.cpu().numpy())
_ = ax[0].set_xlabel("Data", fontsize=14)
_ = ax[1].boxplot(st)
_ = ax[1].set_xlabel("Sample mean", fontsize=14)
_ = ax[2].boxplot(det)
_ = ax[2].set_xlabel("Most probable mean", fontsize=14)
_ = ax[0].set_ylabel(r"$Y_k$", fontsize=14)
fig.tight_layout(w_pad=2, h_pad=2)
fig.savefig('marginal-yk.pdf')

fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 4))
_ = ax[0].plot(np.arange(5), (Y_val == 0).float().mean(0).cpu().numpy(), 'x', label='data')
_ = ax[0].plot(np.arange(5), (st == 0).mean(0), '^', alpha=0.5, label='sample mean')
_ = ax[0].plot(np.arange(5), (det == 0).mean(0), 's', alpha=0.5, label='most probable mean')
_ = ax[0].set_xticks(np.arange(5))
_ = ax[0].set_xticklabels(np.arange(1, 6), fontsize=14)
_ = ax[0].set_xlabel("Party", fontsize=14)
_ = ax[0].set_ylabel("Proportion of 0s", fontsize=14)
_ = ax[1].plot(np.arange(5), (Y_val > 0).float().mean(0).cpu().numpy(), 'x')
_ = ax[1].plot(np.arange(5), (st > 0).mean(0), '^', alpha=0.5)
_ = ax[1].plot(np.arange(5), (det > 0).mean(0), 's', alpha=0.5)
_ = ax[1].set_xticks(np.arange(5))
_ = ax[1].set_xticklabels(np.arange(1, 6), fontsize=14)
_ = ax[1].set_xlabel("Party", fontsize=14)
_ = ax[1].set_ylabel("Proportion of non-zero", fontsize=14)
_ = fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          fancybox=True, shadow=False, ncol=3, fontsize=14)
fig.tight_layout(w_pad=2, h_pad=2)
fig.savefig('sparsity.pdf')

fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(5, 4))
_ = ax.plot(np.arange(5), (Y_val == 0).float().mean(0).cpu().numpy(), 'x', label='data')
_ = ax.plot(np.arange(5), (st == 0).mean(0), '^', alpha=0.5, label='sample mean')
_ = ax.plot(np.arange(5), (det == 0).mean(0), 's', alpha=0.5, label='most probable mean')
_ = ax.set_xticks(np.arange(5))
_ = ax.set_xticklabels(np.arange(1, 6), fontsize=14)
_ = ax.set_xlabel("Party", fontsize=14)
_ = ax.set_ylabel("Proportion of 0s", fontsize=14)
_ = fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          fancybox=True, shadow=False, ncol=3, fontsize=12)
fig.tight_layout(w_pad=2, h_pad=2)
fig.savefig('sparsity0.pdf')


from tabulate import tabulate


rows = [
    ['CC', cc_results['cc']['RMSE'][-1], cc_results['cc']['MAE'][-1]],
    ['Mixed Dir (sample mean)', log['mean_RMSE'][-1], log['mean_MAE'][-1]],
    ['Mixed Dir (most probable mean)', log['viterbi_RMSE'][-1], log['viterbi_MAE'][-1]],
]
print(tabulate(rows, headers=['Model', 'RMSE', 'MAE'], floatfmt=["", ".4f", ".4f"]))
#print("\nSparsity accuracy - most probable mean:", ((det > 0) == (gold > 0)).mean())
#print("\nSparsity accuracy - sample mean:", ((st > 0) == (gold > 0)).mean())

print("\nData sparsity: 0s", (gold == 0).mean())

from sklearn.metrics import classification_report

print("\nMost probable mean (classification report for y > 0):")
print(classification_report((gold > 0).flatten(), (det > 0).flatten()))

print("\nSample mean (classification report for y > 0):")
print(classification_report((gold > 0).flatten(), (st > 0).flatten()))
