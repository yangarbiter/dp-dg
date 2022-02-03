import sys
sys.path.append("../")

import wilds
from wilds.common.grouper import CombinatorialGrouper
import torch
from opacus.accountants.rdp import RDPAccountant

def get_sampling_weights(ds_name):
    full_dataset = wilds.get_dataset(
        dataset=ds_name,
        version="1.0",
        root_dir="../data",
        download=True,
        split_scheme="official")

    train_grouper = CombinatorialGrouper(
            dataset=full_dataset,
            groupby_fields=full_dataset._metadata_fields)

    trn_dset = full_dataset.get_subset(
                "train",
                train_grouper=train_grouper,
                frac=1.0,
                subsample_to_minority=False)

    groups, group_counts = train_grouper.metadata_to_group(
                    trn_dset.metadata_array,
                    return_counts=True)
    group_weights = 1 / group_counts
    weights = group_weights[groups]
    weights = weights / weights.sum() * len(trn_dset)
    return weights
    
def get_privacy_spent(sigma, epochs, n_iters, sample_rate):
    accountant = RDPAccountant()
    for _ in range(epochs):
        for _ in range(n_iters):
            accountant.step(noise_multiplier=sigma, sample_rate=sample_rate)
    return accountant.get_privacy_spent(delta=1e-5, alphas=alphas)

def get_sigma_epsilon(ds_name, epochs, sample_rate, sigmas, weighted_sampling=True):
    alphas = [1 + x / 2000.0 for x in range(1, 20000)] + list(range(12, 64))

    if weighted_sampling:
        weight = get_sampling_weights(ds_name).max().item()
    else:
        weight = 1.0

    data = {}
    for sigma in sigmas:
        epsilon, alpha = get_privacy_spent(sigma, epochs, int(1 / sample_rate), sample_rate * weight)
        data[sigma] = epsilon
    return data