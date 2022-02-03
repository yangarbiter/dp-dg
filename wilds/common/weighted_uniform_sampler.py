#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Sampler


class WeightedUniformWithReplacementSampler(Sampler):
    r"""
    This sampler samples elements according to the Sampled Gaussian Mechanism.
    Each sample is selected with a probability equal to ``sample_rate``.
    """

    def __init__(self, weights: Sequence[int], num_samples: int,
        sample_rate: float, clip_sample_rate: float, generator=None):
        r"""
        Args:
            num_samples (int): number of samples to draw.
            sample_rate (float): probability used in sampling.
            generator (Generator): Generator used in sampling.
        """
        assert (weights.sum() + .5).long() == num_samples, (weights.sum().long(), num_samples)
        self.weights = weights
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.clip_sample_rate = clip_sample_rate
        self.generator = generator
        if self.generator is None:
            generator = torch.Generator()
            generator.manual_seed(
                int(torch.empty((), dtype=torch.int64).random_().item())
            )

        if self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer "
                "value, but got num_samples={}".format(self.num_samples)
            )

    def __len__(self):
        return int(1 / self.sample_rate)

    def __iter__(self):
        num_batches = int(1 / self.sample_rate)
        sampling_probas = (self.weights * self.sample_rate)
        if self.clip_sample_rate is not None:
            sampling_probas = np.clip(sampling_probas, 0, self.clip_sample_rate)
        while num_batches > 0:
            mask = (
                torch.rand(self.num_samples, generator=self.generator)
                < sampling_probas
            )
            indices = mask.nonzero(as_tuple=False).reshape(-1).tolist()
            if len(indices) != 0:
                # We only output non-empty list of indices, otherwise the dataloader is unhappy
                # This is compensated by the privacy engine
                yield indices
            num_batches -= 1
