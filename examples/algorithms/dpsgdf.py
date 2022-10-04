#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from ipaddress import ip_address
import warnings
from typing import List, Optional, Tuple, Union, Callable

import torch
from opacus.accountants import create_accountant
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from opacus.grad_sample.grad_sample_module import GradSampleModule
from opacus.optimizers import DPOptimizer
from opacus.optimizers.optimizer import (
    _generate_noise, _check_processed_flag, _get_flat_grad_sample, _mark_as_processed)
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from opacus import PrivacyEngine
from opacus.privacy_engine import forbid_accumulation_hook


class FairDPOptimizer(DPOptimizer):
    """
    ``torch.optim.Optimizer`` wrapper that adds additional functionality to clip per
    sample gradients and add gaissian noise.

    Can be used with any ``torch.optim.Optimizer`` subclass as an underlying optimizer.
    ``DPOptimzer`` assumes that parameters over which it performs optimization belong
    to GradSampleModule and therefore have ``grad_sample`` attribute.

    On a high lever ``DPOptimizer``'s step looks like this:
    1) Aggregate ``p.grad_sample`` over all parameters to calculate per sample norms
    2) Clip ``p.grad_sample`` so that per sample norm is not above threshold
    3) Aggregate clipped per sample gradients into ``p.grad``
    4) Add Gaussian noise to ``p.grad`` calibrated to a given noise multiplier and
    max grad norm limit (``std = noise_multiplier * max_grad_norm``).
    5) Call underlying optimizer to perform optimization step

    Examples:
        >>> module = MyCustomModel()
        >>> optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
        >>> dp_optimzer = DPOptimizer(
        ...     optimizer=optimizer,
        ...     noise_multiplier=1.0,
        ...     max_grad_norm=1.0,
        ...     expected_batch_size=4,
        ... )
    """

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        n_groups,
        group_info,
        C0: float,
        sigma2: float,
        noise_multiplier: float,
        max_grad_norm: float,
        expected_batch_size: Optional[int],
        loss_reduction: str = "mean",
        generator=None,
        secure_mode: bool = False,
    ):
        """

        Args:
            optimizer: wrapped optimizer.
            noise_multiplier: noise multiplier
            max_grad_norm: max grad norm used for gradient clpping
            expected_batch_size: batch_size used for averaging gradients. When using
                Poisson sampling averaging denominator can't be inferred from the
                actual batch size. Required is ``loss_reduction="mean"``, ignored if
                ``loss_reduction="sum"``
            loss_reduction: Indicates if the loss reduction (for aggregating the gradients)
                is a sum or a mean operation. Can take values "sum" or "mean"
            generator: torch.Generator() object used as a source of randomness for
                the noise
            secure_mode: if ``True`` uses noise generation approach robust to floating
                point arithmetic attacks.
                See :meth:`~opacus.optimizers.optimizer._generate_noise` for details
        """
        self.group_info = group_info
        self.n_groups = n_groups
        self.C0 = C0
        self.sigma2 = sigma2
        self.C = None

        super().__init__(
            optimizer=optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=secure_mode
        )

    def compute_clipping(self):
        self.m = torch.zeros(self.n_groups).float()
        self.o = torch.zeros(self.n_groups).float()

        per_param_norms = [
            g.view(len(g), -1).norm(2, dim=-1) for g in self.grad_samples
        ]
        per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
        batch_group = self.sample_groups

        for k in range(self.n_groups):
            self.m[k] = float((per_sample_norms[batch_group == k] > self.C0).sum())
            self.o[k] = float((per_sample_norms[batch_group == k] <= self.C0).sum())

        self.m = self.m + _generate_noise(
            std=self.sigma2,
            reference=self.m,
            generator=self.generator,
            secure_mode=self.secure_mode,
        )
        self.o = self.o + _generate_noise(
            std=self.sigma2,
            reference=self.m,
            generator=self.generator,
            secure_mode=self.secure_mode,
        )

        device = per_sample_norms.device

        m_sum = self.m.sum()
        bk = self.m + self.o
        Ck = self.C0 * (1 + (self.m / bk) / (m_sum / (bk).sum())).to(device)
        batch_group.to(device)

        per_sample_clip_factor = (Ck[batch_group] / (per_sample_norms + 1e-6)).clamp(
            max=1.0
        )
        for p in self.params:
            _check_processed_flag(p.grad_sample)

            grad_sample = _get_flat_grad_sample(p)
            grad = torch.einsum("i,i...", per_sample_clip_factor, grad_sample)

            if p.summed_grad is not None:
                p.summed_grad += grad
            else:
                p.summed_grad = grad

            _mark_as_processed(p.grad_sample)

        return torch.max(Ck).cpu().item()

    def add_noise(self, clip_val):
        """
        Adds noise to clipped gradeints. Stores clipped and noised result in ``p.grad``
        """
        for p in self.params:
            _check_processed_flag(p.summed_grad)

            noise = _generate_noise(
                std=self.noise_multiplier * clip_val,
                reference=p.summed_grad,
                generator=self.generator,
                secure_mode=self.secure_mode,
            )
            p.grad = p.summed_grad + noise

            _mark_as_processed(p.summed_grad)

    def pre_step(
        self, closure: Optional[Callable[[], float]] = None
    ) -> Optional[float]:
        """
        Perform actions specific to ``DPOptimizer`` before calling
        underlying  ``optimizer.step()``

        Args:
            closure: A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        """
        clip_val = max(self.compute_clipping(), 0.)
        #self.clip_and_accumulate()
        if self._check_skip_next_step():
            self._is_last_step_skipped = True
            return False

        self.add_noise(clip_val)
        self.scale_grad()

        if self.step_hook:
            self.step_hook(self)

        self._is_last_step_skipped = False
        return True


class FairPrivacyEngine(PrivacyEngine):

    def __init__(self, *, accountant: str = "rdp", secure_mode: bool = False):
        super().__init__(accountant=accountant, secure_mode=secure_mode)

    def _prepare_optimizer(
        self,
        optimizer: optim.Optimizer,
        *,
        noise_multiplier: float,
        max_grad_norm: Union[float, List[float]],
        expected_batch_size: int,
        loss_reduction: str = "mean",
        distributed: bool = False,
        clipping: str = "flat",
        noise_generator=None,
        group_info,
        n_groups,
        C0: float,
        sigma2: float,
    ) -> DPOptimizer:
        if isinstance(optimizer, DPOptimizer):
            optimizer = optimizer.original_optimizer

        generator = None
        if self.secure_mode:
            generator = self.secure_rng
        elif noise_generator is not None:
            generator = noise_generator

        optim_class = FairDPOptimizer

        return optim_class(
            optimizer=optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=self.secure_mode,
            group_info=group_info,
            n_groups=n_groups,
            C0=C0,
            sigma2=sigma2,
        )


    def make_private(
        self,
        *,
        module: nn.Module,
        optimizer: optim.Optimizer,
        data_loader: DataLoader,
        noise_multiplier: float,
        max_grad_norm: Union[float, List[float]],
        batch_first: bool = True,
        loss_reduction: str = "mean",
        poisson_sampling: bool = True,
        clipping: str = "flat",
        noise_generator=None,
        group_info,
        n_groups,
        C0: float,
        sigma2: float,
    ) -> Tuple[GradSampleModule, DPOptimizer, DataLoader]:
        """
        Add privacy-related responsibilites to the main PyTorch training objects:
        model, optimizer, and the data loader.

        All of the returned objects act just like their non-private counterparts
        passed as arguments, but with added DP tasks.

        - Model is wrapped to also compute per sample gradients.
        - Optimizer is now responsible for gradient clipping and adding noise to the gradients.
        - DataLoader is updated to perform Poisson sampling.

        Notes:
            Using any other models, optimizers, or data sources during training
            will invalidate stated privacy guarantees.

        Args:
            module: PyTorch module to be used for training
            optimizer: Optimizer to be used for training
            data_loader: DataLoader to be used for training
            noise_multiplier: The ratio of the standard deviation of the Gaussian noise to
                the L2-sensitivity of the function to which the noise is added
                (How much noise to add)
            max_grad_norm: The maximum norm of the per-sample gradients. Any gradient with norm
                higher than this will be clipped to this value.
            batch_first: Flag to indicate if the input tensor to the corresponding module
                has the first dimension representing the batch. If set to True, dimensions on
                input tensor are expected be ``[batch_size, ...]``, otherwise
                ``[K, batch_size, ...]``
            loss_reduction: Indicates if the loss reduction (for aggregating the gradients)
                is a sum or a mean operation. Can take values "sum" or "mean"
            noise_generator: torch.Generator() used as a source of randomness for
                gaussian noise generation
            poisson_sampling: ``True`` if you want to use standard sampling required
                for DP guarantees. Setting ``False`` will leave provided data_loader
                unchanged. Technically this doesn't fit the assumptions made by
                privacy accounting mechanism, but it can be a good approximation when
                using Poisson sampling is unfeasible.
            clipping: Per sample gradient clipping mechanism ("flat" or "per_layer").
                Flat clipping calculates the norm of the entire gradient over
                all parameters, while per layer clipping sets individual norms for
                every parameter tensor. Flat clipping is usually preferred, but using
                per layer clipping in combination with distributed training can provide
                notable performance gains.
            noise_generator: torch.Generator() object used as a source of randomness for
                the noise

        Returns:
            Tuple of (model, optimizer, data_loader).

            Model is a wrapper around the original model that also computes per sample
                gradients
            Optimizer is a wrapper around the original optimizer that also does
             gradient clipping and noise addition to the gradients
            DataLoader is a brand new DataLoader object, constructed to behave as
                equivalent to the original data loader, possibly with updated
                sampling mechanism. Points to the same dataset object.
        """
        if noise_generator and self.secure_mode:
            raise ValueError("Passing seed is prohibited in secure mode")

        distributed = type(module) is DPDDP

        module = self._prepare_model(
            module, batch_first=batch_first, loss_reduction=loss_reduction
        )
        if poisson_sampling:
            module.register_forward_pre_hook(forbid_accumulation_hook)

        data_loader = self._prepare_data_loader(
            data_loader, distributed=distributed, poisson_sampling=poisson_sampling
        )

        sample_rate = 1 / len(data_loader)
        expected_batch_size = int(len(data_loader.dataset) * sample_rate)

        optimizer = self._prepare_optimizer(
            optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            noise_generator=noise_generator,
            distributed=distributed,
            clipping=clipping,
            group_info=group_info,
            n_groups=n_groups,
            C0=C0,
            sigma2=sigma2,
        )

        optimizer.attach_step_hook(
            self.accountant.get_optimizer_hook_fn(sample_rate=sample_rate)
        )

        return module, optimizer, data_loader
