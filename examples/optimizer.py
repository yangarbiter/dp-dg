from typing import Optional, Callable, List

import torch
from torch import nn
from torch.optim import SGD, Adam, Optimizer
from transformers import AdamW


def initialize_optimizer(config, model):
    # initialize optimizers
    if config.optimizer=='SGD':
        params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = SGD(
            params,
            lr=config.lr,
            weight_decay=config.weight_decay,
            **config.optimizer_kwargs)
    elif config.optimizer=='AdamW':
        if 'bert' in config.model or 'gpt' in config.model:
            no_decay = ['bias', 'LayerNorm.weight']
        else:
            no_decay = []

        params = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': config.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(
            params,
            lr=config.lr,
            **config.optimizer_kwargs)
    elif config.optimizer == 'Adam':
        params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = Adam(
            params,
            lr=config.lr,
            weight_decay=config.weight_decay,
            **config.optimizer_kwargs)
    else:
        raise ValueError(f'Optimizer {config.optimizer} not recognized.')

    return optimizer

def _generate_noise(
    std: float,
    reference: torch.Tensor,
) -> torch.Tensor:
    """
    """
    zeros = torch.zeros(reference.shape, device=reference.device)
    if std == 0:
        return zeros
    return torch.normal(
        mean=0,
        std=std,
        size=reference.shape,
        device=reference.device,
    )

def params(optimizer: Optimizer) -> List[nn.Parameter]:
    """
    Return all parameters controlled by the optimizer
    Args:
        optimizer: optimizer
    Returns:
        Flat list of parameters from all ``param_groups``
    """
    ret = []
    for param_group in optimizer.param_groups:
        ret += [p for p in param_group["params"] if p.requires_grad]
    return ret

class NoisyOptimizerWrapper(Optimizer):
    """
    ``torch.optim.Optimizer`` wrapper that adds additional functionality to clip per
    sample gradients and add Gaussian noise.
    Can be used with any ``torch.optim.Optimizer`` subclass as an underlying optimizer.
    ``DPOptimzer`` assumes that parameters over which it performs optimization belong
    to GradSampleModule and therefore have the ``grad_sample`` attribute.
    On a high level ``DPOptimizer``'s step looks like this:
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
        noise_multiplier: float,
    ):
        """
        Args:
            optimizer: wrapped optimizer.
            noise_multiplier: noise multiplier
            max_grad_norm: max grad norm used for gradient clipping
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
        self.original_optimizer = optimizer
        self.noise_multiplier = noise_multiplier

        self.param_groups = optimizer.param_groups
        self.state = optimizer.state

    @property
    def params(self) -> List[nn.Parameter]:
        """
        Returns a flat list of ``nn.Parameter`` managed by the optimizer
        """
        return params(self)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        if closure is not None:
            with torch.enable_grad():
                closure()

        for p in self.params:
            noise = _generate_noise(
                std=self.noise_multiplier,
                reference=p.grad,
            )
            p.grad = p.grad + noise

        return self.original_optimizer.step(closure)

    def __repr__(self):
        return self.original_optimizer.__repr__()

    def state_dict(self):
        return self.original_optimizer.state_dict()

    def load_state_dict(self, state_dict) -> None:
        self.original_optimizer.load_state_dict(state_dict)