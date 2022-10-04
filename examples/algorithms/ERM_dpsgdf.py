import torch
from algorithms.single_model_algorithm import SingleModelAlgorithm
from models.initializer import initialize_model
from dpsgdf import FairDPOptimizer
from torch.nn.utils import clip_grad_norm_

class ERMDPSGDf(SingleModelAlgorithm):
    def __init__(self, config, d_out, grouper, loss,
            metric, n_train_steps):
        model = initialize_model(config, d_out).to(config.device)
        # initialize module
        super().__init__(
            config=config,
            model=model,
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )

    def objective(self, results):
        return self.loss.compute(results['y_pred'], results['y_true'], return_dict=False)

    def _update(self, results):
        """
        Computes the objective and updates the model.
        Also updates the results dictionary yielded by process_batch().
        Should be overridden to change algorithm update beyond modifying the objective.
        """
        # compute objective
        objective = self.objective(results)
        results['objective'] = objective.item()
        # update
        self.model.zero_grad()
        if isinstance(self.optimizer, FairDPOptimizer):
            self.optimizer.zero_grad()
            self.optimizer.sample_groups = results['g']
        else:
            raise
        objective.backward()
        if self.max_grad_norm:
            clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.step_schedulers(
            is_epoch=False,
            metrics=results,
            log_access=False)
