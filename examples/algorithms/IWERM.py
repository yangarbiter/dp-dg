import torch
from algorithms.single_model_algorithm import SingleModelAlgorithm
from models.initializer import initialize_model

class IWERM(SingleModelAlgorithm):
    def __init__(self, config, d_out, grouper, loss,
            metric, group_weights, n_train_steps):
        self.group_weights = group_weights
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
        losses = self.loss.loss_fn(results['y_pred'], results['y_true'])
        losses = losses * self.group_weights[results['g']].to(self.device)
        return losses.mean()
