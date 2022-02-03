import os
import torch
import pandas as pd
import numpy as np

from wilds.datasets.wilds_dataset import WILDSDataset
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.all_metrics import Accuracy

import utils_glue

class MultiNLIDataset(WILDSDataset):
    """
    MultiNLI dataset.
    label_dict = {
        'contradiction': 0,
        'entailment': 1,
        'neutral': 2
    }
    # Negation words taken from https://arxiv.org/pdf/1803.02324.pdf
    negation_words = ['nobody', 'no', 'never', 'nothing']
    """

    _dataset_name = 'multinli'
    _versions_dict = {
        '1.0': {
            'download_url': '',
            'compressed_size': 0}}

    def __init__(self, version=None, root_dir="data", download=False, split_scheme='official'):
        self._version = version
        self._data_dir = self.initialize_data_dir(root_dir, download)

        self.root_dir = root_dir
        self._metadata_path = os.path.join(self._data_dir, "metadata_multinli.csv")
        self._data_dir = os.path.join(self._data_dir, 'glue_data', 'MNLI')

        self.features_array = []
        for feature_file in [
            "cached_train_bert-base-uncased_128_mnli",
            "cached_dev_bert-base-uncased_128_mnli",
            "cached_dev_bert-base-uncased_128_mnli-mm",
        ]:
            features = torch.load(os.path.join(self._data_dir, feature_file))
            self.features_array += features

        self.all_input_ids = torch.tensor([f.input_ids for f in self.features_array], dtype=torch.long)
        self.all_input_masks = torch.tensor([f.input_mask for f in self.features_array], dtype=torch.long)
        self.all_segment_ids = torch.tensor([f.segment_ids for f in self.features_array], dtype=torch.long)
        #self.all_label_ids = torch.tensor([f.label_id for f in self.features_array], dtype=torch.long)

        self.x_array = torch.stack((
            self.all_input_ids,
            self.all_input_masks,
            self.all_segment_ids), dim=2)

        #assert np.all(np.array(self.all_label_ids) == self.y_array)

        # Read in metadata
        self.metadata_df = pd.read_csv(self._metadata_path)

        # Get the y values
        # gold_label is hardcoded
        self._y_array = torch.LongTensor(self.metadata_df['y'].values)
        self._y_size = 1
        self._n_classes = len(np.unique(self._y_array))

        self.confounder_array = self.metadata_df['a'].values
        self.n_confounders = 1
        confounder_names = ['sentence2_has_negation']

        # Map to groups
        #self.n_groups = len(np.unique(self.confounder_array)) * self.n_classes
        #self.group_array = (self.y_array*(self.n_groups/self.n_classes) + self.confounder_array).astype('int')

        # Extract splits
        self._split_array = self.metadata_df['split'].values
        self._split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }

        self._metadata_array = torch.cat(
            (torch.LongTensor(self.confounder_array.reshape(-1, 1)), self._y_array.reshape((-1, 1))),
            dim=1)
        confounder_names = [s.lower() for s in confounder_names]
        self._metadata_fields = confounder_names + ['y']
        self._identity_vars = confounder_names

        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=(confounder_names + ['y']))

    def get_input(self, idx):
        return self.x_array[idx]

    def eval(self, y_pred, y_true, metadata, prediction_fn=None):
        """
        Computes all evaluation metrics.
        Args:
            - y_pred (Tensor): Predictions from a model. By default, they are predicted labels (LongTensor).
                               But they can also be other model outputs such that prediction_fn(y_pred)
                               are predicted labels.
            - y_true (LongTensor): Ground-truth labels
            - metadata (Tensor): Metadata
            - prediction_fn (function): A function that turns y_pred into predicted labels
        Output:
            - results (dictionary): Dictionary of evaluation metrics
            - results_str (str): String summarizing the evaluation metrics
        """
        metric = Accuracy(prediction_fn=prediction_fn)
        return self.standard_group_eval(
            metric,
            self._eval_grouper,
            y_pred, y_true, metadata)
