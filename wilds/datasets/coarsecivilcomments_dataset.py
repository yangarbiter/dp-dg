import os
import torch
import pandas as pd
import numpy as np
from wilds.datasets.wilds_dataset import WILDSDataset
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.all_metrics import Accuracy

class CoarseCivilCommentsDataset(WILDSDataset):
    """
    The CivilComments-wilds toxicity classification dataset.
    This is a modified version of the original CivilComments dataset.

    Supported `split_scheme`:
        'official'

    Input (x):
        A comment on an online article, comprising one or more sentences of text.

    Label (y):
        y is binary. It is 1 if the comment was been rated as toxic by a majority of the crowdworkers who saw that comment, and 0 otherwise.

    Metadata:
        Each comment is annotated with the following binary indicators:
            - male
            - female
            - LGBTQ
            - christian
            - muslim
            - other_religions
            - black
            - white
            - identity_any
            - severe_toxicity
            - obscene
            - threat
            - insult
            - identity_attack
            - sexual_explicit

    Website:
        https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification

    Original publication:
        @inproceedings{borkan2019nuanced,
          title={Nuanced metrics for measuring unintended bias with real data for text classification},
          author={Borkan, Daniel and Dixon, Lucas and Sorensen, Jeffrey and Thain, Nithum and Vasserman, Lucy},
          booktitle={Companion Proceedings of The 2019 World Wide Web Conference},
          pages={491--500},
          year={2019}
        }

    License:
        This dataset is in the public domain and is distributed under CC0.
        https://creativecommons.org/publicdomain/zero/1.0/
    """

    _dataset_name = 'coarsecivilcomments'
    _versions_dict = {
        '1.0': {
            'download_url': '',
            'compressed_size': None}}

    def __init__(self, version=None, root_dir='data', download=False, split_scheme='official'):
        self._version = version
        self._data_dir = self.initialize_data_dir(root_dir, download)

        metadata = os.path.join(self._data_dir, "metadata_civilcomments_coarse.csv")

        text = pd.read_csv(
            os.path.join(
                self._data_dir, "civilcomments_coarse.csv"
            )
        )

        self._text_array = list(text["comment_text"])

        # Read in metadata
        self._metadata_df = pd.read_csv(metadata)

        # Get the y values
        self._y_array = torch.LongTensor(self._metadata_df['y'].values)
        self._y_size = 1
        self._n_classes = len(np.unique(self._y_array))

        # Extract splits
        self._split_scheme = split_scheme
        if self._split_scheme != 'official':
            raise ValueError(f'Split scheme {self._split_scheme} not recognized')
        self._split_array = self._metadata_df['split'].values

        self._metadata_array = torch.cat(
            (
                torch.LongTensor(self._metadata_df["a"].values.reshape(-1, 1)),
                self._y_array.reshape((-1, 1))
            ),
            dim=1
        )
        self._metadata_fields = ["group", 'y']
        self._metadata_map = {
            'group': [
                "male",
                "female",
                "LGBTQ",
                "christian",
                "muslim",
                "other_religions",
                "black",
                "white",
            ],
            'y': ['not toxic', 'toxic'] # Padding for str formatting
        }

        self._eval_grouper = CombinatorialGrouper(
                dataset=self, groupby_fields=self._metadata_fields)

        super().__init__(root_dir, download, split_scheme)

    def get_input(self, idx):
        return self._text_array[idx]

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
