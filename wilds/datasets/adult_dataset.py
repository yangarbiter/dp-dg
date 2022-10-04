import os

import torch
import numpy as np
import pandas as pd
from wilds.datasets.wilds_dataset import WILDSDataset
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.all_metrics import Accuracy
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


class AdultDataset(WILDSDataset):
    """
    A variant of the UTKFace dataset.
    This dataset is not part of the official WILDS benchmark.
    We provide it for convenience and to facilitate comparisons to previous work.

    Supported `split_scheme`:
        'official'

    Input (x):
        Images of celebrity faces that have already been cropped and centered.

    Label (y):
        y is binary. It is 1 if the celebrity in the image has blond hair, and is 0 otherwise.

    Metadata:
        Each image is annotated with whether the celebrity has been labeled 'Male' or 'Female'.
    """
    _dataset_name = 'adult'
    _versions_dict = {'1.0': {'download_url': None, 'compressed_size': 0}}

    def __init__(self, version=None, download=False, root_dir='data', split_scheme='official'):
        self._version = version
        confounder_names = ['sex']
        self._data_dir = self.initialize_data_dir(root_dir, False)

        train_data = pd.read_csv(os.path.join(self._data_dir, 'train.csv'))
        test_data = pd.read_csv(os.path.join(self._data_dir, 'test.csv'))

        df = pd.concat([train_data, test_data], ignore_index=True)
        df["income"] = df["income"].map({"<=50K": 0, ">50K": 1})

        random_state = np.random.RandomState(0)
        self._split_array = np.zeros(len(df))
        self._split_array[:35000] = 0 # train
        self._split_array[40000:43000] = 1 # val
        self._split_array[43000:] = 2 # test
        random_state.shuffle(self._split_array)

        num_cols = ["hours-per-week", "capital-loss", "capital-gain"]

        cat_cols = [
            "work-class",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ]

        num_cols.append("education-num")
        #df[num_cols] = pd.DataFrame(df[num_cols], columns=num_cols)
        scaler = StandardScaler().fit(df[num_cols].to_numpy()[self._split_array == 0])
        df[num_cols] = pd.DataFrame(scaler.transform(df[num_cols]), columns=num_cols)

        one_hot_encoded = pd.get_dummies(df[cat_cols])
        df = pd.concat([df[["income"] + num_cols], one_hot_encoded], axis=1)

        feat_columns = num_cols + list(one_hot_encoded.columns)
        del feat_columns[feat_columns.index('sex_Female')]
        self._input_array = torch.FloatTensor(df[feat_columns].to_numpy())
        self._y_array = torch.LongTensor(df['income'])
        confounders = np.array(df['sex_Male']).reshape((-1, 1))

        self._y_size = 1
        self._n_classes = 2

        self._metadata_array = torch.cat(
            (torch.LongTensor(confounders), self._y_array.reshape((-1, 1))),
            dim=1)
        confounder_names = [s.lower() for s in confounder_names]
        self._metadata_fields = confounder_names + ['y']
        self._metadata_map = {
            'sex_Male': ['Female', 'Male'], # Padding for str formatting
            'y': ['<=50k', '>50'] # Padding for str formatting
        }

        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=(confounder_names + ['y']))

        # Extract splits
        self._split_scheme = split_scheme
        if self._split_scheme != 'official':
            raise ValueError(f'Split scheme {self._split_scheme} not recognized')

        super().__init__(root_dir, True, split_scheme)

    def get_input(self, idx):
       # Note: idx and filenames are off by one.
       return self._input_array[idx]

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
