import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
from torchvision.datasets import INaturalist

from wilds.datasets.wilds_dataset import WILDSDataset
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.all_metrics import Accuracy

class INatSubDataset(WILDSDataset):
    """
    A variant of the CelebA dataset.
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

    Website:

    """
    _dataset_name = 'inatsub'
    _versions_dict = {
        '1.0': {
            'download_url': '',
            'compressed_size': 0}}

    def __init__(self, version=None, root_dir='data', download=False, split_scheme='official'):
        self._version = version
        self._data_dir = self.initialize_data_dir(root_dir, download)
        target_name = 'Blond_Hair'
        confounder_names = [
            "Plantae",
            "Insecta",
            "Aves",
            "Reptilia",
            "Mammalia",
            #"Fungi",
            "Amphibia",
            "Mollusca",
            #"Animalia",
            #"Arachnida",
            "Actinopterygii",
            #"Chromista",
            #"Protozoa",
        ]

        if os.path.exists(root_dir):
            download = False

        subclasses = [0, 1, 2, 3, 4, 6, 7, 10]
        self._data_dir = self._data_dir.replace("inatsub_v1.0", "inaturalist_v1.0")

        self.dset = INaturalist(root=self._data_dir, version="2017", target_type="super", download=download)
        y_array = np.array([self.dset.categories_map[y]['super'] for (y, _) in self.dset.index])

        self.idx = np.concatenate([np.where(y_array == c)[0] for c in subclasses])
        y_array = np.array([subclasses.index(y_array[i]) for i in self.idx])

        self._y_array = torch.LongTensor(y_array)
        self._y_size = 1
        self._n_classes = len(np.unique(self._y_array))

        self._metadata_fields = ['y']
        self._metadata_map = {
            'y': sorted(self.dset.categories_index['super'].keys()),
        }

        self._metadata_array = self._y_array.reshape(-1, 1)

        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=['y']
        )

        # Extract splits
        self._split_scheme = split_scheme
        if self._split_scheme != 'official':
            raise ValueError(f'Split scheme {self._split_scheme} not recognized')

        random_state = np.random.RandomState(0)
        self._split_array = np.zeros(len(self._y_array))
        self._split_array[:450000] = 0 # train
        self._split_array[450000:500000] = 1 # val
        self._split_array[500000:591166] = 2 # test
        random_state.shuffle(self._split_array)

        super().__init__(root_dir, download, split_scheme)

    def get_input(self, idx):
        img = self.dset[self.idx[idx]][0]
        return img.convert('RGB')

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
