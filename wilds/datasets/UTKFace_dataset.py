import os
from glob import glob

import torch
import numpy as np
import pandas as pd
from PIL import Image
from wilds.datasets.wilds_dataset import WILDSDataset
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.all_metrics import Accuracy

class UTKFaceDataset(WILDSDataset):
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

    Website:
        https://susanqq.github.io/UTKFace/

    Original publication:
        @inproceedings{liu2015faceattributes,
          title = {Deep Learning Face Attributes in the Wild},
          author = {Liu, Ziwei and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou},
          booktitle = {Proceedings of International Conference on Computer Vision (ICCV)},
          month = {December},
          year = {2015}
        }

    This variant of the dataset is identical to the setup in:
        @inproceedings{sagawa2019distributionally,
          title = {Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization},
          author = {Sagawa, Shiori and Koh, Pang Wei and Hashimoto, Tatsunori B and Liang, Percy},
          booktitle = {International Conference on Learning Representations},
          year = {2019}
        }

    License:
        This version of the dataset was originally downloaded from Kaggle
        https://www.kaggle.com/jessicali9530/celeba-dataset

        It is available for non-commercial research purposes only.
    """
    _dataset_name = 'UTKFace'
    _versions_dict = {'1.0': {'download_url': '', 'compressed_size': 0}}

    def __init__(self, version=None, root_dir='data', download=False, split_scheme='official'):
        self._version = version
        self._data_dir = self.initialize_data_dir(root_dir, download)
        confounder_names = ['Race']

        self._input_array = []
        self._y_array = []
        _race_array = []
        for im_path in glob(os.path.join(self.data_dir, "*.jpg")):
            self._input_array.append(im_path)
            _, filename = os.path.split(im_path)
            try:
                age, gender, race, _ = filename.split("_")
            except:
                print("problem with: ", im_path)
            self._y_array.append(int(gender))
            _race_array.append(int(race))
        self._y_array = torch.LongTensor(self._y_array)
        self._y_size = 1
        self._n_classes = 2

        confounders = np.array(_race_array).reshape((-1, 1))

        self._metadata_array = torch.cat(
            (torch.LongTensor(confounders), self._y_array.reshape((-1, 1))),
            dim=1)
        confounder_names = [s.lower() for s in confounder_names]
        self._metadata_fields = confounder_names + ['y']
        self._metadata_map = {
            'y': ['male', 'female'], # Padding for str formatting
            'race': ['White', 'Black', 'Asian', 'Indian', 'Others'] # Padding for str formatting
        }

        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=(confounder_names + ['y']))

        # Extract splits
        self._split_scheme = split_scheme
        if self._split_scheme != 'official':
            raise ValueError(f'Split scheme {self._split_scheme} not recognized')
        random_state = np.random.RandomState(0)
        self._split_array = np.zeros(len(self._y_array))
        self._split_array[:17000] = 0 # train
        self._split_array[17000:19000] = 1 # val
        self._split_array[19000:] = 2 # test
        random_state.shuffle(self._split_array)

        super().__init__(root_dir, download, split_scheme)

    def get_input(self, idx):
       # Note: idx and filenames are off by one.
       img_filename = self._input_array[idx]
       x = Image.open(img_filename).convert('RGB')
       return x

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
