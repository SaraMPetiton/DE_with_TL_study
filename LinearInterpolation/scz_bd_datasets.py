from torch.utils.data.dataset import Dataset
from abc import ABC, abstractmethod
import os, pickle
import pandas as pd
import numpy as np
import bisect, logging
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from typing import Callable, List, Type, Sequence, Dict

"""
All functions from this script are (sometimes simplified) functions of from https://github.com/Duplums/SMLvsDL.
"""

class ClinicalBase(ABC, Dataset):
    """
        A generic clinical Dataset written in a torchvision-like manner. It parses a .pkl file defining the different
        splits based on a <unique_key>. All clinical datasets must have:
        - a training set
        - a validation set
        - a test set
        - (eventually) an other intra-test set

        This generic dataset is memory-efficient, taking advantage of memory-mapping implemented with NumPy.
        It always comes with:
        ... pre-processing:
            - VBM
        ... And 2 differents tasks:
            - Diagnosis prediction (classification)
            - Site prediction (classification)
        ... With meta-data:
            - user-defined unique identifier across pre-processing and split
            - TIV + ROI measures based on Neuromorphometrics atlas
    Attributes:
          * target, list[int]: labels to predict
          * all_labels, pd.DataFrame: all labels stored in a pandas DataFrame containing ["diagnosis", "site", "age", "sex]
          * shape, tuple: shape of the data
          * metadata: pd DataFrame: Age + Sex + TIV + ROI measures extracted for each image
          * id: pandas DataFrame, each row contains a unique identifier for an image

    """
    def __init__(self, root: str, preproc: str='vbm', target: [str, List[str]]='diagnosis',
                 split: str='train',transforms: Callable[[np.ndarray], np.ndarray]=None,  load_data: bool=False):
        """
        :param root: str, path to the root directory containing the different .npy and .csv files
        :param target: str or [str], either 'dx' or 'site'.
        :param split: str, either 'train', 'val', 'test' (inter) or (eventually) 'test_intra'
        :param transforms (callable, optional): A function/transform that takes in
            a 3D MRI image and returns a transformed version.
        :param load_data (bool, optional): If True, loads all the data in memory
               --> WARNING: it can be time/memory-consuming
        """
        if isinstance(target, str):
            target = [target]
        assert set(target) <= {'diagnosis', 'site'}, "Unknown target: %s"%target
        assert split in ['train', 'val', 'test', 'validation'], "Unknown split: %s"%split

        self.root = root
        self.preproc = preproc
        self.split = split
        self.transforms = transforms
        self.target_name = target

        if self.split == "val": self.split = "validation"

        if not self._check_integrity():
            raise RuntimeError("Files not found. Check the the root directory %s"%root)

        self.scheme = self.load_pickle(os.path.join(
            root, self._train_val_test_scheme))[self.split]
        

        ## 1) Loads globally all the data for a given pre-processing
        _root = os.path.join(root, "cat12vbm")
        
        df = pd.concat([pd.read_csv(os.path.join(_root, "%s_t1mri_mwp1_participants.csv" % db)) for db in self._studies],
                       ignore_index=True, sort=False)
        data = [np.load(os.path.join(_root, "%s_t1mri_mwp1_gs-raw_data64.npy" % db), mmap_mode='r')
                         for db in self._studies]
        cumulative_sizes = np.cumsum([len(db) for db in data])

        ## 2) Selects the data to load in memory according to selected scheme
        mask = self._extract_mask(df, unique_keys=self._unique_keys, check_uniqueness=self._check_uniqueness)

        # Get TIV and tissue volumes according to the Neuromorphometrics atlas
        self.metadata = self._extract_metadata(df[mask]).reset_index(drop=True)
        self.id = df[mask][self._unique_keys].reset_index(drop=True)

        # Get the labels to predict
        assert set(self.target_name) <= set(df.keys()), \
            "Inconsistent files: missing %s in pandas DataFrame"%self.target_name
        self.target = df[mask][self.target_name]
        assert self.target.isna().sum().sum() == 0, "Missing values for '%s' label"%self.target_name
        self.target = self.target.apply(self.target_transform_fn, axis=1, raw=True).values.ravel().astype(np.float32)

        all_keys = ["age", "sex", "diagnosis", "site"]
        self.all_labels = df[mask][all_keys].reset_index(drop=True)

        # Transforms (dx, site) according to _dx_site_mappings
        self.all_labels = self.all_labels.apply(lambda row: [row[0], row[1],
                                                             self._dx_site_mappings["diagnosis"][row[2]],
                                                             self._dx_site_mappings["site"][row[3]]],
                                                axis=1, raw=True, result_type="broadcast")
        
        # Prepares private variables to build mapping target_idx -> img_idx
        self.shape = (mask.sum(), *data[0][0].shape)
        self._mask_indices = np.arange(len(df))[mask]
        self._cumulative_sizes = cumulative_sizes
        self._data = data
        self._data_loaded = None

        # Loads all in memory to retrieve it rapidly when needed
        if load_data:
            self._data_loaded = self.get_data()[0]

    @property
    @abstractmethod
    def _studies(self) -> List[str]:
        ...
    @property
    @abstractmethod
    def _train_val_test_scheme(self) -> str:
        ...
    @property
    @abstractmethod
    def _unique_keys(self) -> List[str]:
        ...
    @property
    @abstractmethod
    def _dx_site_mappings(self) -> Dict[str, Dict[str, int]]:
        ...
    @property
    def _check_uniqueness(self) -> bool:
        return True

    def _check_integrity(self):
        is_complete = os.path.isdir(self.root)
        is_complete &= os.path.isfile(os.path.join(self.root, self._train_val_test_scheme))

        # TODO: change the formatted names
        dir_files = {
            "cat12vbm": ["%s_t1mri_mwp1_participants.csv", "%s_t1mri_mwp1_gs-raw_data64.npy"],
        }

        for (dir, files) in dir_files.items():
            for file in files:
                for db in self._studies:
                    is_complete &= os.path.isfile(os.path.join(self.root, dir, file%db))
        return is_complete

    def _extract_mask(self, df: pd.DataFrame, unique_keys: Sequence[str], check_uniqueness: bool=True):
        """
        :param df: pandas DataFrame
        :param unique_keys: list of str
        :param check_uniqueness: if True, check the unique_keys identified uniquely an image in the dataset
        :return: a binary mask indicating, for each row, if the participant belongs to the current scheme or not.
        """
        _source_keys = df[unique_keys].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
        if check_uniqueness:
            assert len(set(_source_keys)) == len(_source_keys), "Multiple identique identifiers found"
        _target_keys = self.scheme[unique_keys].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
        mask = _source_keys.isin(_target_keys).values.astype(np.bool)
        return mask

    def _extract_metadata(self, df: pd.DataFrame):
        """
        :param df: pandas DataFrame
        :return: TIV and tissue volumes defined by the Neuromorphometrics atlas
        """
        metadata = ["age", "sex", "tiv"] + [k for k in df.keys() if "GM_Vol" in k or "WM_Vol" in k or "CSF_Vol" in k]
        assert len(metadata) == 290, "Missing meta-data values (%i != %i)"%(len(metadata), 290)
        assert set(metadata) <= set(df.keys()), "Missing meta-data columns: {}".format(set(metadata) - set(df.keys))
        if df[metadata].isna().sum().sum() > 0:
            self.logger.warning("NaN values found in meta-data")
        return df[metadata]

    def target_transform_fn(self, target):
        ## Transforms the target according to mapping site <-> int and dx <-> int
        target = target.copy()
        for i, name in enumerate(self.target_name):
            target[i] = self._dx_site_mappings[name][target[i]]
        return target

    def load_pickle(self, path: str):
        with open(path, 'rb') as f:
            pkl = pickle.load(f)
        return pkl

    def _mapping_idx(self, idx: int):
        """
        :param idx: int ranging from 0 to len(dataset)-1
        :return: integer that corresponds to the original image index to load
        """
        idx = self._mask_indices[idx]
        dataset_idx = bisect.bisect_right(self._cumulative_sizes, idx)
        sample_idx = idx - self._cumulative_sizes[dataset_idx - 1] if dataset_idx > 0 else idx
        return (dataset_idx, sample_idx)

    def copy(self):
        """
        :return: a deep copy of this
        """

        this = self.__class__(self.root, self.preproc, self.target_name,
                              self.split, self.transforms)
        return this
    
    def __getitem__(self, idx: int):
        if self._data_loaded is not None:
            sample, target = self._data_loaded[idx], self.target[idx]
        else:
            (dataset_idx, sample_idx) = self._mapping_idx(idx)
            sample, target = self._data[dataset_idx][sample_idx], self.target[idx]
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample, target.astype(np.float32)

    def __len__(self):
        return len(self.target)

    def __str__(self):
        return "%s-%s-%s"%(type(self).__name__, self.preproc, self.split)


class SCZDataset(ClinicalBase):

    @property
    def _studies(self):
        return ["schizconnect-vip", "bsnip", "cnp", "candi"]

    @property
    def _train_val_test_scheme(self):
        return "train_val_test_test-intra_scz_stratified.pkl"

    @property
    def _unique_keys(self):
        return ["participant_id", "session", "study"]

    @property
    def _dx_site_mappings(self):
        return dict(diagnosis={"control": 0, "schizophrenia": 1},
                    site=self._site_mapping)

    def _extract_mask(self, df: pd.DataFrame, unique_keys: Sequence[str], check_uniqueness: bool=True):
        # Little hack
        df = df.copy()
        df.loc[df['session'].isna(), 'session'] = 1
        df.loc[df['session'].isin(['v1', 'V1']), 'session'] = 1
        df["session"] = df["session"].astype(int)
        self.scheme['session'] = self.scheme['session'].astype(int)
        return super()._extract_mask(df, unique_keys, check_uniqueness=check_uniqueness)

    def _check_integrity(self):
        return super()._check_integrity() & os.path.isfile(os.path.join(self.root, "mapping_site_name-class_scz.pkl"))

    def __init__(self, root: str, *args, **kwargs):
        self._site_mapping = self.load_pickle(os.path.join(root, "mapping_site_name-class_scz.pkl"))
        super().__init__(root, *args, **kwargs)

class BipolarDataset(ClinicalBase):

    @property
    def _studies(self):
        return ["biobd", "bsnip", "cnp", "candi"]

    @property
    def _train_val_test_scheme(self):
        return "train_val_test_test-intra_bip_stratified.pkl"

    @property
    def _unique_keys(self):
        return ["participant_id", "session", "study"]

    @property
    def _dx_site_mappings(self):
        return dict(diagnosis={"control": 0, "bipolar": 1, "bipolar disorder": 1, "psychotic bipolar disorder": 1},
                    site=self._site_mapping)

    def _extract_mask(self, df: pd.DataFrame, unique_keys: Sequence[str], check_uniqueness: bool=True):
        df = df.copy()
        df.loc[df['session'].isna(), 'session'] = 1
        df.loc[df['session'].isin(['v1', 'V1']), 'session'] = 1
        df["session"] = df["session"].astype(int)
        self.scheme['session'] = self.scheme['session'].astype(int)
        return super()._extract_mask(df, unique_keys, check_uniqueness=check_uniqueness)

    def _check_integrity(self):
        return super()._check_integrity() & os.path.isfile(os.path.join(self.root, "mapping_site_name-class_bip.pkl"))

    def __init__(self, root: str, *args, **kwargs):
        self._site_mapping = self.load_pickle(os.path.join(root, "mapping_site_name-class_bip.pkl"))
        super().__init__(root, *args, **kwargs)


