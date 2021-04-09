from typing import MutableMapping
from enum import Enum

import pandas as pd
import numpy as np
from scipy import sparse

from ._s7 import S7
from ._hooks import WithInitHook
from ._matrix import DataMatrix


class ModeType(Enum):
    NONE = "_"
    RNA = "rna"
    ATAC = "atac"


class Mode(S7):
    def __init__(self, mode_type: ModeType, data_matrix: DataMatrix):
        """
        constructor for Mode
        """
        self._mode_type = mode_type
        self._data_matrix = data_matrix

    @property
    def X(self):
        return self._data_matrix


class Modes(MutableMapping[str, object], metaclass=WithInitHook):
    def __init__(self):
        """"""
        self._keys = []
        self._mode_types = [item.value for item in ModeType]

    def __setattr__(self, name, value):
        if not hasattr(self, "_initialized"):
            print(f"DEBUG: constructor call: set attr with name {name}")
            super().__setattr__(name, value)
        else:
            self.__setitem__(name=name, value=value)

    def __iter__(self):
        """"""
        raise Exception("Iterator not implemented.")

    def __len__(self):
        """"""
        raise Exception("Length not implemented.")

    def __delitem__(self, name) -> None:
        """"""
        delattr(self, name)

    def __getitem__(self, name) -> object:
        """"""
        return getattr(self, name)

    def _validate_key(self, key):
        _key = None
        if key.startswith("_"):
            raise Exception(
                f"Cannot add Mode with key {key}. Not a valid key. Expects key not to start with an underscore ('_')."
            )

        if not isinstance(key, ModeType) and not str:
            raise Exception("Not a valid ModeType.")

        if isinstance(key, ModeType):
            if key in self._mode_types:
                _key = key
            else:
                raise Exception("Not a valid ModeType.")

        if isinstance(key, str):
            if key in self._mode_types:
                _key = key
            else:
                raise Exception(
                    f"Not a valid ModeType: {key}. Choose one ModeType from {', '.join(self._mode_types)}"
                )
        return _key

    @staticmethod
    def _validate_value(value):
        if not type(value) in [tuple, np.matrix, pd.DataFrame]:
            raise Exception(
                f"""
Got {type(value)} but expecting either:
- Tuple: (<SciPy CSR matrix>, <feature names>, <observation names>) or
- Tuple: (<NumPy 2D matrix>, <feature names>, <observation names>) or
- pandas.DataFrame
                """
            )

        if isinstance(value, sparse.csr_matrix):
            raise Exception(
                "If your data matrix is a SciPy CSR matrix, use tuple: (<SciPy CSR matrix>, <feature names>, <observation names>)."
            )

        if isinstance(value, sparse.csc_matrix):
            raise Exception(
                "If your data matrix is a SciPy CSC matrix, use tuple: (<SciPy CSC matrix>, <feature names>, <observation names>)."
            )

        if isinstance(value, np.matrix):
            raise Exception(
                "If your data matrix is a NumPy 2D matrix, use tuple: (<NumPy 2D matrix>, <feature names>, <observation names>)."
            )

        if isinstance(value, tuple):
            if len(value) != 3:
                raise Exception(
                    "If your data matrix is a NumPy 2D matrix or SciPy CSR matrix, use tuple: (<NumPy 2D matrix | SciPy CSR matrix>, <feature names>, <observation names>)."
                )

    def __setitem__(self, name, value) -> None:
        """"""
        print(f"DEBUG: instance call: set attr with name {name}")
        print(f"adding new {name} mode")
        _key = self._validate_key(key=name)
        Modes._validate_value(value=value)

        _mode = None
        _data_matrix = None

        if isinstance(value, tuple):
            _matrix, _feature_names, _observation_names = value
            _data_matrix = DataMatrix(
                data_matrix=_matrix,
                feature_names=_feature_names,
                observation_names=_observation_names,
            )

        if isinstance(value, pd.DataFrame):
            _data_matrix = DataMatrix(
                data_matrix=value.values,
                feature_names=value.columns,
                observation_names=value.index,
            )

        if _data_matrix is None:
            raise Exception("Invalid type of the given data natrix.")

        _mode = Mode(mode_type=ModeType(_key), data_matrix=_data_matrix)
        self._keys.append(_key)
        super().__setattr__(_key, _mode)

    def __repr__(self) -> str:
        _mode_keys = f"{', '.join(self._keys)}" if len(self._keys) > 0 else "none"
        return f"Modalities: {_mode_keys}"