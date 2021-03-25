from typing import MutableMapping
from scipy import sparse
import numpy as np
import pandas as pd
from ._mode import Mode, ModeType
from ._matrix import DataMatrix

class Modes(MutableMapping[str, Mode]):

    def __init__(self):
        '''
        '''
        self._keys = []
        self._mode_types = [item.value for item in ModeType]

    def check_valid_key(self, key):
        _key = None
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
                raise Exception(f"Not a valid ModeType: {key}. Choose one ModeType from {', '.join(self._mode_types)}")
        return _key
    
    def __delitem__(self, key) -> None:
        '''
        '''
        delattr(self, key)

    def __getitem__(self, key) -> Mode:
        '''
        '''
        return getattr(self, key)

    def __setattr__(self, key, value) -> None:
        if not key.startswith("_") and not isinstance(value, Mode):
            self.__setitem__(key=key, value=value)
        else:
            super().__setattr__(key, value)

    def __setitem__(self, key, value) -> None:
        '''
        '''
        _key = self.check_valid_key(key=key)
        _mode = None
        _data_matrix = None

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
            raise Exception("If your data matrix is a SciPy CSR matrix, use tuple: (<SciPy CSR matrix>, <feature names>, <observation names>).")
        
        if isinstance(value, sparse.csc_matrix):
            raise Exception("If your data matrix is a SciPy CSC matrix, use tuple: (<SciPy CSC matrix>, <feature names>, <observation names>).")

        if isinstance(value, np.matrix):
            raise Exception("If your data matrix is a NumPy 2D matrix, use tuple: (<NumPy 2D matrix>, <feature names>, <observation names>).")
        
        if isinstance(value, tuple):
            if len(value) != 3:
                raise Exception("If your data matrix is a NumPy 2D matrix or SciPy CSR matrix, use tuple: (<NumPy 2D matrix | SciPy CSR matrix>, <feature names>, <observation names>).")
            
            _matrix, _feature_names, _observation_names = value
            _data_matrix = DataMatrix(
                data_matrix=_matrix,
                feature_names=_feature_names,
                observation_names=_observation_names
            )

        if isinstance(value, pd.DataFrame):
            _data_matrix = DataMatrix(
                data_matrix=value.values,
                feature_names=value.columns,
                observation_names=value.index
            )

        _mode = Mode(
            mode_type=ModeType(_key),
            data_matrix=_data_matrix
        )
        self._keys.append(_key)
        setattr(self, key, _mode)

    def __iter__(self):
        '''
        '''
        raise Exception("Cannot iterate on Modes.")

    def __len__(self):
        '''
        '''
        return len(self._keys)

    def keys(self):
        return frozenset(self._keys)


class LoomX():
    
    def __init__(self):
        '''
        constructor for Mode
        '''
        self.modes: Modes = Modes()    