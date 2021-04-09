from typing import MutableMapping
from enum import Enum

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
        raise Exception("Iterator not implemented.")

    def __delitem__(self, name) -> None:
        """"""
        delattr(self, name)

    def __getitem__(self, name) -> object:
        """"""
        return getattr(self, name)

    def __setitem__(self, name, value) -> None:
        """"""
        print(f"DEBUG: instance call: set attr with name {name}")
        print(f"adding new {name} mode")
        _key = name
        _mode = None
        _data_matrix = None

        if isinstance(value, tuple):
            if len(value) != 3:
                raise Exception(
                    "If your data matrix is a NumPy 2D matrix or SciPy CSR matrix, use tuple: (<NumPy 2D matrix | SciPy CSR matrix>, <feature names>, <observation names>)."
                )

            _matrix, _feature_names, _observation_names = value
            _data_matrix = DataMatrix(
                data_matrix=_matrix,
                feature_names=_feature_names,
                observation_names=_observation_names,
            )

        if _data_matrix is None:
            raise Exception("Invalid type of the given data natrix.")

        _mode = Mode(mode_type=ModeType(_key), data_matrix=_data_matrix)
        self._keys.append(_key)
        super().__setattr__(_key, _mode)

    def __repr__(self) -> str:
        _mode_keys = f"{', '.join(self._keys)}" if len(self._keys) > 0 else "none"
        return f"Modalities: {_mode_keys}"