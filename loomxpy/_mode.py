import abc
import warnings
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
        self._feature_attrs = FeatureAttributes(mode=self)
        self._feature_annotation_attrs_proxy: FeatureAnnotationAttributes = (
            FeatureAnnotationAttributes(mode=self)
        )

    @property
    def X(self):
        return self._data_matrix

    @property
    def f(self):
        return self._feature_attrs

    @property
    def annotations(self):
        return self._feature_annotation_attrs_proxy


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
        raise NotImplementedError

    def __len__(self):
        """"""
        raise NotImplementedError

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


class Axis(Enum):
    OBSERVATIONS = 0
    FEATURES = 1


class AttributeType(Enum):
    ANNOTATION = 0


class Attribute:
    def __init__(
        self, key: str, mode_type: ModeType, attr_type: AttributeType, axis: Axis, data
    ):
        """"""
        self.key = key
        self.mode_type = mode_type
        self.attr_type = attr_type
        self.axis = axis
        self.data = data

    def __repr__(self):
        """"""
        return self.data.__repr__()


class Attributes(MutableMapping[str, Attribute], metaclass=WithInitHook):
    def __init__(self, mode: Mode):
        """"""
        self._keys = []
        self._mode = mode
        self._mode_type = mode._mode_type if mode is not None else ModeType.NONE

    def __delitem__(self, key):
        """"""
        self._keys.remove(key)
        super().__delattr__(key)

    def __getitem__(self, key):
        """"""
        return super().__getattribute__(key).data

    def __iter__(self):
        """"""
        return iter(AttributesIterator(self))

    def __len__(self):
        """"""
        return len(self._keys)

    def _add_item(self, key, value):
        self._keys.append(key)
        super().__setattr__(key, value)

    @abc.abstractclassmethod
    def __setitem__(self, key, value):
        """"""
        raise NotImplementedError


class AttributesIterator:

    """Class to implement an iterator of Attributes """

    def __init__(self, attrs: Attributes):
        self._attrs = attrs

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n <= len(self._attrs._keys):
            self.n += 1
            current_key = self._attrs._keys[self.n]
            return current_key, self._attrs[current_key]
        else:
            raise StopIteration


class FeatureAttributes(Attributes):
    def __init__(self, mode):
        """"""
        super().__init__(mode=mode)

    def _validate_key(self, key: str):
        if key.startswith("_"):
            raise Exception(
                f"Cannot add attribute with key {key}. Not a valid key. Expects key not to start with an underscore ('_')."
            )
        if not isinstance(key, str):
            raise Exception(
                f"Cannot add attribute with key of type ({type(key).__name__}) to {type(self).__name__}. Not a valid key. Expects key of type str."
            )

    def _validate_value(self, value):
        if not isinstance(value, pd.core.frame.DataFrame):
            raise Exception(
                f"Cannot add attribute of type {type(value).__name__} to {type(self).__name__}. Expects a pandas.core.frame.DataFrame."
            )
        # Check if all observations from the given value are present in the DataMatrix of this mode
        _observations = self._mode.X._observation_names
        if not all(np.in1d(value.index.astype(str), _observations)):
            raise Exception(
                f"Cannot add attribute of type {type(value).__name__} to {type(self).__name__}. Index of the given pandas.core.frame.DataFrame does not fully match with the DataMatrix of mode."
            )

    def __getattribute__(self, key):
        """"""
        if key in super().__getattribute__("_keys"):
            return super().__getattribute__(key).data
        return super().__getattribute__(key)

    def __setattr__(self, name, value):
        if not hasattr(self, "_initialized"):
            print(f"DEBUG: constructor call: set attr with name {name}")
            super().__setattr__(name, value)
        else:
            self.__setitem__(name=name, value=value)

    def __setitem__(self, name, value):
        """"""
        self._validate_key(key=name)
        self._validate_value(value=value)

        value = Attribute(
            key=name,
            mode_type=self._mode_type,
            attr_type=AttributeType.ANNOTATION,  # This should be inferred
            axis=Axis.FEATURES,
            data=value,
        )
        super()._add_item(key=name, value=value)


class FeatureAnnotationAttributes(FeatureAttributes):
    def __init__(self, mode):
        """"""
        super().__init__(mode=mode)

    def __getattribute__(self, key):
        """"""
        if key in super().__getattribute__("_keys"):
            return self._mode._feature_attrs[key]
        return super().__getattribute__(key)

    def __setattr__(self, name, value) -> None:
        if not hasattr(self, "_initialized"):
            print(f"DEBUG: constructor call: set attr with name {name}")
            super().__setattr__(name, value)
        else:
            self.__setitem__(name=name, value=value)

    def _validate_value(self, value):
        super()._validate_value(value=value)

    def __setitem__(self, name, value):
        """"""
        super()._validate_key(key=name)
        self._validate_value(value=value)

        # Convert to Categorical
        warnings.warn(f"Converting {name} annotation to categorical type...")
        value = value.astype("category")

        value = Attribute(
            key=name,
            mode_type=self._mode_type,
            attr_type=AttributeType.ANNOTATION,
            axis=Axis.FEATURES,
            data=value,
        )
        self._mode._feature_attrs._add_item(key=name, value=value)
        super()._add_item(key=name, value=value)
