import abc
import warnings
from typing import MutableMapping
from enum import Enum

import pandas as pd
import numpy as np
from scipy import sparse

from loomxpy import __DEBUG__
from ._s7 import S7
from ._hooks import WithInitHook
from ._matrix import DataMatrix


##########################################
# MODES                                  #
##########################################


class ModeType(Enum):
    NONE = "_"
    RNA = "rna"


class Mode(S7):
    def __init__(self, mode_type: ModeType, data_matrix: DataMatrix):
        """
        constructor for Mode
        """
        self._mode_type = mode_type
        # Data Matrix
        self._data_matrix = data_matrix
        # Features
        self._feature_attrs = FeatureAttributes(mode=self)
        self._fa_annotations = FeatureAnnotationAttributes(mode=self)
        self._fa_metrics = FeatureMetricAttributes(mode=self)
        # Observations
        self._observation_attrs = ObservationAttributes(mode=self)
        self._oa_annotations = ObservationAnnotationAttributes(mode=self)

    @property
    def X(self):
        return self._data_matrix

    @property
    def f(self):
        return self._feature_attrs

    @property
    def o(self):
        return self._observation_attrs


class Modes(MutableMapping[str, object], metaclass=WithInitHook):
    def __init__(self):
        """"""
        self._keys = []
        self._mode_types = [item.value for item in ModeType]

    def __setattr__(self, name, value):
        if not hasattr(self, "_initialized"):
            if __DEBUG__:
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

    def __getitem__(self, name) -> Mode:
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
        if __DEBUG__:
            print(f"DEBUG: instance call: set attr with name {name}")
        print(f"INFO: adding new {name} mode")
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


##########################################
# ATTRIBUTES                             #
##########################################


class Axis(Enum):
    OBSERVATIONS = 0
    FEATURES = 1


class AttributeType(Enum):
    ANNOTATION = 0
    METRIC = 1


class Attribute:
    def __init__(
        self, key: str, mode_type: ModeType, attr_type: AttributeType, axis: Axis, data
    ):
        """"""
        self._key = key
        self._mode_type = mode_type
        self._attr_type = attr_type
        self._axis = axis
        self._data = data

    @property
    def key(self):
        return self._key

    @property
    def data(self):
        return self._data

    def __repr__(self):
        """"""
        return self.data.__repr__()


class Attributes(MutableMapping[str, Attribute], metaclass=WithInitHook):
    def __init__(self, mode: Mode, axis: Axis, is_proxy: bool = False):
        """"""
        self._keys = []
        self._mode = mode
        self._mode_type = mode._mode_type if mode is not None else ModeType.NONE
        self._axis = axis
        self._is_proxy = is_proxy

    def __getattribute__(self, key):
        """"""
        if not super().__getattribute__("_is_proxy"):
            if key in super().__getattribute__("_keys"):
                return super().__getattribute__(key).data
            return super().__getattribute__(key)
        else:
            """
            This is a proxy. Override __getattribute__ of Attributes class
            """
            if key in super().__getattribute__("_keys"):
                if super().__getattribute__("_axis") == Axis.FEATURES:
                    return super().__getattribute__("_mode")._feature_attrs[key]
                elif super().__getattribute__("_axis") == Axis.OBSERVATIONS:
                    return super().__getattribute__("_mode")._observation_attrs[key]
                else:
                    raise Exception("Invalid axis.")
            return super().__getattribute__(key)

    def __setattr__(self, name, value):
        if not hasattr(self, "_initialized"):
            if __DEBUG__:
                print(f"DEBUG: constructor call: set attr with name {name}")
            super().__setattr__(name, value)
        else:
            self.__setitem__(name=name, value=value)

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

    def _add_item(self, key: str, attr_type: AttributeType, attr_value) -> Attribute:
        self._keys.append(key)
        value = Attribute(
            key=key,
            mode_type=self._mode_type,
            attr_type=attr_type,
            axis=self._axis,
            data=attr_value,
        )
        super().__setattr__(key, value)
        return value

    def _add_item_by_ref(self, attr: Attribute):
        super().__setattr__(attr.key, attr.data)

    @abc.abstractclassmethod
    def __setitem__(self, name, value):
        """"""
        raise NotImplementedError

    def _validate_key(self, key):
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

        if value.shape[1] > 1:
            raise Exception(
                f"Cannot add attribute of shape {value.shape[1]}. Currently, allows only {type(value).__name__} with maximally 1 feature (i.e.: column)."
            )


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


##########################################
# ATTRIBUTE TYPES                        #
##########################################


class AnnotationAttributes(Attributes):
    def __init__(self, mode: Mode, axis: Axis, is_proxy: bool = False):
        """"""
        super().__init__(mode=mode, axis=axis, is_proxy=is_proxy)
        self._force_conversion_to_categorical = False

    def _validate_value(self, value: pd.core.frame.DataFrame):
        if __DEBUG__:
            print(f"DEBUG: _validate_value ({type(self).__name__})")
        # Do some checks and processing for attribute of type ANNOTATION
        if (
            not self._force_conversion_to_categorical
            and not pd.api.types.is_categorical_dtype(arr_or_dtype=value.values)
            and not pd.api.types.is_bool_dtype(arr_or_dtype=value.values)
        ):
            _dtype = value.infer_objects().dtypes[0]
            raise Exception(
                f"Expects value to be categorical or bool but its dtype is {_dtype}"
            )

    def _normalize_value(self, name: str, value: pd.core.frame.DataFrame):
        if self._force_conversion_to_categorical:
            # Convert to Categorical
            warnings.warn(f"Converting {name} annotation to categorical type...")
            return value.astype("category")
        return value

    @property
    def force(self):
        return self._force_conversion_to_categorical

    @force.setter
    def force(self, force_conversion_to_categorical: bool):
        if not isinstance(force_conversion_to_categorical, bool):
            raise Exception(f"Expects a bool dtype value.")
        self._force_conversion_to_categorical = force_conversion_to_categorical


class MetricAttributes(Attributes):
    def __init__(self, mode: Mode, axis: Axis, is_proxy: bool = False):
        """"""
        super().__init__(mode=mode, axis=axis, is_proxy=is_proxy)
        self._force_conversion_to_numeric = False

    def _validate_value(self, value: pd.core.frame.DataFrame):
        if __DEBUG__:
            print(f"DEBUG: _validate_value ({type(self).__name__})")
        # Do some checks and processing for attribute of type METRIC
        if not self._force_conversion_to_numeric and not pd.api.types.is_numeric_dtype(
            arr_or_dtype=value.values
        ):
            _dtype = value.infer_objects().dtypes[0]
            raise Exception(f"Expects value to be numeric but its dtype is {_dtype}")

    def _normalize_value(self, name: str, value: pd.core.frame.DataFrame):
        if self._force_conversion_to_numeric:
            # Convert to metric
            warnings.warn(f"Converting {name} metric to numeric type...")
            return pd.to_numeric(value)
        return value

    @property
    def force(self):
        return self._force_conversion_to_numeric

    @force.setter
    def force(self, force_conversion_to_numeric: bool):
        if not isinstance(force_conversion_to_numeric, bool):
            raise Exception(f"Expects a bool dtype value.")
        self._force_conversion_to_numeric = force_conversion_to_numeric


##########################################
# AXIS ATTRIBUTES                        #
##########################################


class FeatureAttributes(Attributes):
    def __init__(self, mode: Mode, is_proxy: bool = False):
        """"""
        super().__init__(mode=mode, axis=Axis.FEATURES, is_proxy=is_proxy)

    def _validate_key(self, key: str):
        super()._validate_key(key=key)

    def _validate_value(self, value):
        if __DEBUG__:
            print(f"DEBUG: _validate_value ({type(self).__name__})")
        # Generic validation
        super()._validate_value(value=value)
        # Check if all observations from the given value are present in the DataMatrix of this mode
        _features = self._mode.X._feature_names
        if not all(np.in1d(value.index.astype(str), _features)):
            raise Exception(
                f"Cannot add attribute of type {type(value).__name__} to {type(self).__name__}. Index of the given pandas.core.frame.DataFrame does not fully match the features in DataMatrix of mode."
            )

    def __setitem__(self, name, value):
        """"""
        self._validate_key(key=name)
        self._validate_value(value=value)
        # TODO: Attribute type should be inferred here
        super()._add_item(
            key=name, attr_value=value, attr_type=AttributeType.ANNOTATION
        )

    @property
    def annotations(self):
        return self._mode._fa_annotations

    @property
    def metrics(self):
        return self._mode._fa_metrics


class FeatureAnnotationAttributes(FeatureAttributes, AnnotationAttributes):
    def __init__(self, mode: Mode):
        """"""
        super().__init__(mode=mode, is_proxy=True)

    def __setitem__(self, name, value):
        """"""
        super()._validate_key(key=name)
        super()._validate_value(value=value)
        value = super()._normalize_value(name=name, value=value)

        _attr = self._mode._feature_attrs._add_item(
            key=name, attr_value=value, attr_type=AttributeType.ANNOTATION
        )
        super()._add_item_by_ref(attr=_attr)


class FeatureMetricAttributes(FeatureAttributes, MetricAttributes):
    def __init__(self, mode):
        """"""
        super().__init__(mode=mode, is_proxy=True)

    def __setitem__(self, name: str, value: pd.core.frame.DataFrame):
        """"""
        super()._validate_key(key=name)
        super()._validate_value(value=value)
        value = super()._normalize_value(name=name, value=value)

        _attr = self._mode._feature_attrs._add_item(
            key=name, attr_value=value, attr_type=AttributeType.METRIC
        )
        super()._add_item_by_ref(attr=_attr)


##########################################
# OBSERVATION ATTRIBUTES                 #
##########################################


class ObservationAttributes(Attributes):
    def __init__(self, mode: Mode, is_proxy: bool = False):
        """"""
        super().__init__(mode=mode, axis=Axis.OBSERVATIONS, is_proxy=is_proxy)

    def _validate_value(self, value):
        super()._validate_value(value=value)

        # Check if all observations from the given value are present in the DataMatrix of this mode
        _observations = self._mode.X._observation_names
        if not all(np.in1d(value.index.astype(str), _observations)):
            raise Exception(
                f"Cannot add attribute of type {type(value).__name__} to {type(self).__name__}. Index of the given pandas.core.frame.DataFrame does not fully match the observations in DataMatrix of mode."
            )

    def __setitem__(self, name, value):
        """"""
        super()._validate_key(key=name)
        self._validate_value(value=value)

        # TODO: Attribute type should be inferred here
        super()._add_item(
            key=name, attr_value=value, attr_type=AttributeType.ANNOTATION
        )

    @property
    def annotations(self):
        return self._mode._oa_annotations


class ObservationAnnotationAttributes(ObservationAttributes, AnnotationAttributes):
    def __init__(self, mode):
        """"""
        super().__init__(mode=mode, is_proxy=True)

    def __setitem__(self, name, value):
        """"""
        super()._validate_key(key=name)
        super()._validate_value(value=value)
        value = super()._normalize_value(name=name, value=value)

        _attr = self._mode._observation_attrs._add_item(
            key=name, attr_value=value, attr_type=AttributeType.ANNOTATION
        )
        super()._add_item_by_ref(attr=_attr)
