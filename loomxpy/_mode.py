import os
import json
import abc
import warnings
from typing import MutableMapping, List
from enum import Enum

import pandas as pd
import numpy as np
from scipy import sparse
import loompy as lp

from loomxpy import __DEBUG__
from ._s7 import S7
from ._hooks import WithInitHook
from ._matrix import DataMatrix
from .utils import df_to_named_matrix, compress_encode


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
        # # Observations
        self._observation_attrs = ObservationAttributes(mode=self)
        self._oa_annotations = ObservationAnnotationAttributes(mode=self)
        self._oa_metrics = ObservationMetricAttributes(mode=self)
        self._oa_embeddings = ObservationEmbeddingAttributes(mode=self)
        self._oa_clusterings = ObservationClusteringAttributes(mode=self)

    @property
    def X(self):
        return self._data_matrix

    @property
    def f(self):
        return self._feature_attrs

    @property
    def o(self):
        return self._observation_attrs

    def export(
        self, filename: str, output_format: str, title: str = None, genome: str = None
    ):
        if output_format == "scope_v1":
            _row_attrs: MutableMapping = {}
            _col_attrs: MutableMapping = {}
            _global_attrs: MutableMapping = {
                "title": os.path.splitext(os.path.basename(filename))[0]
                if title is None
                else title,
                "MetaData": {
                    "annotations": [],
                    "metrics": [],
                    "embeddings": [],
                    "clusterings": [],
                },
                "Genome": genome,
            }
            # Add row attributes (in Loom specifications)
            for k, attr in self._feature_attrs:
                _col_name = attr.data.columns[0]
                _row_attrs[k] = attr.data[_col_name].values

            # Add columns attributes (in Loom specifications)
            _default_embedding = None
            _embeddings_X = pd.DataFrame(index=self._data_matrix._observation_names)
            _embeddings_Y = pd.DataFrame(index=self._data_matrix._observation_names)
            _clusterings = pd.DataFrame(index=self._data_matrix._observation_names)

            for k, attr in self._observation_attrs:
                if attr.attr_type.value == AttributeType.ANNOTATION.value:
                    _col_name = attr.data.columns[0]
                    # Categorical not valid, ndarray is required
                    _col_attrs[k] = np.asarray(attr.data[_col_name].values)
                    _global_attrs["MetaData"]["annotations"].append(
                        {
                            "name": k,
                            "values": list(
                                map(
                                    lambda x: x.item()
                                    if type(x).__module__ == "numpy"
                                    else x,
                                    sorted(
                                        np.unique(attr.data[_col_name].values),
                                        reverse=False,
                                    ),
                                )
                            ),
                        }
                    )
                if attr.attr_type.value == AttributeType.METRIC.value:
                    _col_name = attr.data.columns[0]
                    _col_attrs[k] = attr.data[_col_name].values
                    _global_attrs["MetaData"]["metrics"].append({"name": k})

                if attr.attr_type.value == AttributeType.EMBEDDING.value:
                    _data = attr.data.loc[:, 0:1].copy()
                    _data.columns = ["_X", "_Y"]

                    _num_embeddings = len(_global_attrs["MetaData"]["embeddings"])
                    _embedding_id = 0 if _num_embeddings == 0 else _num_embeddings + 1
                    _embeddings_X = pd.merge(
                        _embeddings_X,
                        _data["_X"]
                        .to_frame()
                        .rename(columns={"_X": str(_embedding_id)})
                        .astype("float32"),
                        left_index=True,
                        right_index=True,
                    )
                    _embeddings_Y = pd.merge(
                        _embeddings_Y,
                        _data["_Y"]
                        .to_frame()
                        .rename(columns={"_Y": str(_embedding_id)})
                        .astype("float32"),
                        left_index=True,
                        right_index=True,
                    )
                    _global_attrs["MetaData"]["embeddings"].append(
                        {
                            "id": str(
                                _embedding_id
                            ),  # TODO: type not consistent with clusterings
                            "name": k,
                        }
                    )

                if attr.attr_type.value == AttributeType.CLUSTERING.value:
                    if attr.name is None:
                        raise Exception(
                            f"The clustering with key '{attr.key}' does not have a name. This is required when exporting to SCope."
                        )
                    _col_name = attr.data.columns[0]
                    _num_clusterings = len(_global_attrs["MetaData"]["clusterings"])
                    _clustering_id = (
                        0 if _num_clusterings == 0 else _num_clusterings + 1
                    )
                    _clusterings = pd.merge(
                        _clusterings,
                        attr.data.rename(columns={_col_name: str(_clustering_id)}),
                        left_index=True,
                        right_index=True,
                    )
                    _global_attrs["MetaData"]["clusterings"].append(
                        {
                            "id": _clustering_id,
                            # "key": cluster.key,
                            "name": attr.name,
                            "group": "",
                            "clusters": [],
                            "clusterMarkerMetrics": [],
                        }
                    )
                    cluster: Cluster
                    for k, cluster in self._oa_clusterings.get_attribute(key=attr.key):
                        _global_attrs["MetaData"]["clusterings"][_clustering_id][
                            "clusters"
                        ].append(
                            {
                                "id": cluster.id.item(),
                                # "name": cluster.name,
                                "description": cluster.description,
                            }
                        )
            _row_attrs["Gene"] = np.asarray(self._data_matrix._feature_names)
            _col_attrs["CellID"] = np.asarray(self._data_matrix._observation_names)
            if _default_embedding is None:
                _col_attrs["Embedding"] = df_to_named_matrix(
                    df=pd.DataFrame(
                        {
                            "_X": _embeddings_X["0"].values,
                            "_Y": _embeddings_Y["0"].values,
                        }
                    )
                )
                _embeddings_X.insert(
                    loc=0, column="-1", value=_embeddings_X["0"].values
                )
                _embeddings_Y.insert(
                    loc=0, column="-1", value=_embeddings_Y["0"].values
                )
                _md_first_embedding = list(
                    filter(
                        lambda x: x["id"] == "0",
                        _global_attrs["MetaData"]["embeddings"],
                    )
                )[0]
                _global_attrs["MetaData"]["embeddings"] = [
                    {"id": "-1", "name": _md_first_embedding["name"]}
                ] + list(
                    filter(
                        lambda x: x["id"] != "0",
                        _global_attrs["MetaData"]["embeddings"],
                    )
                )
            _col_attrs["Embeddings_X"] = df_to_named_matrix(df=_embeddings_X)
            _col_attrs["Embeddings_Y"] = df_to_named_matrix(df=_embeddings_Y)
            _col_attrs["Clusterings"] = df_to_named_matrix(
                df=_clusterings.astype(np.int16)
            )
            _global_attrs["MetaData"] = json.dumps(_global_attrs["MetaData"])
            _global_attrs["MetaData"] = compress_encode(value=_global_attrs["MetaData"])
            lp.create(
                filename=filename,
                layers=self._data_matrix._data_matrix.transpose(),
                row_attrs=_row_attrs,
                col_attrs=_col_attrs,
                file_attrs=_global_attrs,
            )
            print("INFO: LoomX successfully exported to SCope-compatible loom file.")

        else:
            raise Exception(
                f"Cannot export LoomX to the given output format '{output_format}'. Invalid output format"
            )


class Modes(MutableMapping[str, object], metaclass=WithInitHook):
    def __init__(self):
        """"""
        self._keys: List[str] = []
        self._mode_types = [item.value for item in ModeType]

    def __setattr__(self, name, value):
        if not hasattr(self, "_initialized"):
            if __DEBUG__:
                print(f"DEBUG: constructor call: set attr with name {name}")
            super().__setattr__(name, value)
        else:
            self.__setitem__(name=name, value=value)

    def __delattr__(self, name: str):
        self._keys.remove(name)
        super().__delattr__(name)

    def __iter__(self):
        """"""
        raise NotImplementedError

    def __len__(self):
        """"""
        raise NotImplementedError

    def __delitem__(self, name) -> None:
        """"""
        self.__delattr__(name)

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
        if _key not in self._keys:
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
    EMBEDDING = 2
    CLUSTERING = 3


class Attribute:
    def __init__(
        self,
        key: str,
        mode_type: ModeType,
        attr_type: AttributeType,
        axis: Axis,
        data,
        name: str = None,
        description: str = None,
    ):
        """"""
        self._key = key
        self._mode_type = mode_type
        self._attr_type = attr_type
        self._axis = axis
        self._data = data
        self._name = name
        self._description = description

    @property
    def key(self):
        return self._key

    @property
    def mode_type(self):
        return self._mode_type

    @property
    def attr_type(self):
        return self._attr_type

    @property
    def axis(self):
        return self._axis

    @property
    def data(self):
        return self._data

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def description(self):
        return self._description

    def __repr__(self):
        """"""
        return f"""
key: {self._key}
mode: {self._mode_type}
type: {self._attr_type}
name: {self._name}
description: {self._description}
        """


class Attributes(MutableMapping[str, Attribute], metaclass=WithInitHook):
    def __init__(self, mode: Mode, axis: Axis, is_proxy: bool = False, **kwargs):
        """"""
        self._keys: List[str] = []
        self._mode = mode
        self._mode_type = mode._mode_type if mode is not None else ModeType.NONE
        self._axis = axis
        self._is_proxy = is_proxy
        self._attr_type = kwargs["attr_type"] if "attr_type" in kwargs else False
        self._is_multi = True if "is_multi" in kwargs and kwargs["is_multi"] else False

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

    def __delattr__(self, name):
        self._keys.remove(name)
        super().__delattr__(name)

    def __delitem__(self, key):
        """"""
        self.__delattr__(key)

    def __getitem__(self, key):
        """"""
        return super().__getattribute__(key).data

    def __iter__(self):
        """"""
        return iter(AttributesIterator(self))

    def __len__(self):
        """"""
        return len(self._keys)

    def _add_key(self, key: str):
        if key not in self._keys:
            self._keys.append(key)

    def _add_item(self, key: str, value: Attribute) -> None:
        self._add_key(key=key)
        super().__setattr__(key, value)

    def _add_item_by_value(self, value: Attribute):
        self._add_key(key=value.key)
        super().__setattr__(value.key, value)

    def get_attribute(self, key):
        """"""
        return super().__getattribute__(key)

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

        if not self._is_multi and value.shape[1] > 1:
            raise Exception(
                f"Cannot add attribute of shape {value.shape[1]}. Currently, allows only {type(value).__name__} with maximally 1 feature (i.e.: column)."
            )


class AttributesIterator:

    """Class to implement an iterator of Attributes """

    def __init__(self, attrs: Attributes):
        self._attrs = attrs

    def __iter__(self):
        self._n = 0
        return self

    def __next__(self):
        if self._n < len(self._attrs._keys):
            current_key = self._attrs._keys[self._n]
            self._n += 1
            return current_key, self._attrs.get_attribute(current_key)
        else:
            raise StopIteration


##########################################
# ATTRIBUTE TYPES                        #
##########################################


class AnnotationAttributes(Attributes):
    def __init__(self, mode: Mode, axis: Axis, is_proxy: bool = False, **kwargs):
        """"""
        super().__init__(
            mode=mode,
            axis=axis,
            is_proxy=is_proxy,
            attr_type=AttributeType.ANNOTATION,
            **kwargs,
        )

    def _validate_value(
        self,
        value: pd.core.frame.DataFrame,
        force_conversion_to_categorical: bool = False,
    ):
        if __DEBUG__:
            print(f"DEBUG: _validate_value ({type(self).__name__})")
        super()._validate_value(value=value)
        # Do some checks and processing for attribute of type ANNOTATION
        if (
            not force_conversion_to_categorical
            and not all(value.apply(pd.api.types.is_categorical_dtype))
            and not all(value.apply(pd.api.types.is_bool_dtype))
        ):
            _dtype = value.infer_objects().dtypes[0]
            raise Exception(
                f"Expects value to be categorical or bool but its dtype is {_dtype}. You can force the conversion to categorical by using <loomx-instance>.modes.<mode>.annotations.add(*, force=True)."
            )

    def _normalize_value(
        self,
        name: str,
        value: pd.core.frame.DataFrame,
        force_conversion_to_categorical: bool = False,
    ):
        if force_conversion_to_categorical:
            # Convert to Categorical
            warnings.warn(f"Converting {name} annotation to categorical type...")
            return value.astype("category")
        return value


class MetricAttributes(Attributes):
    def __init__(self, mode: Mode, axis: Axis, is_proxy: bool = False, **kwargs):
        """"""
        super().__init__(
            mode=mode,
            axis=axis,
            is_proxy=is_proxy,
            attr_type=AttributeType.METRIC,
            **kwargs,
        )

    def _validate_value(
        self, value: pd.core.frame.DataFrame, force_conversion_to_numeric: bool = False
    ):
        if __DEBUG__:
            print(f"DEBUG: _validate_value ({type(self).__name__})")
        super()._validate_value(value=value)
        # Do some checks and processing for attribute of type METRIC
        if not force_conversion_to_numeric and not all(
            value.apply(pd.api.types.is_numeric_dtype)
        ):
            _dtype = value.infer_objects().dtypes[0]
            raise Exception(f"Expects value to be numeric but its dtype is {_dtype}")

    def _normalize_value(
        self,
        name: str,
        value: pd.core.frame.DataFrame,
        force_conversion_to_numeric: bool = False,
    ):
        if force_conversion_to_numeric:
            # Convert to metric
            warnings.warn(f"Converting {name} metric to numeric type...")
            return pd.to_numeric(value)
        return value


class EmbeddingAttributes(Attributes):
    def __init__(self, mode: Mode, axis: Axis, is_proxy: bool = False, **kwargs):
        """"""
        super().__init__(
            mode=mode,
            axis=axis,
            is_proxy=is_proxy,
            attr_type=AttributeType.EMBEDDING,
            **kwargs,
        )


class ClusteringAttributes(Attributes):
    def __init__(self, mode: Mode, axis: Axis, is_proxy: bool = False, **kwargs):
        """"""
        super().__init__(
            mode=mode,
            axis=axis,
            is_proxy=is_proxy,
            attr_type=AttributeType.CLUSTERING,
            **kwargs,
        )


##########################################
# AXIS ATTRIBUTES - FEATURES             #
##########################################


class FeatureAttributes(Attributes):
    def __init__(self, mode: Mode, is_proxy: bool = False):
        """"""
        super().__init__(mode=mode, axis=Axis.FEATURES, is_proxy=is_proxy)

    def _validate_key(self, key: str):
        super()._validate_key(key=key)

    def _validate_value(self, value: pd.DataFrame, **kwargs):
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
        _attr = Attribute(
            key=name,
            attr_type=AttributeType.ANNOTATION,
            mode_type=self._mode_type,
            axis=self._axis,
            data=value,
        )
        super()._add_item(key=name, value=_attr)

    @property
    def annotations(self):
        return self._mode._fa_annotations

    def add_annotation(self, key: str, value: pd.core.frame.DataFrame):
        self._mode._fa_annotations.add(key=key, value=value)

    @property
    def metrics(self):
        return self._mode._fa_metrics

    def add_metric(self, key: str, value: pd.core.frame.DataFrame):
        self._mode._fa_metrics.add(key=key, value=value)


class FeatureAnnotationAttributes(FeatureAttributes, AnnotationAttributes):
    def __init__(self, mode: Mode):
        """"""
        super().__init__(mode=mode, is_proxy=True)

    def __setitem__(self, name, value):
        """"""
        self.add(key=name, value=value)

    def add(self, key: str, value: pd.core.frame.DataFrame):
        """"""
        super()._validate_key(key=key)
        super()._validate_value(value=value)
        _data = super()._normalize_value(name=key, value=value)

        _attr = Attribute(
            key=key,
            mode_type=self._mode_type,
            attr_type=self._attr_type,
            axis=self._axis,
            data=_data,
        )

        self._mode._feature_attrs._add_item(key=key, value=_attr)
        super()._add_item_by_value(value=_attr)


class FeatureMetricAttributes(FeatureAttributes, MetricAttributes):
    def __init__(self, mode):
        """"""
        super().__init__(mode=mode, is_proxy=True)

    def __setitem__(self, name: str, value: pd.core.frame.DataFrame):
        """"""
        self.add(key=name, value=value)

    def add(self, key: str, value: pd.core.frame.DataFrame):
        """"""
        super()._validate_key(key=key)
        super()._validate_value(value=value)
        _data = super()._normalize_value(name=key, value=value)

        _attr = Attribute(
            key=key,
            mode_type=self._mode_type,
            attr_type=self._attr_type,
            axis=self._axis,
            data=_data,
        )

        self._mode._feature_attrs._add_item(key=key, value=_attr)
        super()._add_item_by_value(value=_attr)


##########################################
# AXIS ATTRIBUTES - OBSERVATIONS         #
##########################################


class ObservationAttributes(Attributes):
    def __init__(self, mode: Mode, is_proxy: bool = False, **kwargs):
        """"""
        super().__init__(mode=mode, axis=Axis.OBSERVATIONS, is_proxy=is_proxy, **kwargs)

    def _validate_value(self, value, **kwargs):
        super()._validate_value(value=value, **kwargs)

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
        _attr = Attribute(
            key=name,
            attr_type=AttributeType.ANNOTATION,
            mode_type=self._mode_type,
            axis=self._axis,
            data=value,
        )
        super()._add_item(key=name, value=_attr)

    @property
    def annotations(self):
        return self._mode._oa_annotations

    def add_annotation(
        self,
        key: str,
        value: pd.core.frame.DataFrame,
        name: str = None,
        description: str = None,
        force: bool = False,
    ) -> None:
        self._mode._oa_annotations.add(
            key=key, value=value, name=name, description=description, force=force
        )

    @property
    def metrics(self):
        return self._mode._oa_metrics

    def add_metric(
        self,
        key: str,
        value: pd.core.frame.DataFrame,
        name: str = None,
        description: str = None,
        force: bool = False,
    ) -> None:
        self._mode._oa_metrics.add(
            key=key, value=value, name=name, description=description, force=force
        )

    @property
    def embeddings(self):
        return self._mode._oa_embeddings

    def add_embedding(
        self,
        key: str,
        value: pd.core.frame.DataFrame,
        name: str = None,
        description: str = None,
    ) -> None:
        self._mode._oa_embeddings.add(
            key=key,
            value=value,
            name=name,
            description=description,
        )

    @property
    def clusterings(self):
        return self._mode._oa_clusterings

    def add_clustering(
        self,
        key: str,
        value: pd.core.frame.DataFrame,
        name: str = None,
        description: str = None,
    ):
        self._mode._oa_clusterings.add(
            key=key, value=value, name=name, description=description
        )


class ObservationAnnotationAttributes(ObservationAttributes, AnnotationAttributes):
    def __init__(self, mode):
        """"""
        super().__init__(mode=mode, is_proxy=True)

    def __setitem__(self, name, value):
        """"""
        self.add(key=name, value=value)

    def add(
        self, key, value, name: str = None, description: str = None, force: bool = False
    ):
        """"""
        super()._validate_key(key=key)
        super()._validate_value(value=value, force_conversion_to_categorical=force)
        _data = super()._normalize_value(
            name=key, value=value, force_conversion_to_categorical=force
        )

        _attr = Attribute(
            key=key,
            mode_type=self._mode_type,
            attr_type=self._attr_type,
            axis=self._axis,
            data=_data,
            name=name,
            description=description,
        )

        self._mode._observation_attrs._add_item(key=key, value=_attr)
        super()._add_item_by_value(value=_attr)


class ObservationMetricAttributes(ObservationAttributes, MetricAttributes):
    def __init__(self, mode):
        """"""
        super().__init__(mode=mode, is_proxy=True)

    def __setitem__(self, name: str, value: pd.core.frame.DataFrame):
        """"""
        self.add(key=name, value=value)

    def add(
        self, key, value, name: str = None, description: str = None, force: bool = False
    ):
        """"""
        super()._validate_key(key=key)
        super()._validate_value(value=value)
        _data = super()._normalize_value(
            name=key, value=value, force_conversion_to_numeric=force
        )

        _attr = Attribute(
            key=key,
            mode_type=self._mode_type,
            attr_type=self._attr_type,
            axis=self._axis,
            data=_data,
            name=name,
            description=description,
        )

        self._mode._observation_attrs._add_item(key=key, value=_attr)
        super()._add_item_by_value(value=_attr)


class ProjectionMethod(Enum):
    PCA = 0
    TSNE = 1
    UMAP = 2


class EmbeddingAttribute(Attribute):
    def __init__(self, projection_method: ProjectionMethod = None, **kwargs):
        super().__init__(**kwargs)
        self._projection_method = projection_method

    @property
    def projection_method(self):
        return self._projection_method

    def __repr__(self):
        return f"""
{super().__repr__()}
projection method: {ProjectionMethod(self._projection_method).name}
        """


class ObservationEmbeddingAttributes(ObservationAttributes, EmbeddingAttributes):
    def __init__(self, mode):
        """"""
        super().__init__(mode=mode, is_proxy=True, is_multi=True)

    def __setitem__(self, name: str, value: pd.core.frame.DataFrame):
        """"""
        self.add(key=name, value=value)

    def add(
        self,
        key: str,
        value: pd.core.frame.DataFrame,
        name: str = None,
        description: str = None,
        projection_method: ProjectionMethod = None,
    ):
        super()._validate_key(key=key)
        super()._validate_value(value=value)

        _projection_method = None
        if projection_method:
            _projection_method = projection_method
        elif "pca" in key.lower() or (name is not None and "pca" in name.lower()):
            _projection_method = ProjectionMethod.PCA
        elif "tsne" in key.lower() or (name is not None and "tsne" in name.lower()):
            _projection_method = ProjectionMethod.TSNE
        elif "umap" in key.lower() or (name is not None and "umap" in name.lower()):
            _projection_method = ProjectionMethod.UMAP

        _attr = EmbeddingAttribute(
            key=key,
            mode_type=self._mode_type,
            attr_type=self._attr_type,
            axis=self._axis,
            data=value,
            name=name,
            description=description,
            projection_method=_projection_method,
        )
        self._mode._observation_attrs._add_item(key=key, value=_attr)
        super()._add_item_by_value(value=_attr)


class Cluster:
    def __init__(self, id: int, name: str = None, description: str = None):
        self._id = id
        self._name = name
        self._description = description

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        raise Exception("The ID of the clustering cannot be changed.")

    @property
    def name(self):
        if self._name is not None:
            return self._name
        return f"Unannotated Cluster {self._id}"

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def description(self):
        if self._description is not None:
            return self._description
        return self.name

    @description.setter
    def description(self, value):
        self._description = value


class ClusteringAttribute(Attribute):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._cluster_ids = sorted(
            np.unique(self._data.values).astype(int),
            reverse=False,
        )
        self._make_clusters()

    def __iter__(self):
        """"""
        return iter(ClusteringAttributeIterator(self))

    def __len__(self):
        """"""
        return len(self._keys)

    def __getattribute__(self, name):
        return super().__getattribute__(name)

    def _make_clusters(self):
        for cluster_id in self._cluster_ids:
            super().__setattr__(f"cluster_{cluster_id}", Cluster(id=cluster_id))


class ClusteringAttributeIterator:

    """Class to implement an iterator of ClusteringAttribute """

    def __init__(self, attr: ClusteringAttribute):
        self._attr = attr

    def __iter__(self):
        self._n = 0
        return self

    def __next__(self):
        if self._n < len(self._attr._cluster_ids):
            current_key = self._attr._cluster_ids[self._n]
            self._n += 1
            return current_key, self._attr.__getattribute__(f"cluster_{current_key}")
        else:
            raise StopIteration


class ObservationClusteringAttributes(ObservationAttributes, ClusteringAttributes):
    def __init__(self, mode):
        """"""
        super().__init__(mode=mode, is_proxy=True)

    def __setitem__(self, name: str, value: pd.core.frame.DataFrame):
        """"""
        self.add(key=name, value=value)

    def add(
        self,
        key: str,
        value: pd.core.frame.DataFrame,
        name: str = None,
        description: str = None,
    ):
        super()._validate_key(key=key)
        super()._validate_value(value=value)

        _attr = ClusteringAttribute(
            key=key,
            mode_type=self._mode_type,
            attr_type=self._attr_type,
            axis=self._axis,
            data=value,
            name=name,
            description=description,
        )
        self._mode._observation_attrs._add_item(key=key, value=_attr)
        super()._add_item_by_value(value=_attr)
