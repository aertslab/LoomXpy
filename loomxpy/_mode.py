import os
import re
import json
import abc
import warnings
from typing import MutableMapping, List, Union
from functools import reduce
from enum import Enum

import pandas as pd
import numpy as np
from scipy import sparse
import loompy as lp

from loomxpy import __DEBUG__
from loomxpy._specifications import (
    ProjectionMethod,
    LoomXMetadataEmbedding,
    LoomXMetadataClustering,
    LoomXMetadataCluster,
    LoomXMetadataClusterMarkerMetric,
)
from loomxpy._s7 import S7
from loomxpy._errors import BadDTypeException
from loomxpy._hooks import WithInitHook
from loomxpy._matrix import DataMatrix
from loomxpy.utils import df_to_named_matrix, compress_encode


def custom_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return str(msg) + "\n"


warnings.formatwarning = custom_formatwarning


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
        # Global
        self._global_attrs = GlobalAttributes(mode=self)
        # Features
        self._feature_attrs = FeatureAttributes(mode=self)
        self._fa_annotations = FeatureAnnotationAttributes(mode=self)
        self._fa_metrics = FeatureMetricAttributes(mode=self)
        # Observations
        self._observation_attrs = ObservationAttributes(mode=self)
        self._oa_annotations = ObservationAnnotationAttributes(mode=self)
        self._oa_metrics = ObservationMetricAttributes(mode=self)
        self._oa_embeddings = ObservationEmbeddingAttributes(mode=self)
        self._oa_clusterings = ObservationClusteringAttributes(mode=self)

    @property
    def X(self):
        return self._data_matrix

    @property
    def g(self):
        return self._global_attrs

    @property
    def f(self):
        return self._feature_attrs

    @property
    def o(self):
        return self._observation_attrs

    def export(
        self,
        filename: str,
        output_format: str,
        title: str = None,
        genome: str = None,
        compress_metadata: bool = False,
        cluster_marker_metrics: List[LoomXMetadataClusterMarkerMetric] = [
            {
                "accessor": "avg_logFC",
                "name": "Avg. logFC",
                "description": f"Average log fold change from Wilcoxon test",
                "threshold": 0,
                "threshold_method": "lte_or_gte",  # lte, lt, gte, gt, lte_or_gte, lte_and_gte
            },
            {
                "accessor": "pval",
                "name": "Adjusted P-Value",
                "description": f"Adjusted P-Value from Wilcoxon test",
                "threshold": 0.05,
                "threshold_method": "lte",  # lte, lt, gte, gt, lte_or_gte, lte_and_gte
            },
        ],
    ):
        """
        Export this LoomX object to Loom file

        Parameters
        ---------
        cluster_marker_metrics: dict, optional
            List of dict (ClusterMarkerMetric) containing metadata of each metric available for the cluster markers.
            Expects each metric to be of type float.
        Return
        ------
        None
        """
        if output_format == "scope_v1":
            #
            _feature_names = self._data_matrix._feature_names
            # Init
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
            for _attr_key, _attr in self._feature_attrs:
                _row_attrs[_attr_key] = _attr.values

            # Add columns attributes (in Loom specifications)
            _default_embedding = None
            _embeddings_X = pd.DataFrame(index=self._data_matrix._observation_names)
            _embeddings_Y = pd.DataFrame(index=self._data_matrix._observation_names)
            _clusterings = pd.DataFrame(index=self._data_matrix._observation_names)

            for _attr_key, _attr in self._observation_attrs:
                if _attr.attr_type.value == AttributeType.ANNOTATION.value:
                    # Categorical not valid, ndarray is required
                    _col_attrs[_attr_key] = np.asarray(_attr.values)
                    _global_attrs["MetaData"]["annotations"].append(
                        {
                            "name": _attr_key,
                            "values": list(
                                map(
                                    lambda x: x.item()
                                    if type(x).__module__ == "numpy"
                                    else x,
                                    sorted(
                                        np.unique(_attr.values),
                                        reverse=False,
                                    ),
                                )
                            ),
                        }
                    )
                if _attr.attr_type.value == AttributeType.METRIC.value:
                    _col_attrs[_attr_key] = np.asarray(_attr.values)
                    _global_attrs["MetaData"]["metrics"].append({"name": _attr_key})

                if _attr.attr_type.value == AttributeType.EMBEDDING.value:
                    _attr: EmbeddingAttribute
                    _data = _attr.data.iloc[:, 0:2]
                    _data.columns = ["_X", "_Y"]

                    # Number of embeddings (don't count the default embedding since this will be use to determine the id of the embedding)
                    _num_embeddings = len(
                        list(
                            filter(
                                lambda x: int(x["id"]) != -1,
                                _global_attrs["MetaData"]["embeddings"],
                            )
                        )
                    )
                    _embedding_id = (
                        _attr.id
                        if _attr.id is not None
                        else (
                            -1
                            if _attr._default
                            else 0
                            if _num_embeddings == 0
                            else _num_embeddings + 1
                        )
                    )
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
                            "name": _attr.name,
                        }
                    )

                    if _attr.default:
                        _default_embedding = _data

                if _attr.attr_type.value == AttributeType.CLUSTERING.value:
                    _attr: ClusteringAttribute
                    if _attr.name is None:
                        raise Exception(
                            f"The clustering with key '{_attr.key}' does not have a name. This is required when exporting to SCope."
                        )

                    # Clustering
                    _col_name = (
                        _attr.data.columns[0]
                        if isinstance(_attr, pd.DataFrame)
                        else _attr.name
                    )
                    _num_clusterings = len(_global_attrs["MetaData"]["clusterings"])
                    _clustering_id = (
                        0 if _num_clusterings == 0 else _num_clusterings + 1
                    )
                    _clustering_data = (
                        _attr.data.rename(columns={_col_name: str(_clustering_id)})
                        if isinstance(_attr.data, pd.DataFrame)  # pd.DataFrame
                        else _attr.data.rename(str(_clustering_id))  # pd.Series
                    )
                    _clusterings = pd.merge(
                        _clusterings,
                        _clustering_data,
                        left_index=True,
                        right_index=True,
                    )
                    _clustering_md = LoomXMetadataClustering.from_dict(
                        {
                            "id": _clustering_id,
                            **_attr.metadata.to_dict(),
                        }
                    ).to_dict()

                    # Markers
                    # Dictionary of DataFrame (value) containing the the values of the different metric (key) across features (rows) and for each cluster (columns)
                    _cluster_markers_dict = {}

                    if cluster_marker_metrics:

                        has_cluster_markers = [
                            cluster.markers is not None
                            for cluster in _attr._metadata.clusters
                        ]
                        if not all(has_cluster_markers):
                            continue

                        # Init DataFrame mask of genes representing markers
                        cluster_markers = pd.DataFrame(
                            index=_feature_names,
                            columns=[str(x.id) for x in _attr._metadata.clusters],
                        ).fillna(0, inplace=False)

                        # Init DataFrame containing metric valuess
                        _cluster_marker_metric: LoomXMetadataClusterMarkerMetric
                        for _cluster_marker_metric in cluster_marker_metrics:
                            _cluster_markers_dict[
                                _cluster_marker_metric["accessor"]
                            ] = pd.DataFrame(
                                index=_feature_names,
                                columns=[str(x.id) for x in _attr._metadata.clusters],
                            ).fillna(
                                0, inplace=False
                            )

                        _cluster: LoomXMetadataCluster
                        for _cluster in _attr._metadata.clusters:

                            _features_df = pd.Series(
                                _cluster.markers.index.values,
                                index=_cluster.markers.index.values,
                            )

                            # Dictionary of Series (value) containing the values of the different metric (key) for the current cluster
                            _cluster_marker_metric_values_dict = {}
                            # Dictionary of Series (value) containing a boolean mask of the features that pass the filter criteria for the different metrics (key)
                            _cluster_marker_metric_masks_dict = {}

                            _cluster_marker_metric: LoomXMetadataClusterMarkerMetric
                            for _cluster_marker_metric in cluster_marker_metrics:

                                # Check if metric exists in markers table
                                if (
                                    _cluster_marker_metric["accessor"]
                                    not in _cluster.markers.columns
                                ):
                                    raise Exception(
                                        f"The cluster_marker_metrics argument was not properly defined. Missing {_cluster_marker_metric['accessor']} metric in the markers table. Available columns in markers table are f{''.join(_cluster.markers.columns)}."
                                    )

                                cluster_marker_metric_values = pd.Series(
                                    _cluster.markers[
                                        _cluster_marker_metric["accessor"]
                                    ].values,
                                    index=_cluster.markers.index.values,
                                ).astype(float)

                                if pd.isnull(cluster_marker_metric_values).any():
                                    raise Exception(
                                        f"NaN detected in {_cluster_marker_metric['accessor']} metric column of the markers table"
                                    )

                                if _cluster_marker_metric["threshold_method"] == "lte":
                                    feature_mask = (
                                        cluster_marker_metric_values
                                        < _cluster_marker_metric["threshold"]
                                    )
                                elif _cluster_marker_metric["threshold_method"] == "lt":
                                    feature_mask = (
                                        cluster_marker_metric_values
                                        <= _cluster_marker_metric["threshold"]
                                    )
                                elif (
                                    _cluster_marker_metric["threshold_method"]
                                    == "lte_or_gte"
                                ):
                                    feature_mask = np.logical_and(
                                        np.logical_or(
                                            cluster_marker_metric_values
                                            >= _cluster_marker_metric["threshold"],
                                            cluster_marker_metric_values
                                            <= -_cluster_marker_metric["threshold"],
                                        ),
                                        np.isfinite(cluster_marker_metric_values),
                                    )
                                else:
                                    raise Exception(
                                        "The given threshold method is not implemented"
                                    )
                                _cluster_marker_metric_masks_dict[
                                    _cluster_marker_metric["accessor"]
                                ] = feature_mask
                                _cluster_marker_metric_values_dict[
                                    _cluster_marker_metric["accessor"]
                                ] = cluster_marker_metric_values

                            # Create a new cluster marker mask based on all feature mask generated using each metric
                            cluster_marker_metrics_mask = np.logical_or.reduce(
                                [
                                    v
                                    for _, v in _cluster_marker_metric_masks_dict.items()
                                ]
                            )

                            marker_names = _features_df[cluster_marker_metrics_mask]

                            # Get a marker mask along all features in the matrix
                            marker_genes_along_data_mask = np.in1d(
                                _feature_names, marker_names
                            )
                            marker_genes_along_data = cluster_markers.index[
                                marker_genes_along_data_mask
                            ]

                            # Populate the marker mask
                            markers_df = pd.DataFrame(
                                1, index=marker_names, columns=["is_marker"]
                            )
                            cluster_markers.loc[
                                marker_genes_along_data_mask, str(_cluster.id)
                            ] = markers_df["is_marker"][marker_genes_along_data]

                            if pd.isnull(cluster_markers[str(_cluster.id)]).any():
                                raise Exception(
                                    f"NaN detected in markers DataFrame of cluster {_cluster.id}."
                                )

                            # Populate the marker nmetrics
                            for _cluster_marker_metric in _cluster_markers_dict.keys():
                                _metric_df = pd.DataFrame(
                                    _cluster_marker_metric_values_dict[
                                        _cluster_marker_metric
                                    ][cluster_marker_metrics_mask],
                                    index=marker_names,
                                    columns=[_cluster_marker_metric],
                                )
                                _cluster_markers_dict[_cluster_marker_metric].loc[
                                    marker_genes_along_data_mask, str(_cluster.id)
                                ] = _metric_df[_cluster_marker_metric][
                                    marker_genes_along_data
                                ]

                                if pd.isnull(
                                    _cluster_markers_dict[_cluster_marker_metric][
                                        str(_cluster.id)
                                    ]
                                ).any():
                                    raise Exception(
                                        f"NaN detected in markers metric {_cluster_marker_metric['accessor']} DataFrame of cluster {_cluster.id}."
                                    )

                        # Add the required global metadata for markes to be visualized in SCope
                        # Encapsule with mixin to avoid properties not required by gRPC

                        _clustering_md["clusterMarkerMetrics"] = [
                            LoomXMetadataClusterMarkerMetric.from_dict(cmm).to_dict()
                            for cmm in cluster_marker_metrics
                        ]

                    _global_attrs["MetaData"]["clusterings"].append(_clustering_md)

                    # Convert all markers related data to Loom compliant format
                    _row_attrs[
                        f"ClusterMarkers_{str(_clustering_id)}"
                    ] = df_to_named_matrix(cluster_markers)

                    for _cluster_marker_metric in _cluster_markers_dict.keys():
                        _row_attrs[
                            f"ClusterMarkers_{str(_clustering_id)}_{_cluster_marker_metric}"
                        ] = df_to_named_matrix(
                            _cluster_markers_dict[_cluster_marker_metric].astype(float)
                        )

            _row_attrs["Gene"] = np.asarray(_feature_names)
            _col_attrs["CellID"] = np.asarray(self._data_matrix._observation_names)

            # If no default embedding, use the first embedding as default
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
            else:
                _col_attrs["Embedding"] = df_to_named_matrix(df=_default_embedding)

            _col_attrs["Embeddings_X"] = df_to_named_matrix(df=_embeddings_X)
            _col_attrs["Embeddings_Y"] = df_to_named_matrix(df=_embeddings_Y)
            _col_attrs["Clusterings"] = df_to_named_matrix(
                df=_clusterings.astype(np.int16)
            )

            _global_attrs["MetaData"] = json.dumps(_global_attrs["MetaData"])

            if compress_metadata:
                _global_attrs["MetaData"] = compress_encode(
                    value=_global_attrs["MetaData"]
                )

            for _ga_key, _ga_value in self._global_attrs:
                _global_attrs[_ga_key] = _ga_value

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


class Modes(MutableMapping[str, Mode], metaclass=WithInitHook):
    def __init__(self):
        """"""
        self._keys: List[str] = []
        self._mode_types = [item.value for item in ModeType]
        # Implemented modes (used here mainly for typing purposes)
        self.rna: Mode = None

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
        # FIXME: Fix for editable mode (get called with name=__class__)
        if key.startswith("__"):
            return

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

        # FIXME: Fix for editable mode (get called with name=__class__)
        if name.startswith("__"):
            return
        print(f"INFO: Adding new {name} mode")
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


class GlobalAttributes(MutableMapping[str, str], metaclass=WithInitHook):
    def __init__(self, mode: Mode):
        """"""
        self._keys: List[str] = []
        # Implemented modes (used here mainly for typing purposes)
        self.rna: Mode = mode

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
        return iter(AttributesIterator(self))

    def __len__(self):
        """"""
        return len(self._keys)

    def __delitem__(self, name: str) -> None:
        """"""
        self.__delattr__(name)

    def __getitem__(self, name: str) -> Mode:
        """"""
        return getattr(self, name)

    def __setitem__(self, name: str, value: str) -> None:
        """"""
        # FIXME: Fix for editable mode (get called with name=__class__)
        if name.startswith("__"):
            return

        if not isinstance(name, str):
            raise Exception("Not a valid key for GlobalAttribute.")

        if not isinstance(value, str):
            raise Exception("Not a valid value for GlobalAttribute.")

        self._add_key(key=name)
        super().__setattr__(name, value)

    def get_attribute(self, key: str):
        """"""
        return super().__getattribute__(key)

    def __repr__(self) -> str:
        _keys = f"{', '.join(self._keys)}" if len(self._keys) > 0 else "none"
        return f"Global attributes: {_keys}"

    def _add_key(self, key: str):
        if key not in self._keys:
            self._keys.append(key)


class GlobalAttributesIterator:

    """Class to implement an iterator of Attributes """

    def __init__(self, attrs: GlobalAttributes):
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
    def values(self):
        if isinstance(self._data, pd.DataFrame):
            _col_name = self._data.columns[0]
            return self._data[_col_name].values
        if isinstance(self._data, pd.Series):
            return self._data
        raise Exception(f"Cannot get values from Attribute with key {self._key}")

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
        # FIXME: Fix for editable mode (get called with name=__class__)
        if key.startswith("__"):
            return

        if key.startswith("_"):
            raise Exception(
                f"Cannot add attribute with key {key}. Not a valid key. Expects key not to start with an underscore ('_')."
            )
        if not isinstance(key, str):
            raise Exception(
                f"Cannot add attribute with key of type ({type(key).__name__}) to {type(self).__name__}. Not a valid key. Expects key of type str."
            )
        # Print a warning in key contains characters not allowed. If any present, this will prevent the user to use dot notation. Brackets access will work.
        pattern = "^[a-zA-Z0-9_]+$"
        if not re.match(pattern, key):
            warnings.warn(
                f"The key '{key}' won't be accessible using the dot notation (containing special characters other than '_')",
            )

    def _validate_value(self, value):
        if not isinstance(value, pd.DataFrame) and not isinstance(value, pd.Series):
            raise Exception(
                f"Cannot add attribute of type {type(value).__name__} to {type(self).__name__}. Expects a pandas.DataFrame or a pandas.Series."
            )

        if (
            isinstance(value, pd.DataFrame)
            and not self._is_multi
            and value.shape[1] > 1
        ):
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

    def _validate_value(self, value: Union[pd.DataFrame, pd.Series], **kwargs):
        if __DEBUG__:
            print(f"DEBUG: _validate_value ({type(self).__name__})")
        super()._validate_value(value=value)
        _force_conversion_to_categorical = (
            kwargs["force_conversion_to_categorical"]
            if "force_conversion_to_categorical" in kwargs
            else False
        )
        # Do some checks and processing for attribute of type ANNOTATION
        if not _force_conversion_to_categorical:
            if (
                isinstance(value, pd.DataFrame)
                and not all(value.apply(pd.api.types.is_categorical_dtype))
                and not all(value.apply(pd.api.types.is_bool_dtype))
            ) or (
                isinstance(value, pd.Series)
                and not pd.api.types.is_categorical_dtype(arr_or_dtype=value)
                and not pd.api.types.is_bool_dtype(arr_or_dtype=value)
            ):
                _dtype = (
                    value.infer_objects().dtypes[0]
                    if isinstance(value, pd.DataFrame)
                    else value.infer_objects().dtype
                )
                raise BadDTypeException(
                    f"Expects value to be categorical or bool but its dtype is {_dtype}. You can force the conversion to categorical by using <loomx-instance>.modes.<mode>.annotations.add(*, force=True)."
                )

    def _normalize_value(
        self,
        name: str,
        value: Union[pd.DataFrame, pd.Series],
        force_conversion_to_categorical: bool = False,
    ) -> pd.DataFrame:
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

    def _validate_value(self, value: Union[pd.DataFrame, pd.Series], **kwargs):
        if __DEBUG__:
            print(f"DEBUG: _validate_value ({type(self).__name__})")
        super()._validate_value(value=value)
        _force_conversion_to_numeric = (
            kwargs["force_conversion_to_numeric"]
            if "force_conversion_to_numeric" in kwargs
            else False
        )
        # Do some checks and processing for attribute of type METRIC
        if not _force_conversion_to_numeric:
            if (
                isinstance(value, pd.DataFrame)
                and not all(value.apply(pd.api.types.is_numeric_dtype))
            ) or (
                isinstance(value, pd.Series)
                and not pd.api.types.is_numeric_dtype(arr_or_dtype=value)
            ):
                _dtype = (
                    value.infer_objects().dtypes[0]
                    if isinstance(value, pd.DataFrame)
                    else value.infer_objects().dtype
                )
                raise BadDTypeException(
                    f"Expects value to be numeric but its dtype is {_dtype}"
                )

    def _normalize_value(
        self,
        name: str,
        value: Union[pd.DataFrame, pd.Series],
        force_conversion_to_numeric: bool = False,
    ) -> Union[pd.DataFrame, pd.Series]:
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

    def _validate_value(self, value: Union[pd.DataFrame, pd.Series], **kwargs):
        if __DEBUG__:
            print(f"DEBUG: _validate_value ({type(self).__name__})")
        # Generic validation
        super()._validate_value(value=value)
        # Check if all observations from the given value are present in the DataMatrix of this mode
        _features = self._mode.X._feature_names
        if not all(np.in1d(value.index.astype(str), _features.astype(str))):
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

    def add_annotation(self, key: str, value: Union[pd.DataFrame, pd.Series]):
        self._mode._fa_annotations.add(key=key, value=value)

    @property
    def metrics(self):
        return self._mode._fa_metrics

    def add_metric(self, key: str, value: Union[pd.DataFrame, pd.Series]):
        self._mode._fa_metrics.add(key=key, value=value)


class FeatureAnnotationAttributes(FeatureAttributes, AnnotationAttributes):
    def __init__(self, mode: Mode):
        """"""
        super().__init__(mode=mode, is_proxy=True)

    def __setitem__(self, name, value):
        """"""
        self.add(key=name, value=value)

    def add(self, key: str, value: Union[pd.DataFrame, pd.Series]):
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

    def __setitem__(self, name: str, value: Union[pd.DataFrame, pd.Series]):
        """"""
        self.add(key=name, value=value)

    def add(self, key: str, value: Union[pd.DataFrame, pd.Series]):
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
        if not all(np.in1d(value.index, _observations)):
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
        value: Union[pd.DataFrame, pd.Series],
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
        value: Union[pd.DataFrame, pd.Series],
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
        value: Union[pd.DataFrame, pd.Series],
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
        value: Union[pd.DataFrame, pd.Series],
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

    def __setitem__(self, name: str, value: Union[pd.DataFrame, pd.Series]):
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


class EmbeddingAttribute(Attribute):
    def __init__(
        self,
        metadata: LoomXMetadataEmbedding,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._metadata = metadata

    @property
    def id(self) -> int:
        return self._metadata._id

    @id.setter
    def id(self, value: int) -> None:
        self._metadata._id = value

    @property
    def default(self) -> bool:
        return self._metadata.default

    @default.setter
    def default(self, value: bool) -> None:
        self._metadata.default = value

    @property
    def projection_method(self) -> str:
        return self._metadata.projection_metod

    @projection_method.setter
    def projection_method(self, value: str):
        self._metadata.projection_metod = value

    def __repr__(self):
        try:
            _projection_method = ProjectionMethod(self._metadata.projection_method).name
        except:
            _projection_method = "n.a."
        return f"""
{super().__repr__()}
default: {self._metadata.default}
projection method: {_projection_method}
        """


class ObservationEmbeddingAttributes(ObservationAttributes, EmbeddingAttributes):
    def __init__(self, mode):
        """"""
        super().__init__(mode=mode, is_proxy=True, is_multi=True)

    def __setitem__(self, name: str, value: Union[pd.DataFrame, pd.Series]):
        """"""
        self.add(key=name, value=value)

    def add(
        self,
        key: str,
        value: Union[pd.DataFrame, pd.Series],
        name: str = None,
        description: str = None,
        metadata: LoomXMetadataEmbedding = None,
    ):
        super()._validate_key(key=key)
        super()._validate_value(value=value)

        if metadata is None:
            metadata = self.make_metadata(key=key, value=value, name=name)

        _attr = EmbeddingAttribute(
            key=key,
            mode_type=self._mode_type,
            attr_type=self._attr_type,
            axis=self._axis,
            data=value,
            name=name,
            description=description,
            metadata=metadata,
        )
        self._mode._observation_attrs._add_item(key=key, value=_attr)
        super()._add_item_by_value(value=_attr)

    def make_metadata(
        self, key: str, value: Union[pd.DataFrame, pd.Series], name: str = None
    ):
        _embedding_id = len(
            list(
                filter(
                    lambda a: a[1].attr_type == AttributeType.EMBEDDING
                    and int(a[1].id) > 0,
                    self._mode.o,
                )
            )
        )
        return LoomXMetadataEmbedding.from_dict(
            {"id": _embedding_id, "name": key if name is None else name}
        )


class ClusteringAttribute(Attribute):
    def __init__(
        self,
        metadata: LoomXMetadataClustering = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._metadata = metadata

    def __iter__(self):
        """"""
        return iter(ClusteringAttributeIterator(self))

    def __len__(self):
        """"""
        return len(self._keys)

    def __getattribute__(self, name):
        return super().__getattribute__(name)

    def __repr__(self):
        return f"""
{super().__repr__()}
number of clusters: {len(self._metadata.clusters)}
        """

    @property
    def id(self):
        return self._metadata.id

    @id.setter
    def id(self, value: int):
        self._metadata.id = value

    @property
    def group(self):
        return self._metadata.group

    @group.setter
    def group(self, value: str):
        self._metadata.group = value

    @property
    def clusters(self):
        return self._metadata.clusters

    @clusters.setter
    def clusters(self, value: List[LoomXMetadataCluster]):
        self._metadata.clusters = value

    @property
    def clusterMarkerMetrics(self):
        return self._metadata.clusterMarkerMetrics

    @property
    def markers(self):
        return self._metadata.markers

    @property
    def metadata(self) -> LoomXMetadataClustering:
        return self._metadata


class ClusteringAttributeIterator:

    """Class to implement an iterator of ClusteringAttribute """

    def __init__(self, attr: ClusteringAttribute):
        self._attr = attr

    def __iter__(self):
        self._n = 0
        return self

    def __next__(self):
        if self._n < len(self._attr.clusters):
            current_key = self._n
            self._n += 1
            return current_key, self._attr.clusters[current_key]
        else:
            raise StopIteration


class ObservationClusteringAttributes(ObservationAttributes, ClusteringAttributes):
    def __init__(self, mode):
        """"""
        super().__init__(mode=mode, is_proxy=True)

    def __setitem__(self, name: str, value: Union[pd.DataFrame, pd.Series]):
        """"""
        self.add(key=name, value=value)

    def add(
        self,
        key: str,
        value: Union[pd.DataFrame, pd.Series],
        name: str = None,
        description: str = None,
        metadata: LoomXMetadataClustering = None,
    ):
        super()._validate_key(key=key)
        super()._validate_value(value=value)

        _attr = ClusteringAttribute(
            key=key,
            mode_type=self._mode_type,
            attr_type=self._attr_type,
            axis=self._axis,
            data=value,
            name=key if name is None else name,
            description=description,
            metadata=self.make_metadata(key=key, value=value, name=name)
            if metadata is None
            else metadata,
        )
        self._mode._observation_attrs._add_item(key=key, value=_attr)
        super()._add_item_by_value(value=_attr)

    def make_metadata(self, key: str, value: Union[pd.DataFrame, pd.Series], name: str):
        _clusters = []
        for cluster_id in sorted(
            np.unique(value.values).astype(int),
            reverse=False,
        ):
            _clusters.append(LoomXMetadataCluster.from_dict({"id": int(cluster_id)}))
        _clustering_id = len(
            list(
                filter(
                    lambda a: a[1].attr_type == AttributeType.CLUSTERING, self._mode.o
                )
            )
        )
        return LoomXMetadataClustering.from_dict(
            {"id": _clustering_id, "name": key, "group": "n.a.", "clusters": _clusters}
        )
