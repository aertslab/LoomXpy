import json
import zlib
import base64
import functools
import pandas as pd
import numpy as np
import loompy as lp
from typing import Dict, Union, Tuple
from scipy import sparse

from loomxpy._errors import BadDTypeException
from loomxpy._loomx import LoomX
from loomxpy._mode import Mode, ModeType
from loomxpy._specifications.v1.metadata import (
    Metadata as LoomXMetadataV1,
    Clustering as LoomXMetadataV1Clustering,
    Cluster as LoomXMetadataV1Cluster,
    GLOBAL_ATTRIBUTE_KEY as GLOBAL_ATTRIBUTE_KEY_V1,
)

# A list of all versions of the MetaData global attribute
GLOBAL_ATTRIBUTE_KEY_VX = [GLOBAL_ATTRIBUTE_KEY_V1]


def decompress_metadata(metadata: str):
    try:
        metadata = metadata.decode("ascii")
        return json.loads(zlib.decompress(base64.b64decode(s=metadata)))
    except AttributeError:
        return json.loads(
            zlib.decompress(base64.b64decode(metadata.encode("ascii"))).decode("ascii")
        )


def get_metadata_from_loom_connection(
    loom_connection: lp.LoomConnection, metadata_global_attribute_key: str
):
    _metadata = loom_connection.attrs[metadata_global_attribute_key]
    if type(_metadata) is np.ndarray:
        _metadata = _metadata[0]
    try:
        return json.loads(s=_metadata)
    except json.decoder.JSONDecodeError:
        return decompress_metadata(metadata=_metadata)


def _read_scope_metadata(
    loom_connection: lp.LoomConnection,
) -> Union[Tuple["v1", Union[LoomXMetadataV1]]]:
    # Read the Metadata from SCope
    try:
        _metadata: LoomXMetadataV1 = LoomXMetadataV1.from_dict(
            get_metadata_from_loom_connection(
                loom_connection=loom_connection,
                metadata_global_attribute_key=GLOBAL_ATTRIBUTE_KEY_V1,
            )
        )
        return "v1", _metadata
    except:
        pass
    raise Exception("Cannot read the LoomX metadata.")


def _read_scope_v1_rna_loom(
    loom_connection: lp.LoomConnection,
    metadata: LoomXMetadataV1,
    force_conversion: Dict,
):
    # Loompy stores the features as rows and the observations as columns
    # LoomX follows Scanpy convention (i.e.: machine learning/statistics standards) i.e.: observations as rows and the features as columns. This requires to transpose the matrix
    _matrix = loom_connection[:, :]
    if "Gene" not in loom_connection.ra:
        raise Exception("The loom file is missing a 'Gene' row attribute")
    if "CellID" not in loom_connection.ca:
        raise Exception("The loom file is missing a 'CellID' column attribute")
    # Create the LoomX in-memory object
    ## Add the expression matrix in the RNA mode
    _lx = LoomX()
    print("Adding data matrix...")
    _lx.modes.rna = (
        sparse.csr_matrix(_matrix).transpose(),
        loom_connection.ra["Gene"],
        loom_connection.ca["CellID"],
    )
    ## Add observation (o) attributes
    ### Add annotations
    try:
        print("Adding annotations...")
        for annotation in metadata.annotations:
            _lx.modes.rna.o.annotations.add(
                key=annotation.name,
                name=annotation.name,
                value=pd.Series(
                    data=loom_connection.ca[annotation.name],
                    index=_lx.modes.rna.X._observation_names,
                    name=annotation.name,
                ),
                force=force_conversion["annotations"]
                if "annotations" in force_conversion
                else False,
            )
    except BadDTypeException:
        raise Exception(
            "You can force the conversion of the annotations to categorical by setting the parameter `force_conversion={'annotations': True}`"
        )
    ### Add metrics
    try:
        print("Adding metrics...")
        for metric in metadata.metrics:
            _lx.modes.rna.o.metrics.add(
                key=metric.name,
                name=metric.name,
                value=pd.Series(
                    data=loom_connection.ca[metric.name],
                    index=_lx.modes.rna.X._observation_names,
                    name=metric.name,
                ),
                force=force_conversion["metrics"]
                if "metrics" in force_conversion
                else False,
            )
    except BadDTypeException:
        raise Exception(
            "You can force the conversion of the metrics to numerical by setting the parameter `force_conversion={'metrics': True}`"
        )
    ### Add embeddings
    if "Embeddings_X" in loom_connection.ca and "Embeddings_Y" in loom_connection.ca:
        print("Adding embeddings...")
        for embedding in metadata.embeddings:
            _embedding_df = pd.DataFrame(
                {
                    "_X": loom_connection.ca["Embeddings_X"][str(embedding.id)],
                    "_Y": loom_connection.ca["Embeddings_Y"][str(embedding.id)],
                },
                index=_lx.modes.rna.X._observation_names,
            )
            _lx.modes.rna.o.embeddings.add(
                key=embedding.name,
                value=_embedding_df,
                name=embedding.name,
                metadata=embedding,
            )
    ## Add clusterings
    if "Clusterings" in loom_connection.ca:
        print("Adding clusterings...")
        for clustering in metadata.clusterings:
            _lx.modes.rna.o.clusterings.add(
                key=clustering.name,
                name=clustering.name,
                value=pd.Series(
                    data=loom_connection.ca["Clusterings"][str(clustering.id)],
                    index=_lx.modes.rna.X._observation_names,
                    name=str(clustering.id),
                ),
                metadata=clustering,
            )

    # # Add cluster markers
    _has_markers = False
    clustering_attr: LoomXMetadataV1Clustering
    for _, clustering_attr in _lx.modes.rna.o.clusterings:
        # Check if any markers exist for the current clustering
        cluster_marker_ra_prefix = "ClusterMarkers_{clustering_attr.id}"
        if cluster_marker_ra_prefix not in loom_connection.ra:
            continue

        _has_markers = True

        # Make markers table in long format (cluster, <metrics>)
        markers_df = (
            functools.reduce(
                lambda left, right: pd.merge(
                    left, right, on=["index", "variable"], how="left"
                ),
                [
                    pd.DataFrame(
                        loom_connection.ra[cluster_marker_ra_prefix],
                        index=loom_connection.ra.Gene,
                    )
                    .reset_index()
                    .melt("index")
                    .rename(columns={"value": "is_marker"})
                ]
                + [
                    pd.DataFrame(
                        loom_connection.ra[
                            f"{cluster_marker_ra_prefix}_{metric.accessor}"
                        ],
                        index=loom_connection.ra.Gene,
                    )
                    .reset_index()
                    .melt("index")
                    .rename(columns={"value": metric.accessor})
                    for metric in clustering_attr.clusterMarkerMetrics
                ],
            )
            .set_index("index")
            .rename(columns={"variable": "cluster"})
            .query("is_marker == 1")
            .drop(columns=["is_marker"])
        )
        cluster: LoomXMetadataV1Cluster
        for cluster in clustering_attr.clusters:
            cluster.markers = markers_df.query(f"cluster == '{cluster.id}'")

    if not _has_markers:
        print("INFO: No markers detected in the loom.")

    return _lx


def _read_scope_rna_loom(loom_connection: lp.LoomConnection, force_conversion: Dict):
    # Read the Metadata from SCope
    _version, _metadata = _read_scope_metadata(loom_connection=loom_connection)
    if _version == "v1":
        _lx = _read_scope_v1_rna_loom(
            loom_connection=loom_connection,
            metadata=_metadata,
            force_conversion=force_conversion,
        )
    else:
        raise Exception("Unable to detect the LoomX version.")

    print("Adding global attributes...")
    for global_attr_key in loom_connection.attrs:
        if global_attr_key in GLOBAL_ATTRIBUTE_KEY_VX:
            continue
        _lx.modes.rna.g[global_attr_key] = loom_connection.attrs[global_attr_key]
    return _lx


def _read_scope_loom(
    loom_connection: lp.LoomConnection, mode_type: ModeType, force_conversion: Dict
):
    if mode_type == ModeType.RNA:
        return _read_scope_rna_loom(
            loom_connection=loom_connection, force_conversion=force_conversion
        )
    # Should be uncommented on production
    # Commented here ease the flow with editable mode
    # raise Exception(f"The given mode type '{mode_type}' has not been implemented yet.")


def read_loom(
    file_path: str,
    mode_type: str = "rna",
    force_conversion={"annotations": False, "metrics": False},
) -> LoomX:

    try:
        _mode_type = ModeType(mode_type)
    except:
        mode_types = list(filter(lambda x: x != "_", [w.value for w in (ModeType)]))
        raise Exception(
            f"The given mode type '{mode_type}' does not exist. Choose one of: {', '.join(mode_types)}."
        )

    with lp.connect(filename=file_path, mode="r", validate=False) as loom_connection:
        if any(
            list(map(lambda x: x in loom_connection.attrs, GLOBAL_ATTRIBUTE_KEY_VX))
        ):
            return _read_scope_loom(
                loom_connection=loom_connection,
                mode_type=_mode_type,
                force_conversion=force_conversion,
            )
        raise Exception(f"Unable to read the loom at {file_path}")
