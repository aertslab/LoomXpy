import json
import zlib
import base64
import pandas as pd
import numpy as np
import loompy as lp
from typing import NamedTuple, Dict
from scipy import sparse

from loomxpy._errors import BadDTypeException
from loomxpy._loomx import LoomX
from loomxpy._mode import Mode, ModeType
from loomxpy._specifications import LoomXMetadata, GLOBAL_ATTRIBUTE_KEY


def decompress_metadata(metadata: str):
    try:
        metadata = metadata.decode("ascii")
        return json.loads(zlib.decompress(base64.b64decode(s=meta)))
    except AttributeError:
        return json.loads(
            zlib.decompress(base64.b64decode(meta.encode("ascii"))).decode("ascii")
        )


def get_metadata_from_loom_connection(loom_connection: lp.LoomConnection):
    _metadata = loom_connection.attrs[GLOBAL_ATTRIBUTE_KEY]
    if type(_metadata) is np.ndarray:
        _metadata = _metadata[0]
    try:
        return json.loads(s=_metadata)
    except json.decoder.JSONDecodeError:
        return decompress_metadata(metadata=_metadata)


def _read_scope_rna_loom(loom_connection: lp.LoomConnection, force_conversion: Dict):
    # Read the Metadata from SCope
    _metadata: LoomXMetadata = LoomXMetadata.from_dict(
        get_metadata_from_loom_connection(loom_connection=loom_connection)
    )
    # Loompy stores the features as rows and the observations as columns
    # LoomX follows Scanpy convention (i.e.: machine learning/statistics standards) i.e.: observations as rows and the features as columns. This requires to transpose the matrix
    _matrix = loom_connection[:, :]
    if "Gene" not in loom_connection.ra:
        raise Exception("The loom file is missing a 'Gene' row attribute")
    if "CellID" not in loom_connection.ca:
        raise Exception("The loom file is missing a 'CellID' column attribute")
    # Create the LoomX in-memory object
    ## Add the expression matrix
    lx = LoomX()
    lx.modes.rna = (
        sparse.csr_matrix(_matrix).transpose(),
        loom_connection.ra["Gene"],
        loom_connection.ca["CellID"],
    )
    ## Add observation (o) attributes
    ### Add annotations
    try:
        for annotation in _metadata.annotations:
            lx.modes.rna.o.annotations.add(
                key=annotation.name,
                name=annotation.name,
                value=pd.Series(
                    data=loom_connection.ca[annotation.name],
                    index=lx.modes.rna.X._observation_names,
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
        for metric in _metadata.metrics:
            lx.modes.rna.o.metrics.add(
                key=metric.name,
                name=metric.name,
                value=pd.Series(
                    data=loom_connection.ca[metric.name],
                    index=lx.modes.rna.X._observation_names,
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
    for embedding in _metadata.embeddings:
        _embedding_df = pd.DataFrame(
            {
                "_X": loom_connection.ca["Embeddings_X"][str(embedding.id)],
                "_Y": loom_connection.ca["Embeddings_Y"][str(embedding.id)],
            },
            index=lx.modes.rna.X._observation_names,
        )
        lx.modes.rna.o.embeddings.add(
            key=embedding.name,
            value=_embedding_df,
            name=embedding.name,
            id=int(embedding.id),
            default=int(embedding.id) == -1,
        )
    return lx


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
        if GLOBAL_ATTRIBUTE_KEY in loom_connection.attrs:
            return _read_scope_loom(
                loom_connection=loom_connection,
                mode_type=_mode_type,
                force_conversion=force_conversion,
            )
        raise Exception(f"Unable to read the loom at {file_path}")
