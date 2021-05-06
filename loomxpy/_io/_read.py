import json
import zlib
import base64
import numpy as np
import loompy as lp
from typing import NamedTuple
from scipy import sparse

from loomxpy._loomx import LoomX
from loomxpy._mode import ModeType
from loomxpy._specifications import LoomXMetadata


@staticmethod
def decompress_metadata(metadata: str):
    try:
        metadata = metadata.decode("ascii")
        return json.loads(zlib.decompress(base64.b64decode(s=meta)))
    except AttributeError:
        return json.loads(
            zlib.decompress(base64.b64decode(meta.encode("ascii"))).decode("ascii")
        )


@staticmethod
def get_metadata_from_loom_connection(loom_connection: lp.LoomConnection):
    _metadata = loom_connection.attrs.MetaData
    if type(md) is np.ndarray:
        _metadata = loom_connection.attrs[GLOBAL_ATTRIBUTE_KEY][0]
    try:
        return json.loads(s=_metadata)
    except json.decoder.JSONDecodeError:
        return decompress_metadata(metadata=_metadata)


def _read_scope_rna_loom(loom_connection: lp.LoomConnection):
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
    return lx


def _read_scope_loom(loom_connection: lp.LoomConnection, mode_type: ModeType):
    # Read the Metadata from SCope
    _metadata: LoomXMetadata = LoomXMetadata.from_dict(
        get_metadata_from_loom_connection(loom_connection=loom_connection)
    )
    if mode_type == ModeType.RNA:
        return _read_scope_rna_loom(loom_connection=loom_connection)
    raise Exception(f"The given mode type '{mode_type}' has not been implemented yet.")


def read_loom(file_path: str, mode_type: str = "rna") -> LoomX:

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
                loom_connection=loom_connection, mode_type=_mode_type
            )
        raise Exception(f"Unable to read the loom at {file_path}")
