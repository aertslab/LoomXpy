import zlib
import base64
import json
import pandas as pd
import numpy as np


def df_to_named_matrix(df: pd.DataFrame):
    # Create meta-data structure.
    # Create a numpy structured array
    return np.array(
        [tuple(row) for row in df.values],
        dtype=np.dtype(list(zip(df.columns, df.dtypes))),
    )


def compress_encode(value):
    """
    Compress using ZLIB algorithm and encode the given value in base64.
    Taken from: https://github.com/aertslab/SCopeLoomPy/blob/5438da52c4bcf48f483a1cf378b1eaa788adefcb/src/scopeloompy/utils/__init__.py#L7
    """
    return base64.b64encode(zlib.compress(value.encode("ascii"))).decode("ascii")


def decompress_decode(value):
    try:
        value = value.decode("ascii")
        return json.loads(zlib.decompress(base64.b64decode(value)))
    except AttributeError:
        return json.loads(
            zlib.decompress(base64.b64decode(value.encode("ascii"))).decode("ascii")
        )
