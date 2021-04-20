import numpy as np
from scipy import sparse


class DataMatrix:
    def __init__(self, data_matrix, feature_names, observation_names):
        self._data_matrix = self._normalize_data_matrix(
            data_matrix=data_matrix,
            feature_names=feature_names,
            observation_names=observation_names,
        )
        self._feature_names = feature_names
        self._observation_names = observation_names

    @staticmethod
    def _validate_data_matrix(data_matrix, feature_names, observation_names):
        if feature_names is None:
            raise Exception(
                "The given `data_matrix` argument is a NumPy 2D matrix. Hence, the `feature_names` argument should be provided."
            )
        if observation_names is None:
            raise Exception(
                "The given `data_matrix` argument is a NumPy 2D matrix. Hence, the `observation_names` argument should be provided."
            )
        if (
            len(observation_names) != data_matrix.shape[0]
            or len(feature_names) != data_matrix.shape[1]
        ):
            raise Exception(
                f"""
Expecting the given `data_matrix` argument to have features as rows and observations as columns. Trying to assign {len(observation_names)} observations
and {len(feature_names)} features to data matrix with {data_matrix.shape[0]} rows and {data_matrix.shape[1]} columns.
            """
            )

    @staticmethod
    def _normalize_data_matrix_from_numpy_matrix(
        data_matrix, feature_names, observation_names
    ) -> sparse.csr_matrix:
        DataMatrix._validate_data_matrix(
            data_matrix=data_matrix,
            feature_names=feature_names,
            observation_names=observation_names,
        )
        return sparse.csr_matrix(data_matrix)

    @staticmethod
    def _normalize_data_matrix_from_sparse_matrix(
        data_matrix, feature_names, observation_names
    ) -> sparse.csr_matrix:
        DataMatrix._validate_data_matrix(
            data_matrix=data_matrix,
            feature_names=feature_names,
            observation_names=observation_names,
        )
        return data_matrix

    def _normalize_data_matrix(
        self, data_matrix, feature_names=None, observation_names=None
    ) -> sparse.csr_matrix:
        if isinstance(data_matrix, np.matrix):
            return DataMatrix._normalize_data_matrix_from_numpy_matrix(
                data_matrix=data_matrix,
                feature_names=feature_names,
                observation_names=observation_names,
            )
        if sparse.issparse(data_matrix):
            return DataMatrix._normalize_data_matrix_from_sparse_matrix(
                data_matrix=data_matrix,
                feature_names=feature_names,
                observation_names=observation_names,
            )
        raise Exception("Not a valid `data_matrix` argument.")

    def __repr__(self):
        return f"DataMatrix with {len(self._feature_names)} features and {len(self._observation_names)} observations."
