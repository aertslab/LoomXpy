from enum import Enum
from scipy import sparse
from typing import Mapping
from ._axis import FeatureAttributes, ObservationAttribute, GlobalAttribute
from ._matrix import DataMatrix

class ModeType(Enum):
    RNA = "rna"
    ATAC = "atac"


class Mode:

    def __init__(self, mode_type: ModeType, data_matrix: DataMatrix):
        '''
        constructor for Mode
        '''
        self._mode_type = mode_type
        self._data_matrix = data_matrix
        self._feature_attrs: FeatureAttributes = FeatureAttributes(mode_type=mode_type)
        self._observation_attrs: Mapping[str, ObservationAttribute] = None
        self._global_attrs: Mapping[str, GlobalAttribute] = None

    @property
    def mode_type(self):
        return self._mode_type
    
    @property
    def data_matrix(self):
        return self._data_matrix
    
    @property
    def X(self):
        return self._data_matrix
    
    @property
    def feature_attrs(self):
        return self._feature_attrs
    
    @property
    def f(self):
        return self._feature_attrs

    @property
    def observation_attrs(self):
        return self._observation_attrs
    
    @property
    def o(self):
        return self._observation_attrs

    @property
    def global_attrs(self):
        return self._global_attrs
    
    @property
    def g(self):
        return self._global_attrs

    def __repr__(self) -> str:
        return "__repr__ of Mode"

    def _repr_html_(self):
        return "<b>__repr__ of Mode</b>"


class RNAMode(Mode):

    def __init__(self, data_matrix: DataMatrix):
        super(RNAMode, self).__init__(mode_type=ModeType.RNA, data_matrix=data_matrix)


class ATACMode(Mode):

    def __init__(self, data_matrix: DataMatrix):
        super(ATACMode, self).__init__(mode_type=ModeType.ATAC, data_matrix=data_matrix)   