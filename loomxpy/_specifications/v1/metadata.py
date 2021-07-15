from enum import Enum
import pandas as pd
from dataclasses_json import dataclass_json, config, DataClassJsonMixin, cfg
from dataclasses import dataclass, field
from typing import Optional, List

GLOBAL_ATTRIBUTE_KEY = "MetaData"


@dataclass_json
@dataclass
class Annotation(DataClassJsonMixin):
    name: str
    values: List[str]


@dataclass_json
@dataclass
class Metric(DataClassJsonMixin):
    name: str


class ProjectionMethod(Enum):
    PCA = 0
    TSNE = 1
    UMAP = 2


@dataclass_json
@dataclass
class Embedding(DataClassJsonMixin):
    _id: int = field(metadata=config(field_name="id"))
    _name: str = field(metadata=config(field_name="name"))
    # Attributes exluded from the metadata
    _default: Optional[bool] = field(
        default=None,
        metadata=config(
            field_name="default", exclude=cfg.Exclude.ALWAYS
        ),  # has to be exluded since not part of SCope gRPC API
    )
    _projection_method: Optional[ProjectionMethod] = field(
        default=None,
        metadata=config(
            field_name="projection_method", exclude=cfg.Exclude.ALWAYS
        ),  # has to be exluded since not part of SCope gRPC API
    )

    def __post_init__(self):
        if self._projection_method is not None:
            if "pca" in self._name.lower():
                self._projection_method = ProjectionMethod.PCA
            elif "tsne" in self._name.lower():
                self._projection_method = ProjectionMethod.TSNE
            elif "umap" in self._name.lower():
                self._projection_method = ProjectionMethod.UMAP

    @property
    def id(self):
        return int(self._id)

    @property
    def default(self):
        if self._default is None:
            return int(self._id) == -1
        return False

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def projection_method(self) -> str:
        if self._projection_method is None:
            return "n.a."
        return self._projection_method.name


@dataclass_json
@dataclass
class RegulonAllThresholds:
    tenPercentOfMax: Optional[float] = None


@dataclass_json
@dataclass
class RegulonThreshold:
    regulon: str
    defaultThresholdValue: float
    defaultThresholdName: str
    allThresholds: RegulonAllThresholds
    motifData: str
    motifVersion: Optional[str] = field(default="")


@dataclass_json
@dataclass
class CellTypeAnnotationVoter:
    voter_name: str
    voter_id: str
    voter_hash: str


@dataclass_json
@dataclass
class CellTypeAnnotationVoters:
    total: int
    voters: List[CellTypeAnnotationVoter]


@dataclass_json
@dataclass
class CellTypeAnnotationVotes:
    votes_for: CellTypeAnnotationVoters
    votes_against: CellTypeAnnotationVoters


@dataclass_json
@dataclass
class CellTypeAnnotationData:
    curator_name: str
    curator_id: str
    timestamp: int
    obo_id: str
    ols_iri: str
    annotation_label: str
    markers: field(default_factory=lambda: [])
    publication: str
    comment: str


@dataclass_json
@dataclass
class CellTypeAnnotation:
    data: CellTypeAnnotationData
    validate_hash: str
    votes: CellTypeAnnotationVotes


@dataclass_json
@dataclass
class ClusterMarkerMetric(DataClassJsonMixin):
    _accessor: str = field(default=None, metadata=config(field_name="accessor"))
    _name: str = field(default=None, metadata=config(field_name="name"))
    _description: str = field(default=None, metadata=config(field_name="description"))
    # Attributes exluded from the metadata
    _threshold: Optional[float] = field(
        default=None,
        metadata=config(field_name="threshold", exclude=cfg.Exclude.ALWAYS),
    )
    _threshold_method: Optional[str] = field(
        default=None,
        metadata=config(field_name="threshold_method", exclude=cfg.Exclude.ALWAYS),
    )

    @property
    def accessor(self) -> str:
        return self._accessor

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description


@dataclass_json
@dataclass
class Cluster(DataClassJsonMixin):
    _id: int = field(metadata=config(field_name="id"))
    _description: str = field(default="", metadata=config(field_name="description"))
    _cell_type_annotation: List[CellTypeAnnotation] = field(
        default_factory=lambda: [], metadata=config(field_name="cell_type_annotation")
    )
    # Attributes exluded from the metadata
    _markers: Optional[pd.DataFrame] = field(
        default=None,
        metadata=config(field_name="markers", exclude=cfg.Exclude.ALWAYS),
    )

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        raise Exception("The ID of the cluster cannot be changed.")

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

    @property
    def markers(self):
        return self._markers

    @markers.setter
    def markers(self, value: pd.DataFrame):
        self._markers = value


@dataclass_json
@dataclass
class Clustering(DataClassJsonMixin):
    _id: int = field(metadata=config(field_name="id"))
    _group: str = field(metadata=config(field_name="group"))
    _name: str = field(metadata=config(field_name="name"))
    _clusters: List[Cluster] = field(metadata=config(field_name="clusters"))
    _clusterMarkerMetrics: List[ClusterMarkerMetric] = field(
        default_factory=lambda: [], metadata=config(field_name="clusterMarkerMetrics")
    )
    # Attributes exluded from the metadata
    _markers: Optional[pd.DataFrame] = field(
        default=None,
        metadata=config(field_name="markers", exclude=cfg.Exclude.ALWAYS),
    )

    @property
    def id(self) -> int:
        return self._id

    @id.setter
    def id(self, value):
        raise Exception("The ID of the clustering cannot be changed.")

    @property
    def group(self) -> str:
        return self._group

    @group.setter
    def group(self, value: str) -> None:
        self._group = value

    @property
    def name(self):
        if self._name is not None:
            return self._name
        return f"Unannotated Clustering {self._id}"

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def clusters(self) -> List[Cluster]:
        return self._clusters

    @property
    def clusterMarkerMetrics(self) -> List[ClusterMarkerMetric]:
        return self._clusterMarkerMetrics

    @property
    def markers(self):
        if self._markers is None:
            self._markers = pd.concat([cluster.markers for cluster in self._clusters])
        return self._markers


@dataclass_json
@dataclass
class Metadata(DataClassJsonMixin):
    metrics: List[Metric]
    annotations: List[Annotation]
    embeddings: List[Embedding]
    clusterings: Optional[List[Clustering]]
    regulonThresholds: List[RegulonThreshold] = field(default_factory=lambda: [])
