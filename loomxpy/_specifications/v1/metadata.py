from dataclasses_json import dataclass_json
from dataclasses import dataclass, field
from typing import Optional, List

GLOBAL_ATTRIBUTE_KEY = "MetaData"


@dataclass_json
@dataclass
class Annotation:
    name: str
    values: List[str]


@dataclass_json
@dataclass
class Metric:
    name: str


@dataclass_json
@dataclass
class Embedding:
    id: int
    name: str


@dataclass_json
@dataclass
class RegulonAllThresholds:
    tenPercentOfMax: float = None


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
class CellTypeAnnotationVotes:
    total: int
    voters: List[CellTypeAnnotationVoter]


@dataclass_json
@dataclass
class CellTypeAnnotationVotes:
    votes_for: CellTypeAnnotationVotes
    votes_against: CellTypeAnnotationVotes


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
class ClusterMarkerMetric:
    accessor: str
    name: str
    description: str


@dataclass_json
@dataclass
class Cluster(DataClassJsonMixin):
    _id: int = field(metadata=config(field_name="id"))
    _description: str = field(default="", metadata=config(field_name="description"))
    _cell_type_annotation: List[CellTypeAnnotation] = field(
        default_factory=lambda: [], metadata=config(field_name="cell_type_annotation")
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


@dataclass_json
@dataclass
class Clustering(DataClassJsonMixin):
    id: int
    group: str
    name: str
    clusters: List[Cluster]
    clusterMarkerMetrics: List[ClusterMarkerMetric] = field(default_factory=lambda: [])


@dataclass_json
@dataclass
class Metadata:
    metrics: List[Metric]
    annotations: List[Annotation]
    embeddings: List[Embedding]
    clusterings: Optional[List[Clustering]]
    regulonThresholds: List[RegulonThreshold] = field(default_factory=lambda: [])
