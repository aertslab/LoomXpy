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
    motifVersion: Optional[str] = None


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
    markers: ["ey"]
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
class Cluster:
    id: int
    description: str
    clusterMarkerMetrics: List[ClusterMarkerMetric] = field(default_factory=lambda: [])
    cell_type_annotation: List[CellTypeAnnotation] = field(default_factory=lambda: [])


@dataclass_json
@dataclass
class Clustering:
    id: int
    group: str
    name: str
    clusters: List[Cluster]


@dataclass_json
@dataclass
class Metadata:
    metrics: Optional[List[Metric]] = field(default_factory=lambda: [])
    annotations: Optional[List[Annotation]] = field(default_factory=lambda: [])
    embeddings: Optional[List[Embedding]] = field(default_factory=lambda: [])
    regulonThresholds: Optional[List[RegulonThreshold]] = field(
        default_factory=lambda: []
    )
    clusterings: Optional[List[Clustering]] = field(default_factory=lambda: [])
