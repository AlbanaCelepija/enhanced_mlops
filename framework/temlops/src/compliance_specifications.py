"""Compliance data model -- §2 (Common Foundations), §3.2 (Requirement),
§3.6 (ComplianceObjective) and §3.8 (Documentation / ComplianceDocumentation)
of the AI Compliance Framework reference model.

Lives alongside the other temlops framework core modules (artifact_types.py,
declarative_specs.py, workflow.py) as the compliance-specific counterpart to
artifact_types.py's pipeline artifact types. Pure data model: no GUI/
Streamlit import here. Visualisation of instances built from these classes
belongs in whichever page renders them (e.g. the "Compliance objectives"
section of gui/pages/3_Compliance_Assessment.py), not in this module.

Implementation notes (deliberate, documented deviations from the literal spec):
  * `Set<T>` is implemented as `List[T]` wherever T is one of this module's
    own dataclasses (ComplianceFinding, ComplianceObjective): plain
    dataclasses aren't hashable, and a list gives deterministic, renderable
    order for the GUI. Where T is a primitive (str) or a frozen/hashable
    value (Ref, an Enum member), a real `Set[T]` is used, matching the spec.
  * Several fields reference entity types defined in OTHER sections of the
    spec that are out of scope here (Metric §3.5, Report §3.7, ComplianceContext
    §3.2, and the generic "Entity" supertype). Rather than embedding those
    (which would pull in their entire dependency graphs), those fields keep
    the spec's `Ref<T>` typing -- `Ref["Metric"]`, `Ref["Report"]`, etc. --
    as a forward-reference string naming the type the ref points to, even
    though that type isn't implemented in this module.
  * `ComplianceObjective.requirements` keeps its `Ref["Requirement"]` typing
    even though `Requirement` IS implemented below: the formal spec types it
    `Set<Requirement>` (embedded), but every worked example in the use-case
    reference material writes `Ref<Requirement>("...")` there instead, and
    embedding full Requirement objects (with their own requirement_text/
    summary prose) inline in every objective would duplicate that prose
    across every objective that cites the same requirement.
  * `Requirement.id` and `ComplianceObjective.id` type as a plain `str` slug
    rather than the formal `Identifier` (UUID + namespace) struct used by
    Documentation/ComplianceDocumentation: every worked example of both in
    the reference material writes its id as a plain quoted string
    (`id: "req_hr_high_risk_classification"`, `id:
    "objective_cv_parse_failure_rate"`), never as an `Identifier(...)`
    construction, and a human-authored aipc_<framework>.yaml -- the
    realistic source for both here -- is far more naturally keyed by slug
    than by UUID. `Requirement` also has no `version`/`audit` fields at all,
    matching the formal spec exactly (§3.2 never declares them for
    Requirement); `ComplianceObjective` keeps its formal `version`/`audit`
    fields, but a loader reading it from YAML (which doesn't carry real
    provenance) is expected to synthesise placeholder values for them,
    exactly as the reference material's own worked examples do with the
    content-free `AuditMeta()` / `VersionInfo("0.1.0")` calls.
  * Open enums written in the spec as `A | B | C | CUSTOM of String` (e.g.
    DocumentSection.format) are modelled as a plain `str` field plus a
    `(str, Enum)` class providing the well-known constants -- any other
    string is a valid "custom" value, exactly like the spec's `CUSTOM of
    String` arm, with no separate wrapper type needed.
  * `StatisticalCondition.test` types as `StatisticalTest` in the spec, but
    that type is referenced nowhere else and never itself defined -- it is
    modelled here as a plain `str`.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, Set, TypeVar

# ============================================================================
# §2 Common Foundations (minimal subset needed by §3.6 / §3.8)
# ============================================================================


@dataclass(frozen=True)
class Identifier:
    """§2.1 -- globally unique, immutable identity token assigned at creation."""

    id: uuid.UUID
    namespace: Optional[str] = None  # scoping context (org, project, registry)


@dataclass(frozen=True)
class VersionInfo:
    """§2.2 -- immutable snapshot descriptor. Every substantive mutation of an
    entity produces a new version."""

    version: str  # MAJOR.MINOR.PATCH


class ActorKind(str, Enum):
    """§2.4 -- kind of actor an ActorRef points to."""

    HUMAN = "HUMAN"
    SERVICE = "SERVICE"
    AGENT = "AGENT"


@dataclass(frozen=True)
class ActorRef:
    """§2.4 -- reference to a human user, service account, or automated agent
    that performed an action."""

    kind: ActorKind
    identity: str  # URI, e.g. OIDC subject or service principal


@dataclass(frozen=True)
class AuditEvent:
    """One `{at, by}` pair as used by AuditMeta.created / AuditMeta.modified."""

    at: datetime
    by: ActorRef


@dataclass
class AuditMeta:
    """§2.3 -- provenance envelope attached to every first-class entity."""

    created: AuditEvent
    modified: AuditEvent
    tags: Set[str] = field(default_factory=set)
    labels: Dict[str, str] = field(default_factory=dict)


T = TypeVar("T")


@dataclass(frozen=True)
class Ref(Generic[T]):
    """§1.3 -- 'Ref<T>: a versioned reference (foreign key) to an entity of
    type T.' `T` is carried only as a typing-level hint (often a forward-ref
    string naming a type this module doesn't implement, e.g. Ref["Metric"])
    -- at runtime a Ref is just an id plus an optional pinned version."""

    id: str
    version: Optional[str] = None


# ============================================================================
# §3.2 Requirement
# ============================================================================


@dataclass(frozen=True)
class RegulatoryRef:
    """§3.2 -- a pointer into a regulatory text (e.g. a specific article of a
    named framework) that a Requirement is normatively sourced from."""

    framework: str  # e.g. "EU_AI_ACT", "GDPR", "NIST_AI_RMF", "ISO_42001"
    article: Optional[str] = None  # e.g. "Article 9", "Annex III, point 4(a)"
    version: Optional[str] = None
    uri: Optional[str] = None  # link to the official text


class RequirementKind(str, Enum):
    """§3.2 -- classification of what kind of obligation a Requirement
    imposes."""

    DESIGN = "DESIGN"
    TESTING = "TESTING"
    DOCUMENTATION = "DOCUMENTATION"
    MONITORING = "MONITORING"
    TRANSPARENCY = "TRANSPARENCY"
    HUMAN_OVERSIGHT = "HUMAN_OVERSIGHT"
    DATA_GOVERNANCE = "DATA_GOVERNANCE"
    RISK_MANAGEMENT = "RISK_MANAGEMENT"
    CYBERSECURITY = "CYBERSECURITY"
    QUALITY_MANAGEMENT = "QUALITY_MANAGEMENT"
    OPERATIONAL_CONTROL = "OPERATIONAL_CONTROL"
    OTHER = "OTHER"


class ApplicabilityStatus(str, Enum):
    """§3.2 -- whether a Requirement actually applies in a given
    ComplianceContext."""

    APPLICABLE = "APPLICABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    PARTIALLY_APPLICABLE = "PARTIALLY_APPLICABLE"
    UNCERTAIN = "UNCERTAIN"
    NEEDS_REVIEW = "NEEDS_REVIEW"


@dataclass
class Requirement:
    """§3.2 -- one normative obligation a Project (via its ComplianceContext)
    must satisfy, sourced from a specific regulatory text. ComplianceObjective
    instances (§3.6) cite Requirements -- via Ref, see the module docstring --
    as the regulatory justification for a measurable acceptance criterion."""

    # identity
    id: str  # see module docstring: a plain slug, not the formal Identifier struct

    # descriptive metadata
    name: str

    # normative source
    source: RegulatoryRef
    requirement_text: str  # the (quoted/paraphrased) text of the obligation itself

    # requirement classification
    kind: RequirementKind

    # context-specific applicability
    context: Ref  # Ref["ComplianceContext"] -- §3.2, out of scope in this module
    applicability_status: ApplicabilityStatus

    # descriptive metadata (optional)
    description: Optional[str] = None
    summary: Optional[str] = None

    # traceability
    related_requirements: Optional[Set[Ref]] = None  # Ref["Requirement"] entries


# ============================================================================
# §3.6 ComplianceObjective
# ============================================================================


class ComparisonOperator(str, Enum):
    GEQ = "GEQ"
    LEQ = "LEQ"
    GT = "GT"
    LT = "LT"
    EQ = "EQ"
    NEQ = "NEQ"


class LogicalOperator(str, Enum):
    AND = "AND"
    OR = "OR"


class ObjectiveCondition:
    """Marker base for the ObjectiveCondition discriminated union (§3.6):
    exactly one of ThresholdCondition | RangeCondition | CategoricalCondition
    | StatisticalCondition | CompositeCondition."""


@dataclass
class ThresholdCondition(ObjectiveCondition):
    operator: ComparisonOperator
    value: float


@dataclass
class RangeCondition(ObjectiveCondition):
    lower: float
    upper: float
    inclusive: bool


@dataclass
class CategoricalCondition(ObjectiveCondition):
    allowed: Set[str]


@dataclass
class StatisticalCondition(ObjectiveCondition):
    test: str  # spec's `StatisticalTest` type is referenced but never defined
    significance: float
    hypothesis: str


@dataclass
class CompositeCondition(ObjectiveCondition):
    operator: LogicalOperator
    conditions: List[ObjectiveCondition] = field(default_factory=list)


class ScopeTarget(str, Enum):
    MODEL = "MODEL"
    DATA = "DATA"
    MODEL_SERVICE = "MODEL_SERVICE"
    ALL = "ALL"


@dataclass
class ObjectiveScope:
    applies_to: Set[ScopeTarget] = field(default_factory=set)


class ObjectivePriority(str, Enum):
    MANDATORY = "MANDATORY"
    RECOMMENDED = "RECOMMENDED"
    INFORMATIONAL = "INFORMATIONAL"


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


@dataclass
class ComplianceObjective:
    """§3.6 -- a normative constraint that binds a Metric to a target value
    or region, defining the acceptance criterion against which compliance is
    judged."""

    # identity & versioning
    id: str  # see module docstring: a plain slug, not the formal Identifier struct
    version: VersionInfo
    audit: AuditMeta

    # descriptive metadata
    name: str
    description: str

    # binding
    metric: Ref  # Ref["Metric"] -- Metric is defined in §3.5, out of scope here

    # target specification
    condition: ObjectiveCondition
    priority: ObjectivePriority
    severity: Severity

    # scope
    scope: Optional[ObjectiveScope] = None

    # requirement connection/mapping. Spec types this `Set<Requirement>`
    # (embedded, not Ref) -- Requirement (§3.2) and its own dependency graph
    # (RegulatoryRef, AttributeRef, GroupSpec, ...) are out of scope here, so
    # this keeps a lightweight Ref["Requirement"] per entry instead.
    requirements: List[Ref] = field(default_factory=list)


# ============================================================================
# §3.8 Documentation
# ============================================================================


class DocumentFormat(str, Enum):
    """Well-known values of DocumentSection.format. Spec: `MODELCARD |
    DATACARD | SYSTEMCARD | CUSTOM of String` -- any string other than these
    three constants is a valid "custom" format, so the field itself is typed
    as plain `str` rather than restricted to this enum."""

    MODELCARD = "MODELCARD"
    DATACARD = "DATACARD"
    SYSTEMCARD = "SYSTEMCARD"


@dataclass
class DocumentSection:
    title: str
    content: str  # markdown content
    order: int
    format: str  # one of DocumentFormat's values, or any custom string
    subsections: Optional[List["DocumentSection"]] = None
    specification: Any = None  # structured representation of the documentation
    reports: List[Ref] = field(default_factory=list)  # Ref["Report"] (§3.7)


@dataclass
class Documentation:
    """§3.8 -- structured, high-level representation consolidating an
    artifact's specification with its associated reports and governance
    information (Model Cards / Data Cards)."""

    # identity & versioning
    id: Identifier
    version: VersionInfo
    audit: AuditMeta

    # descriptive metadata
    name: str
    description: str

    # subject binding -- Ref["Entity"]: the spec's generic entity supertype
    # is never itself given fields, so subjects are kept as lightweight refs.
    subjects: List[Ref] = field(default_factory=list)

    # structure
    sections: List[DocumentSection] = field(default_factory=list)


# ============================================================================
# §3.8 ComplianceDocumentation
# ============================================================================


class MitigationStatus(str, Enum):
    NOT_REQUIRED = "NOT_REQUIRED"
    PROPOSED = "PROPOSED"
    PLANNED = "PLANNED"
    IN_PROGRESS = "IN_PROGRESS"
    APPLIED = "APPLIED"
    VERIFIED = "VERIFIED"
    REJECTED = "REJECTED"
    DEFERRED = "DEFERRED"
    ACCEPTED_RISK = "ACCEPTED_RISK"


class ResidualStatus(str, Enum):
    OPEN = "OPEN"
    PARTIALLY_RESOLVED = "PARTIALLY_RESOLVED"
    RESOLVED = "RESOLVED"
    UNRESOLVED = "UNRESOLVED"
    ACCEPTED_RESIDUAL_RISK = "ACCEPTED_RESIDUAL_RISK"
    NEEDS_RETEST = "NEEDS_RETEST"


@dataclass
class MitigationOperation:
    """§3.8 -- tracks the remediation engaged in reaction to a
    ComplianceObjective violation (as referenced by
    ComplianceFinding.mitigation). Distinct from the operation-taxonomy
    `MitigationOperation[extends Operation]` in §3.10, which is out of scope
    for this module (it belongs to the Operation/Run entities of §3.9-3.10,
    not to Documentation/ComplianceDocumentation)."""

    status: MitigationStatus
    rationale: Optional[str] = None
    owner: Optional[ActorRef] = None
    due_at: Optional[datetime] = None
    applied_at: Optional[datetime] = None
    evidence: Optional[List[Ref]] = None  # Ref["Report"] entries (§3.7)
    verification_report: Optional[Ref] = None  # Ref["Report"] (§3.7)
    residual_status: Optional[ResidualStatus] = None


@dataclass
class ComplianceFinding:
    category: str
    description: str
    objective: Ref  # Ref["ComplianceObjective"] -- kept as a ref per spec,
    # even though ComplianceObjective is defined above in this same module
    report: Ref  # Ref["Report"] (§3.7)
    mitigation: Ref  # Ref["MitigationOperation"] -- action taken


class ComplianceStatusValue(str, Enum):
    COMPLIANT = "COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    PARTIALLY_COMPLIANT = "PARTIALLY_COMPLIANT"
    PENDING_REVIEW = "PENDING_REVIEW"


@dataclass
class ComplianceStatus:
    status: ComplianceStatusValue
    assessed_at: datetime
    assessor: ActorRef
    findings: Optional[List[ComplianceFinding]] = None


@dataclass
class ComplianceDocumentation(Documentation):
    """§3.8 -- a Documentation specialised to carry the compliance-related
    assertion about a Project or one of its constituent entities. Should be
    assigned to the project as a whole AND to every constituent entity
    declared, so compliance status is tracked and remediation is explicitly
    proposed wherever a ComplianceObjective is violated."""

    objectives: Optional[List[ComplianceObjective]] = None
    compliance_status: Optional[ComplianceStatus] = None


__all__ = [
    # §2 common foundations
    "Identifier",
    "VersionInfo",
    "ActorKind",
    "ActorRef",
    "AuditEvent",
    "AuditMeta",
    "Ref",
    # §3.2 Requirement
    "RegulatoryRef",
    "RequirementKind",
    "ApplicabilityStatus",
    "Requirement",
    # §3.6 ComplianceObjective
    "ComparisonOperator",
    "LogicalOperator",
    "ObjectiveCondition",
    "ThresholdCondition",
    "RangeCondition",
    "CategoricalCondition",
    "StatisticalCondition",
    "CompositeCondition",
    "ScopeTarget",
    "ObjectiveScope",
    "ObjectivePriority",
    "Severity",
    "ComplianceObjective",
    # §3.8 Documentation
    "DocumentFormat",
    "DocumentSection",
    "Documentation",
    # §3.8 ComplianceDocumentation
    "MitigationStatus",
    "ResidualStatus",
    "MitigationOperation",
    "ComplianceFinding",
    "ComplianceStatusValue",
    "ComplianceStatus",
    "ComplianceDocumentation",
]
