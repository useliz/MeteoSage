from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Mapping

from ._serialization import to_primitive
from .capability_catalog import (
    BindingFailure,
    BindingFailureCode,
    BindingRequest,
    CapabilityQuery,
)
from .data_products import DataModality
from .evidence import EvidenceRole
from .forecasting import ForecastCandidate, ForecastField
from .forecast_selection import ForecastSelectionRecord
from .model_runtime import ModelAttemptRef
from .reliability import RunManifest, deterministic_identity
from .runtime_contracts import RunStatus, RuntimeTask, TypedOutput
from .runtime_records import (
    EvidenceInterpretation,
    EvidenceRecord,
    ForecastContextRef,
    KnowledgeEntryRef,
    LLMAttemptRef,
    StructuredScientificClaim,
)
from .scientific_products import (
    DiagnosticResult,
    EventHypothesis,
    FeatureTrack,
    HazardAssessment,
    HazardSignal,
    IngredientAssessment,
    MeteorologicalFeature,
    ProcessHypothesis,
    ReportArtifactRef,
    ScientificProduct,
    SituationAnalysis,
)


RUN_STATE_SCHEMA_VERSION = "weather-agent-run-state-v1"
EVIDENCE_GAP_STATUS_VALUES = ("open", "satisfied", "blocked")
PLANNED_ACTION_STATUS_VALUES = (
    "planned",
    "ready",
    "running",
    "completed",
    "failed",
    "cancelled",
)


class PlanFieldContractError(ValueError):
    def __init__(
        self,
        code: str,
        *,
        field_name: str,
        expected: Mapping[str, Any],
        observed: Any,
    ) -> None:
        self.code = code
        self.field_name = field_name
        self.expected = dict(expected)
        self.observed = observed
        super().__init__(code)


def _require_aware(value: datetime, label: str) -> None:
    if value.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")


@dataclass(frozen=True)
class EvidenceRequirement:
    summary: str
    variables: tuple[str, ...]
    products: tuple[str, ...]
    modalities: tuple[str, ...]
    evidence_roles: tuple[str, ...]
    valid_start: datetime
    valid_end: datetime
    spatial: Mapping[str, Any]
    schema_version: str = "weather-agent-evidence-requirement-v1"

    def __post_init__(self) -> None:
        if not self.summary:
            raise ValueError("evidence requirement summary is required")
        selectors = (
            (self.variables, "variables"),
            (self.products, "products"),
            (self.modalities, "modalities"),
            (self.evidence_roles, "evidence roles"),
        )
        if not any(values for values, _label in selectors):
            raise ValueError("evidence requirement needs at least one typed selector")
        for values, label in selectors:
            if any(not value for value in values) or len(values) != len(set(values)):
                raise ValueError(
                    f"evidence requirement {label} must be unique and non-empty"
                )
        for value in self.modalities:
            DataModality(value)
        for value in self.evidence_roles:
            EvidenceRole(value)
        _require_aware(self.valid_start, "requirement valid_start")
        _require_aware(self.valid_end, "requirement valid_end")
        if self.valid_start > self.valid_end:
            raise ValueError("requirement valid_start must not be after valid_end")
        if not self.spatial:
            raise ValueError("evidence requirement spatial selection is required")
        deterministic_identity(self)

    @property
    def requirement_identity(self) -> str:
        return deterministic_identity(self)

    @classmethod
    def from_runtime_task(
        cls,
        task: RuntimeTask,
        *,
        summary: str,
    ) -> "EvidenceRequirement":
        return cls(
            summary=summary,
            variables=task.target_scope.variables,
            products=(),
            modalities=(),
            evidence_roles=(),
            valid_start=task.target_scope.valid_start,
            valid_end=task.target_scope.valid_end,
            spatial=task.target_scope.spatial,
        )

    @classmethod
    def for_forecast_field(cls, task: RuntimeTask) -> "EvidenceRequirement":
        if task.answer_contract.output_kind.value != "field":
            raise ValueError("forecast field requirement requires field output")
        variables = task.answer_contract.required_fields
        if not set(variables).issubset(task.target_scope.variables):
            raise ValueError(
                "forecast field requirements must belong to the task target variables"
            )
        return cls(
            summary=f"Forecast field evidence for {', '.join(variables)}",
            variables=variables,
            products=("forecast_field",),
            modalities=(),
            evidence_roles=(EvidenceRole.FORECAST.value,),
            valid_start=task.target_scope.valid_start,
            valid_end=task.target_scope.valid_end,
            spatial=task.target_scope.spatial,
        )

    def to_query(self) -> CapabilityQuery:
        return CapabilityQuery(
            variables=self.variables,
            products=self.products,
            modalities=self.modalities,
            evidence_roles=self.evidence_roles,
            valid_start=self.valid_start,
            valid_end=self.valid_end,
            spatial=self.spatial,
            limit=None,
        )

    def to_binding_request(self, task: RuntimeTask) -> BindingRequest:
        target = task.target_scope
        if (
            self.valid_start != target.valid_start
            or self.valid_end != target.valid_end
            or dict(self.spatial) != dict(target.spatial)
            or not set(self.variables).issubset(target.variables)
        ):
            raise ValueError("evidence requirement differs from its RuntimeTask")
        return BindingRequest(
            variables=self.variables,
            valid_start=self.valid_start,
            valid_end=self.valid_end,
            spatial=self.spatial,
            decision_time=task.decision_time,
        )


@dataclass(frozen=True)
class EvidenceGap:
    gap_id: str
    requirement: EvidenceRequirement
    status: str = "open"
    related_claim_refs: tuple[str, ...] = ()
    transition_evidence_refs: tuple[str, ...] = ()
    terminal_proof_ref: str | None = None
    schema_version: str = RUN_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RUN_STATE_SCHEMA_VERSION:
            raise PlanFieldContractError(
                "evidence_gap_schema_version_invalid",
                field_name="schema_version",
                expected={"allowed_values": (RUN_STATE_SCHEMA_VERSION,)},
                observed=self.schema_version,
            )
        if not self.gap_id or not isinstance(
            self.requirement, EvidenceRequirement
        ):
            raise ValueError("gap_id and requirement are required")
        if self.status not in EVIDENCE_GAP_STATUS_VALUES:
            raise PlanFieldContractError(
                "evidence_gap_status_invalid",
                field_name="status",
                expected={"allowed_values": EVIDENCE_GAP_STATUS_VALUES},
                observed=self.status,
            )
        if self.status == "blocked" and self.terminal_proof_ref is None:
            raise ValueError("blocked evidence gap requires a terminal proof")
        if self.status != "blocked" and self.terminal_proof_ref is not None:
            raise ValueError(
                f"{self.status} evidence gap cannot carry a terminal proof"
            )
        if (
            self.terminal_proof_ref is not None
            and len(self.terminal_proof_ref) != 64
        ):
            raise ValueError("terminal proof ref must be SHA-256")
        for values, label in (
            (self.related_claim_refs, "related claim refs"),
            (self.transition_evidence_refs, "transition evidence refs"),
        ):
            if any(not value for value in values) or len(values) != len(set(values)):
                raise ValueError(f"evidence gap {label} must be unique and non-empty")
        deterministic_identity(self)


@dataclass(frozen=True)
class CapabilityExhaustionProof:
    gap_id: str
    requirement_identity: str
    catalog_snapshot_identity: str
    discovery_identity: str
    unavailability_refs: tuple[str, ...]
    schema_version: str = "weather-agent-capability-exhaustion-proof-v1"

    def __post_init__(self) -> None:
        if not self.gap_id:
            raise ValueError("exhaustion proof gap id is required")
        for label, value in (
            ("requirement_identity", self.requirement_identity),
            ("catalog_snapshot_identity", self.catalog_snapshot_identity),
            ("discovery_identity", self.discovery_identity),
        ):
            if len(value) != 64:
                raise ValueError(f"{label} must be SHA-256")
        if len(self.unavailability_refs) != len(set(self.unavailability_refs)):
            raise ValueError("exhaustion proof refs must be unique")
        if any(len(value) != 64 for value in self.unavailability_refs):
            raise ValueError("exhaustion proof refs must be SHA-256")
        deterministic_identity(self)

    @property
    def proof_identity(self) -> str:
        return deterministic_identity(self)


@dataclass(frozen=True)
class PlannedAction:
    action_id: str
    purpose: str
    capability_ref: str | None
    arguments: Mapping[str, Any] = field(default_factory=dict)
    depends_on: tuple[str, ...] = ()
    status: str = "planned"
    schema_version: str = RUN_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RUN_STATE_SCHEMA_VERSION:
            raise PlanFieldContractError(
                "planned_action_schema_version_invalid",
                field_name="schema_version",
                expected={"allowed_values": (RUN_STATE_SCHEMA_VERSION,)},
                observed=self.schema_version,
            )
        if not self.action_id or not self.purpose:
            raise ValueError("action_id and purpose are required")
        if self.capability_ref == "":
            raise ValueError("capability_ref must be non-empty when provided")
        if self.status not in PLANNED_ACTION_STATUS_VALUES:
            raise PlanFieldContractError(
                "planned_action_status_invalid",
                field_name="status",
                expected={"allowed_values": PLANNED_ACTION_STATUS_VALUES},
                observed=self.status,
            )
        if self.action_id in self.depends_on:
            raise ValueError("an action cannot depend on itself")
        deterministic_identity(self)


@dataclass(frozen=True)
class PlanState:
    objective: str
    subgoals: tuple[str, ...]
    evidence_gaps: tuple[EvidenceGap, ...]
    steps: tuple[PlannedAction, ...]
    revision: int
    revision_reason: str
    stop_conditions: tuple[str, ...]
    schema_version: str = RUN_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RUN_STATE_SCHEMA_VERSION:
            raise PlanFieldContractError(
                "plan_schema_version_invalid",
                field_name="schema_version",
                expected={"allowed_values": (RUN_STATE_SCHEMA_VERSION,)},
                observed=self.schema_version,
            )
        if not self.objective or not self.revision_reason:
            raise ValueError("plan objective and revision_reason are required")
        if self.revision < 0:
            raise ValueError("plan revision cannot be negative")
        step_ids = tuple(step.action_id for step in self.steps)
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("plan action ids must be unique")
        gap_ids = tuple(gap.gap_id for gap in self.evidence_gaps)
        if len(gap_ids) != len(set(gap_ids)):
            raise ValueError("evidence gap ids must be unique")
        if any(dependency not in step_ids for step in self.steps for dependency in step.depends_on):
            raise ValueError("plan dependencies must reference actions in the same plan")
        deterministic_identity(self)

    @property
    def plan_identity(self) -> str:
        return deterministic_identity(self)


@dataclass(frozen=True)
class ActionAttempt:
    attempt_id: str
    action_id: str
    plan_revision: int
    started_at: datetime
    bound_capability_identity: str
    arguments: Mapping[str, Any]
    schema_version: str = RUN_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.attempt_id or not self.action_id or not self.bound_capability_identity:
            raise ValueError("attempt, action, and binding identities are required")
        if self.plan_revision < 0:
            raise ValueError("attempt plan revision cannot be negative")
        _require_aware(self.started_at, "attempt started_at")
        deterministic_identity(self)


@dataclass(frozen=True)
class Observation:
    observation_id: str
    attempt_id: str
    status: str
    summary: str
    observed_at: datetime
    artifact_refs: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = RUN_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.observation_id or not self.attempt_id or not self.summary:
            raise ValueError("observation_id, attempt_id, and summary are required")
        if self.status not in {"success", "unavailable", "unsupported", "timeout", "failed"}:
            raise ValueError("invalid observation status")
        _require_aware(self.observed_at, "observation observed_at")
        if any(not value for value in self.artifact_refs + self.evidence_refs):
            raise ValueError("observation references must not be empty")
        deterministic_identity(self)


@dataclass(frozen=True)
class CapabilityDiscovery:
    gap_id: str
    requirement_identity: str
    query_identity: str
    query: CapabilityQuery
    catalog_snapshot_identity: str
    capability_refs: tuple[str, ...]
    summary_identities: tuple[str, ...]
    schema_version: str = "weather-agent-capability-discovery-v3"

    def __post_init__(self) -> None:
        if not self.gap_id:
            raise ValueError("discovery gap id is required")
        for label, value in (
            ("requirement identity", self.requirement_identity),
            ("query identity", self.query_identity),
            ("Catalog snapshot identity", self.catalog_snapshot_identity),
        ):
            if len(value) != 64:
                raise ValueError(f"discovery {label} must be SHA-256")
        if self.query_identity != self.query.query_identity:
            raise ValueError("discovery query identity differs from its canonical query")
        if self.query.limit is not None:
            raise ValueError("gap discovery cannot truncate capability matches")
        if len(self.capability_refs) != len(self.summary_identities):
            raise ValueError("discovery refs and summary identities must align")
        if len(self.capability_refs) != len(set(self.capability_refs)):
            raise ValueError("discovery capability refs must be unique")
        if any(not value for value in self.capability_refs):
            raise ValueError("discovery capability refs must not be empty")
        if (
            any(len(value) != 64 for value in self.summary_identities)
            or len(self.summary_identities) != len(set(self.summary_identities))
        ):
            raise ValueError("discovery summary identities must be unique SHA-256")
        if tuple(zip(self.capability_refs, self.summary_identities)) != tuple(
            sorted(zip(self.capability_refs, self.summary_identities))
        ):
            raise ValueError("discovery matches must use canonical order")
        deterministic_identity(self)

    @property
    def discovery_identity(self) -> str:
        return deterministic_identity(self)


_TERMINAL_UNAVAILABILITY_CODES = frozenset(
    {
        BindingFailureCode.ACCESS_DENIED,
        BindingFailureCode.NWP_DENIED,
        BindingFailureCode.VARIABLE_UNAVAILABLE,
        BindingFailureCode.TIME_UNAVAILABLE,
        BindingFailureCode.SPACE_UNAVAILABLE,
    }
)


@dataclass(frozen=True)
class CapabilityUnavailability:
    gap_id: str
    requirement_identity: str
    catalog_snapshot_identity: str
    discovery_identity: str
    capability_ref: str
    request_identity: str
    code: BindingFailureCode
    observation_ref: str
    schema_version: str = "weather-agent-capability-unavailability-v1"

    def __post_init__(self) -> None:
        if not self.gap_id or not self.capability_ref:
            raise ValueError("capability unavailability owner refs are required")
        for label, value in (
            ("requirement_identity", self.requirement_identity),
            ("catalog_snapshot_identity", self.catalog_snapshot_identity),
            ("discovery_identity", self.discovery_identity),
            ("request_identity", self.request_identity),
            ("observation_ref", self.observation_ref),
        ):
            if len(value) != 64:
                raise ValueError(f"unavailability {label} must be SHA-256")
        if self.code not in _TERMINAL_UNAVAILABILITY_CODES:
            raise ValueError(
                "binding failure code cannot prove terminal capability unavailability"
            )
        deterministic_identity(self)

    @property
    def unavailability_identity(self) -> str:
        return deterministic_identity(self)

    @classmethod
    def from_binding_failure(
        cls,
        *,
        gap_id: str,
        requirement_identity: str,
        catalog_snapshot_identity: str,
        discovery_identity: str,
        failure: BindingFailure,
    ) -> "CapabilityUnavailability | None":
        if failure.code not in _TERMINAL_UNAVAILABILITY_CODES:
            return None
        return cls(
            gap_id=gap_id,
            requirement_identity=requirement_identity,
            catalog_snapshot_identity=catalog_snapshot_identity,
            discovery_identity=discovery_identity,
            capability_ref=failure.capability_ref,
            request_identity=failure.request_identity,
            code=failure.code,
            observation_ref=deterministic_identity(
                {
                    "kind": "binding_failure_observation",
                    "discovery_identity": discovery_identity,
                    "failure": failure.to_dict(),
                }
            ),
        )


@dataclass(frozen=True)
class ModelSchedulingRecord:
    capability_ref: str
    input_product_ref: str
    attempt_id: str
    status: str
    candidate_identities: tuple[str, ...] = ()
    model_attempt_identity: str | None = None
    schema_version: str = "weather-agent-model-scheduling-v1"

    def __post_init__(self) -> None:
        if not self.capability_ref or not self.input_product_ref or not self.attempt_id:
            raise ValueError("model scheduling identities are required")
        if self.status not in {"success", "failed", "timeout", "unsupported"}:
            raise ValueError("model scheduling status is invalid")
        if self.status == "success" and not self.candidate_identities:
            raise ValueError("successful model scheduling requires forecast candidates")
        if self.model_attempt_identity is not None and len(self.model_attempt_identity) != 64:
            raise ValueError("model attempt identity must be SHA-256")
        for identity in self.candidate_identities:
            if len(identity) != 64:
                raise ValueError("scheduled candidate identities must be SHA-256")

    @property
    def scheduling_identity(self) -> str:
        return deterministic_identity(self)


@dataclass(frozen=True)
class MaterializedProductRecord:
    capability_ref: str
    artifact_ref: str
    query_identity: str
    binding_identity: str
    invocation_identity: str
    evidence_ref: str
    schema_version: str = "weather-agent-materialized-product-v1"

    def __post_init__(self) -> None:
        if not self.capability_ref:
            raise ValueError("materialized product capability ref is required")
        for label, value in (
            ("artifact_ref", self.artifact_ref),
            ("query_identity", self.query_identity),
            ("binding_identity", self.binding_identity),
            ("invocation_identity", self.invocation_identity),
            ("evidence_ref", self.evidence_ref),
        ):
            if len(value) != 64:
                raise ValueError(f"materialized product {label} must be SHA-256")
        deterministic_identity(self)

    @property
    def materialization_identity(self) -> str:
        return deterministic_identity(self)


@dataclass(frozen=True)
class ModelApplicabilityRecord:
    capability_ref: str
    input_product_ref: str
    target_identity: str
    applicable: bool
    reason_codes: tuple[str, ...] = ()
    schema_version: str = "weather-agent-model-applicability-v1"

    def __post_init__(self) -> None:
        if not self.capability_ref or not self.input_product_ref:
            raise ValueError("model applicability requires capability and input product refs")
        if len(self.target_identity) != 64:
            raise ValueError("model applicability target identity must be SHA-256")
        if self.applicable == bool(self.reason_codes):
            raise ValueError("applicable models have no rejection reasons; rejected models require reasons")
        deterministic_identity(self)

    @property
    def applicability_identity(self) -> str:
        return deterministic_identity(self)


@dataclass
class RunState:
    task: RuntimeTask
    manifest: RunManifest
    plan: PlanState
    phase: str = "running"
    iteration: int = 0
    catalog_snapshot_identity: str | None = None
    discovered_capabilities: list[str] = field(default_factory=list)
    capability_discoveries: list[CapabilityDiscovery] = field(default_factory=list)
    capability_unavailabilities: list[CapabilityUnavailability] = field(default_factory=list)
    capability_exhaustion_proofs: list[CapabilityExhaustionProof] = field(default_factory=list)
    bound_capabilities: list[str] = field(default_factory=list)
    attempts: list[ActionAttempt] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    artifact_refs: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    claim_refs: list[str] = field(default_factory=list)
    forecast_candidates: dict[str, ForecastCandidate] = field(default_factory=dict)
    forecast_fields: dict[str, ForecastField] = field(default_factory=dict)
    forecast_selections: dict[str, ForecastSelectionRecord] = field(default_factory=dict)
    forecast_context_refs: dict[str, ForecastContextRef] = field(default_factory=dict)
    materialized_product_refs: list[str] = field(default_factory=list)
    materialized_products: dict[str, MaterializedProductRecord] = field(default_factory=dict)
    applicable_model_refs: list[str] = field(default_factory=list)
    model_applicability_records: dict[str, ModelApplicabilityRecord] = field(default_factory=dict)
    model_scheduling_records: dict[str, ModelSchedulingRecord] = field(default_factory=dict)
    model_attempt_refs: dict[str, ModelAttemptRef] = field(default_factory=dict)
    scientific_products: dict[str, ScientificProduct] = field(default_factory=dict)
    reasoning_claims: dict[str, StructuredScientificClaim] = field(default_factory=dict)
    knowledge_refs: dict[str, KnowledgeEntryRef] = field(default_factory=dict)
    situations: dict[str, SituationAnalysis] = field(default_factory=dict)
    situation_history: list[str] = field(default_factory=list)
    active_situation_ref: str | None = None
    event_hypotheses: dict[str, EventHypothesis] = field(default_factory=dict)
    event_hypothesis_history: dict[str, list[str]] = field(default_factory=dict)
    hazard_assessments: dict[str, HazardAssessment] = field(default_factory=dict)
    report_refs: dict[str, ReportArtifactRef] = field(default_factory=dict)
    llm_attempt_refs: list[LLMAttemptRef] = field(default_factory=list)
    validation_records: list[Mapping[str, Any]] = field(default_factory=list)
    accounting: dict[str, Any] = field(
        default_factory=lambda: {"llm_calls": 0, "actions": 0, "resource_units": 0}
    )
    final_output: TypedOutput | None = None
    failure: Mapping[str, Any] | None = None
    schema_version: str = RUN_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.manifest.task_identity != self.task.task_identity:
            raise ValueError("manifest task identity does not match RuntimeTask")
        if self.phase not in {"running", "completed", "run_failed"}:
            raise ValueError("invalid run phase")
        if self.iteration < 0:
            raise ValueError("run iteration cannot be negative")
        if (
            self.catalog_snapshot_identity is not None
            and len(self.catalog_snapshot_identity) != 64
        ):
            raise ValueError("catalog_snapshot_identity must be a SHA-256 identity")
        deterministic_identity(self.snapshot())

    def _require_active(self) -> None:
        if self.phase != "running":
            raise RuntimeError("terminal RunState cannot be mutated")

    def revise_plan(
        self,
        *,
        subgoals: tuple[str, ...] | None = None,
        evidence_gaps: tuple[EvidenceGap, ...] | None = None,
        steps: tuple[PlannedAction, ...] | None = None,
        stop_conditions: tuple[str, ...] | None = None,
        reason: str,
    ) -> PlanState:
        self._require_active()
        if not reason:
            raise ValueError("plan revision reason is required")
        self.plan = PlanState(
            objective=self.plan.objective,
            subgoals=self.plan.subgoals if subgoals is None else subgoals,
            evidence_gaps=self.plan.evidence_gaps if evidence_gaps is None else evidence_gaps,
            steps=self.plan.steps if steps is None else steps,
            revision=self.plan.revision + 1,
            revision_reason=reason,
            stop_conditions=self.plan.stop_conditions if stop_conditions is None else stop_conditions,
        )
        self.iteration += 1
        return self.plan

    def record_attempt_before_execution(self, attempt: ActionAttempt) -> None:
        self._require_active()
        if attempt.plan_revision != self.plan.revision:
            raise ValueError("attempt plan revision does not match current PlanState")
        if attempt.action_id not in {step.action_id for step in self.plan.steps}:
            raise ValueError("attempt action is not present in current PlanState")
        if attempt.attempt_id in {item.attempt_id for item in self.attempts}:
            raise ValueError("attempt identity already recorded")
        self.attempts.append(attempt)
        self.accounting["actions"] = int(self.accounting.get("actions", 0)) + 1

    def record_observation(self, observation: Observation) -> None:
        self._require_active()
        if observation.attempt_id not in {item.attempt_id for item in self.attempts}:
            raise ValueError("observation must reference a recorded attempt")
        if observation.observation_id in {item.observation_id for item in self.observations}:
            raise ValueError("observation identity already recorded")
        self.observations.append(observation)
        for reference in observation.artifact_refs:
            if reference not in self.artifact_refs:
                self.artifact_refs.append(reference)
        for reference in observation.evidence_refs:
            if reference not in self.evidence_refs:
                self.evidence_refs.append(reference)

    def attach_evidence_record(
        self,
        observation_id: str,
        record: EvidenceRecord,
    ) -> None:
        """Attach one ledger-owned evidence record to its action observation."""

        self._require_active()
        matches = tuple(
            index
            for index, observation in enumerate(self.observations)
            if observation.observation_id == observation_id
        )
        if len(matches) != 1:
            raise ValueError("evidence record requires one current-run observation")
        index = matches[0]
        observation = self.observations[index]
        artifact_refs = tuple(
            artifact.artifact_ref for artifact in record.envelope.artifacts
        )
        self.observations[index] = replace(
            observation,
            artifact_refs=tuple(
                dict.fromkeys(observation.artifact_refs + artifact_refs)
            ),
            evidence_refs=tuple(dict.fromkeys(
                tuple(record.record_id if ref == record.envelope.envelope_identity else ref
                    for ref in observation.evidence_refs) + (record.record_id,)
            )),
        )
        for artifact_ref in artifact_refs:
            if artifact_ref not in self.artifact_refs:
                self.artifact_refs.append(artifact_ref)
        if record.record_id not in self.evidence_refs:
            self.evidence_refs.append(record.record_id)

    def reconcile_observation_evidence_ref(
        self,
        observed_ref: str,
        canonical_ref: str,
    ) -> None:
        """Replace a tool-local evidence identity with its RunLedger identity."""

        self._require_active()
        for label, reference in (
            ("observed", observed_ref),
            ("canonical", canonical_ref),
        ):
            if len(reference) != 64:
                raise ValueError(f"{label} evidence ref must be a SHA-256 identity")
        if observed_ref == canonical_ref:
            return
        if observed_ref not in self.evidence_refs:
            raise ValueError("observed evidence ref is absent from RunState")

        def reconcile_result(value: Any) -> Any:
            if not isinstance(value, Mapping):
                return value
            document = dict(value)
            payload = document.get("value")
            if not isinstance(payload, Mapping):
                return document
            rebound = dict(payload)
            if rebound.get("evidence_ref") == observed_ref:
                rebound["evidence_ref"] = canonical_ref
            if isinstance(rebound.get("evidence_refs"), (tuple, list)):
                rebound["evidence_refs"] = tuple(
                    canonical_ref if item == observed_ref else item
                    for item in rebound["evidence_refs"]
                )
            document["value"] = rebound
            return document

        rebound_count = 0
        observations: list[Observation] = []
        for observation in self.observations:
            if observed_ref not in observation.evidence_refs:
                observations.append(observation)
                continue
            details = dict(observation.details)
            for key in ("action_result", "tool_result"):
                if key in details:
                    details[key] = reconcile_result(details[key])
            observations.append(
                replace(
                    observation,
                    evidence_refs=tuple(
                        canonical_ref if item == observed_ref else item
                        for item in observation.evidence_refs
                    ),
                    details=details,
                )
            )
            rebound_count += 1
        if rebound_count != 1:
            raise ValueError("tool-local evidence ref must belong to one observation")
        self.observations = observations
        self.evidence_refs = list(
            dict.fromkeys(
                canonical_ref if item == observed_ref else item
                for item in self.evidence_refs
            )
        )

    def record_validation(self, record: Mapping[str, Any]) -> None:
        self._require_active()
        deterministic_identity(record)
        self.validation_records.append(dict(record))

    def _evidence_gap(self, gap_id: str) -> EvidenceGap:
        matches = tuple(
            gap for gap in self.plan.evidence_gaps if gap.gap_id == gap_id
        )
        if len(matches) != 1:
            raise ValueError("record must reference one current evidence gap")
        return matches[0]

    def _plan_replacing_gap(
        self, gap: EvidenceGap, reason: str
    ) -> PlanState:
        return replace(
            self.plan,
            evidence_gaps=tuple(
                gap if item.gap_id == gap.gap_id else item
                for item in self.plan.evidence_gaps
            ),
            revision=self.plan.revision + 1,
            revision_reason=reason,
        )

    def record_gap_interpretation(
        self,
        gap_id: str,
        interpretation: EvidenceInterpretation,
    ) -> EvidenceGap:
        self._require_active()
        if not isinstance(interpretation, EvidenceInterpretation):
            raise TypeError("gap transition requires an EvidenceInterpretation")
        gap = self._evidence_gap(gap_id)
        if gap.status != "open":
            raise ValueError("evidence interpretation requires an open gap")
        if interpretation.record_ref not in self.evidence_refs:
            raise ValueError(
                "gap interpretation must reference canonical current-run evidence"
            )
        if interpretation.record_ref in gap.transition_evidence_refs:
            return gap
        next_gap = replace(
            gap,
            transition_evidence_refs=(
                *gap.transition_evidence_refs,
                interpretation.record_ref,
            ),
        )
        self.plan = self._plan_replacing_gap(
            next_gap,
            f"Evidence interpretation recorded for open gap {gap_id}",
        )
        self.iteration += 1
        return next_gap

    def record_gap_satisfaction(
        self,
        gap_ids: tuple[str, ...],
        evidence_refs: tuple[str, ...],
        claim_ref: str,
    ) -> tuple[EvidenceGap, ...]:
        self._require_active()
        if not gap_ids or len(gap_ids) != len(set(gap_ids)) or any(
            not gap_id for gap_id in gap_ids
        ):
            raise ValueError("gap satisfaction requires unique gap ids")
        if not evidence_refs or len(evidence_refs) != len(set(evidence_refs)) or any(
            not reference for reference in evidence_refs
        ):
            raise ValueError("gap satisfaction requires unique evidence refs")
        gaps = tuple(self._evidence_gap(gap_id) for gap_id in gap_ids)
        if any(gap.status != "open" for gap in gaps):
            raise ValueError("gap satisfaction requires every gap to be open")
        if claim_ref not in self.claim_refs:
            raise ValueError("gap satisfaction requires a current-run claim")
        if not set(evidence_refs).issubset(self.evidence_refs):
            raise ValueError(
                "gap satisfaction requires canonical current-run evidence"
            )
        next_gaps = tuple(
            replace(
                gap,
                status="satisfied",
                related_claim_refs=tuple(
                    dict.fromkeys((*gap.related_claim_refs, claim_ref))
                ),
                transition_evidence_refs=tuple(
                    dict.fromkeys((*gap.transition_evidence_refs, *evidence_refs))
                ),
            )
            for gap in gaps
        )
        replacements = {gap.gap_id: gap for gap in next_gaps}
        self.plan = replace(
            self.plan,
            evidence_gaps=tuple(
                replacements.get(gap.gap_id, gap)
                for gap in self.plan.evidence_gaps
            ),
            revision=self.plan.revision + 1,
            revision_reason=f"Evidence sufficiency satisfied gaps {', '.join(gap_ids)}",
        )
        self.iteration += 1
        return next_gaps

    def record_capability_discovery(
        self, discovery: CapabilityDiscovery
    ) -> CapabilityExhaustionProof | None:
        self._require_active()
        if (
            self.catalog_snapshot_identity is not None
            and discovery.catalog_snapshot_identity != self.catalog_snapshot_identity
        ):
            raise ValueError("discovery does not belong to the active Catalog snapshot")
        gap = self._evidence_gap(discovery.gap_id)
        if discovery.requirement_identity != gap.requirement.requirement_identity:
            raise ValueError("discovery requirement differs from its evidence gap")
        expected_query = gap.requirement.to_query()
        if discovery.query != expected_query:
            raise ValueError("discovery query differs from its evidence requirement")
        existing = tuple(
            item
            for item in self.capability_discoveries
            if item.gap_id == discovery.gap_id
            and item.requirement_identity == discovery.requirement_identity
            and item.catalog_snapshot_identity
            == discovery.catalog_snapshot_identity
        )
        if existing:
            if len(existing) == 1 and existing[0] == discovery:
                return next(
                    (
                        proof
                        for proof in self.capability_exhaustion_proofs
                        if proof.discovery_identity == discovery.discovery_identity
                    ),
                    None,
                )
            raise ValueError("capability discovery owner-key collision")
        if gap.status != "open":
            raise ValueError("capability discovery requires an open evidence gap")

        proof = None
        next_plan = None
        if not discovery.capability_refs:
            proof = CapabilityExhaustionProof(
                gap_id=discovery.gap_id,
                requirement_identity=discovery.requirement_identity,
                catalog_snapshot_identity=discovery.catalog_snapshot_identity,
                discovery_identity=discovery.discovery_identity,
                unavailability_refs=(),
            )
            next_plan = self._plan_replacing_gap(
                replace(
                    gap,
                    status="blocked",
                    terminal_proof_ref=proof.proof_identity,
                ),
                f"Catalog alternatives exhausted for evidence gap {gap.gap_id}",
            )
        if self.catalog_snapshot_identity is None:
            self.catalog_snapshot_identity = discovery.catalog_snapshot_identity
        self.capability_discoveries.append(discovery)
        for capability_ref in discovery.capability_refs:
            if capability_ref not in self.discovered_capabilities:
                self.discovered_capabilities.append(capability_ref)
        if proof is not None:
            self.capability_exhaustion_proofs.append(proof)
            self.plan = next_plan
            self.iteration += 1
        return proof

    def record_discovered_binding_failure(
        self,
        discovery_identity: str,
        failure: BindingFailure,
    ) -> tuple[
        CapabilityUnavailability | None,
        CapabilityExhaustionProof | None,
    ]:
        """Record one linked terminal failure and owner-generated exhaustion."""

        self._require_active()
        if not isinstance(failure, BindingFailure):
            raise TypeError("discovered binding failure requires BindingFailure")
        matches = tuple(
            item
            for item in self.capability_discoveries
            if item.discovery_identity == discovery_identity
        )
        if len(matches) != 1:
            raise ValueError("binding failure must reference one canonical discovery")
        discovery = matches[0]
        gap = self._evidence_gap(discovery.gap_id)
        if discovery.requirement_identity != gap.requirement.requirement_identity:
            raise ValueError(
                "binding failure discovery requirement differs from current gap"
            )
        if failure.capability_ref not in discovery.capability_refs:
            raise ValueError("binding failure capability was not discovered")
        expected_request = gap.requirement.to_binding_request(self.task)
        if failure.request_identity != expected_request.request_identity:
            raise ValueError("binding failure request differs from its evidence gap")
        record = CapabilityUnavailability.from_binding_failure(
            gap_id=gap.gap_id,
            requirement_identity=discovery.requirement_identity,
            catalog_snapshot_identity=discovery.catalog_snapshot_identity,
            discovery_identity=discovery.discovery_identity,
            failure=failure,
        )
        if record is None:
            return None, None
        collision = tuple(
            item
            for item in self.capability_unavailabilities
            if item.discovery_identity == discovery.discovery_identity
            and item.capability_ref == record.capability_ref
        )
        if collision:
            if len(collision) == 1 and collision[0] == record:
                proof = next(
                    (
                        item
                        for item in self.capability_exhaustion_proofs
                        if item.discovery_identity == discovery.discovery_identity
                    ),
                    None,
                )
                return collision[0], proof
            raise ValueError("capability unavailability owner-key collision")
        if gap.status != "open":
            raise ValueError("binding failure requires an open evidence gap")

        unavailable = tuple(
            item
            for item in self.capability_unavailabilities
            if item.discovery_identity == discovery.discovery_identity
        ) + (record,)
        proof = None
        next_plan = None
        if {item.capability_ref for item in unavailable} == set(
            discovery.capability_refs
        ):
            proof = CapabilityExhaustionProof(
                gap_id=discovery.gap_id,
                requirement_identity=discovery.requirement_identity,
                catalog_snapshot_identity=discovery.catalog_snapshot_identity,
                discovery_identity=discovery.discovery_identity,
                unavailability_refs=tuple(
                    sorted(item.unavailability_identity for item in unavailable)
                ),
            )
            next_plan = self._plan_replacing_gap(
                replace(
                    gap,
                    status="blocked",
                    terminal_proof_ref=proof.proof_identity,
                ),
                f"Catalog alternatives exhausted for evidence gap {gap.gap_id}",
            )

        self.capability_unavailabilities.append(record)
        if proof is not None:
            self.capability_exhaustion_proofs.append(proof)
            self.plan = next_plan
            self.iteration += 1
        return record, proof

    def record_forecast_candidate(self, candidate: ForecastCandidate) -> None:
        self._require_active()
        identity = candidate.candidate_identity
        existing = self.forecast_candidates.get(identity)
        if existing is not None and existing != candidate:
            raise ValueError("forecast candidate identity collision")
        self.forecast_candidates[identity] = candidate

    def record_forecast_field(self, forecast_field: ForecastField) -> None:
        self._require_active()
        identity = forecast_field.field_identity
        existing = self.forecast_fields.get(identity)
        if existing is not None and existing != forecast_field:
            raise ValueError("forecast field identity collision")
        self.forecast_fields[identity] = forecast_field

    def record_forecast_selection(self, selection: ForecastSelectionRecord) -> None:
        self._require_active()
        existing = self.forecast_selections.get(selection.selection_identity)
        if existing is not None and existing != selection:
            raise ValueError("forecast selection identity collision")
        self.forecast_selections[selection.selection_identity] = selection

    def record_model_scheduling(self, record: ModelSchedulingRecord) -> None:
        self._require_active()
        existing = self.model_scheduling_records.get(record.capability_ref)
        if existing is not None and existing != record:
            raise ValueError("one applicable model may be dispatched only once in Phase 5")
        self.model_scheduling_records[record.capability_ref] = record

    def record_materialized_product(self, record: MaterializedProductRecord) -> None:
        self._require_active()
        existing = self.materialized_products.get(record.capability_ref)
        if existing is not None and existing != record:
            raise ValueError("one product capability may materialize only once in Phase 5")
        self.materialized_products[record.capability_ref] = record
        if record.capability_ref not in self.materialized_product_refs:
            self.materialized_product_refs.append(record.capability_ref)

    def record_model_applicability(self, record: ModelApplicabilityRecord) -> None:
        self._require_active()
        existing = self.model_applicability_records.get(record.capability_ref)
        if existing is not None and existing.target_identity != record.target_identity:
            raise ValueError("model applicability target changed within one run")
        self.model_applicability_records[record.capability_ref] = record

    def record_model_attempt(self, reference: ModelAttemptRef) -> None:
        self._require_active()
        existing = self.model_attempt_refs.get(reference.attempt_identity)
        if existing is not None and existing != reference:
            raise ValueError("model attempt identity collision")
        self.model_attempt_refs[reference.attempt_identity] = reference

    def record_forecast_context_ref(self, reference: ForecastContextRef) -> None:
        self._require_active()
        existing = self.forecast_context_refs.get(reference.context_ref)
        if existing is not None and existing != reference:
            raise ValueError("forecast context ref identity collision")
        self.forecast_context_refs[reference.context_ref] = reference

    def record_knowledge_ref(self, reference: KnowledgeEntryRef) -> None:
        self._require_active()
        existing = self.knowledge_refs.get(reference.entry_ref)
        if existing is not None and existing != reference:
            raise ValueError("knowledge entry identity collision")
        self.knowledge_refs[reference.entry_ref] = reference

    def _require_known_refs(
        self,
        references: tuple[str, ...],
        *,
        allowed: set[str],
        label: str,
    ) -> None:
        unknown = tuple(reference for reference in references if reference not in allowed)
        if unknown:
            raise ValueError(f"{label} contains unknown current-run refs: {unknown}")

    def record_scientific_product(self, product: ScientificProduct) -> None:
        self._require_active()
        identity = product.product_ref
        existing = self.scientific_products.get(identity)
        if existing is not None:
            if existing != product:
                raise ValueError("scientific product identity collision")
            return

        if isinstance(product, DiagnosticResult):
            self._require_known_refs(
                product.input_artifact_refs,
                allowed=set(self.artifact_refs),
                label="diagnostic artifacts",
            )
            self._require_known_refs(
                product.input_evidence_refs,
                allowed=set(self.evidence_refs),
                label="diagnostic evidence",
            )
        elif isinstance(product, MeteorologicalFeature):
            self._require_known_refs(
                (product.geometry_ref,),
                allowed=set(self.artifact_refs),
                label="feature geometry",
            )
            known_records = (
                set(self.evidence_refs) | set(self.claim_refs) | set(self.scientific_products)
            )
            self._require_known_refs(
                product.supporting_record_refs + product.opposing_record_refs,
                allowed=known_records,
                label="feature records",
            )
        elif isinstance(product, FeatureTrack):
            feature_refs = {
                reference
                for reference, item in self.scientific_products.items()
                if isinstance(item, MeteorologicalFeature)
            }
            self._require_known_refs(
                product.feature_refs, allowed=feature_refs, label="track features"
            )
        elif isinstance(product, (IngredientAssessment, ProcessHypothesis)):
            raise ValueError("reasoning products require SituationStateReducer")
        elif isinstance(product, HazardSignal):
            self._require_known_refs(
                (product.definition_ref,),
                allowed=set(self.knowledge_refs),
                label="hazard definition",
            )
            diagnostics = {
                reference
                for reference, item in self.scientific_products.items()
                if isinstance(item, DiagnosticResult)
            }
            features = {
                reference
                for reference, item in self.scientific_products.items()
                if isinstance(item, MeteorologicalFeature)
            }
            self._require_known_refs(
                product.diagnostic_refs, allowed=diagnostics, label="hazard diagnostics"
            )
            self._require_known_refs(
                product.feature_refs, allowed=features, label="hazard features"
            )

        self.scientific_products[identity] = product

    def _record_reasoning_claim(self, claim: StructuredScientificClaim) -> None:
        self._require_active()
        existing = self.reasoning_claims.get(claim.claim_ref)
        if existing is not None and existing != claim:
            raise ValueError("reasoning claim identity collision")
        self.reasoning_claims[claim.claim_ref] = claim
        if claim.claim_ref not in self.claim_refs:
            self.claim_refs.append(claim.claim_ref)

    def _record_reasoning_product(
        self,
        product: IngredientAssessment | ProcessHypothesis,
    ) -> None:
        self._require_active()
        existing = self.scientific_products.get(product.product_ref)
        if existing is not None and existing != product:
            raise ValueError("reasoning product identity collision")
        self.scientific_products[product.product_ref] = product

    def _record_situation(self, situation: SituationAnalysis) -> None:
        self._require_active()
        identity = situation.product_ref
        self.situations[identity] = situation
        self.situation_history.append(identity)
        self.active_situation_ref = identity

    def _record_event_hypothesis(self, hypothesis: EventHypothesis) -> None:
        self._require_active()
        history = self.event_hypothesis_history.setdefault(hypothesis.hypothesis_id, [])
        identity = hypothesis.product_ref
        self.event_hypotheses[identity] = hypothesis
        history.append(identity)

    def _record_hazard_assessment(self, assessment: HazardAssessment) -> None:
        self._require_active()
        self.hazard_assessments[assessment.product_ref] = assessment

    def record_report_ref(self, reference: ReportArtifactRef) -> None:
        self._require_active()
        self._require_known_refs(
            (reference.artifact_ref,),
            allowed=set(self.artifact_refs),
            label="report artifact",
        )
        self._require_known_refs(
            reference.claim_refs, allowed=set(self.claim_refs), label="report claims"
        )
        self._require_known_refs(
            reference.evidence_refs, allowed=set(self.evidence_refs), label="report evidence"
        )
        existing = self.report_refs.get(reference.report_identity)
        if existing is not None and existing != reference:
            raise ValueError("report identity collision")
        self.report_refs[reference.report_identity] = reference

    def record_llm_attempt(self, reference: LLMAttemptRef) -> None:
        self._require_active()
        if reference.attempt_ref in {item.attempt_ref for item in self.llm_attempt_refs}:
            raise ValueError("duplicate LLM attempt identity")
        if self.llm_attempt_refs and reference.started_at < self.llm_attempt_refs[-1].started_at:
            raise ValueError("LLM attempts must be recorded chronologically")
        self.llm_attempt_refs.append(reference)
        self.accounting["llm_calls"] = int(self.accounting.get("llm_calls", 0)) + 1
        self.accounting["prompt_tokens"] = int(
            self.accounting.get("prompt_tokens", 0)
        ) + int(reference.prompt_tokens or 0)
        self.accounting["completion_tokens"] = int(
            self.accounting.get("completion_tokens", 0)
        ) + int(reference.completion_tokens or 0)

    def complete(self, output: TypedOutput) -> None:
        self._require_active()
        self.final_output = output
        self.failure = None
        self.phase = RunStatus.COMPLETED.value

    def fail(self, failure: Mapping[str, Any]) -> None:
        self._require_active()
        if not failure:
            raise ValueError("failure details are required")
        deterministic_identity(failure)
        self.failure = dict(failure)
        self.final_output = None
        self.phase = RunStatus.RUN_FAILED.value

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_identity": self.task.task_identity,
            "manifest_identity": self.manifest.run_identity,
            "phase": self.phase,
            "iteration": self.iteration,
            "catalog_snapshot_identity": self.catalog_snapshot_identity,
            "plan": to_primitive(self.plan),
            "discovered_capabilities": list(self.discovered_capabilities),
            "capability_discoveries": to_primitive(tuple(self.capability_discoveries)),
            "capability_unavailabilities": to_primitive(
                tuple(self.capability_unavailabilities)
            ),
            "capability_exhaustion_proofs": to_primitive(tuple(self.capability_exhaustion_proofs)),
            "bound_capabilities": list(self.bound_capabilities),
            "attempts": to_primitive(tuple(self.attempts)),
            "observations": to_primitive(tuple(self.observations)),
            "artifact_refs": list(self.artifact_refs),
            "evidence_refs": list(self.evidence_refs),
            "claim_refs": list(self.claim_refs),
            "forecast_candidates": {
                identity: candidate.to_dict()
                for identity, candidate in sorted(self.forecast_candidates.items())
            },
            "forecast_fields": {
                identity: forecast_field.to_dict()
                for identity, forecast_field in sorted(self.forecast_fields.items())
            },
            "forecast_selections": {
                identity: selection.to_dict()
                for identity, selection in sorted(self.forecast_selections.items())
            },
            "forecast_context_refs": {
                identity: reference.to_dict()
                for identity, reference in sorted(self.forecast_context_refs.items())
            },
            "materialized_product_refs": tuple(self.materialized_product_refs),
            "materialized_products": {
                capability_ref: to_primitive(record)
                for capability_ref, record in sorted(self.materialized_products.items())
            },
            "applicable_model_refs": tuple(self.applicable_model_refs),
            "model_applicability_records": {
                identity: to_primitive(record)
                for identity, record in sorted(self.model_applicability_records.items())
            },
            "model_scheduling_records": {
                identity: to_primitive(record)
                for identity, record in sorted(self.model_scheduling_records.items())
            },
            "model_attempt_refs": {
                identity: reference.to_dict()
                for identity, reference in sorted(self.model_attempt_refs.items())
            },
            "scientific_products": {
                identity: product.to_dict()
                for identity, product in sorted(self.scientific_products.items())
            },
            "reasoning_claims": {
                identity: claim.to_dict()
                for identity, claim in sorted(self.reasoning_claims.items())
            },
            "knowledge_refs": {
                identity: reference.to_dict()
                for identity, reference in sorted(self.knowledge_refs.items())
            },
            "situations": {
                identity: situation.to_dict()
                for identity, situation in sorted(self.situations.items())
            },
            "situation_history": tuple(self.situation_history),
            "active_situation_ref": self.active_situation_ref,
            "event_hypotheses": {
                identity: hypothesis.to_dict()
                for identity, hypothesis in sorted(self.event_hypotheses.items())
            },
            "event_hypothesis_history": {
                identity: tuple(history)
                for identity, history in sorted(self.event_hypothesis_history.items())
            },
            "hazard_assessments": {
                identity: item.to_dict()
                for identity, item in sorted(self.hazard_assessments.items())
            },
            "report_refs": {
                identity: item.to_dict()
                for identity, item in sorted(self.report_refs.items())
            },
            "llm_attempt_refs": tuple(item.to_dict() for item in self.llm_attempt_refs),
            "validation_records": to_primitive(tuple(self.validation_records)),
            "accounting": to_primitive(self.accounting),
            "final_output": None if self.final_output is None else self.final_output.to_dict(),
            "failure": None if self.failure is None else to_primitive(self.failure),
        }
