from __future__ import annotations

from .host_control import raise_control_error
from .material_readers import is_material_reader, is_material_code

from dataclasses import dataclass, field, replace
from time import monotonic
from datetime import datetime, timezone
from uuid import uuid4
from typing import Any, Callable, Mapping, Protocol, Sequence

from .runtime_errors import error_details
from .context_limits import ContextLimits
from .working_state import WorkingStateConfig, working_state_view, WorkingStateInputUnavailable
from .capability_catalog import CapabilityDescriptor
from ._serialization import canonical_json, to_primitive
from .c1_records import (
    C1ControlRecords,
    MAX_ARGUMENT_BYTES,
    MAX_RESPONSE_BYTES,
    json_bytes,
)
from .executable_operations import (
    ArgumentValidationError,
    ChoiceSpec,
    ExecutableOperation,
    OperationContext,
    OperationExecutionContext,
    OperationObservation,
    OperationProvider,
    TransitionResult,
    _EpisodeResources,
)
from .public_payload_identity import public_payload_identity
from .reliability import ModelIdentity, RunManifest, deterministic_identity
from .record_inspection import (
    publish_current_turn_record_refs, public_record_index, recent_operation_history,
)
from .run_state import PlanState, RunState
from .runtime_contracts import RuntimeTask, TypedOutput
from .retrieval_scope import ResolvedRetrievalScope
from .runtime_records import LLMAttemptRef, LLMProtocolOutcome, RunLedger
from .source_coverage import (
    SourceCoverageInputs,
    SourceCoverageOperationalAnchor,
    SourceCoverageSnapshot,
    SourceCoverageTurnView,
    SourceOwnerOutcome,
    derive_source_coverage,
    project_source_coverage,
)


EXECUTABLE_LOOP_SCHEMA_VERSION = "weather-agent-executable-loop-v1"


_OMITTED_UPDATE = object()


@dataclass(frozen=True)
class Submission:
    choice: str
    arguments: Mapping[str, object]
    record_delta: object | None = None
    schema_version: str = EXECUTABLE_LOOP_SCHEMA_VERSION
    notebook_update: object = field(default_factory=lambda: _OMITTED_UPDATE)

    def __post_init__(self) -> None:
        if not isinstance(self.choice, str) or not self.choice:
            raise ValueError("submission choice must be a non-empty string")
        if not isinstance(self.arguments, Mapping):
            raise TypeError("submission arguments must be an object")

    def to_dict(self) -> Mapping[str, Any]:
        value = {"choice": self.choice, "arguments": to_primitive(self.arguments)}
        if self.record_delta is not None:
            value["record_delta"] = to_primitive(self.record_delta)
        if self.notebook_update is not _OMITTED_UPDATE:
            value["notebook_update"] = to_primitive(self.notebook_update)
        return value


@dataclass(frozen=True)
class ProtocolCorrection:
    code: str
    field_path: str
    instruction: str
    recovery_kind: str = "repair_fields"
    details: Mapping[str, Any] | None = None

    def to_dict(self) -> Mapping[str, Any]:
        return {key: value for key, value in to_primitive(self).items() if value is not None}


@dataclass(frozen=True)
class Turn:
    objective: str
    fixed_scope: Mapping[str, Any]
    open_requirements: tuple[str, ...]
    latest_observation: Mapping[str, Any] | None
    evidence: tuple[Mapping[str, Any], ...]
    science: Mapping[str, Any] | None
    choices: tuple[ChoiceSpec, ...]
    remaining_budget: Mapping[str, int]
    state_revision: int
    correction: ProtocolCorrection | None = None
    task_context: Mapping[str, Any] | None = None
    catalog_overview: Mapping[str, Any] | None = None
    catalog_navigation: Mapping[str, Any] | None = None
    notebook: Mapping[str, Any] | None = None
    hidden_counts: Mapping[str, int] = field(default_factory=dict)
    hidden_refs: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    source_coverage: SourceCoverageTurnView | None = None
    recent_history: tuple[Mapping[str, Any], ...] = ()
    records_index: Mapping[str, Any] | None = None
    maintenance_status: Mapping[str, Any] | None = None
    working_state_mode: str = "single"
    notebook_identity: str | None = None
    update_limits: Mapping[str, Any] | None = None
    previous_update_result: Mapping[str, Any] | None = None
    pinned_notes: tuple[Mapping[str, Any], ...] = ()
    materials: tuple[Mapping[str, Any], ...] = ()
    material_attachments: tuple[Mapping[str, Any], ...] = ()
    work_facts: Mapping[str, Any] | None = None
    last_work_result: Mapping[str, Any] | None = None
    schema_version: str = EXECUTABLE_LOOP_SCHEMA_VERSION
    context_limits: ContextLimits | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.objective or self.state_revision < 0:
            raise ValueError("turn objective and non-negative revision are required")
        if not self.choices or any(not item.choice for item in self.choices):
            raise ValueError("turn requires complete bound executable choices")
        handles = tuple(item.choice for item in self.choices)
        keys = tuple(item.semantic_key for item in self.choices)
        if len(handles) != len(set(handles)) or len(keys) != len(set(keys)):
            raise ValueError("turn choices and semantic keys must be unique")

    @property
    def identity(self) -> str:
        return public_payload_identity(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> Mapping[str, Any]:
        value: dict[str, Any] = {
            "objective": self.objective,
            "fixed_scope": to_primitive(self.fixed_scope),
            "open_requirements": self.open_requirements,
            "latest_observation": self.latest_observation,
            "evidence": self.evidence,
            "choices": tuple(item.to_dict() for item in self.choices),
            **({"remaining_budget": dict(self.remaining_budget)} if self.remaining_budget else {}),
            "state_revision": self.state_revision,
            "wire": {
                "choice": "opaque-current-operation",
                "arguments": {},
                "record_delta": "optional",
            },
            "schema_version": self.schema_version,
        }
        if self.working_state_mode == "single":
            value["wire"] = {"choice": "opaque-current-operation", "arguments": {}, "notebook_update": "optional"}
            value.update(working_state_mode="single", notebook_identity=self.notebook_identity,
                update_limits=self.update_limits, previous_update_result=self.previous_update_result,
                response_limits={"response_bytes": MAX_RESPONSE_BYTES, "arguments_bytes": MAX_ARGUMENT_BYTES},
                materials=to_primitive(self.materials), material_attachments=to_primitive(self.material_attachments),
                pinned_notes=to_primitive(self.pinned_notes))
        report_choices = {item.choice: {"response_bytes": None, "arguments_bytes": None}
            for item in self.choices if item.semantic_key == "final:submit-answer"
            and item.arguments_schema.get("x-report-format") == "report-object-v1"}
        if report_choices and "response_limits" in value:
            value["response_limits"]["choice_overrides"] = report_choices
        if self.working_state_mode == "maintained":
            value["wire"].pop("record_delta")
            value["working_state_mode"] = "maintained"
            if self.maintenance_status is not None:
                value["maintenance_status"] = dict(self.maintenance_status)
        if self.correction is not None:
            value["correction"] = self.correction.to_dict()
        if self.task_context is not None:
            value["task_context"] = to_primitive(self.task_context)
        if self.catalog_overview is not None:
            value["catalog_overview"] = to_primitive(self.catalog_overview)
        if self.catalog_navigation is not None:
            value["catalog_navigation"] = to_primitive(self.catalog_navigation)
        if self.science is not None:
            value["science"] = to_primitive(self.science)
        if self.notebook is not None:
            value["notebook"] = to_primitive(self.notebook)
        if self.hidden_counts:
            value["hidden_counts"] = dict(self.hidden_counts)
        if self.hidden_refs:
            value["hidden_refs"] = to_primitive(self.hidden_refs)
        if self.source_coverage is not None:
            value["source_coverage"] = self.source_coverage.to_dict()
        if self.recent_history:
            value["recent_history"] = to_primitive(self.recent_history)
        if self.records_index is not None:
            value["records_index"] = to_primitive(self.records_index)
        if self.materials:
            value["materials"] = to_primitive(self.materials)
        if self.material_attachments:
            value["material_attachments"] = to_primitive(self.material_attachments)
        if self.last_work_result is not None:
            value["last_work_result"] = to_primitive(self.last_work_result)
        if self.work_facts is not None:
            value["work_facts"] = to_primitive(self.work_facts)
        if include_identity:
            value["turn_identity"] = self.identity
        return value


class AdapterProtocolError(RuntimeError):
    """A correctable adapter failure before a typed Submission exists."""

    def __init__(
        self,
        code: str,
        field_path: str,
        instruction: str,
        *,
        attempt_ref: str | None = None,
        update_issue: Mapping[str, Any] | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        if not code or not field_path or not instruction:
            raise ValueError("adapter protocol issue fields must be non-empty")
        super().__init__(code)
        self.code = code
        self.field_path = field_path
        self.instruction = instruction
        self.attempt_ref = attempt_ref
        self.update_issue = update_issue
        self.details = details


class DecisionAdapter(Protocol):
    def choose(
        self,
        turn: Turn,
        *,
        record_attempt: Callable[[LLMAttemptRef], None],
    ) -> Submission: ...



@dataclass(frozen=True)
class EpisodeResult:
    status: str
    observations: tuple[OperationObservation, ...]
    accounting: Mapping[str, Any]
    state_identity: str
    state_snapshot: Mapping[str, Any]
    ledger_snapshot: Mapping[str, Any]
    output: TypedOutput | None = None
    control_snapshot: Mapping[str, Any] = field(default_factory=dict)
    navigation_snapshot: Mapping[str, Any] = field(default_factory=dict)
    terminal_code: str | None = None
    schema_version: str = EXECUTABLE_LOOP_SCHEMA_VERSION

    @property
    def completed(self) -> bool:
        return self.status == "completed"


@dataclass(frozen=True)
class _SubmissionIssue:
    code: str
    field_path: str
    instruction: str
    adapter_classification: str | None = None
    system_defect: bool = False
    transport_retryable: bool = False
    provider_failure: Mapping[str, Any] | None = None
    shared_provider_failure: Exception | None = None
    update_issue: Mapping[str, Any] | None = None
    recovery_kind: str = "repair_fields"
    details: Mapping[str, Any] | None = None

    def correction(self) -> ProtocolCorrection:
        from .action_diagnostics import copy_text
        instruction = (copy_text('correction', self.details['error_kind'])
            if self.details and self.details.get('diagnostic_version') == 'action-correction-v1'
            and self.code not in {'invalid_public_source_request','arguments_contract_mismatch',
                'action_operation_shape_invalid','arguments_not_object'}
            else self.instruction)
        return ProtocolCorrection(self.code, self.field_path, instruction,
            recovery_kind=self.recovery_kind, details=self.details)



@dataclass
class _LoopState:
    resources: _EpisodeResources
    revision: int = 0
    c1_open_control: bool = False
    observations: list[OperationObservation] = field(default_factory=list)
    started_monotonic: float = field(default_factory=monotonic)
    maintained_turn: Turn | None = None

    @property
    def task(self) -> RuntimeTask:
        return self.resources.state.task

    @property
    def accounting(self) -> dict[str, Any]:
        return self.resources.state.accounting

    def operation_context(self) -> OperationContext:
        return OperationContext(
            task=self.task,
            state_revision=self.revision,
            observations=tuple(self.observations),
            evidence=self.resources.ledger.project(),
            terminal_output=self.resources.state.final_output,
            resources=self.resources,
            c1_open_control=self.c1_open_control,
            deadline=self.task.budget.deadline(self.started_monotonic),
        )

    def execution_context(self) -> OperationExecutionContext:
        return OperationExecutionContext(
            task=self.task,
            state_revision=self.revision,
            observations=tuple(self.observations),
            evidence=self.resources.ledger.project(),
            resources=self.resources,
            c1_open_control=self.c1_open_control,
            deadline=self.task.budget.deadline(self.started_monotonic),
        )

    def commit(self, result: TransitionResult) -> None:
        canonical_output = self.resources.state.final_output
        if result.terminal_output != canonical_output:
            raise RuntimeError(
                "transition terminal output differs from canonical RunState"
            )
        self.observations.append(result.observation)
        self.revision += 1

    @property
    def state_identity(self) -> str:
        return public_payload_identity(
            {
                **({"memory": self.resources.memory_state.snapshot()} if self.resources.memory_state is not None else {}),
                "state_revision": self.revision,
                "run_state": self.resources.state.snapshot(),
                "ledger": self.resources.ledger.snapshot(),
                "observations": tuple(item.to_dict() for item in self.observations),
                "control": (
                    {}
                    if self.resources.control_records is None
                    else self.resources.control_records.snapshot()
                ),
                "catalog_navigation": (
                    {}
                    if self.resources.catalog_navigation is None
                    else self.resources.catalog_navigation.snapshot()
                ),
            }
        )


EpisodeResourcesFactory = Callable[[RuntimeTask], _EpisodeResources]


def _phase_remaining(state):
    result={}
    config=state.accounting.get('working_state',{})
    for phase in ('action','maintenance'):
        limit=config.get('max_'+phase+'_calls')
        if limit is not None:result[phase+'_calls']=max(0,limit-int(state.accounting.get(phase+'_calls',0)))
    return result


def _control_remaining(state):
    result={}
    for public,name,counter in [('turns','max_iterations','turns'),('adapter_calls','max_llm_calls','adapter_calls'),('actions','max_actions','actions')]:
        remaining=state.task.budget.remaining(name,int(state.accounting.get(counter,0)))
        if remaining is not None:result[public]=remaining
    result.update(_phase_remaining(state))
    return result


def _maintenance_allowed(state):
    config=state.accounting['working_state']
    for phase in ('action','maintenance'):
        cap=config['max_'+phase+'_calls']
        if cap is not None and int(state.accounting.get(phase+'_calls',0))>=cap:return False
    left=state.task.budget.remaining('max_llm_calls',int(state.accounting.get('adapter_calls',0)))
    return (left is None or left>1) and not state.task.budget.timed_out(monotonic()-state.started_monotonic)


class EpisodeRunner:
    """The sole orchestration authority for the executable-operation loop."""

    def __init__(
        self,
        providers: Sequence[OperationProvider],
        *,
        max_turn_characters: int = 32_000,
        max_protocol_corrections: int = 2,
        max_transport_retries: int = 2,
        resources_factory: EpisodeResourcesFactory | None = None,
        working_state: WorkingStateConfig | None = None,
        maintenance_adapter: Any = None,
        source_coverage_descriptors: tuple[CapabilityDescriptor, ...] = (),
        source_coverage_owner_outcomes: tuple[SourceOwnerOutcome, ...] = (),
    ) -> None:
        if not providers:
            raise ValueError("EpisodeRunner requires at least one injected provider")
        if max_turn_characters <= 0:
            raise ValueError("max_turn_characters must be positive")
        if (
            type(max_protocol_corrections) is not int
            or max_protocol_corrections not in {0, 1, 2, 3, 4}
        ):
            raise ValueError("max_protocol_corrections must be an integer from zero through four")
        if isinstance(max_transport_retries, bool) or max_transport_retries not in {0, 1, 2}:
            raise ValueError("max_transport_retries must be zero, one, or two")
        self._max_transport_retries = max_transport_retries
        descriptor_refs = tuple(
            descriptor.capability_ref for descriptor in source_coverage_descriptors
        )
        if len(descriptor_refs) != len(set(descriptor_refs)):
            raise ValueError("source coverage descriptors must have unique refs")
        self._working_state = working_state or WorkingStateConfig()
        self._maintenance_adapter = maintenance_adapter
        if (self._working_state.mode == "maintained") != (maintenance_adapter is not None):
            raise ValueError("maintained mode requires exactly one maintenance adapter")
        self._providers = tuple(providers)
        self._max_turn_characters = max_turn_characters
        self._max_protocol_corrections = max_protocol_corrections
        self._resources_factory = resources_factory or _new_episode_resources
        self._source_coverage_descriptors = tuple(source_coverage_descriptors)
        self._source_coverage_owner_outcomes = tuple(
            source_coverage_owner_outcomes
        )

    def run(
        self,
        task: RuntimeTask,
        adapter: DecisionAdapter,
        *,
        c1_open_control: bool = False,
        resolved_retrieval_scope: ResolvedRetrievalScope | None = None,
    ) -> EpisodeResult:
        if self._working_state.mode == "maintained" and not c1_open_control:
            raise ValueError("maintenance requires the public notebook loop")
        if self._working_state.mode == "maintained":
            from .working_state import MAINTENANCE_PROMPT_VERSION, MAINTAINED_ACTION_PROMPT_VERSION
            action_identity = getattr(getattr(adapter, "hosted", adapter), "prompt_version", None)
            maintenance_identity = getattr(self._maintenance_adapter, "prompt_version", None)
            if action_identity not in {None, MAINTAINED_ACTION_PROMPT_VERSION} or maintenance_identity not in {None, MAINTENANCE_PROMPT_VERSION}:
                raise ValueError("maintained phase prompt identity mismatch")
        if c1_open_control and self._working_state.mode == "single":
            from .working_state import SINGLE_ACTION_PROMPT_VERSIONS
            action_identity = getattr(getattr(adapter, "hosted", adapter), "prompt_version", None)
            if action_identity not in {None, *SINGLE_ACTION_PROMPT_VERSIONS}:
                raise ValueError("single hosted prompt must identify scientific-openness-single-action-v6")
        resources = self._resources_factory(task)
        if resolved_retrieval_scope is not None:
            if (
                resources.resolved_retrieval_scope is not None
                and resources.resolved_retrieval_scope != resolved_retrieval_scope
            ):
                raise ValueError("episode retrieval scope sources disagree")
            resources.resolved_retrieval_scope = resolved_retrieval_scope
        if c1_open_control:
            if resources.control_records is None:
                resources.control_records = C1ControlRecords()
            resources.control_records.ensure_durable(task.task_identity)
        resources.knowledge_delivery_required = True
        resources.knowledge_pending_delivery = {}
        resources.working_state_config = self._working_state
        state = _LoopState(resources, c1_open_control=c1_open_control)
        if c1_open_control:
            state.accounting["working_state"] = {"mode": self._working_state.mode,
                "max_action_calls": self._working_state.max_action_calls,
                "max_maintenance_calls": self._working_state.max_maintenance_calls,
                "max_total_calls": task.budget.max_llm_calls,
                **{key: getattr(self._working_state, key) for key in ("notebook_bytes", "note_characters", "update_bytes", "plan_characters", "max_changes", "max_basis_per_note", "maintenance_max_tokens")}}

        if state.task.task_identity != task.task_identity:
            raise ValueError("episode resources belong to a different RuntimeTask")
        try:
            scope = resources.resolved_retrieval_scope
            if task.resolved_scope_identity is None:
                if scope is not None:
                    raise ValueError("unscoped task received a resolved retrieval scope")
            elif (
                not isinstance(scope, ResolvedRetrievalScope)
                or scope.scope_ref != task.retrieval_scope_ref
                or scope.resolved_scope_identity != task.resolved_scope_identity
            ):
                raise ValueError("RuntimeTask retrieval scope content is unavailable or differs")
            if c1_open_control:
                final_policy = getattr(resources, 'evaluation_final_policy', 'first-explicit-final-v1')
                state.resources.control_records.record_event(
                    "episode_started",
                    {"task_identity": task.task_identity, "question": task.question,
                     "evaluation_audit_version": ('evaluation-host-facts-memory-v1'
                         if final_policy == 'committed-after-memory-v1' else 'evaluation-host-facts-v1'),
                     **({'evaluation_final_policy': final_policy}
                        if final_policy == 'committed-after-memory-v1' else {}),
                     "budget_seconds": task.budget.timeout_seconds},
                )
            preflight = state.resources.validation_gate.validate_preflight(
                task, state.resources.state.manifest
            )
            state.resources.state.record_validation(preflight.to_dict())
        except Exception as error:
            raise_control_error(error)
            from .runtime_errors import AuditWriteError
            if isinstance(error, AuditWriteError): raise
            if state.resources.control_records is not None:
                state.resources.control_records.record_event("operation_preflight_failed", error_details(error, "preflight"))
            return self._terminal(
                state, "system_defect", f"preflight_{type(error).__name__}"
            )
        if preflight.status.value != "valid":
            return self._terminal(
                state, "validation_failed", "preflight_validation_failed"
            )

        if c1_open_control and resources.memory_state is not None:
            memory = resources.memory_state
            deadlines = [value for value in (task.budget.deadline(state.started_monotonic),
                memory.host_deadline) if value is not None]
            memory.set_deadline(min(deadlines) if deadlines else None)
            memory.initialize(resources, task.question)

        while state.resources.state.final_output is None:
            if getattr(state.resources, "host_control", None) is not None:state.resources.host_control.check()
            if task.answer_contract.report_format == "report-object-v1" and getattr(resources,"report_delivery",None) is not None:
                return self._terminal(state,"report_invalid",state.resources.report_delivery.get("terminal_code") or "report_object_invalid")
            action_limit=state.accounting.get('working_state',{}).get('max_action_calls')
            if action_limit is not None and int(state.accounting.get('action_calls',0))>=action_limit:
                return self._terminal(state,'budget_exhausted','action_call_budget_exhausted')
            if not task.budget.allows("max_iterations", int(state.accounting.get("turns", 0))):
                return self._terminal(state, "budget_exhausted", "turn_budget_exhausted")
            try:
                current = self._materialize(state)
            except Exception as error:
                raise_control_error(error)
                from .runtime_errors import AuditWriteError
                if isinstance(error, AuditWriteError): raise
                if state.resources.control_records is not None:
                    state.resources.control_records.record_event('operation_registration_failed',
                        error_details(error, 'materialize'))
                return self._terminal(
                    state, "system_defect", f"registration_{type(error).__name__}"
                )
            if not current:
                if c1_open_control and not task.budget.allows("max_actions", int(state.accounting.get("actions", 0))):
                    return self._terminal(
                        state, "budget_exhausted", "action_budget_exhausted"
                    )
                return self._terminal(
                    state, "no_executable_operation", "no_executable_operation"
                )
            state.accounting["turns"] = int(state.accounting.get("turns", 0)) + 1
            from .working_state_delivery import WorkingStateDeliveryUnavailable
            try:
                issue = self._maintain(state, current) if self._working_state.mode == "maintained" else None
                if issue is None:
                    selected, arguments, issue = self._select(state, adapter, current)
                else:
                    selected, arguments = None, None
            except (WorkingStateDeliveryUnavailable, WorkingStateInputUnavailable) as error:
                state.resources.control_records.record_event("input_delivery_failed", {
                    "code": str(error), "issue": error.issue,
                    "notebook_identity": state.resources.control_records.notebook_identity,
                    "pending_note_refs": state.accounting.get("pending_note_refs", ())})
                return self._terminal(state, "system_defect", "input_delivery_failure"
                    if self._working_state.mode == "single" else "working_state_projection_unavailable")
            except Exception as error:
                raise_control_error(error)
                from .runtime_errors import AuditWriteError, SharedProviderFailure
                if isinstance(error, (AuditWriteError, SharedProviderFailure)): raise
                if state.resources.control_records is not None:
                    state.resources.control_records.record_event('preinvocation_failed',
                        error_details(error, 'select_public_action'))
                return self._terminal(
                    state, "system_defect", f"preinvocation_audit_{type(error).__name__}"
                )
            if issue is not None:
                status = (
                    "system_defect"
                    if issue.system_defect
                    else "turn_too_large"
                    if issue.code == "turn_too_large"
                    else "budget_exhausted" if issue.code in {"adapter_budget_exhausted","episode_timeout"}
                    else "provider_request_failed" if issue.provider_failure is not None
                    else "protocol_failure"
                )
                return self._terminal(state, status, issue.code)
            assert selected is not None
            actions_before = int(state.accounting.get("actions", 0))
            try:
                if c1_open_control and selected.action_budget_cost:
                    if not task.budget.allows("max_actions", actions_before, selected.action_budget_cost):
                        return self._terminal(
                            state, "budget_exhausted", "action_budget_exhausted"
                        )
                    if not selected.manages_action_accounting:
                        state.accounting["actions"] = (
                            actions_before + selected.action_budget_cost
                        )
                if c1_open_control:
                    state.resources.control_records.record_event(
                        "invocation_started",
                        {
                            "choice": selected.public.choice,
                            "semantic_key": selected.public.semantic_key,
                            "state_revision": state.revision,
                            "action_budget_cost": selected.action_budget_cost,
                            "actions_before": actions_before,
                            "actions_reserved": int(
                                state.accounting.get("actions", 0)
                            ),
                        },
                    )
                state.accounting["operation_invocations"] = int(
                    state.accounting.get("operation_invocations", 0)
                ) + 1
                if getattr(state.resources, "host_control", None) is not None:state.resources.host_control.admit("action")
                result = selected.invoke(state.execution_context(), arguments)
                if c1_open_control and selected.action_budget_cost:
                    actions_after = int(state.accounting.get("actions", 0))
                    expected_actions = actions_before + selected.action_budget_cost
                    if selected.manages_action_accounting and actions_after == actions_before:
                        state.accounting["actions"] = expected_actions
                    elif actions_after != expected_actions:
                        raise RuntimeError("operation action accounting diverged from its contract")
                if not isinstance(result, TransitionResult):
                    raise TypeError("operation must return TransitionResult")
                state.commit(result)
                if (c1_open_control and selected.public.semantic_key == 'final:submit-answer'
                        and getattr(resources, 'evaluation_final_policy',
                                    'first-explicit-final-v1') == 'committed-after-memory-v1'
                        and getattr(resources, 'report_delivery', None) is not None):
                    resources.control_records.record_event('memory_report_committed', {
                        'attempt_ref': state.accounting.get('last_invocation_attempt_ref'),
                        'scored_attempt_ref': state.accounting.get('evaluation_final_intent_ref'),
                        'report_delivery_status': resources.report_delivery.get('status'),
                        'report_delivery_identity': resources.report_delivery.get('identity')})
                if c1_open_control:
                    state.resources.control_records.record_event(
                        "operation_observation",
                        {
                            "state_revision": state.revision,
                            "observation": result.observation.to_dict(),
                        },
                    )
            except Exception as error:
                raise_control_error(error)
                from .runtime_errors import AuditWriteError
                if isinstance(error, AuditWriteError): raise
                if c1_open_control and selected.action_budget_cost:
                    actions_after = int(state.accounting.get("actions", 0))
                    expected_actions = actions_before + selected.action_budget_cost
                    if actions_after == actions_before:
                        state.accounting["actions"] = expected_actions
                code = f"invocation_{type(error).__name__}"
                if c1_open_control:
                    failure = OperationObservation(
                        status="failed",
                        code=code,
                        summary=error_details(error, selected.public.semantic_key)["feedback"],
                    )
                    state.observations.append(failure)
                    try:
                        state.resources.control_records.record_event(
                            "operation_execution_failed",
                            {
                                "state_revision": state.revision,
                                "observation": failure.to_dict(),
                                **error_details(error, selected.public.semantic_key),
                            },
                        )
                    except Exception as audit_error:
                        code = f"postinvocation_audit_{type(audit_error).__name__}"
                from .runtime_errors import SharedComputeCleanupUncertain
                if isinstance(error, SharedComputeCleanupUncertain):
                    raise
                return self._terminal(state, "system_defect", code)
        return self._result(state, "completed", output=state.resources.state.final_output)


    def _maintain(self, state, current):
        from .working_state_delivery import (pack_maintained_action, fallback_maintained_action,
            WorkingStateDeliveryUnavailable)
        from .record_inspection import material_location_available
        from .c1_records import DeltaDisposition, _DeltaIssue
        from copy import deepcopy
        control = state.resources.control_records
        previous = deepcopy(state.accounting.get("maintenance_status"))
        state.maintained_turn = None
        foundation = self._render_turn(state, current,
            source_coverage_descriptors=self._source_coverage_descriptors,
            source_coverage_owner_outcomes=self._source_coverage_owner_outcomes)
        before = control.notebook_identity
        started = monotonic()
        view, attempt_ref = None, None
        disposition = {"attempt_ref": None, "status": "skipped", "code": "maintenance_budget_exhausted",
            "applied_refs": [], "closed_refs": [], "issue": None}
        try:
            view = working_state_view(foundation, state.resources, config=self._working_state,
                previous_update_result=previous)
        except _DeltaIssue as error:
            disposition = DeltaDisposition("rejected", str(error), issue=error.issue).public_result()
            state.maintained_turn = fallback_maintained_action(foundation,
                remaining_budget=_control_remaining(state), status=disposition, config=self._working_state)
        else:
            if _maintenance_allowed(state):
                transport_retries = 0
                while True:
                    update, issue, attempt_ref = self._choose(self._maintenance_adapter, view, state, phase="maintenance")
                    if (issue is None or not issue.transport_retryable or transport_retries >= self._max_transport_retries
                            or not _maintenance_allowed(state)):
                        break
                    transport_retries += 1
                if issue is not None:
                    if issue.provider_failure and issue.provider_failure.get("provider_type") == "finish_reason":
                        state.accounting["provider_failure"] = issue.provider_failure
                        return issue
                    error = _DeltaIssue(issue.code, field_path="$",
                        kind="transport" if issue.field_path == "provider" else "parse",
                        recovery="No update was applied. Use the current input at the next natural maintenance opportunity.")
                    disposition = DeltaDisposition("rejected", issue.code, issue=issue.update_issue or error.issue).public_result(attempt_ref)
                else:
                    prior_basis = {canonical_json(location) for note in control.current_view(limit=len(control._records))['active_records']
                        for location in note.get('basis', ())}
                    materials = {handle: location for handle, location in view.materials.items()
                        if canonical_json(location) in prior_basis or material_location_available(state.resources, location)}
                    remaining = _control_remaining(state)
                    def prove_delivery(candidate, changed):
                        payload = candidate._trajectory[-1]['payload']
                        status = DeltaDisposition('applied', 'notebook_local_update_applied',
                            tuple(payload['applied_refs']), tuple(payload['closed_refs'])).public_result(attempt_ref)
                        state.maintained_turn = pack_maintained_action(foundation, candidate, view,
                            preferred_refs=changed, remaining_budget=remaining, status=status)
                    result = control.apply_local_update(update, expected_identity=view.document["notebook_identity"],
                        visible_notes=view.visible_notes, materials=materials,
                        attempt_ref=attempt_ref, delivery_check=prove_delivery, config=self._working_state,
                        unavailable_materials={handle for handle, location in materials.items()
                            if not material_location_available(state.resources, location)})
                    disposition = result.public_result(attempt_ref)
                    if disposition['status'] == 'applied':
                        from .record_inspection import publish_current_turn_record_refs
                        publish_current_turn_record_refs(state.resources, latest_observation=None,
                            evidence=(), notebook=None, catalog_navigation=None)
            if state.maintained_turn is None:
                remaining = _control_remaining(state)
                try:
                    state.maintained_turn = pack_maintained_action(foundation, control, view,
                        remaining_budget=remaining, status=disposition)
                except WorkingStateDeliveryUnavailable as error:
                    disposition = DeltaDisposition('rejected', str(error), issue=error.issue).public_result(attempt_ref)
                    state.maintained_turn = fallback_maintained_action(foundation,
                        remaining_budget=remaining, status=disposition, config=self._working_state)
        state.accounting["maintenance_status"] = disposition
        elapsed = monotonic() - started
        state.accounting["maintenance_seconds"] = state.accounting.get("maintenance_seconds", 0) + elapsed
        if disposition["status"] == "rejected":
            state.accounting["maintenance_rejections"] = state.accounting.get("maintenance_rejections", 0) + 1
        control.record_event("maintenance_completed", {"phase": "maintenance",
            "decision_index": state.accounting["turns"], "turn_identity": view.identity if view else None,
            "attempt_ref": attempt_ref, "disposition": disposition,
            "notebook_before": before, "notebook_after": control.notebook_identity,
            "fully_shown_note_refs": view.close_refs if view else (), "visible_evidence_refs": view.evidence_refs if view else (),
            "elapsed_seconds": elapsed})

    @staticmethod
    def _skip_single_update(state, issue, attempt_ref):
        from .c1_records import DeltaDisposition, _DeltaIssue
        detail = issue.update_issue or _DeltaIssue(issue.code, field_path=issue.field_path,
            kind="transport" if issue.field_path == "provider" else "parse",
            recovery=issue.instruction).issue
        result = DeltaDisposition("skipped", issue.code, issue=detail).public_result(attempt_ref)
        state.resources.control_records.record_event("single_update_completed", {
            "phase": "action", "attempt_ref": attempt_ref, "disposition": result})
        state.accounting["previous_update_result"] = result

    def _apply_single_update(self, state, turn, view, submission, attempt_ref):
        from .c1_records import DeltaDisposition
        from .working_state_delivery import pack_working_action
        from .record_inspection import unavailable_material_handles
        control = state.resources.control_records
        before = control.notebook_identity
        started = monotonic()
        authorization_started = monotonic()
        unavailable = unavailable_material_handles(state.resources, view.materials)
        state.accounting["update_authorization_seconds"] = state.accounting.get("update_authorization_seconds", 0) + monotonic() - authorization_started
        def prove_delivery(candidate, changed):
            payload = candidate._trajectory[-1]['payload']
            status = DeltaDisposition('applied', 'notebook_local_update_applied',
                tuple(payload['applied_refs']), tuple(payload['closed_refs'])).public_result(attempt_ref)
            pack_working_action(turn, candidate, view, preferred_refs=changed, status=status)
        raw = {} if submission.notebook_update is _OMITTED_UPDATE else submission.notebook_update
        result = control.apply_local_update(raw, expected_identity=view.document['notebook_identity'],
            visible_notes=view.visible_notes, materials=view.materials, unavailable_materials=unavailable,
            attempt_ref=attempt_ref, delivery_check=prove_delivery, config=self._working_state)
        disposition = result.public_result(attempt_ref)
        pending = tuple(ref for ref in state.accounting.get('pending_note_refs', ())
            if ref not in result.closed_refs)
        state.accounting['pending_note_refs'] = tuple(dict.fromkeys((*pending, *result.applied_refs)))
        state.accounting['previous_update_result'] = disposition
        elapsed = monotonic() - started
        state.accounting['notebook_update_seconds'] = state.accounting.get('notebook_update_seconds', 0) + elapsed
        control.record_event('single_update_completed', {'phase': 'action', 'attempt_ref': attempt_ref,
            'turn_identity': turn.identity, 'disposition': disposition, 'notebook_before': before,
            'notebook_after': control.notebook_identity, 'pending_note_refs': state.accounting['pending_note_refs'],
            'elapsed_seconds': elapsed})
        return disposition

    def _select(
        self,
        state: _LoopState,
        adapter: DecisionAdapter,
        current: Mapping[str, ExecutableOperation],
    ) -> tuple[ExecutableOperation | None, object | None, _SubmissionIssue | None]:
        correction: ProtocolCorrection | None = None
        corrections_used = 0
        transport_retries_used = 0
        generation_retries_used = 0
        physical_requests = 0
        t3_retry = getattr(state.resources, 't3_t4_node', None) is not None
        correction_limit = self._max_protocol_corrections if state.c1_open_control else 1
        physical_request_limit = 1 + correction_limit + 2 + 4
        reuse_turn = False
        turn = None
        def commit_proposal(proposal):
            if proposal is None or state.accounting.get('evaluation_final_intent_ref') is not None:
                return
            state.resources.control_records.record_event('evaluation_final_intent', proposal)
            state.accounting['evaluation_final_intent_ref'] = proposal['attempt_ref']
            memory = state.resources.memory_state
            if memory is not None and memory.pre_final is not None:
                memory.pre_final.submission_attempt_ref = proposal['attempt_ref']
        while True:
            if state.task.budget.timed_out(monotonic() - state.started_monotonic):
                return None, None, _SubmissionIssue("episode_timeout", "provider", "The episode time budget is exhausted.")
            if not reuse_turn:
                if state.maintained_turn is not None:
                    turn = state.maintained_turn
                    if correction is not None:
                        from .working_state_delivery import bounded_correction
                        remaining = _control_remaining(state)
                        turn = replace(turn, correction=bounded_correction(correction), remaining_budget=remaining)
                else:
                    render_started = monotonic()
                    turn = self._render_turn(
                        state, current, correction=correction,
                        source_coverage_descriptors=self._source_coverage_descriptors,
                        source_coverage_owner_outcomes=self._source_coverage_owner_outcomes)
                    state.accounting["render_turn_seconds"] = state.accounting.get("render_turn_seconds", 0) + monotonic() - render_started
                if state.c1_open_control and self._working_state.mode == "single":
                    from .working_state import freeze_action_update_context
                    from .working_state_delivery import pack_working_action
                    materials_started = monotonic()
                    view = working_state_view(turn, state.resources, config=self._working_state,
                        previous_update_result=state.accounting.get("previous_update_result"),
                        preferred_refs=state.accounting.get("pending_note_refs", ()))
                    state.accounting["working_materials_seconds"] = state.accounting.get("working_materials_seconds", 0) + monotonic() - materials_started
                    packing_started = monotonic()
                    turn = pack_working_action(turn, state.resources.control_records, view,
                        preferred_refs=state.accounting.get("pending_note_refs", ()),
                        status=state.accounting.get("previous_update_result"), correction=correction)
                    update_context = freeze_action_update_context(turn, view)
                    state.accounting["final_packing_seconds"] = state.accounting.get("final_packing_seconds", 0) + monotonic() - packing_started
            assert turn is not None
            reuse_turn = False
            turn_size = json_bytes(turn.to_dict())
            turn_limit = (
                state.resources.context_limits.turn_bytes
                if state.c1_open_control
                else self._max_turn_characters
            )
            if turn_size > turn_limit:
                return None, None, _SubmissionIssue(
                    "turn_too_large", "turn", "The complete current Turn exceeds its budget."
                )
            if state.c1_open_control and turn.source_coverage is not None:
                state.resources.control_records.record_event(
                    "source_coverage_snapshot",
                    {
                        "snapshot_identity": turn.source_coverage.snapshot_identity,
                        "projection_identity": turn.source_coverage.view_identity,
                        "projection_bytes": json_bytes(
                            turn.source_coverage.to_dict()
                        ),
                    },
                )
            physical_requests += 1
            submission, issue, attempt_ref = self._choose(adapter, turn, state)
            proposal = state.accounting.pop('pending_final_proposal', None)
            if issue is not None or submission is None:
                commit_proposal(proposal)
            if issue is not None and issue.field_path == "provider":
                failure = issue.provider_failure or {}
                generation = failure.get('retry_class') == 'generation'
                used = generation_retries_used if generation else transport_retries_used
                limit = (2 if generation else 4) if t3_retry else self._max_transport_retries
                if (not issue.transport_retryable or used >= limit or (t3_retry and physical_requests >= physical_request_limit)
                        or not state.task.budget.allows('max_llm_calls', int(state.accounting.get('adapter_calls', 0)))):
                    if failure:
                        code = ('provider_generation_exhausted' if generation else 'provider_transport_exhausted') if issue.transport_retryable else failure['code']
                        issue = replace(issue, code=code, provider_failure=dict(failure, code=code, failure_scope='node'))
                        state.accounting['provider_failure'] = issue.provider_failure
                    return None, None, issue
                if generation:
                    generation_retries_used += 1
                    # Short cancellable generation delay never changes account cooldown.
                    import time
                    end = monotonic() + generation_retries_used
                    while monotonic() < end:
                        control = getattr(state.resources, 'host_control', None)
                        if control is not None: control.check()
                        if state.task.budget.timed_out(monotonic() - state.started_monotonic):
                            return None, None, _SubmissionIssue('episode_timeout', 'provider', 'The node deadline expired.')
                        time.sleep(min(.05, max(0, end - monotonic())))
                else:
                    transport_retries_used += 1
                if state.c1_open_control:
                    state.resources.control_records.record_event('provider_request_retry', {
                        'attempt_ref': attempt_ref, 'turn_identity': turn.identity,
                        'retry_class': failure.get('retry_class', 'transport'),
                        'transport_retries_used': transport_retries_used,
                        'generation_retries_used': generation_retries_used,
                        'protocol_corrections_used': corrections_used, 'physical_requests': physical_requests})
                reuse_turn = True
                continue
            if not t3_retry: transport_retries_used = 0
            selected: ExecutableOperation | None = None
            arguments: object | None = None
            if state.c1_open_control and submission is not None:
                proposed = current.get(submission.choice)
                report_final = (state.task.answer_contract.report_format == "report-object-v1"
                    and proposed is not None and proposed.public.semantic_key == "final:submit-answer")
                if report_final:
                    from .t3_t4.report_intent import audit_intent
                    audited_arguments,arguments_size,response_size=audit_intent(submission)
                else:
                    audited_arguments=to_primitive(submission.arguments)
                    arguments_size=json_bytes(submission.arguments)
                    response_size=json_bytes(submission.to_dict())
                state.resources.control_records.record_event(
                    "response_received",
                    {
                        "turn_identity": turn.identity,
                        "response_bytes": response_size,
                        "llm_attempt_count": len(
                            state.resources.state.llm_attempt_refs
                        ),
                    },
                )
                from .action_diagnostics import copy_text as recovery_copy
                if not report_final and response_size > MAX_RESPONSE_BYTES:
                    issue = _SubmissionIssue(
                        "response_oversize",
                        "submission",
                        recovery_copy("correction","response_too_large"),
                        details={"diagnostic_version":"action-correction-v1","error_kind":"response_too_large","actual_bytes":response_size,"allowed_bytes":MAX_RESPONSE_BYTES},
                    )
                elif not report_final and json_bytes(submission.arguments) > MAX_ARGUMENT_BYTES:
                    issue = _SubmissionIssue(
                        "arguments_oversize",
                        "arguments",
                        recovery_copy("correction","arguments_too_large"),
                        details={"diagnostic_version":"action-correction-v1","error_kind":"arguments_too_large","actual_bytes":json_bytes(submission.arguments),"allowed_bytes":MAX_ARGUMENT_BYTES},
                    )
            if (issue is None and submission is not None and state.c1_open_control
                    and self._working_state.mode == "single" and submission.record_delta is not None):
                issue = _SubmissionIssue("legacy_record_delta_forbidden", "record_delta",
                    "Use optional notebook_update with plan_text and changes in single v6.")
            if issue is not None:
                commit_proposal(proposal)
            if issue is None:
                assert submission is not None
                selected = current.get(submission.choice)
                if selected is None:
                    commit_proposal(proposal)
                    issue = _SubmissionIssue(
                        "stale_or_unknown_choice",
                        "choice",
                        "Select one opaque choice from this current Turn.",
                    )
                else:
                    memory = state.resources.memory_state
                    eligible_defer = (state.c1_open_control and proposal is not None and memory is not None
                        and memory.pre_final is not None and memory.pre_final.status == 'pending'
                        and state.accounting.get('evaluation_final_intent_ref') is None
                        and state.task.answer_contract.report_format == 'report-object-v1'
                        and self._working_state.mode == 'single'
                        and selected.public.semantic_key == 'final:submit-answer'
                        and isinstance(submission.arguments, Mapping)
                        and 'report' in submission.arguments)
                    if eligible_defer:
                        next_turn = memory.before_final(state.resources, turn,
                            submission.arguments, attempt_ref,
                            remaining_budget=_control_remaining(state),
                            deadline=state.task.budget.deadline(state.started_monotonic),
                            turn_bytes=turn_limit,
                            response_digest=proposal.get('response_digest'),
                            audited_arguments=audited_arguments)
                        if next_turn is not None:
                            try:
                                selected.argument_contract.validate(submission.arguments)
                                assessment = {'valid': True, 'code': None, 'field_path': None}
                            except ArgumentValidationError as error:
                                assessment = {'valid': False, 'code': error.code,
                                              'field_path': error.field_path}
                            state.resources.control_records.record_event(
                                'memory_deferred_action_assessment',
                                {'attempt_ref': attempt_ref, 'semantic_key': selected.public.semantic_key,
                                 'turn_identity': turn.identity, **assessment})
                            disposition = next_turn.previous_update_result
                            control = state.resources.control_records
                            control.record_event('single_update_completed', to_primitive({
                                'phase': 'action', 'attempt_ref': attempt_ref,
                                'disposition': disposition, 'notebook_before': control.notebook_identity,
                                'notebook_after': control.notebook_identity}))
                            control.record_event('action_deferred', {'attempt_ref': attempt_ref,
                                'turn_identity': turn.identity, 'reason': 'memory_pre_final_deferred'})
                            control.record_event('memory_report_deferred', {
                                'attempt_ref': attempt_ref, 'turn_identity': turn.identity,
                                'read_ref': memory.pre_final.read_ref,
                                'draft_original_sha256': proposal.get('response_digest')})
                            state.accounting['previous_update_result'] = disposition
                            turn = next_turn
                            reuse_turn = True
                            continue
                    commit_proposal(proposal)
                    try:
                        arguments = selected.argument_contract.validate(
                            submission.arguments
                        )
                    except ArgumentValidationError as error:
                        issue = _SubmissionIssue(
                            error.code,
                            error.field_path,
                            str(error),
                            recovery_kind=error.recovery_kind, details=error.details,
                        )
                    except Exception as error:
                        raise_control_error(error)
                        from .runtime_errors import AuditWriteError
                        if isinstance(error, AuditWriteError): raise
                        return None, None, _SubmissionIssue(
                            f"argument_contract_{type(error).__name__}",
                            "arguments",
                            "The operation registration has a broken argument contract.",
                            system_defect=True,
                        )
            if (issue is None and state.c1_open_control and selected is not None
                    and selected.public.semantic_key == "records.inspect"):
                from .record_inspection import directory_no_progress
                repeated = directory_no_progress(state.resources, arguments)
                if repeated is not None:
                    issue = _SubmissionIssue(
                        "directory_no_progress", "arguments.ref",
                        "This directory has already been read without new task information. "
                        "Read an entry body using its reading_arguments, choose a different useful action, "
                        "or submit the report if ready. Do not reread this directory unchanged.",
                        recovery_kind="choose_alternative", details=repeated)
            if issue is None:
                if state.c1_open_control:
                    assert submission is not None and selected is not None
                    state.resources.control_records.record_event(
                        "action_validation",
                        {
                            "status": "valid",
                            "attempt_ref": attempt_ref,
                            "choice": submission.choice,
                            "semantic_key": selected.public.semantic_key,
                            "arguments_bytes": arguments_size,
                        },
                    )
                    if attempt_ref is not None:
                        outcome = LLMProtocolOutcome(
                            attempt_ref=attempt_ref,
                            classification="valid_action",
                            field_path=None,
                            origin="action_validation",
                        )
                        state.resources.control_records.record_event(
                            "llm_protocol_outcome",
                            {"outcome": outcome.to_dict()},
                        )
                    if self._working_state.mode == "maintained":
                        disposition = {"status": "ignored", "code": "ignored_in_maintained_mode"}
                        if submission.record_delta is not None:
                            state.resources.control_records.record_event("action_delta_ignored", {
                                "attempt_ref": attempt_ref, "code": "ignored_in_maintained_mode"})
                    else:
                        disposition = self._apply_single_update(state, turn, update_context,
                            submission, attempt_ref)
                    actions_before = int(state.accounting.get("actions", 0))
                    state.resources.control_records.record_event(
                        "invocation_ready",
                        {
                            "choice": submission.choice,
                            "semantic_key": selected.public.semantic_key,
                            "arguments": audited_arguments,
                            "access_policy_identity": deterministic_identity(state.task.access_policy),
                            "resolved_scope_identity": state.task.resolved_scope_identity,
                            "delta": disposition,
                            "action_budget_cost": selected.action_budget_cost,
                            "actions_before": actions_before,
                            "actions_after_reservation": (
                                actions_before + selected.action_budget_cost
                            ),
                        },
                    )
                state.accounting['last_invocation_attempt_ref'] = attempt_ref
                return selected, arguments, None
            if issue.system_defect:
                return None, None, issue
            if state.c1_open_control:
                if self._working_state.mode == "single":
                    self._skip_single_update(state, issue, attempt_ref)
                state.resources.control_records.record_event(
                    "action_validation",
                    {
                        "status": "rejected",
                        "attempt_ref": attempt_ref,
                        "code": issue.code,
                        "field_path": issue.field_path,
                    },
                )
                if attempt_ref is not None:
                    outcome = LLMProtocolOutcome(
                        attempt_ref=attempt_ref,
                        classification=(
                            issue.adapter_classification
                            if issue.adapter_classification is not None
                            else "invalid_action"
                        ),
                        field_path=issue.field_path,
                        origin=(
                            "adapter"
                            if issue.adapter_classification is not None
                            else "action_validation"
                        ),
                    )
                    state.resources.control_records.record_event(
                        "llm_protocol_outcome", {"outcome": outcome.to_dict()}
                    )
            if (corrections_used >= correction_limit or (t3_retry and physical_requests >= physical_request_limit) or issue.code == "adapter_budget_exhausted"
                    or not state.task.budget.allows("max_llm_calls", int(state.accounting.get("adapter_calls", 0)))):
                return None, None, issue
            corrections_used += 1
            from .action_diagnostics import candidate_details, diagnostic
            details = dict(issue.details or {})
            if issue.code == 'stale_or_unknown_choice':
                details.update(diagnostic('stale_choice', menu_identity=turn.identity))
            if submission is not None: details.update(candidate_details(submission.to_dict()))
            details.update(attempt_ref=attempt_ref, correction_index=corrections_used,
                corrections_remaining=correction_limit-corrections_used)
            issue=replace(issue,details=details)
            from .working_state_delivery import bounded_correction
            correction = bounded_correction(issue.correction())
            turn=replace(turn,correction=correction,remaining_budget=_control_remaining(state))
            reuse_turn=True

    @staticmethod
    def _choose(
        adapter: DecisionAdapter, turn: Turn, state: _LoopState, *, phase: str = "action"
    ) -> tuple[Submission | None, _SubmissionIssue | None, str | None]:
        adapter_calls = int(state.accounting.get("adapter_calls", 0))
        phase_calls = int(state.accounting.get(phase + "_calls", 0))
        config = state.accounting.get("working_state", {})
        phase_limit = config.get("max_" + phase + "_calls", state.task.budget.max_llm_calls)
        if not state.task.budget.allows("max_llm_calls", adapter_calls) or (phase_limit is not None and phase_calls >= phase_limit):
            return (
                None,
                _SubmissionIssue(
                    "adapter_budget_exhausted", "choice", "No adapter calls remain."
                ),
                None,
            )
        state.accounting["adapter_calls"] = adapter_calls + 1
        state.accounting[phase + "_calls"] = phase_calls + 1
        recorded_attempt: LLMAttemptRef | None = None
        if phase == 'action':
            # The baseline is query-only. A future M9 state can use this one
            # seam after packing; its preparation must not mutate the payload.
            memory = state.resources.memory_state
            prepare = getattr(memory, 'mark_prepared', None)
            if prepare is not None:
                identity_before = turn.identity
                prepare(turn)
                if turn.identity != identity_before:
                    raise ValueError('Memory preparation changed the final action payload')
        if state.c1_open_control:
            state.resources.control_records.record_event("maintenance_emitted" if phase == "maintenance" else "turn_emitted", {
                "turn": turn.to_dict(), "phase": phase, "decision_index": state.accounting["turns"],
                "notebook_identity": deterministic_identity(state.resources.control_records.snapshot()["notebook"])})


        audit_failure = None
        def audit(kind, payload):
            nonlocal audit_failure
            try:
                if state.c1_open_control:
                    state.resources.control_records.record_event(kind, dict(payload,
                        phase=phase, decision_index=state.accounting["turns"],
                        elapsed_seconds=monotonic()-state.started_monotonic,
                        timestamp=datetime.now(timezone.utc).isoformat()))
            except Exception as error:
                raise_control_error(error)
                from .runtime_errors import AuditWriteError
                if isinstance(error, AuditWriteError): raise
                audit_failure = error
                raise
        try:
            def record_attempt(attempt: LLMAttemptRef) -> None:
                nonlocal recorded_attempt
                if recorded_attempt is not None:
                    raise ValueError("one adapter call cannot record multiple attempts")
                recorded_attempt = attempt
                if state.c1_open_control:
                    protected_response_ref = (
                        None
                        if attempt.response_digest is None
                        else f"sha256:{attempt.response_digest}"
                    )
                    if protected_response_ref is not None:
                        state.resources.protected_record_refs.add(
                            protected_response_ref
                        )
                    audit(
                        "llm_attempt",
                        {
                            "attempt": attempt.to_dict(),
                            "protected_raw_response_ref": protected_response_ref,
                        },
                    )
                try:
                    state.resources.state.record_llm_attempt(attempt)
                except Exception as error:
                    raise_control_error(error)
                    from .runtime_errors import AuditWriteError
                    if isinstance(error, AuditWriteError): raise
                    nonlocal audit_failure
                    audit_failure = error
                    raise

            def record_public_request(request):
                if state.c1_open_control:
                    audit("public_request", {"turn_identity": turn.identity, "request": request})
                    if phase == "action" and config.get("mode") == "single":
                        pending = state.accounting.get("pending_note_refs", ())
                        shown = {note["record_ref"] for note in (turn.notebook or {}).get("active_records", ())}
                        delivered = tuple(ref for ref in pending if ref in shown)
                        audit("single_notebook_sent", {
                            "turn_identity": turn.identity, "note_refs": delivered,
                            "notebook_identity": turn.notebook_identity, "reception": "sent_understanding_unknown"})
                        state.accounting["pending_note_refs"] = tuple(ref for ref in pending if ref not in shown)
            record_attempt.record_public_request = record_public_request
            record_attempt.record_attempt_start = lambda payload: audit("llm_attempt_started", payload)
            def record_public_response(payload):
                audit("public_response", dict(payload, turn_identity=turn.identity))
                if phase == "action" and recorded_attempt is not None and recorded_attempt.provider_failure is None:
                    # The normal model response proves reception even when its
                    # action envelope is subsequently rejected. Keep raw text
                    # in the existing protected response event, not the notebook.
                    from .t3_t4.evaluation_export import explicit_final_choice
                    if explicit_final_choice(payload.get("content"), turn.to_dict()):
                        proposal = {"attempt_ref": payload.get("attempt_ref"),
                            "response_digest": payload.get("response_digest"),
                            "turn_identity": turn.identity,
                            "elapsed_seconds": monotonic()-state.started_monotonic}
                        if getattr(state.resources, 'evaluation_final_policy',
                                   'first-explicit-final-v1') == 'committed-after-memory-v1':
                            audit('report_proposal_received', proposal)
                            state.accounting['pending_final_proposal'] = proposal
                        else:
                            audit("evaluation_final_intent", proposal)
            record_attempt.record_public_response = record_public_response
            record_attempt.decision_index = state.accounting["turns"]
            invoke = adapter.maintain if phase == "maintenance" else adapter.choose
            if getattr(state.resources, "host_control", None) is not None:state.resources.host_control.check()
            submission = invoke(
                turn,
                record_attempt=record_attempt,
            )
            from .knowledge_read_transaction import confirm_delivery
            if phase == 'action':
                confirm_delivery(state.resources, turn)
            if getattr(state.resources, "host_control", None) is not None:state.resources.host_control.check()
        except AdapterProtocolError as error:
            if audit_failure is not None:
                raise RuntimeError("provider audit persistence failed") from audit_failure
            attempt_ref = error.attempt_ref
            if error.attempt_ref is not None:
                if (
                    recorded_attempt is None
                    or recorded_attempt.attempt_ref != error.attempt_ref
                ):
                    raise RuntimeError(
                        "adapter protocol issue differs from its recorded attempt"
                    )
            return (
                None,
                _SubmissionIssue(
                    error.code,
                    error.field_path,
                    error.instruction,
                    adapter_classification=error.code,
                    update_issue=error.update_issue, details=error.details,
                ),
                attempt_ref,
            )
        except Exception as error:
            raise_control_error(error)
            from .runtime_errors import AuditWriteError
            if isinstance(error, AuditWriteError): raise
            if audit_failure is not None:
                raise RuntimeError("provider audit persistence failed") from audit_failure
            from .runtime_errors import SharedProviderFailure
            if isinstance(error, SharedProviderFailure):
                raise
            from .hosted_agent_policy import HostedTransportError
            if isinstance(error, HostedTransportError) and recorded_attempt is not None:
                failure = error.failure
                return (None, _SubmissionIssue(failure['code'] if failure else 'provider_transport_exhausted',
                    'provider', 'The provider request failed; see its saved audit.',
                    transport_retryable=error.retryable, provider_failure=failure), recorded_attempt.attempt_ref)
            return (
                None,
                _SubmissionIssue(
                    f"adapter_{type(error).__name__}",
                    "submission",
                    "The adapter failed internally; inspect its audit.",
                    system_defect=True,
                ),
                None if recorded_attempt is None else recorded_attempt.attempt_ref,
            )
        if phase == "maintenance":
            return submission, None, None if recorded_attempt is None else recorded_attempt.attempt_ref
        if not isinstance(submission, Submission):
            return (
                None,
                _SubmissionIssue(
                    "submission_not_typed",
                    "submission",
                    "Return exactly one JSON object with choice and arguments.",
                ),
                None if recorded_attempt is None else recorded_attempt.attempt_ref,
            )
        return (
            submission,
            None,
            None if recorded_attempt is None else recorded_attempt.attempt_ref,
        )


    def _materialize(self, state: _LoopState) -> dict[str, ExecutableOperation]:
        context = state.operation_context()
        offered = tuple(
            operation
            for provider in self._providers
            for operation in provider.offer(context)
            if not state.c1_open_control
            or state.task.budget.allows("max_actions", int(state.accounting.get("actions", 0)), operation.action_budget_cost)
        )
        semantic_keys = tuple(item.public.semantic_key for item in offered)
        audit_identities = tuple(item.audit_identity for item in offered)
        if len(semantic_keys) != len(set(semantic_keys)):
            raise ValueError("offered semantic keys must be unique")
        if len(audit_identities) != len(set(audit_identities)):
            raise ValueError("offered audit identities must be unique")
        current: dict[str, ExecutableOperation] = {}
        if state.c1_open_control:
            namespace = state.accounting.setdefault("choice_namespace", uuid4().hex[:12])
            ordered = sorted(offered, key=lambda operation: operation.public.semantic_key)
            signature = deterministic_identity({"revision": state.revision,
                "operations": tuple(operation.audit_identity for operation in ordered)})
            if state.accounting.get("choice_menu_signature") != signature:
                state.accounting["choice_menu_generation"] = int(state.accounting.get("choice_menu_generation", -1)) + 1
                state.accounting["choice_menu_signature"] = signature
            generation = state.accounting["choice_menu_generation"]
            for index, operation in enumerate(ordered, 1):
                handle = f"op_{namespace}_{generation}_{index}"
                current[handle] = operation.bind(handle)
        else:
            for operation in offered:
                handle = "op_" + deterministic_identity({"task_identity": state.task.task_identity,
                    "state_revision": state.revision, "audit_identity": operation.audit_identity})[:32]
                current[handle] = operation.bind(handle)
        if state.c1_open_control and json_bytes(
            tuple(operation.public.to_dict() for operation in current.values())
        ) > state.resources.context_limits.choices_bytes:
            raise ValueError("C1 choice catalog exceeds its fixed projection budget")
        return current


    @staticmethod
    def _render_turn(
        state: _LoopState,
        current: Mapping[str, ExecutableOperation],
        *,
        correction: ProtocolCorrection | None = None,
        source_coverage_descriptors: tuple[CapabilityDescriptor, ...] = (),
        source_coverage_owner_outcomes: tuple[SourceOwnerOutcome, ...] = (),
    ) -> Turn:
        from .record_inspection import public_observation
        observations = tuple(public_observation(state.resources, item.to_dict())
            for item in state.observations)
        task = state.task
        sorted_choices = tuple(
            item.public
            for item in sorted(
                current.values(), key=lambda value: value.public.semantic_key
            )
        )
        remaining = {public: value for public, name, counter in (
            ('turns', 'max_iterations', 'turns'), ('adapter_calls', 'max_llm_calls', 'adapter_calls'),
            ('actions', 'max_actions', 'actions'))
            if (value := task.budget.remaining(name, int(state.accounting.get(counter, 0)))) is not None}
        config = state.accounting.get("working_state")
        if config is not None and config["mode"] == "maintained":
            remaining.update(_phase_remaining(state))
        fixed_scope = {
            "decision_time": task.decision_time.isoformat(),
            "target": task.target_scope.to_dict() if task.target_scope else None,
        }
        if task.retrieval_scope_ref is not None:
            fixed_scope["retrieval_scope_ref"] = task.retrieval_scope_ref
        task_context = task.disclosed_context.get("task_context")
        if task.answer_contract.report_required:
            from .report_submission import REPORT_GUIDANCE
            task_context = dict(task_context or {}) | {"report_instructions": REPORT_GUIDANCE}
        if task_context is not None and not isinstance(task_context, Mapping):
            raise TypeError("RuntimeTask task_context projection must be an object")
        catalog_overview = None
        catalog_navigation = None
        navigation_owner = state.resources.catalog_navigation
        resolved_scope = state.resources.resolved_retrieval_scope
        if navigation_owner is not None and navigation_owner.overview is not None:
            catalog_overview = navigation_owner.overview.to_dict()
            catalog_snapshot = state.resources.catalog_snapshot
            if catalog_snapshot is None or resolved_scope is None:
                raise RuntimeError("Catalog navigation lacks its scope-bound snapshot")
            catalog_navigation = navigation_owner.turn_view(
                resolved_scope_identity=resolved_scope.resolved_scope_identity,
                snapshot_identity=catalog_snapshot.snapshot_identity,
                current_binding_identities=tuple(state.resources.bindings),
                operations=tuple(state.resources.control_records.public_operations()) if state.resources.control_records else (),
            )
        if not state.c1_open_control:
            turn = Turn(
                objective=task.question,
                fixed_scope=fixed_scope,
                open_requirements=tuple(task.answer_contract.required_fields),
                latest_observation=observations[-1] if observations else None,
                evidence=state.resources.ledger.context_view().records,
                science=EpisodeRunner._science_projection(state),
                choices=sorted_choices,
                remaining_budget=remaining,
                state_revision=state.revision,
                correction=correction,
                task_context=task_context,
                catalog_overview=catalog_overview,
                catalog_navigation=catalog_navigation,
            )
            publish_current_turn_record_refs(
                state.resources,
                latest_observation=turn.latest_observation,
                evidence=turn.evidence,
                notebook=turn.notebook,
                catalog_navigation=turn.catalog_navigation,
            )
            return turn

        source_coverage = None
        catalog_snapshot = state.resources.catalog_snapshot
        expected_snapshot_identity = (
            state.resources.state.catalog_snapshot_identity
        )
        if (
            source_coverage_descriptors
            and task.target_scope is not None
            and catalog_snapshot is not None
            and expected_snapshot_identity is not None
        ):
            derived = derive_source_coverage(
                SourceCoverageInputs(
                    operational_anchor=SourceCoverageOperationalAnchor(
                        decision_time=task.decision_time,
                        valid_start=task.target_scope.valid_start,
                        valid_end=task.target_scope.valid_end,
                        spatial=task.target_scope.spatial,
                    ),
                    catalog_snapshot=catalog_snapshot,
                    expected_catalog_snapshot_identity=expected_snapshot_identity,
                    expected_access_policy_identity=deterministic_identity(
                        task.access_policy
                    ),
                    descriptors=source_coverage_descriptors,
                    bindings=tuple(
                        sorted(
                            state.resources.bindings.values(),
                            key=lambda binding: binding.binding_identity,
                        )
                    ),
                    evidence_records=tuple(
                        state.resources.ledger.get(reference)
                        for reference in state.resources.ledger.record_refs
                    ),
                    typed_owner_outcomes=source_coverage_owner_outcomes,
                )
            )
            if isinstance(derived, SourceCoverageSnapshot):
                source_coverage = project_source_coverage(derived)

        from .c1_final import eligible_evidence_refs
        allowed_evidence = set(eligible_evidence_refs(state.operation_context()))
        all_evidence = tuple(item for item in state.resources.ledger.context_view().records
            if item["record_ref"] in allowed_evidence)
        full_notebook = state.resources.control_records.current_view()
        publish_current_turn_record_refs(state.resources,
            latest_observation=observations[-1] if observations else None,
            evidence=all_evidence, notebook=full_notebook,
            catalog_navigation=catalog_navigation)
        from .record_inspection import recent_evidence_reads, working_evidence, result_card, notebook_working_view, last_work_result
        record_index = dict(public_record_index(state.resources))
        cards = []
        for item in reversed(all_evidence):
            card = result_card(state.resources, item["record_ref"], max_bytes=500)
            if json_bytes([*cards, card]) <= 3072:
                cards.append(card)
        record_index["recent_results"] = tuple(cards)
        recent_history = []
        for item in reversed(recent_operation_history(state.resources)):
            item = {key: value for key, value in item.items() if key != "canonical_refs" and value is not None}
            if json_bytes([*recent_history, item]) <= 1300:
                recent_history.insert(0, item)
        from .notebook_display import notebook_window
        notebook = (notebook_window(state.resources.control_records, budget=config["notebook_bytes"], config=state.resources.working_state_config) if config is not None
            else notebook_working_view(state.resources))
        latest_observation = observations[-1] if observations else None
        if latest_observation is not None and latest_observation["status"] in {"failed", "rejected"}:
            details = latest_observation.get("details", {})
            if details.get("analysis_executed") is False:
                feedback = "Analysis did not execute. " + ("A later explicit attempt may retry startup." if details.get("retryable") else "Resolve the reported startup limitation before retrying.")
            else:
                feedback = "Use the returned failure and previous attempts to decide whether to revise or retry. A failed attempt does not establish a scientific result."
            latest_observation = dict(latest_observation, feedback=feedback)
        operations = state.resources.control_records.public_operations()
        observation_ref = operations[-1]["ref"] if operations else None
        if latest_observation is not None and latest_observation.get("details", {}).get("diagnostics_display") == "partial":
            latest_observation = dict(latest_observation, details=dict(latest_observation["details"], read_ref=observation_ref))
        if (latest_observation is not None and not is_material_code(latest_observation["code"])
                and not (latest_observation.get("details", {}).get("trust") == "advisory_memory"
                    and state.resources.memory_state is not None
                    and state.resources.memory_state.config.retrieval_mode == "contextual")
                and json_bytes(latest_observation) > state.resources.context_limits.navigation_bytes + 204):
            details = latest_observation.get("details", {})
            brief = {key: details[key] for key in ("trust", "stage", "reason", "analysis_executed", "retryable", "diagnostics", "diagnostics_display") if key in details}
            if json_bytes(details.get("result")) <= 2800 and "result" in details:
                brief["result"] = details["result"]
            latest_observation = dict(latest_observation, details=brief | {"display": "partial", "omitted_from_turn": True, "read_ref": observation_ref})
        latest_refs = tuple(item["identity"] for item in (latest_observation or {}).get("canonical_refs", ()))
        prioritized = tuple(dict.fromkeys((*latest_refs, *recent_evidence_reads(state.resources),
            *(item["record_ref"] for item in reversed(all_evidence)))))
        by_ref = {item["record_ref"]: working_evidence(item, state.resources) for item in all_evidence}
        evidence = tuple(by_ref[ref] for ref in prioritized if ref in by_ref)[:4]
        from .record_inspection import recent_memory_cards, persistent_memory_advice
        memory = state.resources.memory_state
        initial = memory.initial_projection(state.revision,max_bytes=state.resources.context_limits.turn_bytes) if memory is not None else None
        if initial is not None:
            record_index["initial_advice"] = initial
        persistent = persistent_memory_advice(state.resources, (initial, latest_observation)) if memory is not None else None
        if persistent is not None and (persistent["items"] or persistent["omitted_count"]):
            record_index["persistent_advice"] = persistent
        record_index["recent_advice"] = recent_memory_cards(state.resources, (initial, latest_observation, persistent))
        # Compact repeated machine navigation first. Advice then yields before
        # current scientific results, working notes, and the latest public read.
        compact_navigation = False
        while True:
            turn = Turn(objective=task.question, fixed_scope=fixed_scope,
                context_limits=state.resources.context_limits,
                working_state_mode=config["mode"] if config is not None else "single",
                maintenance_status=state.accounting.get("maintenance_status"),
                open_requirements=tuple(task.answer_contract.required_fields),
                latest_observation=latest_observation, evidence=evidence, science=None,
                choices=sorted_choices, remaining_budget=remaining, state_revision=state.revision,
                correction=correction, task_context=task_context, catalog_overview=catalog_overview,
                catalog_navigation=catalog_navigation, notebook=notebook if config is not None or notebook.get("plan_text") or notebook["active_records"] else None, source_coverage=source_coverage,
                recent_history=tuple(recent_history), records_index=record_index,
                last_work_result=last_work_result(state.resources),
                hidden_counts={key: value for key, value in {"evidence": len(all_evidence)-len(evidence),
                    "notebook_records": notebook["hidden_active_record_count"],
                    "observation_details": int(bool((latest_observation or {}).get("details", {}).get("omitted_from_turn")))}.items() if value})
            if json_bytes(turn.to_dict()) <= state.resources.context_limits.turn_bytes:
                return turn
            if source_coverage is not None:
                source_coverage = None
                continue
            if not compact_navigation and catalog_navigation is not None:
                catalog_navigation = {key: value for key, value in catalog_navigation.items()
                    if key not in {"activated_binding_identities", "inspection_refs", "search_refs"}}
                browse = dict(catalog_navigation.get("current_page") or {})
                browse.pop("cards", None)
                browse.update(cards_omitted=True, read_ref=browse.get("search_ref"))
                catalog_navigation["current_page"] = browse
                compact_navigation = True
                continue
            if record_index.get("persistent_advice", {}).get("items"):
                available = json_bytes({"persistent_advice":record_index["persistent_advice"]}) - (json_bytes(turn.to_dict()) - state.resources.context_limits.turn_bytes)
                projected = persistent_memory_advice(state.resources, (record_index.get("initial_advice"), latest_observation),
                    max_bytes=max(0,available))
                if projected is None:
                    record_index.pop("persistent_advice")
                else:
                    record_index["persistent_advice"] = projected
                record_index["recent_advice"] = recent_memory_cards(state.resources,
                    (record_index.get("initial_advice"),latest_observation,projected))
                continue
            if record_index.get("initial_advice", {}).get("advice"):
                available = json_bytes(record_index["initial_advice"]) - (json_bytes(turn.to_dict()) - state.resources.context_limits.turn_bytes)
                projected = memory.initial_projection(state.revision, max_bytes=max(0, available))
                if projected is None:
                    record_index.pop("initial_advice")
                else:
                    record_index["initial_advice"] = projected
                record_index["recent_advice"] = recent_memory_cards(state.resources,
                    (record_index.get("initial_advice"),latest_observation,record_index.get("persistent_advice")))
                continue
            if record_index.get("recent_advice"):
                record_index["recent_advice"] = record_index["recent_advice"][:-1]
                continue
            if "initial_advice" in record_index:
                record_index.pop("initial_advice")
                continue
            if "persistent_advice" in record_index:
                record_index.pop("persistent_advice")
                continue
            if (memory is not None and memory.config.retrieval_mode == "contextual" and latest_observation is not None
                    and not is_material_code(latest_observation.get("code"))):
                from .record_inspection import project_memory_observation
                available = json_bytes(latest_observation) - (json_bytes(turn.to_dict()) - state.resources.context_limits.turn_bytes)
                projected = project_memory_observation(latest_observation,
                    max_bytes=max(0, available), read_ref=observation_ref)
                if projected != latest_observation:
                    latest_observation = projected
                    continue
            if evidence:
                # Keep the original owner result and support before dropping an old result.
                if any(item.get("artifact_contracts") for item in evidence):
                    evidence = tuple(dict(item, artifact_contracts=(),
                        contract_omitted="Inspect read_ref for the owner input contract.") for item in evidence)
                else:
                    evidence = evidence[:-1]
                continue
            if record_index["recent_results"]:
                record_index["recent_results"] = record_index["recent_results"][:-1]
                continue
            if config is not None:
                from .working_state_delivery import WorkingStateDeliveryUnavailable
                raise WorkingStateDeliveryUnavailable("working_state_foundation_oversize")
            raise RuntimeError("bounded C1 turn cannot be projected within 24 KiB")

    @staticmethod
    def _science_projection(state: _LoopState) -> Mapping[str, Any]:
        canonical = state.resources.state
        active = (
            None
            if canonical.active_situation_ref is None
            else canonical.situations[canonical.active_situation_ref].to_ref().to_dict()
        )
        return {
            "claims": tuple(
                claim.to_dict()
                for _reference, claim in sorted(canonical.reasoning_claims.items())
            ),
            "products": tuple(
                product.to_ref().to_dict()
                for _reference, product in sorted(canonical.scientific_products.items())
            ),
            "active_situation": active,
        }


    @staticmethod
    def _terminal(state: _LoopState, status: str, code: str) -> EpisodeResult:
        from .knowledge_read_transaction import rollback_delivery
        rollback_delivery(state.resources)
        canonical = state.resources.state
        if canonical.phase == "running":
            canonical.fail(
                {
                    "kind": "executable_operation_terminal",
                    "status": status,
                    "sanitized_error_code": code,
                }
            )
        return EpisodeRunner._result(state, status, terminal_code=code)

    @staticmethod
    def _result(
        state: _LoopState,
        status: str,
        *,
        output: TypedOutput | None = None,
        terminal_code: str | None = None,
    ) -> EpisodeResult:
        if state.c1_open_control:
            final_policy = getattr(state.resources, 'evaluation_final_policy', 'first-explicit-final-v1')
            state.resources.control_records.record_event("evaluation_episode_closed", {
                "evaluation_audit_version": ('evaluation-host-facts-memory-v1'
                    if final_policy == 'committed-after-memory-v1' else 'evaluation-host-facts-v1'),
                "status": status, "terminal_code": terminal_code,
                "elapsed_seconds": monotonic()-state.started_monotonic})
        return EpisodeResult(
            status=status,
            terminal_code=terminal_code,
            observations=tuple(state.observations),
            accounting=to_primitive(state.accounting),
            state_identity=state.state_identity,
            state_snapshot=dict(state.resources.state.snapshot()) | ({"memory": state.resources.memory_state.snapshot()} if state.resources.memory_state is not None else {}),
            ledger_snapshot=state.resources.ledger.snapshot(),
            control_snapshot=(
                {}
                if state.resources.control_records is None
                else state.resources.control_records.snapshot()
            ),
            navigation_snapshot=(
                {}
                if state.resources.catalog_navigation is None
                else state.resources.catalog_navigation.snapshot()
            ),
            output=output,
        )


def _new_episode_resources(task: RuntimeTask) -> _EpisodeResources:
    from .validation import SystemValidationGate

    config_identity = deterministic_identity(
        {"control": EXECUTABLE_LOOP_SCHEMA_VERSION, "task": task.task_identity}
    )
    manifest = RunManifest(
        code_commit="uncommitted-executable-operation-loop",
        baseline_tag=EXECUTABLE_LOOP_SCHEMA_VERSION,
        model=ModelIdentity(
            provider="decision-adapter",
            requested_model="runtime-selected",
            thinking_mode="adapter-owned",
        ),
        evaluator_identity={"name": "episode-runner"},
        tool_set=(),
        budgets=to_primitive(task.budget),
        task_identity=task.task_identity,
        config_identity=config_identity,
        data_identity=deterministic_identity(
            {"task": task.task_identity, "disclosed_context": task.disclosed_context}
        ),
    )
    state = RunState(
        task=task,
        manifest=manifest,
        plan=PlanState(
            objective=task.question,
            subgoals=(),
            evidence_gaps=(),
            steps=(),
            revision=0,
            revision_reason="executable-operation episode initialization",
            stop_conditions=("contract-valid typed output exists",),
        ),
    )
    return _EpisodeResources(
        state=state,
        ledger=RunLedger(),
        validation_gate=SystemValidationGate(),
        control_records=None,
    )


__all__ = [
    "DecisionAdapter",
    "EpisodeResult",
    "EpisodeRunner",
    "EXECUTABLE_LOOP_SCHEMA_VERSION",
    "ProtocolCorrection",
    "Submission",
    "Turn",
]
