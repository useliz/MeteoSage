"""Neutral action requests, observations and owner protocol.

No scientific method selection or domain implementation belongs here. The
historical ``actions`` imports remain aliases for saved records and consumers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Protocol

from ._serialization import to_primitive
from .reliability import deterministic_identity

if TYPE_CHECKING:
    from .capability_catalog import BoundCapability
    from .executable_operations import OperationContext, OperationExecutionContext
    from .execution import ExecutionRecord
    from .run_state import RunState
    from .runtime import ToolResult
    from .tooling import ExecutionContext

ACTIONS_SCHEMA_VERSION = "weather-agent-actions-v3"
ACTION_EFFECT_CONTRACT_SCHEMA_VERSION = "weather-agent-action-effect-contract-v1"

class ActionEffectContractCode(str, Enum):
    UNKNOWN_ARGUMENT_CONTRACT = "unknown_action_argument_contract"
    INVALID_ARGUMENTS = "invalid_action_effect_arguments"
    INVALID_BINDING_PROJECTION = "invalid_action_effect_binding_projection"
    SUCCESS_REPLAY = "action_success_replay_not_allowed"
    DUPLICATE_PLAN_EFFECT = "action_plan_duplicate_effect"
    OBSERVATION_MISMATCH = "completed_action_effect_observation_mismatch"


class ActionEffectContractError(ValueError):
    def __init__(self, code: ActionEffectContractCode, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class ActionOwnerPreparation:
    """One serializable owner-prepared input behind the Action operation seam."""

    identity_inputs: Mapping[str, Any]
    additions: Mapping[str, Any] = field(default_factory=dict)
    arguments_schema: Mapping[str, Any] | None = None
    frozen_arguments: Mapping[str, Any] | None = None
    input_artifact_refs: tuple[str, ...] = ()
    selected_artifact_refs_path: tuple[str, ...] = ()
    decision_ref: str | None = None
    owner_summary: str | None = None

    def __post_init__(self) -> None:
        if self.decision_ref is None and not self.owner_summary:
            raise ValueError("prepared projection requires a decision ref or owner summary")
        if (self.arguments_schema is None) != (self.frozen_arguments is None):
            raise ValueError(
                "owner preparation schema and frozen arguments must be supplied together"
            )
        if (
            self.arguments_schema is not None
            and self.arguments_schema.get("type") != "object"
        ):
            raise ValueError("owner preparation schema must describe one object")
        if len(self.input_artifact_refs) != len(set(self.input_artifact_refs)):
            raise ValueError("owner preparation artifact refs must be unique")
        deterministic_identity(self)

    @property
    def projection_identity(self) -> str:
        return deterministic_identity(self.identity_inputs)


class ActionOperationOwner(Protocol):
    """Owner behavior for preparation, validation, and bounded result projection."""

    contract_identity: str

    def prepare(
        self,
        bound_arguments: Mapping[str, Any],
        binding: BoundCapability,
        context: OperationContext | OperationExecutionContext,
    ) -> tuple[ActionOwnerPreparation, ...]: ...

    def canonicalize(
        self,
        arguments: Mapping[str, object],
        preparation: ActionOwnerPreparation,
        binding: BoundCapability,
    ) -> Mapping[str, Any]: ...

    def binding_error(
        self,
        action: ActionRequest,
        preparation: ActionOwnerPreparation,
        binding: BoundCapability,
        state: RunState,
        context: ExecutionContext,
    ) -> ActionBindingFailure | str | None: ...

    def project_result(self, result: ToolResult) -> Mapping[str, Any] | None: ...


class ActionKind(str, Enum):
    TOOL_CALL = "tool_call"
    SCRIPT_RUN = "script_run"
    CODE_RUN = "code_run"


class ActionStatus(str, Enum):
    SUCCESS = "success"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"
    TIMEOUT = "timeout"
    FAILED = "failed"
    UNSUPPORTED_IN_PHASE = "unsupported_in_phase"
    DEPENDENCY_FAILED = "dependency_failed"
    BINDING_FAILURE = "binding_failure"


@dataclass(frozen=True)
class ActionBindingFailure:
    reason_code: str
    message: str
    field_path: str
    expected: str
    observed: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.reason_code or not self.message or not self.field_path or not self.expected:
            raise ValueError(
                "action binding failure requires code, message, field path, and expected value"
            )
        deterministic_identity(self)

    def to_dict(self) -> Mapping[str, Any]:
        return to_primitive(self)


class FailureKind(str, Enum):
    TOOL_EXECUTION = "tool_execution"


class FailureCorrection(str, Enum):
    CHANGE_ARGUMENTS_OR_BINDING = "change_arguments_or_binding"
    REDUCE_RESOURCE_REQUEST = "reduce_resource_request"
    ACQUIRE_OTHER_EVIDENCE = "acquire_other_evidence"


class FailureDisposition(str, Enum):
    REVISE_ACTION = "revise_action"
    SELECT_OTHER_BINDING = "select_other_binding"
    ACQUIRE_DESCRIPTOR_OR_EVIDENCE = "acquire_descriptor_or_evidence"
    STOP = "stop"


@dataclass(frozen=True)
class FailureEnvelope:
    kind: FailureKind
    action_status: ActionStatus
    reason_code: str
    correction_class: FailureCorrection
    disposition: FailureDisposition
    field_path: str
    observed: Mapping[str, Any]
    expected: Mapping[str, Any]
    budget_source: str
    failed_action_retry_identity: str
    material_state_epoch_identity: str
    schema_version: str = "weather-agent-action-failure-envelope-v1"

    def __post_init__(self) -> None:
        if self.reason_code not in {
            "tool_unavailable",
            "source_unsupported",
            "source_missing",
            "source_validation_failed",
            "source_failure",
            "adapter_exception",
            "materialization_context_missing",
            "materialization_failed",
            "timeout",
            "execution_failed",
        }:
            raise ValueError("failure envelope reason code is not in the closed vocabulary")
        if self.field_path != "action_plan.actions[].arguments":
            raise ValueError("failure envelope field path is not canonical")
        if self.budget_source != "controller_context.remaining_budget/accounting":
            raise ValueError("failure envelope budget source is not canonical")
        if len(self.observed) > 8 or len(self.expected) > 8:
            raise ValueError("failure envelope shape is not bounded")
        for value in (
            self.failed_action_retry_identity,
            self.material_state_epoch_identity,
        ):
            if len(value) != 64:
                raise ValueError("failure envelope identities must be SHA-256")
        deterministic_identity(self)

    def to_dict(self) -> Mapping[str, Any]:
        return to_primitive(self)


@dataclass(frozen=True)
class ActionRequest:
    action_id: str
    kind: ActionKind
    bound_capability_ref: str
    arguments: Mapping[str, Any]
    dependencies: tuple[str, ...]
    timeout_seconds: float
    resource_kind: str
    resource_units: int
    max_retries: int = 0
    schema_version: str = ACTIONS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.action_id or not self.bound_capability_ref:
            raise ValueError("action_id and bound_capability_ref are required")
        action_path = Path(self.action_id)
        if (
            action_path.is_absolute()
            or len(action_path.parts) != 1
            or action_path.parts[0] in {".", ".."}
        ):
            raise ValueError("action_id must be a safe single path component")
        if self.action_id in self.dependencies:
            raise ValueError("an action cannot depend on itself")
        if self.timeout_seconds <= 0 or self.resource_units <= 0 or self.max_retries < 0:
            raise ValueError("action timeout/resource units must be positive and retries non-negative")
        if self.resource_kind not in {"cpu", "gpu"}:
            raise ValueError("action resource_kind must be cpu or gpu")
        deterministic_identity(self)

    @property
    def action_identity(self) -> str:
        return deterministic_identity(self)


def action_retry_identity(
    action: ActionRequest,
    *,
    bound_capability_identity: str | None = None,
) -> str:
    """Identify fields that can change a pre-dispatch binding validation result."""

    return deterministic_identity(
        {
            "kind": action.kind.value,
            "bound_capability_identity": (
                bound_capability_identity or action.bound_capability_ref
            ),
            "arguments": action.arguments,
            "timeout_seconds": float(action.timeout_seconds),
            "resource_kind": action.resource_kind,
            "resource_units": action.resource_units,
        }
    )


@dataclass(frozen=True)
class CanonicalActionEffect:
    binding_identity: str
    action_contract_identity: str
    argument_contract: str
    canonical_arguments: Mapping[str, Any]
    input_summary_seed: Mapping[str, Any]
    schema_version: str = ACTION_EFFECT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for label, value in (
            ("binding_identity", self.binding_identity),
            ("action_contract_identity", self.action_contract_identity),
        ):
            if not _is_sha256(value):
                raise ValueError(f"{label} must be a SHA-256 identity")
        if not self.argument_contract:
            raise ValueError("argument_contract is required")
        deterministic_identity(self)

    @property
    def effect_identity(self) -> str:
        return deterministic_identity(
            {
                "schema_version": self.schema_version,
                "binding_identity": self.binding_identity,
                "action_contract_identity": self.action_contract_identity,
                "canonical_arguments": self.canonical_arguments,
            }
        )

    def successful_record(self) -> Mapping[str, Any]:
        """Persist only the public first-stage projection after typed success."""

        return {
            "schema_version": self.schema_version,
            "effect_identity": self.effect_identity,
            "binding_identity": self.binding_identity,
            "action_contract_identity": self.action_contract_identity,
            "argument_contract": self.argument_contract,
            "input_summary_seed": to_primitive(self.input_summary_seed),
        }


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class ActionPlan:
    actions: tuple[ActionRequest, ...]
    plan_revision: int
    schema_version: str = ACTIONS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.plan_revision < 0:
            raise ValueError("action plan revision cannot be negative")
        action_ids = tuple(action.action_id for action in self.actions)
        if not action_ids or len(action_ids) != len(set(action_ids)):
            raise ValueError("action plan ids must be non-empty and unique")
        seen: set[str] = set()
        for action in self.actions:
            missing = set(action.dependencies).difference(seen)
            if missing:
                raise ValueError(
                    f"action dependencies must refer to earlier actions: {sorted(missing)}"
                )
            seen.add(action.action_id)
        deterministic_identity(self)

    @property
    def plan_identity(self) -> str:
        return deterministic_identity(self)


@dataclass(frozen=True)
class ActionObservation:
    action_id: str
    attempt_id: str
    status: ActionStatus
    summary: str
    action_result: Mapping[str, Any] | None = None
    tool_result: Mapping[str, Any] | None = None
    artifact_refs: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    resource_units_consumed: int = 0
    schema_version: str = ACTIONS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.action_id or not self.attempt_id or not self.summary:
            raise ValueError("action observation ids and summary are required")
        if self.resource_units_consumed < 0:
            raise ValueError("action observation resource usage cannot be negative")
        deterministic_identity(self)

    @property
    def observation_identity(self) -> str:
        return deterministic_identity(self)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self) | {"observation_identity": self.observation_identity}


@dataclass(frozen=True)
class ExecutionBatch:
    plan_identity: str
    status: str
    observations: tuple[ActionObservation, ...]
    execution_records: tuple[ExecutionRecord, ...]
    attempted_actions: int
    attempted_tool_calls: int
    resumed_tool_calls: int
    restart_count: int
    resource_units_consumed: int
    attempted_by_kind: Mapping[str, int] = field(default_factory=dict)
    resumed_by_kind: Mapping[str, int] = field(default_factory=dict)
    schema_version: str = ACTIONS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.status not in {"completed", "incomplete"}:
            raise ValueError("execution batch status must be completed or incomplete")
        if min(
            self.attempted_actions,
            self.attempted_tool_calls,
            self.resumed_tool_calls,
            self.restart_count,
            self.resource_units_consumed,
            *self.attempted_by_kind.values(),
            *self.resumed_by_kind.values(),
        ) < 0:
            raise ValueError("execution batch accounting cannot be negative")
        deterministic_identity(self)

    @property
    def batch_identity(self) -> str:
        return deterministic_identity(self)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self) | {"batch_identity": self.batch_identity}


def _identity_refs(value: Any) -> tuple[str, ...]:
    references: list[str] = []

    def collect(item: Any) -> None:
        if _is_sha256(item):
            references.append(str(item))
        elif isinstance(item, Mapping):
            for key, nested in item.items():
                collect(key)
                collect(nested)
        elif isinstance(item, (tuple, list)):
            for nested in item:
                collect(nested)

    collect(value)
    return tuple(dict.fromkeys(references))


def canonical_owner_action_effect(action, binding, owner_contract_identity):
    argument_contract = binding.execution_binding.get("argument_contract")
    if not isinstance(argument_contract, str) or not argument_contract:
        raise ActionEffectContractError(ActionEffectContractCode.UNKNOWN_ARGUMENT_CONTRACT,
                                        "owner operation has no argument contract identity")
    contract_identity = deterministic_identity({
        "schema_version": ACTION_EFFECT_CONTRACT_SCHEMA_VERSION,
        "action_kind": action.kind.value,
        "argument_contract": argument_contract,
        "owner_contract_identity": owner_contract_identity,
        "execution_contract": binding.execution_binding})
    return CanonicalActionEffect(
        binding_identity=binding.binding_identity, action_contract_identity=contract_identity,
        argument_contract=argument_contract, canonical_arguments=to_primitive(action.arguments),
        input_summary_seed={"argument_contract":argument_contract,"semantic_field_paths":(),
                            "effect_input_refs":_identity_refs(action.arguments),
                            "owner_contract_identity":owner_contract_identity})


class ActionExecutionContract(Protocol):
    """A bound owner's checked selection handed to the common runtime."""
    owner_managed: bool
    def canonical_effect(self, action: ActionRequest) -> CanonicalActionEffect: ...
    def binding_error(self, action: ActionRequest, *, state: RunState, context: ExecutionContext) -> ActionBindingFailure | str | None: ...
    def invocation_arguments(self, action: ActionRequest) -> Mapping[str, Any]: ...
    def requested_artifact_refs(self, action: ActionRequest) -> tuple[str, ...]: ...
    def project_result(self, result: ToolResult) -> Mapping[str, Any] | None: ...
