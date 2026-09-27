from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Mapping


from ._serialization import to_primitive
from .action_contracts import (ActionKind, ActionPlan, ActionRequest, ActionStatus, canonical_owner_action_effect)
from .actions import (ActionRuntime)
from .c1_records import json_bytes
from .capability_catalog import BoundCapability
from .computation_evidence import ComputationEvidenceEnvelope
from .executable_operations import (
    ArgumentValidationError,
    CanonicalRecordRef,
    ChoiceSpec,
    C1OperationProvider,
    ExecutableOperation,
    MappingArgumentContract,
    OperationContext,
    OperationExecutionContext,
    OperationObservation,
    TransitionResult,
)
from .reliability import deterministic_identity
from .run_state import PlannedAction
from .runtime_records import (
    ArtifactRef,
    EvidenceRecord,
    ScientificPayloadRef,
    derived_evidence_times,
    artifact_contract,
    _validate_compact,
)
from .tooling import ExecutionContext
from .coderun_stages import check_deadline, save_stage
from .coderun_failure import MESSAGES
from .host_control import raise_control_error


@dataclass(frozen=True)
class _CodeRunExecutionContract:
    binding: BoundCapability
    selected: ActionRequest
    owner_managed = True
    contract_identity = deterministic_identity({"owner":"artifact_coderun", "contract":"selected_code_and_inputs_v1"})

    def canonical_effect(self, action):
        return canonical_owner_action_effect(action, self.binding, self.contract_identity)

    def binding_error(self, action, *, state, context):
        if action != self.selected:
            return "CodeRun action differs from the authorized selection"
        return None

    def invocation_arguments(self, action):
        return dict(self.binding.parameters) | dict(action.arguments)

    def requested_artifact_refs(self, action):
        return tuple(action.arguments["artifact_refs"])

    def project_result(self, result):
        # Full values remain in the ToolResult; the CodeRun provider publishes
        # its own bounded observation after artifact/evidence admission.
        return None


CODERUN_DESCRIPTION = "Run Python analysis on selected authorized artifacts and save derived results for reporting or further computation.\n\nSelect current artifact_refs with a supported CodeRun input contract; a listed file may be provenance-only. Inspect its input status if unsure. Read paths with artifacts[ref]. Assign a finite JSON object to result; write supported files under output_dir. Read a saved result through its result_read_ref instead of recomputing merely to see its contents; supported results can be reused as artifact inputs. Inputs may differ in time, region or grid; perform any required alignment in code. Inspect the files' coordinates, units and time support. Current libraries, formats and limits are listed below; the automatic result file uses one output-file slot.\n\nTo reuse a derived field with field tools, save it as NetCDF with truthful variable units, coordinates, CRS/grid metadata and any required temporal support. JSON summaries and missing metadata do not supply a field contract. Reusing a computed field preserves its computed origin; it does not make the result an observation or verify its scientific correctness."

CODERUN_OPERATION_SCHEMA_VERSION = "weather-agent-artifact-coderun-operation-v1"
MAX_CODERUN_CHOICE_BYTES = 2 * 1024
MAX_CODERUN_CODE_BYTES = 65536
MAX_CODERUN_ARTIFACT_REFS = 16



def computation_working_summary(result: Mapping[str, Any], limit: int = 2500) -> Mapping[str, Any]:
    """Exact bounded structural preview; no weather-field ranking or new conclusions.

    Large indivisible values remain in the original public record. Omitted JSON
    paths are explicit; this preview never replaces the stored computation result.
    """
    def fits(value, budget):
        try:
            _validate_compact(value, "computation preview")
            return json_bytes(value) <= budget
        except (ValueError, TypeError):
            return False
    if fits(result, limit):
        return result
    omitted = []
    def select(value, path=()):
        if isinstance(value, Mapping):
            return {key: selected for key, item in value.items()
                if (selected := select(item, (*path, key))) is not missing}
        if not fits(value, limit // 2):
            omitted.append(list(path))
            return missing
        return value
    missing = object()
    preview = select(result)
    value = {"display": "partial", "result": preview, "omitted_paths": omitted,
        "reading_guidance": "Exact retained values only. Read this evidence's read_ref for the complete result and omitted fields."}
    if not fits(value, limit):
        return {"display": "reference-only", "reading_guidance": "Read this evidence's read_ref for the full computation result."}
    return value

def authorized_time_summary(result, source_times):
    """Keep mechanical provenance outside the model result and within a small preview."""
    from .t3_t4.public_json_access import TIME_BASIS
    note = {'basis': TIME_BASIS, 'physical_release_time_inferred': False,
        'source_refs': tuple(source_times)}
    return {'result': computation_working_summary(result, limit=max(256, 2400-json_bytes(note))),
        '_host_input_time_basis': note}


@dataclass(frozen=True)
class ArtifactCodeRunPolicy:
    timeout_seconds: float
    cpu_time_seconds: int
    memory_limit_bytes: int
    max_output_bytes: int
    max_output_files: int
    max_stdout_bytes: int
    network_policy: str = "deny"
    filesystem_policy: str = "bound_handles_and_run_artifacts"
    isolation_backend: str = "bwrap"
    expected_result_kind: str = "json_object"
    schema_version: str = CODERUN_OPERATION_SCHEMA_VERSION
    generation_identity: str | None = None
    runtime_libraries: Mapping[str, str] = field(default_factory=dict)
    max_call_seconds: float = 240.0

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0 or self.cpu_time_seconds <= 0:
            raise ValueError("CodeRun time budgets must be positive")
        if min(
            self.memory_limit_bytes,
            self.max_output_bytes,
            self.max_output_files,
            self.max_stdout_bytes,
        ) <= 0:
            raise ValueError("CodeRun resource budgets must be positive")
        if (
            self.network_policy != "deny"
            or self.filesystem_policy != "bound_handles_and_run_artifacts"
            or self.isolation_backend not in ("bwrap", "chroot_seccomp")
            or self.expected_result_kind != "json_object"
        ):
            raise ValueError("CodeRun policy must preserve the guarded native sandbox")

    @property
    def invocation_parameters(self) -> Mapping[str, Any]:
        return {
            "timeout_seconds": self.timeout_seconds,
            "cpu_time_seconds": self.cpu_time_seconds,
            "memory_limit_bytes": self.memory_limit_bytes,
            "max_output_bytes": self.max_output_bytes,
            "max_output_files": self.max_output_files,
            "max_stdout_bytes": self.max_stdout_bytes,
            "network_policy": self.network_policy,
            "filesystem_policy": self.filesystem_policy,
            "isolation_backend": self.isolation_backend,
            "expected_result_kind": self.expected_result_kind,
        }

    @property
    def policy_identity(self) -> str:
        return deterministic_identity(self)


@dataclass(frozen=True)
class _SelectedCodeRun:
    code: str
    artifact_refs: tuple[str, ...]

    @property
    def code_sha256(self) -> str:
        return hashlib.sha256(self.code.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _EligibleArtifact:
    artifact: ArtifactRef
    record: EvidenceRecord
    path: Path


def authorized_artifact_inputs(context: OperationContext | OperationExecutionContext, *, for_reading: bool = False) -> Mapping[str, _EligibleArtifact]:
    """Current, equivalent public artifact parents, before format eligibility.

    Both inspection and CodeRun use this authority check. A provenance file may
    be readable without supplying a supported computational input contract.
    """
    resources = context.resources
    workspace = resources.scientific_workspace
    if workspace is None or workspace.run_identity != resources.state.manifest.run_identity:
        return {}
    from .c1_final import eligible_evidence_refs

    # Evidence query identities belong to their source/derivation contract,
    # not universally to BindingRequest. Reuse the same current authority,
    # source selection and lineage checks as public evidence projection.
    legal_records = set(eligible_evidence_refs(context))
    candidates: dict[str, list[_EligibleArtifact]] = {}
    for reference in resources.ledger.record_refs:
        if reference in resources.protected_record_refs or reference not in resources.state.evidence_refs or reference not in legal_records:
            continue
        record = resources.ledger.get(reference)
        envelope = record.envelope
        available_time = getattr(envelope, "available_time", None)
        from .t3_t4.source_evidence import SourceQueryEvidenceEnvelope
        from .t3_t4.public_json_access import artifact_access_time
        raw_access = isinstance(envelope, SourceQueryEvidenceEnvelope) and any(
            artifact_access_time(envelope, a) == context.task.decision_time for a in envelope.artifacts)
        readable_source = for_reading and isinstance(envelope, SourceQueryEvidenceEnvelope)
        if (
            (available_time is None and not readable_source and not raw_access)
            or (available_time is not None and available_time > context.task.decision_time)
            or envelope.evidence_role == "future_truth"
        ):
            continue
        for artifact in envelope.artifacts:
            if (available_time is None and not for_reading
                    and artifact_access_time(envelope, artifact) != context.task.decision_time):
                continue
            if artifact.artifact_ref in resources.protected_record_refs or artifact.artifact_ref not in resources.state.artifact_refs:
                continue
            try:
                path = workspace.artifacts.resolve(artifact)
            except (FileNotFoundError, PermissionError, TypeError, ValueError):
                continue
            candidates.setdefault(artifact.artifact_ref, []).append(
                _EligibleArtifact(artifact, record, path)
            )
    from .equivalent_inputs import equivalence_key
    return {
        reference: values[0]
        for reference, values in candidates.items()
        if len(values) == 1 or len({equivalence_key(item.record, resources)
            for item in values}) == 1
    }


def artifact_input_status(item: _EligibleArtifact) -> Mapping[str, object]:
    from .t3_t4.public_json_access import artifact_access_time
    if (getattr(item.record.envelope, "available_time", None) is None
            and artifact_access_time(item.record.envelope, item.artifact) is None):
        return {"eligible": False, "reason": "source_readable_without_computational_availability"}
    supported = artifact_contract(item.record, item.artifact) is not None
    return {"eligible": supported, "reason": "supported_input_contract" if supported else "missing_supported_input_contract"}


def _argument_contract(policy: ArtifactCodeRunPolicy) -> MappingArgumentContract:
    from .action_diagnostics import copy_text as recovery_copy
    schema = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "pattern": r"\S",
                "x-maxUtf8Bytes": MAX_CODERUN_CODE_BYTES,
                "description": recovery_copy("operations","coderun","code_description"),
            },
            "artifact_refs": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": 1,
                "maxItems": MAX_CODERUN_ARTIFACT_REFS,
                "uniqueItems": True,
                "description": recovery_copy("operations","coderun","artifact_refs_description"),
            },
        },
        "required": ("code", "artifact_refs"),
        "additionalProperties": False,
    }
    from importlib.metadata import PackageNotFoundError, version
    libraries = dict(policy.runtime_libraries)
    for name in ("numpy", "xarray", "h5netcdf", "scipy"):
        if policy.runtime_libraries:
            break
        try:
            libraries[name] = version(name)
        except PackageNotFoundError:
            pass
    schema["x-runtime"] = {"libraries": libraries, "formats": ("JSON object", "NetCDF", "PNG"),
        "timeout_seconds": policy.timeout_seconds, "cpu_time_seconds": policy.cpu_time_seconds,
        "memory_limit_bytes": policy.memory_limit_bytes, "max_output_bytes": policy.max_output_bytes,
        "max_output_files": policy.max_output_files, "automatic_result_file_slots": 1}
    if policy.isolation_backend == "chroot_seccomp":
        schema["x-runtime"].update(mode="serial; dask=sync", workspace_bytes=256*1024**2)
    if json_bytes(schema) > MAX_CODERUN_CHOICE_BYTES:
        raise ValueError("CodeRun choice schema exceeds 2 KiB")

    def validate(arguments: Mapping[str, object]) -> _SelectedCodeRun:
        from .action_diagnostics import validate_envelope
        validate_envelope(arguments, schema)
        from .action_diagnostics import diagnostic
        code = arguments['code']
        if not code.strip():
            raise ArgumentValidationError('arguments_contract_mismatch', 'arguments.code',
                recovery_copy('contract_corrections', 'blank_code'),
                details=diagnostic('field_value', reason_code='blank_string',
                    actual_length=len(code), actual_bytes=len(code.encode('utf-8'))))
        refs = arguments['artifact_refs']
        duplicates = [[i,j] for j in range(len(refs)) for i in range(j) if refs[i] == refs[j]]
        if duplicates:
            raise ArgumentValidationError('arguments_contract_mismatch', 'arguments.artifact_refs',
                recovery_copy('contract_corrections', 'duplicate_artifact_refs'),
                details=diagnostic('field_value', reason_code='duplicate_items',
                    duplicate_positions=duplicates[:16], omitted_count=max(0,len(duplicates)-16)))
        try:
            return _selected(arguments)
        except (TypeError, ValueError) as error:
            raise ArgumentValidationError(
                "arguments_contract_mismatch",
                "arguments",
                "CodeRun arguments must contain only bounded code and artifact_refs",
            ) from error

    return MappingArgumentContract(schema=schema, validator=validate)


def _selected(arguments: object) -> _SelectedCodeRun:
    if not isinstance(arguments, Mapping) or set(arguments) != {
        "code",
        "artifact_refs",
    }:
        raise ValueError("CodeRun accepts only code and artifact_refs")
    code = arguments.get("code")
    refs = arguments.get("artifact_refs")
    if (
        not isinstance(code, str)
        or not code.strip()
        or len(code.encode("utf-8")) > MAX_CODERUN_CODE_BYTES
    ):
        raise ValueError(f"CodeRun code must be nonempty and at most {MAX_CODERUN_CODE_BYTES} UTF-8 bytes")
    if isinstance(refs, (str, bytes, Mapping)) or not isinstance(refs, (tuple, list)):
        raise ValueError("CodeRun artifact_refs must be an array")
    if any(not isinstance(item, str) or not item for item in refs):
        raise ValueError("CodeRun artifact_refs items must be non-empty strings")
    normalized = tuple(refs)
    if (
        not normalized
        or len(normalized) > MAX_CODERUN_ARTIFACT_REFS
        or len(normalized) != len(set(normalized))
        or any(not item for item in normalized)
    ):
        raise ValueError("CodeRun artifact_refs must be bounded and unique")
    return _SelectedCodeRun(code, tuple(sorted(normalized)))


def frozen_code_binding(descriptor, snapshot,
                        task, policy: ArtifactCodeRunPolicy):
    from .capability_catalog import BoundCapability
    return BoundCapability(
        capability_ref=descriptor.capability_ref, descriptor_identity=descriptor.descriptor_identity,
        request_identity=deterministic_identity({"owner": descriptor.capability_ref, "task": task.task_identity}),
        namespace=descriptor.namespace, variables=descriptor.variables,
        valid_start=task.target_scope.valid_start if task.target_scope else None, valid_end=task.target_scope.valid_end if task.target_scope else None,
        spatial=task.target_scope.spatial if task.target_scope else {}, parameters=policy.invocation_parameters,
        execution_binding=descriptor.execution_binding,
        resource={"kind": "cpu", "max_timeout_seconds": policy.max_call_seconds, "max_resource_units": 1},
        decision_time=task.decision_time, available_time=task.decision_time,
        catalog_snapshot_identity=snapshot.snapshot_identity, access_policy_identity=snapshot.access_policy_identity,
        resolved_scope_identity=snapshot.resolved_scope_identity)


class ArtifactCodeRunProvider(C1OperationProvider):
    """Generic current-artifact CodeRun owner backed by ActionRuntime."""

    def __init__(
        self,
        runtime: ActionRuntime,
        *,
        capability_ref: str,
        policy: ArtifactCodeRunPolicy,
        clock: Callable[[], datetime] | None = None,
        monotonic_clock: Callable[[], float] | None = None,
    ) -> None:
        if not capability_ref:
            raise ValueError("CodeRun capability_ref is required")
        self._runtime = runtime
        self._capability_ref = capability_ref
        self._policy = policy
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._monotonic_clock = monotonic_clock or time.monotonic
        self._runner_identity = deterministic_identity(
            {
                "owner": "ArtifactCodeRunProvider",
                "adapter": "NativeCodeRunAdapter",
                "capability_ref": capability_ref,
                "policy_identity": policy.policy_identity,
            }
        )

    def offer(self, context: OperationContext) -> tuple[ExecutableOperation, ...]:
        resources = context.resources
        if (
            not self._execution_budget_available(context)
            or not context.task.access_policy.allow_generated_code
            or resources.scientific_workspace is None
            or resources.catalog_snapshot is None
            or context.task.resolved_scope_identity is None
            or resources.catalog_snapshot.resolved_scope_identity
            != context.task.resolved_scope_identity
        ):
            return ()
        # Historical records remain auditable; only currently valid host-policy
        # bindings can authorize execution. Never choose among conflicting policies.
        bindings = []
        for candidate in resources.bindings.values():
            if candidate.capability_ref != self._capability_ref:
                continue
            validation = resources.validation_gate.validate_binding(
                context.task, candidate,
                catalog_snapshot_identity=resources.catalog_snapshot.snapshot_identity,
            )
            if (validation.status.value == "valid"
                    and candidate.execution_binding.get("action_kind") == "code_run"
                    and candidate.execution_binding.get("argument_contract") == "code_run_v1"
                    and candidate.execution_binding.get("code_runner_id") is not None
                    and to_primitive(candidate.parameters) == to_primitive(self._policy.invocation_parameters)):
                bindings.append(candidate)
        if not bindings:
            return ()
        equivalent = {deterministic_identity({key: value for key, value in to_primitive(item).items()
            if key != "request_identity"}) for item in bindings}
        if len(equivalent) != 1:
            return ()
        binding = min(bindings, key=lambda item: item.binding_identity)
        eligible = self._eligible_artifacts(context)
        if not eligible:
            return ()
        contract = _argument_contract(self._policy)

        def invoke(
            execution_context: OperationExecutionContext,
            arguments: object,
        ) -> TransitionResult:
            if not isinstance(arguments, _SelectedCodeRun):
                return _failure(
                    "coderun_arguments_invalid",
                    "CodeRun arguments were not validated by the owner contract.",
                    status="rejected",
                )
            selected = arguments
            if not self._execution_budget_available(execution_context):
                return _failure(
                    "coderun_budget_no_longer_available",
                    "Remaining action, resource, or wall budget cannot cover this CodeRun.",
                    status="rejected",
                )
            current_eligible = self._eligible_artifacts(execution_context)
            selected_inputs = tuple(
                current_eligible.get(reference) for reference in selected.artifact_refs
            )
            if any(item is None for item in selected_inputs):
                public_inputs = authorized_artifact_inputs(execution_context)
                missing = tuple(ref for ref in selected.artifact_refs
                    if ref in public_inputs and not artifact_input_status(public_inputs[ref])["eligible"])
                if missing and all(ref in public_inputs for ref in selected.artifact_refs):
                    return _failure("coderun_missing_input_contract",
                        "These selected artifacts have public records but no supported CodeRun input "
                        "contract: " + ", ".join(missing) + ". They were not executed. Select supported "
                        "data/result artifacts from their reading contracts; the other source records "
                        "have not become stale.")
                return _failure(
                    "coderun_artifact_stale_or_unauthorized",
                    "A selected artifact is unknown, stale, future, cross-run, or unprovenanced.",
                )
            parents = tuple(item for item in selected_inputs if item is not None)
            return self._execute(
                execution_context,
                selected,
                parents,
                binding,
            )

        return (
            ExecutableOperation(
                public=ChoiceSpec(
                    semantic_key="runtime.coderun",
                    purpose=CODERUN_DESCRIPTION.split("\n\n", 1)[0],
                    arguments_schema=contract.schema,
                    expected_observation="Derived result with input lineage, or typed sandbox failure.",
                ),
                argument_contract=contract,
                invoke=invoke,
                audit_identity=deterministic_identity(
                    {
                        "owner": "artifact_coderun",
                        "binding": binding.binding_identity,
                        "eligible_artifacts": tuple(sorted(eligible)),
                        "state_revision": context.state_revision,
                        "policy": self._policy.policy_identity,
                    }
                ),
                manages_action_accounting=True,
            ),
        )

    def _execution_budget_available(
        self,
        context: OperationContext | OperationExecutionContext,
    ) -> bool:
        accounting = context.resources.state.accounting
        if not context.task.budget.allows('max_actions', int(accounting.get('actions', 0))):
            return False
        resource_limit = context.task.budget.max_resource_units
        if resource_limit is not None and (
            resource_limit - int(accounting.get("resource_units", 0)) < 1
        ):
            return False
        current = float(self._monotonic_clock())
        if not math.isfinite(current):
            return False
        remaining_wall = (context.deadline - current if context.deadline is not None
                          else context.task.budget.timeout_seconds)
        return remaining_wall is None or remaining_wall > 0

    def _eligible_artifacts(
        self,
        context: OperationContext | OperationExecutionContext,
    ) -> Mapping[str, _EligibleArtifact]:
        return {ref: item for ref, item in authorized_artifact_inputs(context).items()
            if artifact_input_status(item)["eligible"]}

    def _execute(
        self,
        execution_context: OperationExecutionContext,
        selected: _SelectedCodeRun,
        parents: tuple[_EligibleArtifact, ...],
        binding: BoundCapability,
    ) -> TransitionResult:
        resources = execution_context.resources
        state = resources.state
        deadline = self._monotonic_clock() + self._policy.max_call_seconds
        if execution_context.deadline is not None:
            deadline = min(deadline, execution_context.deadline)
        check_deadline(deadline)
        current_binding = resources.bindings.get(binding.binding_identity)
        if current_binding != binding:
            return _failure(
                "coderun_binding_stale", "CodeRun binding changed before invocation."
            )
        parent_binding_refs = tuple(
            sorted(
                {
                    parent.record.envelope.binding_identity
                    for parent in parents
                    if parent.record.envelope.binding_identity in resources.bindings
                }
            )
        )
        action_id = "coderun-" + deterministic_identity(
            {
                "revision": execution_context.state_revision,
                "code": selected.code_sha256,
                "artifacts": selected.artifact_refs,
            }
        )[:20]
        action = ActionRequest(
            action_id=action_id,
            kind=ActionKind.CODE_RUN,
            bound_capability_ref=binding.binding_identity,
            arguments={
                "code": selected.code,
                "capability_handles": parent_binding_refs,
                "artifact_refs": selected.artifact_refs,
            },
            dependencies=(),
            timeout_seconds=max(.001, deadline - self._monotonic_clock()),
            resource_kind="cpu",
            resource_units=1,
            max_retries=0,
        )
        state.revise_plan(
            steps=state.plan.steps
            + (
                PlannedAction(
                    action_id=action.action_id,
                    purpose="Execute one guarded current-artifact computation.",
                    capability_ref=binding.binding_identity,
                    status="ready",
                ),
            ),
            reason="selected generic artifact CodeRun operation",
        )
        plan = ActionPlan((action,), plan_revision=state.plan.revision)
        workspace = resources.scientific_workspace
        assert workspace is not None
        run_scope = workspace.artifacts.run_root
        base_context = ExecutionContext(
            task_identity=state.task.task_identity,
            plan_identity=plan.plan_identity,
            run_root=run_scope,
            run_scope_root=run_scope,
            config_identity=state.manifest.config_identity,
            data_identity=state.manifest.data_identity,
            code_identity=deterministic_identity(
                {"code_commit": state.manifest.code_commit}
            ),
            max_tool_calls=state.task.budget.remaining("max_actions", int(state.accounting.get("actions", 0))),
            max_resource_units=state.task.budget.max_resource_units,
            run_identity=state.manifest.run_identity,
            artifact_paths={
                parent.artifact.artifact_ref: parent.path for parent in parents
            },
            artifact_digests={
                parent.artifact.artifact_ref: parent.artifact.sha256
                for parent in parents
            },
            runtime_documents={
                "evidence_records": {
                    parent.record.record_id: parent.record.to_dict()
                    for parent in parents
                },
                "coderun_authority": {"owner_managed": True, "action_id": action.action_id,
                    "runner_identity": self._runner_identity, "policy_identity": self._policy.policy_identity,
                    "input_provenance": tuple({"artifact_ref": parent.artifact.artifact_ref,
                        "sha256": parent.artifact.sha256, "evidence_ref": parent.record.record_id} for parent in parents),
                    "attached_evidence_refs": tuple(ref for ref in state.evidence_refs if ref in resources.ledger.record_refs),
                    "authorized_binding_refs": tuple(ref for ref, candidate in resources.bindings.items()
                        if resources.validation_gate.validate_binding(state.task, candidate,
                            catalog_snapshot_identity=resources.catalog_snapshot.snapshot_identity).status.value == "valid")},
            },
            deadline=deadline,
        )
        batch = self._runtime.execute(
            plan,
            state=state,
            bindings=resources.bindings,
            context=base_context,
            catalog_snapshot=resources.catalog_snapshot,
            operation_contracts={action.action_id: _CodeRunExecutionContract(binding, action)},
            allow_success_replay=execution_context.c1_open_control,
        )
        action_observation = batch.observations[0]
        canonical_observation = next(
            item
            for item in reversed(state.observations)
            if item.attempt_id == action_observation.attempt_id
        )
        canonical_refs = (
            CanonicalRecordRef("bound_capability", binding.binding_identity),
            CanonicalRecordRef("action_attempt", action_observation.attempt_id),
            CanonicalRecordRef(
                "action_observation", canonical_observation.observation_id
            ),
        )
        if (
            action_observation.status is not ActionStatus.SUCCESS
            or not batch.execution_records
        ):
            native_failure_code = action_observation.status.value
            public_failure = {'reason':action_observation.summary,'stage':None,'analysis_executed':None,
                'retryable':False,'audit_record_ref':action_observation.attempt_id}
            if batch.execution_records:
                failed_result = batch.execution_records[-1].results[-1]
                native_failure_code = getattr(failed_result.error_code,"value",failed_result.error_code) or native_failure_code
                public_failure['reason'] = failed_result.message or public_failure['reason']
                for provenance in failed_result.provenance:
                    outcome = provenance.get("code_run_outcome")
                    failure = (
                        outcome.get("failure")
                        if isinstance(outcome, Mapping)
                        else None
                    )
                    if isinstance(failure, Mapping) and isinstance(
                        failure.get("code"), str
                    ):
                        native_failure_code = str(failure["code"])
                        detail = failure.get("details", {})
                        public_failure = {"reason": failure["message"],
                            "stage": detail.get("stage"),
                            "analysis_executed": detail.get("analysis_executed"),
                            "retryable": detail.get("retryable", False),
                            "audit_record_ref": outcome.get("audit_record_ref")}
                        capacity = detail.get("capacity")
                        if native_failure_code == "output_contract_invalid" and isinstance(capacity, Mapping):
                            public_failure["capacity"] = dict(capacity)
                        if public_failure["analysis_executed"]:
                            # Native outcomes already contain bounded, redacted worker output.
                            public_failure["diagnostics"] = {key: outcome[key]
                                for key in ("stdout", "traceback") if outcome.get(key)}
                        break
            from .runtime_errors import safe_text
            from .action_diagnostics import copy_text as recovery_copy
            diagnostics=public_failure.pop('diagnostics',{})
            diagnostic_text=str(diagnostics.get('traceback',''))
            reason=str(public_failure.get('reason') or '')
            error_text=reason+' '+diagnostic_text
            selected_refs=selected.artifact_refs
            bad_ref=next((ref for ref in selected_refs if ('FileNotFoundError' in error_text and
                (repr(ref) in error_text or '/workspace/artifacts/'+ref in error_text))),None)
            if bad_ref is not None:
                native_failure_code='input_path'
                reason=recovery_copy('errors','coderun_input_path')
                public_failure['input_ref']=bad_ref
            elif any(text in error_text for text in ('generated code must assign a structured result dict','generated code result must be a JSON object','JSON output must be an object','JSON output requires finite basic values','ValueError: Out of range float values are not JSON compliant')):
                native_failure_code='result_contract_invalid'
                reason=recovery_copy('errors','coderun_result_contract_invalid')
            else:
                reason=safe_text(reason).encode('utf-8')[:450].decode('utf-8',errors='ignore')
                public_failure['recovery']=recovery_copy('errors','coderun_fallback')
            public_failure['reason']=reason
            while json_bytes(public_failure)>1024 and len(public_failure['reason'])>32:
                public_failure['reason']=public_failure['reason'][:-16]
            if json_bytes(public_failure)>1024:
                public_failure.pop('capacity',None)
            return TransitionResult(
                OperationObservation(
                    status="failed",
                    code=f"coderun_{native_failure_code}",
                    summary=public_failure.get("reason", action_observation.summary),
                    canonical_refs=canonical_refs,
                    details={
                        "action_status": action_observation.status.value,
                        "failure_code": native_failure_code,
                        **public_failure,
                    },
                )
            )
        result = batch.execution_records[-1].results[-1]
        value = result.value
        if not isinstance(value, Mapping) or not isinstance(value.get("result"), Mapping):
            return _failure(
                "coderun_result_protocol_invalid",
                "CodeRun succeeded without its canonical JSON-object result.",
            )
        return self._register_result(execution_context, selected, parents, binding, action,
            canonical_observation, result, canonical_refs, run_scope, deadline=deadline)

    def _register_result(self, execution_context, selected, parents, binding, action,
            canonical_observation, result, canonical_refs, run_scope, deadline=None) -> TransitionResult:
        """Register this saved successful execution after rechecking current authority."""
        resources = execution_context.resources
        state = resources.state
        workspace = resources.scientific_workspace
        value = result.value
        if state.phase != "running":
            return _failure("coderun_registration_run_closed", "The saved run is closed; registration requires an active authorized run.",
                {"stage": "result_registration", "analysis_executed": True, "result_registered": False})
        current = self._eligible_artifacts(execution_context)
        if resources.bindings.get(binding.binding_identity) != binding or any(
            parent.artifact.artifact_ref not in current
            or current[parent.artifact.artifact_ref].artifact != parent.artifact for parent in parents):
            return _failure("coderun_registration_inputs_withdrawn",
                "Saved execution cannot be registered after input authority changed.",
                {"stage": "result_registration", "analysis_executed": True, "result_registered": False})
        outcome = value.get("code_run_outcome", {})
        stage_path = Path(outcome.get("stage_record_path") or "")
        try:
            if stage_path.is_symlink() or not stage_path.resolve().is_relative_to(run_scope.resolve()):
                raise ValueError("saved execution stage is outside this run")
            stage = json.loads(stage_path.read_bytes())
            origin = stage["request"]["authority"]
            original_binding = resources.bindings.get(origin["binding_identity"])
            if original_binding is None or resources.validation_gate.validate_binding(
                    state.task, original_binding,
                    catalog_snapshot_identity=resources.catalog_snapshot.snapshot_identity).status.value != "valid":
                raise ValueError("original execution binding is no longer authorized")
            binding = original_binding
            from .c1_final import eligible_evidence_refs
            legal = set(eligible_evidence_refs(execution_context))
            original_parents = []
            for item in origin["input_provenance"]:
                record = resources.ledger.get(item["evidence_ref"])
                artifact = next(a for a in record.envelope.artifacts if a.artifact_ref == item["artifact_ref"])
                if (record.record_id not in legal or artifact.sha256 != item["sha256"]
                    or item["artifact_ref"] not in current or current[item["artifact_ref"]].artifact != artifact):
                    raise ValueError("original input evidence is no longer authorized; recovery cannot change lineage")
                original_parents.append(_EligibleArtifact(artifact, record, workspace.artifacts.resolve(artifact)))
            if tuple(p.artifact.artifact_ref for p in original_parents) != selected.artifact_refs:
                raise ValueError("original input provenance differs from this request")
            parents = tuple(original_parents)
            attempts = {a.attempt_id for a in state.attempts if a.action_id == origin["action_id"]}
            observations = [o for o in state.observations if o.attempt_id in attempts]
            if len(observations) != 1:
                raise ValueError("original execution observation is absent or ambiguous")
            canonical_observation = observations[0]
            invocation_identity = origin["invocation_identity"]
            audit_record_ref = stage["generation_audit_ref"]
            original_action_identity = origin["action_identity"]
            original_runner_identity = origin["runner_identity"]
            original_policy_identity = origin["policy_identity"]
        except (OSError, ValueError, KeyError, StopIteration) as error:
            raise_control_error(error)
            return _failure("coderun_recovery_origin_unavailable", str(error)[:600],
                {"stage": "result_registration", "analysis_executed": True, "result_registered": False})
        result_summary = computation_working_summary(dict(value["result"]))
        result_schema_identity = deterministic_identity(
            {
                "kind": "json_object",
                "fields": tuple(
                    sorted(
                        (str(name), type(item).__name__)
                        for name, item in value["result"].items()
                    )
                ),
            }
        )
        producer_identity = deterministic_identity(
            {
                "binding": binding.binding_identity,
                "action": original_action_identity,
                "invocation": invocation_identity,
                "attempt": canonical_observation.attempt_id,
                "audit": audit_record_ref,
                "code": selected.code_sha256,
                "runner": original_runner_identity,
            }
        )
        try:
            publication = stage.setdefault("publication", {"retrieved_time": self._clock().isoformat()})
            save_stage(stage_path, stage, deadline)
            output_artifacts, output_payloads, result_artifact_ref = self._publish_outputs(
                workspace=workspace,
                action=action,
                value=value,
                base_run_scope=run_scope,
                producer_identity=producer_identity,
                parent_digests=tuple(dict.fromkeys(parent.artifact.sha256 for parent in parents)),
                policy=self._policy,
                deadline=deadline,
            )
            evidence_records = {
                reference: resources.ledger.get(reference)
                for reference in resources.ledger.record_refs
            }
            scientific_scope = None
            parent_refs = tuple(
                dict.fromkeys(parent.record.record_id for parent in parents)
            )
            from .t3_t4.public_json_access import artifact_access_time
            authorized_source_times = {
                parent.record.record_id: artifact_access_time(parent.record.envelope, parent.artifact)
                for parent in parents if getattr(parent.record.envelope, 'available_time', None) is None
            }
            if authorized_source_times:
                result_summary = authorized_time_summary(result_summary, authorized_source_times)
            available_time, retrieved_time = derived_evidence_times(
                evidence_records,
                parent_refs,
                retrieved_time=datetime.fromisoformat(publication["retrieved_time"]),
                authorized_source_times=authorized_source_times,
            )
            envelope = ComputationEvidenceEnvelope(
                producer_kind="guarded_model_code",
                producer_id=str(binding.execution_binding["code_runner_id"]),
                producer_version="1.0.0",
                evidence_role="derived_computation",
                binding_identity=binding.binding_identity,
                action_identity=original_action_identity,
                invocation_identity=invocation_identity,
                attempt_identity=canonical_observation.attempt_id,
                audit_record_ref=audit_record_ref,
                code_sha256=selected.code_sha256,
                runner_identity=original_runner_identity,
                sandbox_policy_identity=original_policy_identity,
                result_schema_identity=result_schema_identity,
                available_time=available_time,
                retrieved_time=retrieved_time,
                input_evidence_refs=parent_refs,
                input_artifact_refs=tuple(
                    parent.artifact.artifact_ref for parent in parents
                ),
                input_artifact_digests=tuple(
                    parent.artifact.sha256 for parent in parents
                ),
                result_summary=result_summary,
                result_artifact_ref=result_artifact_ref,
                artifacts=output_artifacts,
                lineage_refs=parent_refs,
                scientific_scope=scientific_scope,
            )
            check_deadline(deadline)
            record_id = deterministic_identity({"envelope_identity": envelope.envelope_identity,
                "payload_ref": None, "payload_refs": tuple(p.payload_ref for p in output_payloads)})
            if publication.get("record_id", record_id) != record_id:
                raise ValueError("saved publication identity changed during recovery")
            publication["record_id"] = record_id
            save_stage(stage_path, stage, deadline)
            record = (resources.ledger.get(record_id) if record_id in resources.ledger.record_refs
                else resources.ledger.append_system_record(envelope, payload_refs=output_payloads))
            check_deadline(deadline)
            if record_id not in state.evidence_refs:
                state.attach_evidence_record(canonical_observation.observation_id, record)
            stage["closed"] = True
            stage["stage"] = "registered"
            save_stage(stage_path, stage, deadline)
        except (KeyError, OSError, TypeError, ValueError) as error:
            raise_control_error(error)
            interrupted = isinstance(error, OSError)
            return TransitionResult(
                OperationObservation(
                    status="failed",
                    code="coderun_evidence_registration_failed",
                    summary=(MESSAGES["registration_interrupted"] if interrupted else
                        "Guarded execution completed but computation evidence validation failed."),
                    canonical_refs=canonical_refs,
                    details={"reason": str(error)[:600], "stage": "result_registration",
                        "failure_class": "registration_interrupted" if interrupted else "registration_invalid",
                        "analysis_executed": True, "result_registered": False,
                        "execution_read_ref": canonical_observation.observation_id,
                        "audit_record_ref": audit_record_ref},
                )
            )
        return TransitionResult(
            OperationObservation(
                status="succeeded",
                code="coderun_succeeded",
                summary="Guarded model code produced lineage-backed computation evidence.",
                canonical_refs=(
                    *canonical_refs,
                    CanonicalRecordRef("coderun_audit", audit_record_ref),
                    CanonicalRecordRef("evidence", record.record_id),
                    *(
                        CanonicalRecordRef("artifact", artifact.artifact_ref)
                        for artifact in output_artifacts
                    ),
                ),
                details={
                    "trust": "guarded-model-code-derived-computation",
                    "result_summary": result_summary,
                    "input_evidence_refs": parent_refs,
                    "output_artifact_refs": tuple(
                        artifact.artifact_ref for artifact in output_artifacts
                    ),
                },
            )
        )

    @staticmethod
    def _publish_outputs(*, workspace: Any, action: ActionRequest, value: Mapping[str, Any],
        base_run_scope: Path, producer_identity: str, parent_digests: tuple[str, ...],
        policy: ArtifactCodeRunPolicy,
        deadline: float | None = None,
    ) -> tuple[tuple[ArtifactRef, ...], tuple[ScientificPayloadRef, ...], str]:
        outcome = value.get("code_run_outcome")
        if not isinstance(outcome, Mapping):
            raise ValueError("CodeRun outcome is missing")
        if outcome.get("execution_root"):
            return ArtifactCodeRunProvider._import_normalized_outputs(workspace=workspace, outcome=outcome,
                base_run_scope=base_run_scope, producer_identity=producer_identity,
                parent_digests=parent_digests, deadline=deadline)
        raise ValueError("Historical raw CodeRun output requires fixed checking before import")

    @staticmethod
    def _import_normalized_outputs(*, workspace, outcome, base_run_scope,
                                   producer_identity, parent_digests, deadline):
        root = Path(outcome["execution_root"])
        if root.is_symlink() or not root.resolve().is_relative_to(base_run_scope.resolve()):
            raise ValueError("saved CodeRun execution is outside this run")
        usage = outcome["resource_usage"]
        if usage["decoded_accounted_bytes"] > workspace.artifacts.max_uncompressed_bytes or \
                usage["elements"] > workspace.artifacts.max_elements:
            raise ValueError("normalized output exceeds this store's decoded quota")
        files = outcome["artifacts"]
        # Legacy outcomes have no per-file version. Recover only from their
        # original checked stage, with matching deployment and saved metadata.
        if any("checker_contract_version" not in item["normalized_metadata"] for item in files):
            from .coderun_stages import checked_files
            import json
            stage_path=Path(outcome["stage_record_path"])
            if stage_path.is_symlink() or stage_path.resolve()!=root.resolve()/"stage.json":
                raise ValueError("outcome stage escaped its execution root")
            stage=json.loads(stage_path.read_bytes())
            if (stage["checked"]["checker_identity"]!=usage["checker_identity"]
                    or stage["request"]["generation_identity"]!=usage["generation_identity"]):
                raise ValueError("legacy outcome stage identity differs")
            saved={str(Path(item["path"]).relative_to(root)):item for item in checked_files(stage_path,stage,deadline)}
            restored=[]
            for item in files:
                original=saved.get(item["relative_path"])
                if original is None:raise ValueError("legacy normalized file absent from checked stage")
                expected={k:v for k,v in original.items() if k not in ("path","checker_contract_version")}
                if item["normalized_metadata"]!=expected:
                    raise ValueError("legacy outcome metadata differs from checked stage")
                restored.append({**item,"normalized_metadata":{**expected,
                    "checker_contract_version":original["checker_contract_version"]}})
            files=restored
            usage={**usage,"checker_contract_version":stage["checked"]["receipt"]["checker_version"]}
        if not 1 <= len(files) <= 4 or sum(item["size_bytes"] for item in files) > workspace.artifacts.max_artifact_bytes:
            raise ValueError("normalized output collection exceeds store quota")
        if sum(bool(item["automatic"]) for item in files) != 1:
            raise ValueError("one normalized automatic result is required")
        groups = {}
        for item in files:
            relative = Path(item["relative_path"])
            path = root / relative
            if relative.is_absolute() or ".." in relative.parts or path.is_symlink() or \
                    not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("normalized file escaped its original execution root")
            metadata = item["normalized_metadata"]
            if metadata.get("checker_contract_version") != usage.get("checker_contract_version"):
                raise ValueError("normalized outcome checker version differs")
            if metadata["sha256"] != item["sha256"] or metadata["bytes"] != item["size_bytes"]:
                raise ValueError("normalized outcome differs from fixed receipt")
            groups.setdefault((item["sha256"], item["media_type"]), []).append((item, path))
        artifacts, payloads, result_ref = [], [], None
        for group in groups.values():
            check_deadline(deadline)
            first, path = group[0]
            names = tuple({"output_name": item["output_name"], "automatic": item["automatic"],
                           "raw_sha256": item["normalized_metadata"]["raw_sha256"]} for item, _ in group)
            stored = workspace.artifacts._import_normalized_file(path, first["normalized_metadata"],
                producer_identity=producer_identity, parent_refs=parent_digests, output_names=names,
                checker_identity=usage["checker_identity"], deadline=deadline)
            artifacts.append(stored.artifact)
            payloads.append(stored.payload)
            if any(item["automatic"] for item, _ in group):
                result_ref = stored.artifact.artifact_ref
        assert result_ref is not None
        return tuple(artifacts), tuple(payloads), result_ref


def _failure(
    code: str,
    summary: str,
    details: Mapping[str, Any] | None = None,
    *,
    status: str = "failed",
) -> TransitionResult:
    return TransitionResult(
        OperationObservation(
            status=status,
            code=code,
            summary=summary,
            details={} if details is None else details,
        )
    )


__all__ = [
    "CODERUN_OPERATION_SCHEMA_VERSION",
    "MAX_CODERUN_ARTIFACT_REFS",
    "MAX_CODERUN_CHOICE_BYTES",
    "MAX_CODERUN_CODE_BYTES",
    "ArtifactCodeRunPolicy",
    "ArtifactCodeRunProvider",
]
