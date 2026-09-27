"""Public action discovery and invocation through explicit capability owners."""
from __future__ import annotations
from .runtime_records import PUBLIC_EVIDENCE_TYPES
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
from ._serialization import to_primitive
from .action_contracts import ActionBindingFailure, ActionEffectContractError, ActionKind, ActionOperationOwner, ActionOwnerPreparation, ActionPlan, ActionRequest, ActionStatus, CanonicalActionEffect
from .action_contracts import _is_sha256, canonical_owner_action_effect
from .actions import ActionRuntime, _unique_typed_result_members, material_state_epoch_identity
from .capability_catalog import BoundCapability
from .catalog_navigation import select_activated_binding_identities
from .executable_operations import (ArgumentValidationError, CanonicalRecordRef, ChoiceSpec, C1OperationProvider, ExecutableOperation, OperationContext, OperationExecutionContext, OperationObservation, TransitionResult)
from .reliability import deterministic_identity
from .run_state import PlannedAction, RunState
from .runtime import ToolResult
from .runtime_records import ArtifactRef, EvidenceEnvelope, LLMAttemptRef, ScientificPayloadRef
from .scientific_products import ScientificProduct
from .tooling import ExecutionContext

@dataclass(frozen=True)
class ActionOperationRegistration:
    """One owner-registered ActionRuntime operation."""

    semantic_key: str
    capability_ref: str
    action_id: str
    purpose: str
    expected_observation: str
    binding_request_identity: str | None = None
    bound_arguments: Mapping[str, Any] = field(default_factory=dict)

    owner: ActionOperationOwner | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not all(
            (
                self.semantic_key,
                self.capability_ref,
                self.action_id,
                self.purpose,
                self.expected_observation,
            )
        ):
            raise ValueError("action operation registration fields are required")
        if self.binding_request_identity is not None and len(
            self.binding_request_identity
        ) != 64:
            raise ValueError("action binding request identity must be SHA-256")
        owner_identity = None
        if self.owner is not None:
            owner_identity = str(getattr(self.owner, "contract_identity", ""))
            if not _is_sha256(owner_identity):
                raise ValueError("action owner contract identity must be SHA-256")
        deterministic_identity(self.audit_values)

    @property
    def audit_values(self) -> Mapping[str, Any]:
        return {
            "semantic_key": self.semantic_key,
            "capability_ref": self.capability_ref,
            "action_id": self.action_id,
            "purpose": self.purpose,
            "expected_observation": self.expected_observation,
            "binding_request_identity": self.binding_request_identity,
            "bound_arguments": to_primitive(self.bound_arguments),
            "owner_contract_identity": (
                None if self.owner is None else self.owner.contract_identity
            ),
        }


class _BoundActionOperationContract:
    """One Action owner contract supplies both minimal schema and ActionRequest."""

    @classmethod
    def materialize(
        cls,
        registration: ActionOperationRegistration,
        binding: BoundCapability,
        context: OperationContext | OperationExecutionContext,
    ) -> tuple["_BoundActionOperationContract", ...]:
        if registration.owner is None:
            raise ValueError("public action operation requires an explicit owner")
        preparations = registration.owner.prepare(
            to_primitive(registration.bound_arguments),
            binding,
            context,
        )
        if not isinstance(preparations, tuple) or any(
            not isinstance(item, ActionOwnerPreparation)
            or item.arguments_schema is None
            or item.frozen_arguments is None
            for item in preparations
        ):
            raise TypeError("action owner must return complete preparations")
        identities = tuple(item.projection_identity for item in preparations)
        if len(identities) != len(set(identities)):
            raise RuntimeError("action owner returned duplicate preparations")
        if len(preparations) > 1:
            decision_refs = tuple(item.decision_ref for item in preparations)
            if None in decision_refs or len(decision_refs) != len(set(decision_refs)):
                raise RuntimeError(
                    "multiple owner preparations require unique decision refs"
                )
        return tuple(
            cls(
                registration,
                binding,
                context,
                projection=preparation,
            )
            for preparation in preparations
        )

    def __init__(
        self,
        registration: ActionOperationRegistration,
        binding: BoundCapability,
        context: OperationContext | OperationExecutionContext,
        *,
        projection: ActionOwnerPreparation,
    ) -> None:
        self._registration = registration
        self._binding = binding
        self._task = context.task
        if registration.owner is None or projection.frozen_arguments is None or projection.arguments_schema is None:
            raise TypeError("owner contract requires one complete preparation")
        self._owner = registration.owner
        self._projection = projection
        self._frozen_arguments = dict(to_primitive(projection.frozen_arguments))
        self._prepared_input_identity = projection.projection_identity
        self._decision_ref = projection.decision_ref
        self._owner_summary = projection.owner_summary
        instance_identity = deterministic_identity({
            "registration_action_id": registration.action_id,
            "binding_identity": binding.binding_identity,
            "prepared_projection_identity": self._prepared_input_identity,
        })
        self._action_id = f"{registration.action_id}-{instance_identity}"
        self._schema = dict(to_primitive(projection.arguments_schema))
        self._action_kind = ActionKind(
            str(binding.execution_binding.get("action_kind"))
        )
        self._timeout = float(binding.resource.get('max_timeout_seconds',
            binding.resource.get('default_timeout_seconds', 120)))
        if context.task.budget.timeout_seconds is not None:
            self._timeout = min(self._timeout, context.task.budget.timeout_seconds)
        self._resource_kind = str(binding.resource.get("kind", "cpu"))

    @property
    def action_id(self) -> str:
        return self._action_id

    @property
    def prepared_input_identity(self) -> str:
        return self._prepared_input_identity

    @property
    def decision_ref(self) -> str | None:
        return self._decision_ref

    @property
    def owner_summary(self) -> str | None:
        return self._owner_summary

    @property
    def schema(self) -> Mapping[str, Any]:
        return self._schema

    @property
    def audit_values(self) -> Mapping[str, Any]:
        return {
            "action_id": self._action_id,
            "binding_identity": self._binding.binding_identity,
            "binding_request_identity": self._binding.request_identity,
            "task_identity": self._task.task_identity,
            "action_kind": self._action_kind.value,
            "timeout_seconds": self._timeout,
            "resource_kind": self._resource_kind,
            "resource_units": 1,
            "max_retries": 0,
            "frozen_arguments": to_primitive(self._frozen_arguments),
            "prepared_input_identity": self._prepared_input_identity,
            "decision_ref": self._decision_ref,
            "argument_contract": self._binding.execution_binding.get(
                "argument_contract"
            ),
        }

    def invocation_arguments(self, action: ActionRequest) -> Mapping[str, Any]:
        return dict(self._binding.parameters) | dict(action.arguments)

    def canonical_effect(self, action: ActionRequest) -> CanonicalActionEffect:
        return canonical_owner_action_effect(action, self._binding, self._owner.contract_identity)

    def binding_error(
        self,
        action: ActionRequest,
        *,
        state: RunState,
        context: ExecutionContext,
    ) -> ActionBindingFailure | str | None:
        return self._owner.binding_error(
            action,
            self._projection,
            self._binding,
            state,
            context,
        )

    @property
    def owner_managed(self) -> bool:
        return True

    def requested_artifact_refs(self, action: ActionRequest) -> tuple[str, ...]:
        if self._projection.selected_artifact_refs_path:
            selected = action.arguments
            for key in self._projection.selected_artifact_refs_path:
                selected = selected[key]
            refs = tuple(selected.values()) if isinstance(selected, Mapping) else tuple(selected)
            if not set(refs).issubset(self._projection.input_artifact_refs):
                raise ValueError("selected inputs are outside the owner preparation")
            return refs
        return self._projection.input_artifact_refs

    def project_result(self, result: ToolResult) -> Mapping[str, Any] | None:
        projected = self._owner.project_result(result)
        if projected is None:
            return None
        if not isinstance(projected, Mapping):
            raise TypeError("action owner result projection must be a mapping")
        return to_primitive(projected)

    def validate(self, arguments: Mapping[str, object]) -> ActionRequest:
        from .action_diagnostics import validate_envelope
        # Only the outer object belongs here. The owner retains all nested semantics.
        properties = self._schema.get("properties", {})
        if not isinstance(properties, Mapping):
            raise RuntimeError("invalid bound operation schema")
        validate_envelope(arguments, {
            "type": "object", "properties": {key: {} for key in properties},
            "required": self._schema.get("required", ()), "additionalProperties": False,
        }, "action_operation_shape_invalid" if isinstance(arguments, Mapping) else "arguments_not_object")
        try:
            canonical_arguments = to_primitive(
                self._owner.canonicalize(arguments, self._projection, self._binding)
            )
            action = ActionRequest(
                action_id=self._action_id,
                kind=self._action_kind,
                bound_capability_ref=self._binding.binding_identity,
                arguments=canonical_arguments,
                dependencies=(),
                timeout_seconds=self._timeout,
                resource_kind=self._resource_kind,
                resource_units=1,
                max_retries=0,
            )
            self.canonical_effect(action)
            return action
        except ArgumentValidationError:
            raise
        except (KeyError, TypeError, ValueError, ActionEffectContractError) as error:
            raise ArgumentValidationError(
                "action_arguments_invalid",
                "arguments",
                f"arguments violate the bound action contract: {type(error).__name__}",
            ) from None


class ActionOperationProvider(C1OperationProvider):
    """Owner-local adapter from bound actions to ActionRuntime operations."""

    requires_explicit_owner = True

    def __init__(
        self,
        runtime: ActionRuntime,
        registrations: tuple[ActionOperationRegistration, ...],
        *,
        run_root: Path,
    ) -> None:
        if not registrations:
            raise ValueError("ActionOperationProvider requires registrations")
        if self.requires_explicit_owner and any(item.owner is None for item in registrations):
            raise ValueError("public ActionOperationProvider requires explicit owners")
        keys = tuple(item.semantic_key for item in registrations)
        action_ids = tuple(item.action_id for item in registrations)
        if len(keys) != len(set(keys)) or len(action_ids) != len(set(action_ids)):
            raise ValueError("action operation registrations must be unique")
        self._runtime = runtime
        self._registrations = registrations
        self._run_root = Path(run_root)

    def _materialize_contracts(self, registration, binding, context):
        return _BoundActionOperationContract.materialize(registration, binding, context)

    def offer(self, context: OperationContext) -> tuple[ExecutableOperation, ...]:
        resources = context.resources
        state = resources.state
        if not context.task.budget.allows("max_actions", int(state.accounting.get("actions", 0))):
            return ()
        snapshot = resources.catalog_snapshot
        access_policy_identity = deterministic_identity(context.task.access_policy)
        if (
            snapshot is None
            or state.catalog_snapshot_identity != snapshot.snapshot_identity
            or snapshot.access_policy_identity != access_policy_identity
        ):
            return ()
        active_bindings = None
        if context.c1_open_control and resources.catalog_navigation is not None:
            active_bindings = set(select_activated_binding_identities(
                resources.catalog_navigation.binding_activations,
                resolved_scope_identity=context.task.resolved_scope_identity,
                snapshot_identity=snapshot.snapshot_identity,
                current_binding_identities=tuple(resources.bindings),
            ))
        candidates: list[
            tuple[
                ActionOperationRegistration,
                BoundCapability,
                _BoundActionOperationContract,
                str,
                str,
            ]
        ] = []
        for registration in self._registrations:
            matching = tuple(
                binding
                for binding in resources.bindings.values()
                if binding.capability_ref == registration.capability_ref
                and (active_bindings is None or binding.binding_identity in active_bindings)
                and (
                    registration.binding_request_identity is None
                    or binding.request_identity == registration.binding_request_identity
                )
            )
            if not matching:
                continue
            if len(matching) != 1:
                raise ValueError("action registration resolved multiple current bindings")
            binding = matching[0]
            if binding.access_policy_identity != access_policy_identity:
                continue
            validation = resources.validation_gate.validate_binding(
                context.task,
                binding,
                catalog_snapshot_identity=snapshot.snapshot_identity,
            )
            if (
                validation.status.value != "valid"
                or snapshot.descriptor_identity_for(binding.capability_ref)
                != binding.descriptor_identity
            ):
                continue
            contracts = self._materialize_contracts(
                registration, binding, context
            )
            for contract in contracts:
                semantic_key = registration.semantic_key
                purpose = registration.purpose
                if len(contracts) > 1:
                    assert contract.decision_ref is not None
                    semantic_key = f"{semantic_key}:{contract.decision_ref[:12]}"
                if len(contracts) > 1 and contract.owner_summary:
                    purpose = f"{purpose} {contract.owner_summary}"
                candidates.append(
                    (registration, binding, contract, semantic_key, purpose)
                )
        semantic_keys = tuple(item[3] for item in candidates)
        if len(semantic_keys) != len(set(semantic_keys)):
            raise RuntimeError("prepared Action semantic keys must be unique")

        operations: list[ExecutableOperation] = []
        for registration, binding, contract, semantic_key, purpose in candidates:
            if not context.c1_open_control and any(
                attempt.action_id == contract.action_id
                for attempt in state.attempts
            ):
                continue

            def invoke(
                execution_context: OperationExecutionContext,
                arguments: object,
                *,
                current_registration: ActionOperationRegistration = registration,
                current_binding: BoundCapability = binding,
                offered_contract: _BoundActionOperationContract = contract,
            ) -> TransitionResult:
                if not isinstance(arguments, ActionRequest):
                    raise TypeError("action operation arguments were not owner-validated")
                current_resources = execution_context.resources
                state = current_resources.state
                if arguments.bound_capability_ref != current_binding.binding_identity:
                    raise RuntimeError("validated action changed its frozen binding")
                current_contract = next(
                    (
                        candidate
                        for candidate in self._materialize_contracts(
                            current_registration,
                            current_binding,
                            execution_context,
                        )
                        if candidate.prepared_input_identity
                        == offered_contract.prepared_input_identity
                    ),
                    None,
                )
                if (
                    current_contract is None
                    or current_contract.prepared_input_identity
                    != offered_contract.prepared_input_identity
                ):
                    return TransitionResult(
                        OperationObservation(
                            status="failed",
                            code="action_projection_stale",
                            summary="Action owner inputs changed before invocation.",
                            canonical_refs=(
                                CanonicalRecordRef(
                                    "bound_capability",
                                    current_binding.binding_identity,
                                ),
                            ),
                        )
                    )
                state.revise_plan(
                    # The plan is a current action projection, not the attempt log.
                    # Explicit retries retain the same action identity; the runtime
                    # appends distinct attempts/observations below.
                    steps=tuple(
                        step for step in state.plan.steps
                        if step.action_id != arguments.action_id
                    )
                    + (
                        PlannedAction(
                            action_id=arguments.action_id,
                            purpose=current_registration.purpose,
                            capability_ref=current_binding.binding_identity,
                            status="ready",
                        ),
                    ),
                    reason=f"selected operation {current_registration.semantic_key}",
                )
                plan = ActionPlan((arguments,), plan_revision=state.plan.revision)
                run_scope = self._run_root / state.manifest.run_identity
                workspace = current_resources.scientific_workspace
                if workspace is not None and (
                    workspace.run_identity != state.manifest.run_identity
                ):
                    raise RuntimeError("scientific workspace belongs to another run")
                evidence_records = {
                    reference: current_resources.ledger.get(reference)
                    for reference in current_resources.ledger.record_refs
                }
                artifact_records = {
                    artifact.artifact_ref: artifact
                    for record in evidence_records.values()
                    for artifact in record.envelope.artifacts
                }
                if artifact_records and workspace is None:
                    raise RuntimeError(
                        "artifact-backed action execution requires the shared workspace"
                    )
                execution = ExecutionContext(
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
                    deadline=execution_context.deadline,
                    run_identity=state.manifest.run_identity,
                    artifact_paths={
                        reference: workspace.artifacts.resolve(artifact)
                        for reference, artifact in artifact_records.items()
                    }
                    if workspace is not None
                    else {},
                    artifact_digests={
                        reference: artifact.sha256
                        for reference, artifact in artifact_records.items()
                    },
                    runtime_documents={
                        "task_authority": {
                            "source_authority": current_resources.source_authority,
                            "decision_time": state.task.decision_time.isoformat(),
                            "resolved_scope_identity": state.task.resolved_scope_identity,
                            "binding_identity": current_binding.binding_identity,
                            "valid_start": to_primitive(current_binding.valid_start),
                            "valid_end": to_primitive(current_binding.valid_end),
                            "spatial": to_primitive(current_binding.spatial),
                        },
                        "artifact_refs": artifact_records,
                        "evidence_records": {
                            reference: record.to_dict()
                            for reference, record in evidence_records.items()
                        },
                        "scientific_products": dict(state.scientific_products),
                        "forecast_contracts": dict(state.task.disclosed_context.get("target_contracts", {})),
                    },
                )
                batch = self._runtime.execute(
                    plan,
                    state=state,
                    bindings=current_resources.bindings,
                    context=execution,
                    catalog_snapshot=current_resources.catalog_snapshot,
                    operation_contracts={
                        arguments.action_id: offered_contract
                    },
                    allow_success_replay=execution_context.c1_open_control,
                )
                action_observation = batch.observations[0]
                canonical_observation = next(
                    item
                    for item in reversed(state.observations)
                    if item.attempt_id == action_observation.attempt_id
                )
                canonical_refs = [
                    CanonicalRecordRef(
                        "bound_capability", current_binding.binding_identity
                    ),
                    CanonicalRecordRef(
                        "action_attempt", action_observation.attempt_id
                    ),
                    CanonicalRecordRef(
                        "action_observation",
                        canonical_observation.observation_id,
                    ),
                ]
                if (
                    action_observation.status is ActionStatus.SUCCESS
                    and batch.execution_records
                ):
                    result = batch.execution_records[-1].results[-1]
                    value = result.value
                    artifacts = _unique_typed_result_members(
                        value,
                        ArtifactRef,
                        "artifact_ref",
                    )
                    envelopes = _unique_typed_result_members(
                        value,
                        PUBLIC_EVIDENCE_TYPES,
                        "envelope_identity",
                    )
                    payloads = _unique_typed_result_members(
                        value,
                        ScientificPayloadRef,
                        "payload_ref",
                    )
                    attempts = _unique_typed_result_members(
                        value,
                        LLMAttemptRef,
                        "attempt_ref",
                    )
                    products = _unique_typed_result_members(
                        value,
                        ScientificProduct,
                        "product_ref",
                    )
                    all_artifacts = {
                        artifact.artifact_ref: artifact
                        for envelope in envelopes
                        for artifact in envelope.artifacts
                    }
                    all_artifacts.update(
                        {artifact.artifact_ref: artifact for artifact in artifacts}
                    )
                    canonical_refs.extend(
                        CanonicalRecordRef("artifact", reference)
                        for reference in all_artifacts
                    )
                    for envelope in envelopes:
                        envelope_artifact_refs = {
                            artifact.artifact_ref for artifact in envelope.artifacts
                        }
                        matching_payloads = tuple(
                            payload
                            for payload in payloads
                            if payload.artifact_ref in envelope_artifact_refs
                        )
                        primary = value.get('payload_ref') if isinstance(value, Mapping) else None
                        if len(matching_payloads) > 1 and isinstance(primary, ScientificPayloadRef):
                            if primary not in matching_payloads:
                                raise TypeError('primary scientific payload is not carried by the result envelope')
                            matching_payloads = (primary, *(p for p in matching_payloads if p != primary))
                        record = current_resources.ledger.append_system_record(
                            envelope,
                            matching_payloads[0] if len(matching_payloads) == 1 else None,
                            payload_refs=matching_payloads if len(matching_payloads) > 1 else (),
                        )
                        observed_ref = envelope.envelope_identity
                        if (
                            observed_ref != record.record_id
                            and observed_ref in action_observation.evidence_refs
                        ):
                            state.reconcile_observation_evidence_ref(
                                observed_ref, record.record_id
                            )
                        state.attach_evidence_record(
                            canonical_observation.observation_id, record
                        )
                        canonical_refs.append(
                            CanonicalRecordRef("evidence", record.record_id)
                        )
                    canonical_refs.extend(
                        CanonicalRecordRef(
                            "tool_llm_attempt",
                            attempt.attempt_ref,
                        )
                        for attempt in attempts
                    )
                    for product in products:
                        state.record_scientific_product(product)
                        canonical_refs.append(
                            CanonicalRecordRef("scientific_product", product.product_ref)
                        )
                succeeded = action_observation.status is ActionStatus.SUCCESS
                details: dict[str, Any] = {
                    "action_status": action_observation.status.value,
                    "batch_identity": batch.batch_identity,
                }
                action_result = action_observation.action_result
                owner_projection = (
                    action_result.get("owner_projection")
                    if isinstance(action_result, Mapping)
                    else None
                )
                if isinstance(owner_projection, Mapping):
                    details["result"] = owner_projection
                return TransitionResult(
                    OperationObservation(
                        status="succeeded" if succeeded else "failed",
                        code=(
                            "action_succeeded"
                            if succeeded
                            else "action_failed"
                        ),
                        summary=action_observation.summary,
                        canonical_refs=tuple(canonical_refs),
                        details=details,
                    )
                )

            operations.append(
                ExecutableOperation(
                    public=ChoiceSpec(
                        semantic_key=semantic_key,
                        purpose=purpose,
                        arguments_schema=contract.schema,
                        expected_observation=registration.expected_observation,
                    ),
                    argument_contract=contract,
                    invoke=invoke,
                    audit_identity=deterministic_identity(
                        {
                            "owner": "action_runtime",
                            "registration": registration.audit_values,
                            "contract": contract.audit_values,
                            "run_identity": resources.state.manifest.run_identity,
                            "run_root": str(self._run_root),
                            "state_epoch": material_state_epoch_identity(
                                resources.state, binding
                            ),
                            "catalog_snapshot": (
                                None
                                if resources.catalog_snapshot is None
                                else resources.catalog_snapshot.snapshot_identity
                            ),
                        }
                    ),
                    manages_action_accounting=True,
                )
            )
        return tuple(operations)
