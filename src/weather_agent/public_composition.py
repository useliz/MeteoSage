"""Public answer assembly from installed, paired scientific capability owners."""
from __future__ import annotations
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from .action_operations import ActionOperationProvider, ActionOperationRegistration
from .actions import ActionRuntime
from .activated_operations import ActivatedOwnerOperationProvider, ActivatedOwnerRegistration
from .c1_final import C1FinalOperationProvider
from .c1_intake import OperationalIntakeContext, OperationalTaskIntake
from .capability_catalog import BoundCapability, CapabilityDescriptor, CatalogSnapshot
from .catalog_navigation import CatalogNavigationProvider
from .coderun import NativeCodeRunAdapter, NativeCodeRunner
from .coderun_operation import ArtifactCodeRunPolicy, ArtifactCodeRunProvider, CODERUN_DESCRIPTION, frozen_code_binding
from .episode_runner import EpisodeResult, _new_episode_resources
from .execution import ExecutionModule
from .production_source_operations import production_source_operation_registrations
from .production_sources import ProductionSourceRegistration
from .record_inspection import RecordsInspectProvider
from .reliability import ModelIdentity, deterministic_identity
from .retrieval_scope import FrozenRetrievalScopeRegistry, ResolvedRetrievalScope
from .runtime_contracts import AccessPolicy, RunBudget, RuntimeTask
from .scientific_composition import build_owned_scientific_composition
from .system import MemoryConfig, WeatherAgent
from .workspace import ScientificWorkspace

CODE_RUNNER_ID = "native-coderun"


def coderun_descriptor(generation_identity: str | None = None) -> CapabilityDescriptor:
    return CapabilityDescriptor(
        capability_id="artifact-python", version="1.0.0", namespace="science.runtime.coderun",
        description=CODERUN_DESCRIPTION, kind="computation", variables=("artifact",),
        products=("derived-artifact",), evidence_roles=("derived_computation",),
        spatial_coverage={"global": True}, temporal_coverage={"timeless": True},
        lead_time_coverage={}, schema_identity=deterministic_identity(("artifact-python", "1.0.0")),
        time_semantics={"kind": "derived_input_valid_interval"},
        access={"enabled": True, "requires_generated_code": True}, resource={"kind": "cpu"},
        limitations=(), execution_binding={"action_kind": "code_run", "code_runner_id": CODE_RUNNER_ID,
            "argument_contract": "code_run_v1", "required_selectors": (),
            **({"generation_identity": generation_identity} if generation_identity else {})})




def _scope(
    code_commit: str,
    name: str,
    descriptors: tuple[CapabilityDescriptor, ...],
    authorized_source_objects: tuple[str, ...] | None = None,
) -> ResolvedRetrievalScope:
    tool_refs = tuple(
        sorted(
            {
                str(descriptor.execution_binding[key])
                for descriptor in descriptors
                for key in ("tool_id", "code_runner_id", "script_id")
                if isinstance(descriptor.execution_binding.get(key), str)
            }
        )
    )
    source_objects = tuple(
        sorted(
            {
                str(reference)
                for descriptor in descriptors
                for reference in descriptor.execution_binding.get("source_object_refs", ())
            }
        )
    )
    if authorized_source_objects is not None:
        source_objects = tuple(sorted(set(authorized_source_objects)))
    scope_ref = f"retrieval-scope:weather-agent-{name}@1"
    return FrozenRetrievalScopeRegistry(
        registry_commit=code_commit,
        resolver_id=f"weather-agent-{name}-resolver",
        resolver_version="1.0.0",
        acl_policy={
            "policy": "installed-scientific-capabilities",
            "descriptor_identities": tuple(
                sorted(item.descriptor_identity for item in descriptors)
            ),
            "source_objects": source_objects,
        },
        entries=(
            {
                "scope_ref": scope_ref,
                "allowed_source_family_refs": tuple(
                    sorted(
                        item.source_family_ref
                        for item in descriptors
                        if item.source_family_ref is not None
                    )
                ),
                "allowed_capability_refs": tuple(
                    sorted(item.capability_ref for item in descriptors)
                ),
                "allowed_tool_refs": tool_refs,
                "allowed_source_object_refs": source_objects,
                "evaluator_only_deny_refs": (),
            },
        ),
    ).resolve(scope_ref)


@dataclass
class PublicComposition:
    question: str
    context: OperationalIntakeContext
    intake: OperationalTaskIntake
    task: RuntimeTask
    scope: ResolvedRetrievalScope
    providers: tuple[Any, ...]
    resources: Any
    run_root: Path
    memory: MemoryConfig | None = None
    code_runner: NativeCodeRunner | None = None
    close_callbacks: tuple[Any, ...] = ()
    max_protocol_corrections: int = 2

    def run(self, adapter: Any, *, working_state=None, maintenance_adapter=None,
            memory_selection_transport=None, host_deadline=None, memory_package_identity=None) -> EpisodeResult:
        try:
            result = WeatherAgent(
                self.providers,
                max_turn_characters=24 * 1024,
                max_protocol_corrections=self.max_protocol_corrections,
                resources_factory=lambda task: self._resources(task),
                task_intake=self.intake,
                memory=self.memory,
                memory_selection_transport=memory_selection_transport, host_deadline=host_deadline,
                memory_package_identity=memory_package_identity,
                working_state=working_state or getattr(self.resources,"working_state_config",None),
                maintenance_adapter=maintenance_adapter,
            ).answer(self.question, self.context, adapter)
        finally:
            from .knowledge_read_transaction import rollback_delivery
            rollback_delivery(self.resources)
            for close in self.close_callbacks:
                close()
        if not isinstance(result, EpisodeResult):
            raise RuntimeError("public answer unexpectedly requested clarification")
        return result

    def _resources(self, task: RuntimeTask) -> Any:
        if task.task_identity != self.task.task_identity:
            raise ValueError("public resources belong to another task")
        return self.resources


def assemble_public_answer(
    *,
    code_commit: str,
    name: str,
    question: str,
    decision_time: datetime,
    valid_start: datetime | None,
    valid_end: datetime | None,
    spatial: Mapping[str, Any] | None,
    budget: RunBudget,
    source: ProductionSourceRegistration | None,
    source_tool: Any,
    descriptors: tuple[CapabilityDescriptor, ...],
    registrations: tuple[ActionOperationRegistration, ...],
    tools: Mapping[str, Any],
    run_root: Path,
    disclosed_context: Mapping[str, Any],
    model: ModelIdentity,
    include_coderun: bool,
    max_protocol_corrections: int = 2,
    report_required: bool = False,
    report_format: str | None = None,
    additional_sources: tuple[tuple[ProductionSourceRegistration, Any], ...] = (),
    weather_target_required: bool = True,
    submission_schema: Mapping[str, Any] | None = None,
    coderun_backend: str = "bwrap",
    coderun_deployment: Path | None = None,
    resolved_scope: ResolvedRetrievalScope | None = None,
    source_authority: Mapping[str, Any] | None = None,
    public_records: Mapping[str, Any] | None = None,
    public_record_descriptors: Mapping[str, Any] | None = None,
    knowledge_registry=None, knowledge_error=None, knowledge_visible_refs=None, knowledge_bound_rules=(), knowledge_service_config=None,
) -> PublicComposition:
    from .t3_t4.model_profiles import validate_protocol_corrections
    validate_protocol_corrections(max_protocol_corrections)
    source_pairs = (() if source is None else ((source, source_tool),)) + additional_sources
    source_descriptors = tuple(item.capability_descriptor() for item, _ in source_pairs)
    if len({item.capability_ref for item in source_descriptors}) != len(source_descriptors):
        raise ValueError("duplicate public source registration")
    knowledge_descriptors = tuple(entry.to_capability_descriptor() for entry in knowledge_registry.entries
        if entry.review_status == 'approved' and (knowledge_visible_refs is None or entry.to_ref().entry_ref in knowledge_visible_refs)) if knowledge_registry else ()
    all_descriptors = (*source_descriptors, *descriptors, *knowledge_descriptors)
    code_runner = NativeCodeRunner(isolation_backend=coderun_backend, deployment=coderun_deployment) if include_coderun else None
    code = coderun_descriptor(code_runner.generation_identity) if code_runner else None
    if code is not None:
        all_descriptors += (code,)
    if resolved_scope is not None and not isinstance(resolved_scope, ResolvedRetrievalScope):
        raise TypeError("Expected a host-verified resolved retrieval scope")
    scope = resolved_scope if resolved_scope is not None else _scope(code_commit, name, all_descriptors,
        tuple(source_authority["source_object_refs"]) if source_authority is not None else None)
    from .task_sources import source_authority as declare_sources, constrain_source_authority
    if source_authority is None:
        declared = tuple({'kind': None, 'source_id': source.card.source_id,
            'capability_ref': descriptor.capability_ref, 'source_family_ref': descriptor.source_family_ref,
            'tool_id': descriptor.execution_binding['tool_id']}
            for (source, _), descriptor in zip(source_pairs, source_descriptors, strict=True))
        source_authority = declare_sources((), None, [item['source_id'] for item in declared],
            scope.allowed_source_object_refs, declared)
    source_authority = constrain_source_authority(source_authority, scope)
    context = OperationalIntakeContext(
        decision_time=decision_time,
        target_spatial=spatial,
        weather_target_required=weather_target_required,
        submission_schema=submission_schema,
        valid_start=valid_start,
        valid_end=valid_end,
        access_policy=AccessPolicy(
            allowed_namespaces=("science.*",),
            denied_evidence_roles=("future_truth", "event_label", "hidden_label"),
            allow_nwp=True,
            allow_generated_code=include_coderun,
        ),
        budget=budget,
        timezone_name="UTC",
        report_required=report_required,
        report_format=report_format,
        report_audience="general" if report_required else None,
        disclosed_context=disclosed_context,
        task_context={
            **dict(disclosed_context),
            "retrieval_universe_spec_id": scope.scope_ref,
        },
        resolved_retrieval_scope=scope,
        workload_tags=(name,),
    )
    intake = OperationalTaskIntake()
    task = intake.intake(question, context)
    if not isinstance(task, RuntimeTask):
        raise RuntimeError("public task unexpectedly requires clarification")
    source_actions = production_source_operation_registrations(tuple(item for item, _ in source_pairs)) if source_pairs else ()
    actions = (*source_actions, *registrations)
    registered_tools = dict(tools)
    for registration, executor in source_pairs:
        registered_tools[str(registration.capability_descriptor().execution_binding["tool_id"])] = executor
    code_policy = None
    if code is not None:
        code_policy = ArtifactCodeRunPolicy(
            timeout_seconds=code_runner.time_limits["generate_wall_seconds"],
            cpu_time_seconds=code_runner.time_limits["generate_cpu_seconds"],
            max_call_seconds=code_runner.time_limits["action_wall_seconds"], memory_limit_bytes=1024 * 1024 * 1024,
            max_output_bytes=8 * 1024 * 1024, max_output_files=4, max_stdout_bytes=32 * 1024,
            isolation_backend=coderun_backend, generation_identity=code_runner.generation_identity,
            runtime_libraries=code_runner.runtime_libraries,
        )
        registered_tools[CODE_RUNNER_ID] = NativeCodeRunAdapter(code_runner)
    science = build_owned_scientific_composition(
        source=source, source_tool=source_tool, descriptors=descriptors,
        registrations=registrations, tools=registered_tools, catalog_version=f"weather-agent-{name}-{code_commit}",
        runtime_capabilities=knowledge_descriptors + (() if code is None else (code,)), additional_sources=additional_sources)
    catalog = science.catalog
    runtime = ActionRuntime(
        ExecutionModule(science.tools)
    )
    resources = _new_episode_resources(task)
    from .task_sources import source_data_access
    resources.source_authority = dict(source_authority or {})
    from .knowledge_access import KnowledgeAccessContext
    resources.knowledge_registry = knowledge_registry
    resources.knowledge_error = knowledge_error
    resources.knowledge_read_locations = {}
    resources.knowledge_disclosed_sections = {}
    resources.knowledge_access = (KnowledgeAccessContext.for_task(task, resources.source_authority,
        knowledge_registry, visible_entry_refs=knowledge_visible_refs, bound_rules=knowledge_bound_rules)
        if knowledge_registry is not None else None)
    resources.source_data_access = tuple(
        {'view_handles':tuple(v['effective_view_handle'] for v in descriptor.execution_binding['source_view_contract']['views']),
         'data_access':source_data_access(descriptor)} for descriptor in all_descriptors
        if descriptor.capability_ref in scope.allowed_capability_refs and descriptor.execution_binding.get('source_view_contract'))
    from .task_sources import validate_public_record_descriptors
    input_descriptors=validate_public_record_descriptors(public_records or {},
        {} if public_record_descriptors is None else public_record_descriptors)
    resources.max_protocol_corrections = max_protocol_corrections
    resources.public_input_records.update(public_records or {})
    resources.public_record_descriptors=input_descriptors
    resources.state.manifest = replace(
        resources.state.manifest,
        code_commit=code_commit,
        baseline_tag=f"weather-agent-{name}-public-acceptance-v1",
        model=model,
        evaluator_identity={"name": f"weather-agent-{name}-public-acceptance"},
        tool_set=tuple(sorted(registered_tools)),
        config_identity=deterministic_identity(
            {
                "max_protocol_corrections": max_protocol_corrections,
                "descriptors": tuple(
                    sorted(item.descriptor_identity for item in all_descriptors)
                ),
                "actions": tuple(sorted(item.semantic_key for item in actions)),
                "scope": scope.resolved_scope_identity,
                "source_authority": source_authority,
                **({"public_record_descriptors":input_descriptors} if input_descriptors else {}),
                "knowledge_packages": knowledge_registry.package_manifests if knowledge_registry else {},
                "knowledge_operations": ("knowledge.search@1", "knowledge.read@1"),
                "knowledge_aliases": getattr(knowledge_registry, "aliases_identity", None),
                **({"knowledge_service": knowledge_service_config} if knowledge_service_config is not None else {}),
                # The host attempt directory is immutable within an episode.
                # Identical public tasks in distinct attempts must not share
                # run-scoped artifact authority; aliases resolve to one attempt.
                "host_attempt_identity": deterministic_identity(str(Path(run_root).resolve())),
            }
        ),
    )
    snapshot = catalog.snapshot(task.access_policy, resolved_scope=scope)
    resources.catalog_snapshot = snapshot
    resources.state.catalog_snapshot_identity = snapshot.snapshot_identity
    resources.resolved_retrieval_scope = scope
    resources.scientific_workspace = ScientificWorkspace(
        run_root / resources.state.manifest.run_identity,
        resources.state.manifest.run_identity,
    )
    action_provider = ActivatedOwnerOperationProvider(
        ActionOperationProvider(runtime, actions, run_root=run_root),
        tuple(
            ActivatedOwnerRegistration(item.semantic_key, item.capability_ref)
            for item in actions
        ),
    )
    input_support = {item.capability_ref: item.owner.input_support for item in actions
        if item.owner is not None and callable(getattr(item.owner, "input_support", None))}
    providers: tuple[Any, ...] = (
        CatalogNavigationProvider(catalog, input_support=input_support,
            host_binding_resolvers=({code.capability_ref: lambda descriptor, snapshot, task, request:
                frozen_code_binding(descriptor, snapshot, task, code_policy)}
                if code is not None and code_policy is not None else {})), action_provider)
    if code is not None and code_policy is not None and dict(snapshot.entries).get(code.capability_ref) == code.descriptor_identity:
        code_binding = frozen_code_binding(code, snapshot, task, code_policy)
        resources.bindings[code_binding.binding_identity] = code_binding
        resources.state.bound_capabilities.append(code_binding.binding_identity)
        providers += (
            ArtifactCodeRunProvider(
                runtime,
                capability_ref=code.capability_ref,
                policy=code_policy,
            ),
        )
    from .knowledge_operations import KnowledgeOperationProvider
    providers += (RecordsInspectProvider(), KnowledgeOperationProvider(), C1FinalOperationProvider())
    return PublicComposition(
        question, context, intake, task, scope, providers, resources, run_root, code_runner=code_runner, max_protocol_corrections=max_protocol_corrections
    )
