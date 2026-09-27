"""One deployment and per-task public composition for ordinary scientific work."""
from __future__ import annotations

from .frozen_json import thaw_structure
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
import json
import hashlib
from typing import Any, Mapping

from ._serialization import parse_datetime, to_primitive
from .container_tool_transport import ContainerToolAdapter
from .installed_science import ScientificAssets
from .reliability import ModelIdentity
from .runtime_contracts import RunBudget
from .retrieval_scope import ResolvedRetrievalScope


@dataclass(frozen=True)
class Deployment:
    project_root: Path
    data_root: Path
    code_commit: str
    coderun_deployment: Path
    source_kinds: tuple[str, ...] = tuple(ScientificAssets.source_configs)
    coderun_backend: str = "chroot_seccomp"
    local_container: bool = True
    scope_registry_path: Path | None = None
    scope_registry_sha256: str | None = None
    gpu_lock_path: Path | None = None
    gpu_device_binding: Any = None
    gpu_profiles: tuple = ()
    gpu_local_supervisor: bool = False
    gpu_capacity_experiment: bool = False
    gpu_capacity_budget: Mapping[str, int] | None = None
    reuse: str = "off"
    reuse_manifest: Path | None = None
    t0_cases: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    manifest_tasks: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    scientific_cpu_profile: Mapping[str, Any] | None = None
    scientific_cpu_runtime: Any = None
    scientific_cpu_dispatch_validator: Any = None
    scientific_action_limits: Mapping[str, Any] | None = None
    forecast_runtime: Mapping[str, Any] | None = None
    benchmark_root: Path | None = None
    client: Mapping[str, Any] = field(default_factory=dict)
    knowledge_packages: Mapping[str, str] | None = None
    knowledge_service: Mapping[str, Any] | None = None

    def __post_init__(self):
        from .scientific_cpu_runtime import ScientificCpuRuntime
        cpu_runtime = ScientificCpuRuntime.from_dict(self.scientific_cpu_runtime)
        if cpu_runtime is not None and self.scientific_cpu_profile is not None:
            raise ValueError('scientific CPU runtime/profile are mutually exclusive')
        object.__setattr__(self, 'scientific_cpu_runtime', cpu_runtime)
        from .forecast_service_runtime import ForecastRuntime
        object.__setattr__(self, 'forecast_runtime', ForecastRuntime.from_dict(self.forecast_runtime))
        from .scientific_action_limits import validate_scientific_action_limits
        object.__setattr__(self, "scientific_action_limits", validate_scientific_action_limits(self.scientific_action_limits))
        from .gpu_resources import GpuDeviceBinding, GpuResourceProfile
        if self.gpu_device_binding is not None:
            binding = (GpuDeviceBinding(**self.gpu_device_binding)
                if isinstance(self.gpu_device_binding, Mapping) else self.gpu_device_binding)
            profiles = tuple(GpuResourceProfile(**value) if isinstance(value, Mapping) else value
                for value in self.gpu_profiles)
            if not isinstance(binding, GpuDeviceBinding) or not profiles or any(
                    not isinstance(value, GpuResourceProfile) for value in profiles):
                raise ValueError('GPU deployment requires a device binding and frozen profiles')
            if len({value.ref for value in profiles}) != len(profiles):
                raise ValueError('GPU deployment profile refs must be unique')
            if self.gpu_lock_path is not None and str(self.gpu_lock_path) != binding.lock_path:
                raise ValueError('GPU deployment must retain the physical device lock')
            if cpu_runtime is not None and (binding.retained_cpu_threads < 4 or binding.retained_host_memory_bytes < 12*1024**3):
                raise ValueError('GPU binding must retain the scientific CPU/control partition')
            object.__setattr__(self, 'gpu_device_binding', binding)
            object.__setattr__(self, 'gpu_profiles', profiles)
        elif self.gpu_profiles or self.gpu_local_supervisor or self.gpu_capacity_experiment:
            raise ValueError('GPU profiles and local supervisor require an explicit device binding')
        if self.gpu_capacity_budget is not None:
            from .frozen_json import freeze_structure
            if not self.gpu_capacity_experiment or self.gpu_device_binding is None:
                raise ValueError('capacity budget requires an explicit GPU capacity experiment')
            binding=self.gpu_device_binding
            ceilings=dict(gpu_reservation_bytes=binding.available_gpu_bytes,
                cpu_threads=binding.cpu_threads-binding.retained_cpu_threads,
                host_memory_bytes=binding.host_memory_bytes-binding.retained_host_memory_bytes)
            if (not isinstance(self.gpu_capacity_budget,Mapping) or set(self.gpu_capacity_budget)!=set(ceilings)
                    or any(type(v) is not int or not 0<v<=ceilings[k] for k,v in self.gpu_capacity_budget.items())):
                raise ValueError('capacity budget exceeds the bound device or retained resources')
            object.__setattr__(self,'gpu_capacity_budget',freeze_structure(self.gpu_capacity_budget))
        if self.client.get('api_key_env','DEEPSEEK_API_KEY') not in {'DEEPSEEK_API_KEY','QWEN_API_KEY','TEAMOROUTER_API_KEY','ZCODE_API_KEY'}:
            raise ValueError('unsupported client credential variable')
        if set(self.client) - {'endpoint', 'proxy', 'credential_file', 'api_key_env'}:
            raise ValueError('Client deployment accepts connection settings and credential file paths only')
        if self.reuse not in {'off', 'compatible'}:
            raise ValueError('Reuse mode must be off or compatible')

    @classmethod
    def from_dict(cls, document):
        values = dict(document)
        for key in ("project_root", "data_root", "coderun_deployment"):
            values[key] = Path(values[key]).resolve(strict=True)
        if values.get("scope_registry_path"):
            values["scope_registry_path"] = Path(values["scope_registry_path"]).resolve(strict=True)
        if "source_kinds" in values:
            values["source_kinds"] = tuple(values["source_kinds"])
        deployment = cls(**values)
        if deployment.scope_registry_path is not None:
            deployment.scope_registry
        return deployment

    @cached_property
    def scope_registry(self):
        from .retrieval_scope_registry import FrozenRetrievalScopeRegistry
        if self.scope_registry_path is None:
            raise ValueError('No trusted retrieval registry is configured')
        raw = Path(self.scope_registry_path).read_bytes()
        if self.scope_registry_sha256 is None:
            raise ValueError('Retrieval registry requires an explicit content identity')
        if hashlib.sha256(raw).hexdigest() != self.scope_registry_sha256:
            raise ValueError('Retrieval registry content changed; use a new deployment and batch group')
        return FrozenRetrievalScopeRegistry(**json.loads(raw))

    @cached_property
    def saved_results(self):
        if self.reuse == "off":
            return None
        if self.reuse != "compatible":
            raise ValueError("Unknown result reuse mode")
        if self.reuse_manifest is None:
            return None
        from .saved_science_results import SavedScienceResults
        try:
            return SavedScienceResults(self.reuse_manifest, project_root=self.project_root)
        except (OSError, ValueError, TypeError, KeyError):
            return None  # An optional unreadable cache cannot disable the live worker.

    def wrap(self, executor, *, gpu=False):
        if self.forecast_runtime is not None:
            from .forecast_service_runtime import forecast_adapter_id
            adapter_id = forecast_adapter_id(executor)
            if adapter_id in self.forecast_runtime.eligible_adapters:
                saved = self.saved_results
                if saved is not None and any(forecast_adapter_id(request['executor']) == adapter_id
                                             for _, request, _ in saved.entries.values()):
                    raise ValueError('shared forecast and legacy saved-result replay overlap for this adapter')
                from .shared_forecast_executor import SharedForecastExecutor
                return SharedForecastExecutor(executor, self._forecast_publication_transport, self.forecast_runtime)
        return self._transport(executor, gpu=gpu)

    def _forecast_publication_transport(self, executor):
        return self._transport(executor, gpu=False, allow_saved=False)

    def scientific_cpu_transport(self, executor, service_compute_seconds=None):
        from .scientific_cpu_runtime import transport_for
        return transport_for(self, executor, service_compute_seconds=service_compute_seconds)

    def _transport(self, executor, *, gpu=False, allow_saved=True):
        if getattr(executor, 'requires_bounded_cpu', False):
            if gpu:
                raise ValueError('bounded CPU executor cannot use GPU placement')
            return self.scientific_cpu_transport(executor)
        if gpu and self.gpu_device_binding is not None:
            from .gpu_deployment import DeploymentGpuAdapter
            return DeploymentGpuAdapter(executor, self)
        worker = ContainerToolAdapter(executor, worktree=self.project_root,
            gpu=gpu, local_container=self.local_container, gpu_lock_path=self.gpu_lock_path)
        return worker if not allow_saved or self.saved_results is None else self.saved_results.wrap(executor, fallback=worker)


def _window(value, label, *, optional=False):
    if optional and value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(label + " requires two timezone-aware timestamps")
    result = tuple(parse_datetime(item) for item in value)
    if any(item.tzinfo is None or item.utcoffset() is None for item in result) or result[0] > result[1]:
        raise ValueError(label + " is unordered or timezone-naive")
    return result


def public_parameters(task: Mapping[str, Any]):
    """Map explicit public fields; experiments retain all evaluation metadata."""
    question = task.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Public question is required")
    decision = parse_datetime(task["as_of"])
    if decision.tzinfo is None or decision.utcoffset() is None:
        raise ValueError("Decision time must be timezone-aware")
    weather = task.get("weather_target_required", True)
    if not isinstance(weather, bool) or not isinstance(task["report_required"], bool):
        raise ValueError("Weather target and report flags must be boolean")
    history = _window(task.get("observation_window"), "observation_window", optional=not weather)
    analysis = _window(task.get("analysis_window"), "analysis_window", optional=not weather)
    forecast = _window(task.get("forecast_window"), "forecast_window", optional=True)
    forecast_mode=task.get('forecast_mode','prospective')
    if forecast_mode not in ('prospective','retrospective_replay'):
        raise ValueError('Unsupported public forecast mode')
    if not weather and any(value is not None for value in (history, analysis, forecast, task.get("region"))):
        raise ValueError("Non-weather task must omit weather selectors")
    if history and history[1] > decision:
        raise ValueError("Observation support exceeds decision time")
    if forecast_mode=='retrospective_replay' and (not weather or forecast is None
            or forecast[1]>decision or history is None or history[1]>forecast[0]):
        raise ValueError('Retrospective forecast requires historical initialization and completed target support')
    if forecast and forecast_mode=='prospective' and forecast[0] < decision:
        raise ValueError("Forecast support precedes decision time")
    context = dict(task.get("public_context", {}))
    context.update(observation_window=to_primitive(history), forecast_window=to_primitive(forecast))
    if forecast_mode=='retrospective_replay':context['forecast_mode']=forecast_mode
    # These are disclosed source selectors, never a deployment tool allow-list.
    binding = task.get("source_binding", {})
    if not isinstance(binding, Mapping):
        raise ValueError("source_binding must be an object")
    if binding:
        context["source_binding"] = dict(binding)
    if "event_id" in binding:
        context["event_id"] = binding["event_id"]
        if history:
            context.update(history_start=history[0].isoformat(), history_end=history[1].isoformat())
    if "arome_reference_time" in binding:
        context["arome"] = {"reference_time": binding["arome_reference_time"]}
    return dict(question=question, decision_time=decision,
        valid_start=analysis[0] if analysis else None,
        valid_end=analysis[1] if analysis else None, spatial=task.get("region"),
        disclosed_context=context, weather_target_required=weather,
        report_required=task["report_required"], submission_schema=task.get("submission_schema"))


def build_task(deployment: Deployment, public_task, run_config, run_root: Path,
               *, resolved_scope: ResolvedRetrievalScope | None = None, report_sources=None):
    from .benchmark_t1_t2 import task_format, normalize_manifest_task
    from .t3_t4.model_profiles import validate_protocol_corrections
    correction_limit = validate_protocol_corrections(run_config.get('max_protocol_corrections', 2))
    input_format = task_format(public_task)
    is_t0 = input_format == 't0'
    if is_t0:
        from .benchmark_t0 import validate_t0_budget
        validate_t0_budget(run_config)
    mode = public_task.get("retrieval_scope_mode")
    if resolved_scope is None and public_task.get("retrieval_scope_ref"):
        resolved_scope = deployment.scope_registry.resolve(public_task["retrieval_scope_ref"])
    if not is_t0 and report_sources is None and resolved_scope is None and mode != "deployment-scoped-development":
        raise ValueError("Task needs a trusted resolved scope or explicit deployment-scoped-development mode")
    from .working_state import WorkingStateConfig, phase_model_configuration, model_identity
    # Default to single: update the notebook in the action response.
    # Keep all notebook features available without a maintenance model call.
    # maintained is explicit opt-in only; never enable it as a default or fallback.
    working_state=WorkingStateConfig.from_public(run_config.get('working_state_mode','single'),run_config.get('working_state'))
    model_config=phase_model_configuration(run_config['model'],working_state)
    effective_models={'action':model_config}
    if working_state.mode=='maintained':
        effective_models['maintenance']=phase_model_configuration(run_config['model'],working_state,'maintenance')
    model = model_identity(model_config,working_state)
    from .trajectory_memory_operations import MemoryConfig
    memory = MemoryConfig.from_public(run_config.get('memory', {'mode': 'off'}))
    if report_sources is not None:
        from .t3_t4.intake import normalize_task
        normalized = normalize_task(public_task,report_sources.capabilities,report_sources,
            budget=RunBudget(**run_config['budget']),source_kinds=())
    elif is_t0:
        from .benchmark_t0 import normalize_t0_task
        normalized = normalize_t0_task(deployment, public_task, run_config, working_state=working_state)
    elif input_format == 'manifest':
        normalized = normalize_manifest_task(deployment, public_task, run_config, resolved_scope=resolved_scope)
    else:
        from .task_sources import NormalizedTaskInput
        parameters = public_parameters(public_task)
        parameters["disclosed_context"]["retrieval_scope_mode"] = mode if resolved_scope is None else "resolved-public-scope"
        normalized = NormalizedTaskInput(parameters, deployment.source_kinds, RunBudget(**run_config["budget"]))
    composition = ScientificAssets(deployment.project_root, deployment.data_root).assemble_answer(
        knowledge_packages=deployment.knowledge_packages, knowledge_service=deployment.knowledge_service,
        scientific_action_limits=deployment.scientific_action_limits,
        wrap=deployment.wrap, wrap_render=deployment.wrap,
        source_kinds=normalized.source_kinds, scoped_sources=normalized.scoped_sources,
        manifest_sources=normalized.manifest_sources, report_sources=normalized.report_sources,
        public_records=normalized.public_records, public_record_descriptors=normalized.public_record_descriptors, code_commit=deployment.code_commit,
        budget=normalized.budget, model=model, run_root=Path(run_root),
        max_protocol_corrections=correction_limit,
        coderun_backend=deployment.coderun_backend, coderun_deployment=deployment.coderun_deployment,
        include_coderun=True, resolved_scope=resolved_scope, memory=memory, **thaw_structure(normalized.parameters))
    from .context_limits import ContextLimits
    composition.resources.context_limits = ContextLimits.for_mode(run_config.get("context_mode", "expanded-v2"))
    composition.resources.working_state_config = working_state
    composition.resources.effective_models = effective_models
    policy = run_config.get('evaluation_final_policy', 'first-explicit-final-v1')
    if policy == 'committed-after-memory-v1':
        if (memory is None or memory.consumption_policy != 'report-pre-final-v1'
                or working_state.mode != 'single'
                or normalized.budget.timeout_seconds != 3600):
            raise ValueError('committed-after-memory-v1 requires the frozen report Memory route')
    elif policy != 'first-explicit-final-v1':
        raise ValueError('unknown final evaluation policy')
    composition.resources.evaluation_final_policy = policy
    return composition


def run_task(deployment, public_task, run_config, output_dir, *, adapter=None, resolved_scope=None,
             result_name="result.json", metadata=None, maintenance_adapter=None,
             memory_selection_transport=None, host_deadline=None, report_sources=None, client_builder=None, host_control=None, provider_admission=None):
    """Persist each terminal before any independently recoverable offline export."""
    import time
    from .hosted_client import build_client_from_environment, AuditedHostedAdapter
    from .hosted_agent_policy import HostedDecisionAdapter
    from .run_bundle import collect_result, publish_terminal
    from .runtime_errors import error_details
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if Path(result_name).name != result_name or (output_dir / result_name).exists() or (output_dir / "runs").exists():
        raise FileExistsError("Attempt already exists or name is invalid; select a new output directory")
    if host_control is not None:host_control.check()
    started = time.time()
    composition = result = error = None
    candidate_commit = deployment.get('code_commit') if isinstance(deployment, Mapping) else deployment.code_commit
    phase = "task_contract"
    try:
        from .benchmark_t1_t2 import task_format, input_parameters
        if report_sources is None and task_format(public_task) != 't0':
            input_parameters(public_task)
        phase = "deployment"
        if isinstance(deployment, Mapping):
            deployment = Deployment.from_dict(deployment)
        if not deployment.project_root.is_dir() or not deployment.data_root.is_dir():
            raise ValueError("Deployment project/data path is unavailable")
        phase = "task_composition"
        composition = build_task(deployment, public_task, run_config, output_dir / "runs", resolved_scope=resolved_scope, report_sources=report_sources)
        composition.resources.host_control = host_control
        # Let external supervision read the existing durable control stream;
        # the pointer does not change task, grant, or runtime budgets.
        from .c1_records import C1ControlRecords
        if composition.resources.control_records is None:
            composition.resources.control_records = C1ControlRecords()
        phase = "audit"
        composition.resources.control_records.ensure_durable(composition.task.task_identity, directory=output_dir / "control-audit")
        publish_terminal(output_dir / 'control-audit-location.json', {
            'task_identity': composition.task.task_identity,
            **composition.resources.control_records.snapshot()['persistence']})
        phase = "isolation"
        readiness = composition.code_runner.readiness()
        publish_terminal(output_dir / 'isolation-readiness.json', readiness)
        if not readiness['ready']:
            raise RuntimeError(readiness['reason'])
        working_state=composition.resources.working_state_config
        if working_state.mode=='single' and maintenance_adapter is not None:
            raise ValueError('single cannot accept a maintenance adapter')
        if adapter is not None and working_state.mode=='maintained' and maintenance_adapter is None:
            raise ValueError('Controlled maintained replay needs its explicit maintenance adapter')
        if adapter is None:
            phase = "client"
            audit_options={}
            if report_sources is not None:
                composition.resources.provider_audit_root=output_dir/'provider-http'
                audit_options['audit_root']=composition.resources.provider_audit_root
            if host_control is not None:audit_options['host_control']=host_control
            if host_deadline is not None and report_sources is not None:audit_options['host_deadline']=host_deadline
            if provider_admission is not None:
                from .t3_t4.provider_admission import validate_binding
                binding=validate_binding(provider_admission)
                audit_options['provider_admission']={'root':binding['root'],'run_group':binding['run_group'],
                    'provider':composition.resources.effective_models['action']['provider']}
            client, _ = (client_builder or build_client_from_environment)(**deployment.client,**audit_options)
            adapter = AuditedHostedAdapter(HostedDecisionAdapter(client, host_deadline=host_deadline,host_control=host_control, **composition.resources.effective_models['action']))
            if composition.memory is not None and composition.memory.retrieval_mode == 'contextual':
                from .trajectory_memory_selection import openai_selection_transport
                memory_selection_transport = openai_selection_transport(client)
            if working_state.mode=='maintained':
                maintenance_adapter=AuditedHostedAdapter(HostedDecisionAdapter(client, host_deadline=host_deadline,host_control=host_control, **composition.resources.effective_models['maintenance']))
        phase = "episode"
        result = composition.run(adapter, working_state=working_state, maintenance_adapter=maintenance_adapter,
            memory_selection_transport=memory_selection_transport, host_deadline=host_deadline,
            memory_package_identity=run_config.get("memory_identity"))
    except Exception as caught:
        if host_control is not None:host_control.failed(caught)
        error = caught
    if composition is not None and composition.resources.control_records is not None and composition.resources.control_records.persistence_failed:
        from .runtime_errors import AuditWriteError
        error = AuditWriteError("Durable control audit failed; stop shared dispatch")
        phase = "audit"
    if composition is not None and adapter is not None:
        record = collect_result(composition, adapter, result, error, maintenance_adapter=maintenance_adapter)
    else:
        record = {"schema_version": "weather-agent-run-bundle-v1", "candidate_commit": candidate_commit,
            "status": "not_started", "output": None, "state": {}, "ledger": {}, "trajectory": [],
            "responses": [], "turns": [], "call_and_token_records": {}}
    if error is not None:
        from .runtime_errors import SharedComputeCleanupUncertain, AuditWriteError
        if isinstance(error, AuditWriteError):
            phase = "audit"
        if isinstance(error, SharedComputeCleanupUncertain):
            phase = "isolation"
        from .trajectory_memory_operations import MemoryIdentityConflict
        if isinstance(error, MemoryIdentityConflict):
            phase = "deployment"
        record["error"] = {**error_details(error, phase), "code": phase + "_unavailable",
            "failure_scope": "deployment" if phase in {"deployment", "isolation", "client", "audit"} else "task"}
        from .runtime_errors import SharedProviderFailure
        if isinstance(error, SharedProviderFailure):
            record["error"].update(error.failure_details())
    if composition is not None and getattr(composition.resources,"report_delivery",None) is not None:
        record["report_delivery"] = to_primitive(composition.resources.report_delivery)
    record.update(metadata or {})
    record.update(elapsed_seconds=time.time() - started,
        runtime_identity={"deployment": to_primitive(deployment), "run_config": to_primitive(run_config),
            "effective_models":to_primitive(composition.resources.effective_models) if composition is not None else None,
            "context_limits":to_primitive(composition.resources.context_limits) if composition is not None else None,
            "working_state":to_primitive(composition.resources.working_state_config) if composition is not None else None},
        artifact_reuse={"mode": deployment.reuse if isinstance(deployment, Deployment) else None,
            "events": deployment.saved_results.events if isinstance(deployment, Deployment) and deployment.saved_results else []})
    publish_terminal(output_dir / result_name, record)
    return record
