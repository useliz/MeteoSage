"""Installed source and science assets; no task text or evaluation routes are consulted."""
from __future__ import annotations
from dataclasses import dataclass, replace
from functools import cached_property
import hashlib
import json
from pathlib import Path
from .reliability import deterministic_identity
from .action_operations import ActionOperationRegistration
from .arome_operations import build_arome_qpf_extension
from .field_profiles import FieldProfileRegistry
from .field_quality import FIELD_QUALITY_TOOL_ID, FieldQualityToolAdapter, field_quality_capability, field_quality_operation_owner
from .forecast_model_cards import load_forecast_model_card
from .learned_nowcast import AlphaPreAdapter
from .meteonet_sources import build_meteonet_sources
from .model_runtime import ModelWorkerSpec
from .production_sources import ProductionSourceRegistration, SourceCard
from .scf_c_composition import build_scf_c_composition_extension
from .scf_c_field_operations import SCFCFieldOperationExtension, build_scf_c_field_operation_extension
from .sevir_sources import build_sevir_sources
from .wb2_sources import WeatherBench2ProductionSource
from .forecast_sources import CachedForecastSource
from .installed_atmospheric import installed_atmospheric_extension
from .standard_view_operations import standard_view_operation_registration
from .vlm_evidence import VLMEvidenceToolAdapter, vlm_evidence_capability
from .vlm_operations import vlm_evidence_operation_registration
from .weather_rendering import StandardWeatherViewToolAdapter, WeatherViewRegistry, standard_weather_view_capability
from .diagnostic_runtime import FunctionDiagnosticOperator, DIAGNOSTIC_TOOL_ID
from .diagnostic_runtime import _calculator_with_project_resources
from .diagnostic_calculations import CALCULATORS
from .domain_knowledge import load_knowledge_document
from .diagnostic_operations import public_diagnostic_descriptor, diagnostic_operation_registration
from .field_reduction_operations import field_reduction_descriptor, field_reduction_registration, FieldReductionAdapter
from .field_matching_operations import matching_descriptors, matching_registration
from .field_alignment import FIELD_ALIGNMENT_TOOL_ID, FieldAlignmentToolAdapter
from .point_grid_collocation import POINT_GRID_COLLOCATION_TOOL_ID, PointGridCollocationToolAdapter
from .feature_operations import installed_feature_operators, feature_registration, ReadableFeatureDiagnosticAdapter
from .association_operations import installed_association, association_descriptor, association_registration
from .multisource_association import ASSOCIATION_TOOL_ID, AssociationToolAdapter


@dataclass(frozen=True)
class ScientificAssets:
    project_root: Path
    data_root: Path

    @property
    def profile_root(self):
        return self.project_root / "reproduction/weather_agent/knowledge/field_profiles"

    @property
    def manifest_root(self):
        return self.project_root / "reproduction/weather_agent/manifests"

    @property
    def card_root(self):
        return self.project_root / "reproduction/weather_agent/model_cards"

    @cached_property
    def simvp_worker(self):
        from .simvp_worker_contract import load_installed_simvp
        return load_installed_simvp(self.manifest_root/'installed-canonical-simvp-v1.json')

    @property
    def source_card_root(self):
        return self.project_root / "reproduction/weather_agent/source_cards"

    @cached_property
    def _profiles(self) -> FieldProfileRegistry:
        return FieldProfileRegistry.from_paths(self.profile_root / 'canonical-field-vocabulary-v1.json',
            tuple(self.profile_root / name for name in self.profile_files))

    profile_files = ('meteonet-field-profiles-v1.json', 'meteonet-environment-field-profiles-v1.json',
        'scf-c-rain-rate-field-profiles-v1.json', 'wb2-field-profiles-v1.json',
        'cached-forecast-field-profiles-v1.json', 'adrian-native-wind-v1.json')

    def profiles(self) -> FieldProfileRegistry:
        return self._profiles

    def environment_profiles(self) -> FieldProfileRegistry:
        return self._profiles

    @cached_property
    def _meteonet_sources(self):
        return build_meteonet_sources(self.manifest_root / 'meteonet-product-index-v1.json',
            self.data_root, field_profiles=self.profiles())

    @cached_property
    def _sevir_sources(self):
        return build_sevir_sources(self.manifest_root / 'sevir-product-index-v1.json', self.data_root)

    @cached_property
    def _wb2_sources(self):
        return {'wb2_era5': WeatherBench2ProductionSource(self.data_root / 'WeatherBench2', field_profiles=self.profiles())}

    @cached_property
    def _cached_forecasts(self):
        return {'pangu_cached_t2m': CachedForecastSource(
            self.manifest_root / 'pangu-cached-forecast-asset-v1.json', field_profiles=self.profiles())}

    def alpha_adapter(self) -> AlphaPreAdapter:
        card_path = self.card_root / "alphapre-v2.json"
        card = load_forecast_model_card(card_path)
        worker = ModelWorkerSpec(
            model_id=card.model_id,
            model_version=card.model_version,
            command=(
                str(self.project_root / "deploy/model_workers/alphapre/.venv/bin/python"),
                str(self.project_root / "deploy/model_workers/alphapre/worker.py"),
            ),
            checkpoint_path=Path(str(card.checkpoint["external_path"])),
            checkpoint_identity=card.checkpoint_identity,
            source_checkout=Path(str(card.source["external_checkout"])),
            source_commit=str(card.source["commit"]),
            timeout_seconds=1200,
            preprocessing_identity=card.preprocessing_identity,
            output_variable=card.single_output.variables[0],
            output_unit=card.single_output.units[card.single_output.variables[0]],
            model_config=card.hyperparameters,
            require_clean_source=True,
            require_cuda=True,
        )
        return AlphaPreAdapter(card_path, worker)


    # Deployment keys select held sources; no task identifier or scientific route.
    source_configs = {
        'radar': ('meteonet', 'radar_reflectivity', 'meteonet-radar-reflectivity-v2.json'),
        'era5': ('meteonet', 'era5_reanalysis', 'meteonet-era5-v1.json'),
        'station': ('meteonet', 'ground_station_observations', 'meteonet-ground-stations-v1.json'),
        'satellite': ('meteonet', 'satellite_imagery', 'meteonet-satellite-v1.json'),
        'nwp_2d': ('meteonet', 'weather_models_2d', 'meteonet-weather-models-2d-v1.json'),
        'nwp_3d': ('meteonet', 'weather_models_3d', 'meteonet-weather-models-3d-v1.json'),
        'static': ('meteonet', 'static_coordinates', 'meteonet-static-v1.json'),
        'sevir': ('sevir', 'vil', 'sevir-vil-v1.json'),
        'sevir_ir069': ('sevir', 'ir069', 'sevir-ir069-v1.json'),
        'sevir_ir107': ('sevir', 'ir107', 'sevir-ir107-v1.json'),
        'sevir_vis': ('sevir', 'vis', 'sevir-vis-v1.json'),
        'wb2': ('wb2', 'wb2_era5', 'wb2-era5-v2.json'),
        'pangu_cached': ('cached_forecast', 'pangu_cached_t2m', 'pangu-cached-t2m-v1.json'),
    }

    def source_registration(self, kind: str, profiles: FieldProfileRegistry | None = None) -> ProductionSourceRegistration:
        if kind not in self.source_configs:
            raise ValueError('unknown production source kind: ' + kind)
        family, product, filename = self.source_configs[kind]
        card_path = self.source_card_root / filename
        card = SourceCard.from_path(card_path)
        sources = {'meteonet': lambda: self._meteonet_sources, 'sevir': lambda: self._sevir_sources,
            'wb2': lambda: self._wb2_sources, 'cached_forecast': lambda: self._cached_forecasts}[family]()
        return ProductionSourceRegistration.create(sources[product], source_card_path=card_path,
            query_manifest_path=self.project_root / card.representative_query_manifest,
            project_root=self.project_root, field_profiles=self.profiles())

    def diagnostic_operators(self):
        operators=[]
        for name, calculator in CALCULATORS.items():
            card=load_knowledge_document(self.project_root / 'reproduction/weather_agent/knowledge/operator_cards' / (name+'-v1.json'))
            if 'field_profile_registry_identity' in card.parameters:
                parameters={name:dict(spec) for name,spec in card.parameters.items()}
                values={'field_profile_registry_identity':self.profiles().registry_identity,
                    'profile_paths':['reproduction/weather_agent/knowledge/field_profiles/'+name for name in self.profile_files]}
                for key,value in values.items():
                    if key in parameters:parameters[key]=dict(parameters[key],default=value,choices=[value])
                card=replace(card,parameters=parameters)
            operators.append(FunctionDiagnosticOperator(card, _calculator_with_project_resources(
                calculator, self.project_root / 'reproduction/weather_agent')))
        return tuple(operators)

    def assemble_answer(self, *, wrap, source_kinds=None, scoped_sources=None, manifest_sources=(), report_sources=None, public_records=None, public_record_descriptors=None,
                        wrap_render=lambda executor: executor,
                        vlm_factory=VLMEvidenceToolAdapter, memory=None,
                        coderun_backend="bwrap", coderun_deployment=None, knowledge_packages=None,
                        knowledge_visible_refs=None, knowledge_bound_rules=(), knowledge_service=None, scientific_action_limits=None, **request):
        """One deployment entry for natural questions and batch clients.

        Source keys are a deployment allow-set, independent of question text.
        All science owners share that deployment and prepare only when activated.
        """
        from .tooling import NativeEvidenceToolAdapter
        from .public_composition import assemble_public_answer

        kinds = tuple(self.source_configs) if source_kinds is None else tuple(source_kinds)
        if len(set(kinds)) != len(kinds):
            raise ValueError('deployment source allow-set must be unique')
        sources = tuple(self.source_registration(kind) for kind in kinds)
        pairs = tuple((source, wrap(NativeEvidenceToolAdapter(
            {source.card.source_id + '@' + source.card.source_version: source.source})))
            for source in sources)
        installed = self.operations(wrap=wrap, wrap_render=wrap_render, vlm_factory=vlm_factory, manifest_sources=manifest_sources, report_sources=report_sources, scientific_action_limits=scientific_action_limits)
        descriptors, registrations, tools = installed.descriptors, installed.registrations, dict(installed.tools)
        if manifest_sources:
            from .manifest_source_operations import manifest_descriptor, manifest_registration, ManifestSourceAdapter, TOOL_ID as MANIFEST_TOOL_ID
            descriptor = manifest_descriptor(manifest_sources)
            descriptors += (descriptor,)
            registrations += (manifest_registration(descriptor),)
            tools[MANIFEST_TOOL_ID] = wrap(ManifestSourceAdapter(manifest_sources))
        scoped_adapter = None
        statistics_route = None
        if scoped_sources is not None:
            from .scoped_evidence_operations import scoped_descriptor, scoped_registration, ScopedEvidenceToolAdapter, TOOL_ID
            derivation_backend = derivation_contract = None
            if scoped_sources.runtime_source and scoped_sources.runtime_source['source'].get('kind') == 'run004_f4':
                from .f4_backend import FixedF4Backend
                from .f4_contract import build_derivation_contract
                if coderun_deployment is None:
                    raise ValueError('F4 requires its fixed shared-compute deployment')
                derivation_backend = FixedF4Backend(coderun_deployment)
                derivation_contract = build_derivation_contract(scoped_sources, derivation_backend)
            descriptor = scoped_descriptor(scoped_sources, self.profiles(), derivation_contract)
            scoped_adapter = ScopedEvidenceToolAdapter(scoped_sources, self.profiles(),
                derivation_backend=derivation_backend, derivation_contract=derivation_contract)
            descriptors += (descriptor,)
            registrations += (scoped_registration(descriptor, scoped_sources.surface),)
            if knowledge_service is not None:
                from copy import copy
                from .knowledge_service_backend import ScopedStatisticsRoutingAdapter
                statistics_route = ScopedStatisticsRoutingAdapter(scoped_adapter, wrap(statistics_fallback := copy(scoped_adapter)),
                    allocation_id=knowledge_service.get('host_allocation_id'), cpu_ids=knowledge_service.get('host_cpu_ids',()),
                    host_binding=knowledge_service.get('host_binding'))
                tools[TOOL_ID] = statistics_route
            else:
                tools[TOOL_ID] = wrap(scoped_adapter)
        from .task_sources import source_authority
        authorized_objects = {ref for source in sources
            for ref in source.capability_descriptor().execution_binding.get('source_object_refs', ())}
        if 'satellite' in kinds:
            authorized_objects.update(ref for item in installed.descriptors
                if item.kind == 'catalog_query'
                for ref in item.execution_binding.get('source_object_refs', ()))
        if scoped_sources is not None:
            authorized_objects.update(view['effective_view_handle'] for view in scoped_sources.trusted_views)
        authorized_objects.update(obj['object_ref'] for declaration in manifest_sources for obj in declaration.objects)
        authority = source_authority(kinds, scoped_sources, [source.card.source_id for source in sources],
            sorted(authorized_objects), tuple({'kind': kind, 'source_id': source.card.source_id,
                'capability_ref': source.capability_descriptor().capability_ref,
                'source_family_ref': source.capability_descriptor().source_family_ref,
                'tool_id': source.capability_descriptor().execution_binding['tool_id']}
                for kind, source in zip(kinds, sources, strict=True)), manifest_sources=manifest_sources)
        from .domain_knowledge import GitKnowledgeRegistry
        knowledge_registry = None
        knowledge_error = None
        if knowledge_packages is not None:
            try:
                knowledge_registry = GitKnowledgeRegistry.load(self.project_root / 'reproduction/weather_agent/knowledge',
                    package_manifests=knowledge_packages)
            except (ValueError, TypeError, KeyError, OSError):
                knowledge_error = 'package_integrity_error'
        composition = assemble_public_answer(name='installed-science',
            knowledge_registry=knowledge_registry, knowledge_error=knowledge_error, knowledge_service_config=knowledge_service,
            knowledge_visible_refs=knowledge_visible_refs, knowledge_bound_rules=knowledge_bound_rules,
            source=pairs[0][0] if pairs else None, source_tool=pairs[0][1] if pairs else None,
            additional_sources=pairs[1:], descriptors=descriptors, registrations=registrations,
            tools=tools, coderun_backend=coderun_backend, source_authority=authority,
            public_records=public_records or {}, public_record_descriptors={} if public_record_descriptors is None else public_record_descriptors, coderun_deployment=coderun_deployment, **request)
        if scoped_adapter is not None:
            scoped_adapter.authorization_identity = composition.scope.resolved_scope_identity
        if statistics_route is not None:
            statistics_fallback.authorization_identity = composition.scope.resolved_scope_identity
            from .knowledge_service_backend import KnowledgeServiceBinding, KnowledgeServiceBackend
            config = knowledge_service
            binding = KnowledgeServiceBinding(snapshot_root=config['snapshot_root'], snapshot_sha256=config['snapshot_sha256'],
                data_root=str(scoped_sources.data_root.parent), cohort='t0', case_id=scoped_sources.surface['case_id'],
                task_identity=composition.task.task_identity, authority_identity=deterministic_identity(composition.resources.source_authority),
                queries=tuple({'action':'climatology','view_handle':v['effective_view_handle']} for v in scoped_sources.trusted_views if v.get('evidence_role')=='climatology'),
                python=str(self.project_root/'.venv/bin/python'), worker=str(self.project_root/'src/weather_agent/knowledge_service_worker.py'),
                host_binding=statistics_route.host_binding)
            statistics_route.backend = KnowledgeServiceBackend(binding, composition.run_root/'knowledge-service')
            composition.close_callbacks = (statistics_route.backend.close,)
            composition.resources.knowledge_service_backend = statistics_route.backend
        if report_sources is not None:
            composition.resources.t3_t4_node = report_sources
        composition.memory = memory
        return composition

    def field_extension(self, profiles: FieldProfileRegistry):
        case = json.loads(
            (self.manifest_root / "scf-c-nowcast-blending-cases-v1.json").read_text()
        )
        return build_scf_c_field_operation_extension(
            model_card_path=self.card_root / "alphapre-v2.json",
            case_manifest_path=self.manifest_root / "scf-c-nowcast-blending-cases-v1.json",
            zr_profile_path=self.manifest_root / "zr-profile-v2.json",
            radar_coordinate_identity=case["meteonet_radar_arome"]["radar"]["coordinates_sha256"],
        )


    def operations(self, *, wrap, wrap_render=lambda executor: executor,
                   vlm_factory=VLMEvidenceToolAdapter, manifest_sources=(), report_sources=None, scientific_action_limits=None) -> SCFCFieldOperationExtension:
        """Wire every installed SCF-C method plus existing QC/render/VLM/AROME owners."""
        profiles = self.profiles()
        atmospheric, atmospheric_registrations, atmospheric_tools = installed_atmospheric_extension(self.project_root, manifest_sources, scientific_action_limits=scientific_action_limits)
        fields = self.field_extension(profiles)
        environment_identity = hashlib.sha256((self.project_root / "uv.lock").read_bytes()).hexdigest()
        forecasts = build_scf_c_composition_extension(
            alphapre_adapter=self.alpha_adapter(),
            pysteps_card_root=self.project_root / "reproduction/weather_agent/pysteps_cards",
            field_profiles=profiles, environment_identity=environment_identity)
        arome, arome_registration, arome_tool = build_arome_qpf_extension(
            self.manifest_root / "scf-c-nowcast-blending-cases-v1.json", self.data_root)
        radar_profile = profiles.profile("meteonet-radar-reflectivity@1.0.0")
        views = WeatherViewRegistry.from_path(self.profile_root / "standard-weather-views-v1.json", profiles)
        satellite_profile = profiles.profile('meteonet-satellite-ir108@1.0.0')
        station_profile = profiles.profile('meteonet-ground-stations@1.0.0')
        view = standard_weather_view_capability((radar_profile, satellite_profile), views, profiles)
        quality = field_quality_capability((radar_profile, satellite_profile, station_profile, profiles.profile("adrian-native-wind@1.0.0")), profiles)
        vlm = vlm_evidence_capability(self.card_root / "qwen3-vl-plus-visual-evidence-v3.json")
        operators = self.diagnostic_operators()
        feature_operators = installed_feature_operators(self.project_root/'reproduction/weather_agent/knowledge')
        association = installed_association(self.project_root/'reproduction/weather_agent/knowledge')
        diagnostic_descriptors = tuple(public_diagnostic_descriptor(item) for item in operators)
        feature_descriptors = tuple(public_diagnostic_descriptor(item) for item in feature_operators)
        matching = matching_descriptors(profiles)
        from .manifest_grib import NATIVE_QUANTITIES
        native_variables = {name for names, *_ in NATIVE_QUANTITIES.values() for name in names}
        field_variables = tuple(sorted({name for profile in (*profiles.profiles, *self.environment_profiles().profiles) for name in profile.canonical_names} | {"vil"} | native_variables))
        reductions = tuple(field_reduction_descriptor(kind, field_variables) for kind in ("geographic-support", "time-reduction", "spatial-reduction"))
        from .forecast_operations import combine_descriptor, combine_registration, ForecastCombineAdapter, TOOL_ID as COMBINE_TOOL_ID
        combination = combine_descriptor(field_variables)
        from .forecast_projection import project_descriptor, project_registration, ForecastProjectAdapter, TOOL_ID as PROJECT_TOOL_ID
        projection = project_descriptor(field_variables)
        from .forecast_query_operations import query_descriptor, query_registration, ForecastQueryAdapter, TOOL_ID as QUERY_TOOL_ID
        query = query_descriptor(field_variables)
        from .forecast_comparison import comparison_descriptor, comparison_registration, ForecastComparisonAdapter, TOOL_ID as COMPARE_TOOL_ID
        comparison = comparison_descriptor(field_variables)
        from .forecast_alignment import alignment_descriptor, alignment_registration, ForecastAlignmentAdapter, TOOL_ID as ALIGN_TOOL_ID
        forecast_alignment = alignment_descriptor(field_variables)
        from .forecast_objects import objects_descriptor, objects_registration, ForecastObjectsAdapter, TOOL_ID as OBJECTS_TOOL_ID
        forecast_objects = objects_descriptor(field_variables)
        from .canonical_radar_operations import radar_descriptor, radar_registration, RadarAdapter, TOOL_ID as RADAR_TOOL_ID
        radar = radar_descriptor()
        from .simvp_operations import simvp_descriptor, simvp_registration, TOOL_ID as SIMVP_TOOL_ID
        from .simvp_adapter import SimVPAdapter
        simvp = simvp_descriptor()
        from .scientific_action_limits import timed_descriptor
        combination, projection, query, comparison, forecast_alignment, forecast_objects, radar, simvp = (
            timed_descriptor(item, scientific_action_limits) for item in
            (combination, projection, query, comparison, forecast_alignment, forecast_objects, radar, simvp))
        region_registry = self.project_root / "reproduction/weather_agent/geography/t0_regions_v1.jsonl"
        descriptors = (*fields.descriptors, *forecasts.capabilities, *atmospheric, arome, quality, view, vlm, *diagnostic_descriptors, *feature_descriptors, association_descriptor(association), *matching, *reductions, combination, projection, query, comparison, forecast_alignment, forecast_objects, radar, simvp)
        registrations = (*fields.registrations, *forecasts.registrations, *atmospheric_registrations, arome_registration,
            ActionOperationRegistration(semantic_key="action:quality:field", capability_ref=quality.capability_ref,
                action_id="summarize-observation-quality", purpose="Run the registered deterministic field-quality summary.",
                expected_observation="One provenance-linked field-quality evidence record.", owner=field_quality_operation_owner()),
            standard_view_operation_registration(view, views), vlm_evidence_operation_registration(vlm),
            *(diagnostic_operation_registration(item) for item in operators),
            *(feature_registration(item) for item in feature_operators),
            association_registration(association),
            *(matching_registration(item, profiles) for item in matching),
            *(field_reduction_registration(item, region_registry) for item in reductions), combine_registration(combination), project_registration(projection), query_registration(query), comparison_registration(comparison), alignment_registration(forecast_alignment, profiles), objects_registration(forecast_objects), radar_registration(radar), simvp_registration(simvp,self.simvp_worker))
        raw_tools = {**fields.tools, **forecasts.tools, **atmospheric_tools,
            COMBINE_TOOL_ID: ForecastCombineAdapter(),
            PROJECT_TOOL_ID: ForecastProjectAdapter(),
            QUERY_TOOL_ID: ForecastQueryAdapter(),
            COMPARE_TOOL_ID: ForecastComparisonAdapter(),
            ALIGN_TOOL_ID: ForecastAlignmentAdapter(profiles),
            OBJECTS_TOOL_ID: ForecastObjectsAdapter(),
            RADAR_TOOL_ID: RadarAdapter(),
            SIMVP_TOOL_ID: SimVPAdapter(self.simvp_worker),
            str(arome.execution_binding["tool_id"]): arome_tool,
            FIELD_QUALITY_TOOL_ID: FieldQualityToolAdapter(profiles),
            DIAGNOSTIC_TOOL_ID: ReadableFeatureDiagnosticAdapter((*operators,*feature_operators)),
            ASSOCIATION_TOOL_ID: AssociationToolAdapter((association,)),
            FIELD_ALIGNMENT_TOOL_ID: FieldAlignmentToolAdapter(profiles),
            POINT_GRID_COLLOCATION_TOOL_ID: PointGridCollocationToolAdapter(profiles),
            **{str(item.execution_binding["tool_id"]): FieldReductionAdapter(item.capability_id) for item in reductions}}
        gpu_tools = {str(item.execution_binding['tool_id']) for item in descriptors
            if item.resource.get('kind') == 'gpu' and 'tool_id' in item.execution_binding}
        tools = {tool_id: wrap(executor, gpu=tool_id in gpu_tools)
                 for tool_id, executor in raw_tools.items()}
        tools[str(view.execution_binding["tool_id"])] = wrap_render(StandardWeatherViewToolAdapter(profiles, views))
        tools[str(vlm.execution_binding["tool_id"])] = vlm_factory(self.card_root / "qwen3-vl-plus-visual-evidence-v3.json")
        metadata_asset = self.manifest_root / 'satellite-catalog-index-v1.json'
        if metadata_asset.is_file():
            from .catalog_metadata_operation import metadata_descriptor, metadata_registration, CatalogMetadataAdapter, TOOL_ID
            from ._serialization import parse_datetime
            asset=json.loads(metadata_asset.read_text())
            metadata=metadata_descriptor(parse_datetime(asset['retrieval_universe']['as_of_time']), asset['retrieval_universe']['source_objects'])
            if metadata.execution_binding['snapshot_identity'] != asset['snapshot_identity']:
                raise ValueError('installed catalog snapshot differs from public asset')
            descriptors=(*descriptors,metadata)
            registrations=(*registrations,metadata_registration(metadata))
            tools[TOOL_ID]=CatalogMetadataAdapter()
        if report_sources is not None:
            from .t3_t4.source_operations import source_registration, TOOL_ID
            descriptor, registration, executor = source_registration(report_sources)
            descriptors=(*descriptors,descriptor)
            registrations=(*registrations,registration)
            tools[TOOL_ID]=executor  # live session stays in the bounded host process
            from .t3_t4.field_operation import registration as field_registration, TOOL_ID as FIELD_SOURCE_TOOL_ID
            field_descriptor, field_reg, field_executor=field_registration(report_sources,profiles.profile('adrian-native-wind@1.0.0'))
            descriptors=(*descriptors,field_descriptor)
            registrations=(*registrations,field_reg)
            tools[FIELD_SOURCE_TOOL_ID]=wrap(field_executor)
        return SCFCFieldOperationExtension(tuple(descriptors), tuple(registrations), tools)
