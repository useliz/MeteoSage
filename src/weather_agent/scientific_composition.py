from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .capability_catalog import CapabilityDescriptor, InMemoryCapabilityCatalog
from .diagnostic_runtime import (
    DIAGNOSTIC_TOOL_ID,
    DiagnosticToolAdapter,
    diagnostic_capabilities,
    load_diagnostic_operators,
)
from .domain_knowledge import (
    GitKnowledgeRegistry,
    HazardDefinition,
    load_knowledge_document,
)
from .field_alignment import (
    FIELD_ALIGNMENT_TOOL_ID,
    FieldAlignmentToolAdapter,
    field_alignment_capability,
)
from .field_profiles import FieldProfileRegistry
from .field_quality import (
    FIELD_QUALITY_TOOL_ID,
    FieldQualityToolAdapter,
    field_quality_capability,
)
from .meteorological_features import load_feature_operators
from .hazard_measurements import (
    HAZARD_TOOL_ID,
    hazard_measurement_capabilities,
    load_hazard_measurement_operators,
)
from .hazard_reasoning import (
    HazardReasoningToolAdapter,
    definition_signal_capability,
)
from .multisource_association import (
    ASSOCIATION_TOOL_ID,
    AssociationToolAdapter,
    association_capabilities,
    load_association_operators,
)
from .point_grid_collocation import (
    POINT_GRID_COLLOCATION_TOOL_ID,
    PointGridCollocationToolAdapter,
    point_grid_collocation_capability,
)
from .production_sources import ProductionSourceRegistration
from .tooling import NativeEvidenceToolAdapter, ToolExecutor, ToolRegistry
from .weather_rendering import (
    STANDARD_WEATHER_VIEW_TOOL_ID,
    StandardWeatherViewToolAdapter,
    WeatherViewRegistry,
    standard_weather_view_capability,
)


@dataclass(frozen=True)
class ScientificComposition:
    """System-level scientific catalog and its executable tool routing."""

    catalog: InMemoryCapabilityCatalog
    tools: ToolRegistry


def build_scientific_composition(
    *,
    registrations: tuple[ProductionSourceRegistration, ...],
    field_profiles: FieldProfileRegistry,
    operator_root: Path,
    view_profile_path: Path,
    additional_capabilities: tuple[CapabilityDescriptor, ...] = (),
    additional_tools: Mapping[str, ToolExecutor] | None = None,
) -> ScientificComposition:
    if not registrations:
        raise ValueError("scientific composition requires production registrations")
    if not isinstance(field_profiles, FieldProfileRegistry):
        raise TypeError("scientific composition requires a field profile registry")

    registration_refs = tuple(
        item.capability_descriptor().capability_ref for item in registrations
    )
    if len(registration_refs) != len(set(registration_refs)):
        raise ValueError("scientific composition production capabilities must be unique")
    source_keys = tuple(
        f"{item.descriptor.source}@{item.descriptor.version}"
        for item in registrations
    )
    if len(source_keys) != len(set(source_keys)):
        raise ValueError("scientific composition production source identities must be unique")
    source_adapter = NativeEvidenceToolAdapter(
        {
            source_key: registration.source
            for source_key, registration in zip(source_keys, registrations)
        }
    )

    profiles = field_profiles.profiles
    quality_profiles = tuple(
        item for item in profiles if "quality" in item.preparation_intents
    )
    alignment_profiles = tuple(
        item
        for item in profiles
        if {"associate", "compare", "blend"}.intersection(
            item.preparation_intents
        )
    )
    point_profiles = tuple(
        item
        for item in profiles
        if item.spatial_support["kind"] == "point_set"
        and "collocate" in item.preparation_intents
    )
    grid_profiles = tuple(
        item
        for item in profiles
        if item.spatial_support["kind"] == "regular_grid"
        and "collocate" in item.preparation_intents
    )
    if not quality_profiles or not alignment_profiles or not point_profiles or not grid_profiles:
        raise ValueError("field profile registry cannot support every SCF-A capability")
    views = WeatherViewRegistry.from_path(view_profile_path, field_profiles)

    field_descriptors = (
        field_quality_capability(quality_profiles, field_profiles),
        field_alignment_capability(alignment_profiles, field_profiles),
        point_grid_collocation_capability(
            point_profiles, grid_profiles, field_profiles
        ),
        standard_weather_view_capability(profiles, views, field_profiles),
    )
    diagnostics = load_diagnostic_operators(operator_root)
    features = load_feature_operators(operator_root)
    associations = load_association_operators(operator_root)
    definitions = tuple(
        document
        for path in sorted((Path(operator_root) / "definitions").glob("*.json"))
        if isinstance((document := load_knowledge_document(path)), HazardDefinition)
    )
    measurements = load_hazard_measurement_operators(operator_root, definitions)
    if not diagnostics or not features or not associations or not measurements:
        raise ValueError(
            "scientific composition requires approved diagnostic, feature, association, and hazard measurement operators"
        )
    scientific_operators = (*diagnostics, *features)
    profile_identities = tuple(
        item.descriptor().profile.profile_identity
        for item in (*associations, *measurements)
    )
    if any(not item for item in profile_identities) or len(profile_identities) != len(
        set(profile_identities)
    ):
        raise ValueError("scientific profile identities must be non-empty and unique")
    operator_descriptors = (
        *diagnostic_capabilities(scientific_operators),
        *association_capabilities(associations),
        *hazard_measurement_capabilities(measurements),
        definition_signal_capability(definitions),
    )
    operator_refs = tuple(item.capability_ref for item in operator_descriptors)
    if any(not item for item in operator_refs) or len(operator_refs) != len(
        set(operator_refs)
    ):
        raise ValueError("scientific operator refs must be non-empty and unique")
    knowledge_descriptors = GitKnowledgeRegistry.load(
        Path(operator_root)
    ).capability_descriptors()
    descriptors = (
        *field_descriptors,
        *operator_descriptors,
        *knowledge_descriptors,
        *additional_capabilities,
    )
    capability_refs = registration_refs + tuple(
        item.capability_ref for item in descriptors
    )
    if len(capability_refs) != len(set(capability_refs)):
        raise ValueError("scientific composition capabilities are registered twice")
    catalog = InMemoryCapabilityCatalog(
        descriptors,
        production_registrations=tuple(
            item.catalog_registration for item in registrations
        ),
    )

    tool_adapters: dict[str, ToolExecutor] = {
        FIELD_QUALITY_TOOL_ID: FieldQualityToolAdapter(field_profiles),
        FIELD_ALIGNMENT_TOOL_ID: FieldAlignmentToolAdapter(field_profiles),
        POINT_GRID_COLLOCATION_TOOL_ID: PointGridCollocationToolAdapter(
            field_profiles
        ),
        STANDARD_WEATHER_VIEW_TOOL_ID: StandardWeatherViewToolAdapter(
            field_profiles, views
        ),
        DIAGNOSTIC_TOOL_ID: DiagnosticToolAdapter(scientific_operators),
        ASSOCIATION_TOOL_ID: AssociationToolAdapter(associations),
        HAZARD_TOOL_ID: HazardReasoningToolAdapter(measurements, definitions),
    }
    for registration in registrations:
        tool_id = str(
            registration.capability_descriptor().execution_binding["tool_id"]
        )
        if tool_id in tool_adapters:
            raise ValueError(f"scientific tool id is registered twice: {tool_id}")
        tool_adapters[tool_id] = source_adapter

    for tool_id, adapter in (additional_tools or {}).items():
        if not isinstance(tool_id, str) or not tool_id:
            raise ValueError("scientific extension tool ids must be non-empty")
        if tool_id in tool_adapters:
            raise ValueError(f"scientific tool id is registered twice: {tool_id}")
        tool_adapters[tool_id] = adapter

    return ScientificComposition(catalog=catalog, tools=ToolRegistry(tool_adapters))


__all__ = [
    "ScientificComposition",
    "build_scientific_composition",
]


def build_owned_scientific_composition(*, source: ProductionSourceRegistration | None,
        source_tool: ToolExecutor, descriptors: tuple[CapabilityDescriptor, ...],
        registrations: tuple, tools: Mapping[str, ToolExecutor], catalog_version: str,
        runtime_capabilities: tuple[CapabilityDescriptor, ...] = (),
        additional_sources: tuple[tuple[ProductionSourceRegistration, ToolExecutor], ...] = ()) -> ScientificComposition:
    """Compose installed owner/descriptor pairs; discovery does not imply prepared inputs."""
    refs = tuple(item.capability_ref for item in descriptors)
    owners = tuple(item.capability_ref for item in registrations)
    if len(set(refs)) != len(refs) or set(refs) != set(owners) or len(owners) != len(refs):
        raise ValueError("installed descriptors require exactly one owner registration each")
    if any(item.owner is None for item in registrations):
        raise ValueError("installed capabilities require explicit owners")
    if any(str(item.execution_binding.get("tool_id", "")) not in tools for item in descriptors):
        raise ValueError("installed capability has no executable adapter")
    registered = dict(tools)
    source_pairs = (() if source is None else ((source, source_tool),)) + additional_sources
    source_refs = [item.capability_descriptor().capability_ref for item, _ in source_pairs]
    if len(set(source_refs)) != len(source_refs) or set(source_refs).intersection(refs):
        raise ValueError("duplicate installed source identity")
    for registration, executor in source_pairs:
        registered[str(registration.capability_descriptor().execution_binding["tool_id"])] = executor
    return ScientificComposition(
        InMemoryCapabilityCatalog((*descriptors, *runtime_capabilities),
            production_registrations=tuple(item.catalog_registration for item, _ in source_pairs), catalog_version=catalog_version),
        ToolRegistry(registered))
