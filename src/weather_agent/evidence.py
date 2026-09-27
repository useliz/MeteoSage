from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Mapping, Protocol, runtime_checkable

from ._serialization import identity, parse_datetime, to_primitive


class EvidenceRole(str, Enum):
    OBSERVATION = "observation"
    ANALYSIS = "analysis"
    REANALYSIS = "reanalysis"
    FORECAST = "forecast"
    FUTURE_TRUTH = "future_truth"
    CLIMATOLOGY = "climatology"
    EVENT_LABEL = "event_label"
    IMPACT_RECORD = "impact_record"


class EvidenceOutcome(str, Enum):
    AVAILABLE = "available"
    UNSUPPORTED = "unsupported"
    MISSING = "missing"
    FAILED = "failed"


class SpatialKind(str, Enum):
    POINT = "point"
    BOUNDING_BOX = "bounding_box"
    NAMED_REGION = "named_region"
    GEOMETRY = "geometry"


SPATIAL_SELECTION_CONTRACT_SCHEMA_VERSION = (
    "weather-agent-spatial-selection-contract-v2"
)


def _validate_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value.lower()):
        raise ValueError(f"{label} must be a 64-character SHA-256 hex digest")


@dataclass(frozen=True)
class VariableDescriptor:
    standard_name: str
    source_name: str
    unit: str
    dimensions: tuple[str, ...]
    vertical_coordinate: str | None = None
    allowed_units: tuple[str, ...] = ()
    physical_quantity: str | None = None
    vertical_semantics: str | None = None

    def __post_init__(self) -> None:
        if not self.standard_name or not self.source_name or not self.unit:
            raise ValueError("variable names and unit are required")
        if not self.dimensions:
            raise ValueError("variable dimensions cannot be empty")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VariableDescriptor":
        return cls(
            standard_name=str(value["standard_name"]),
            source_name=str(value["source_name"]),
            unit=str(value["unit"]),
            dimensions=tuple(value["dimensions"]),
            vertical_coordinate=value.get("vertical_coordinate"),
            allowed_units=tuple(value.get("allowed_units", ())),
            physical_quantity=value.get("physical_quantity"),
            vertical_semantics=value.get("vertical_semantics"),
        )


@dataclass(frozen=True)
class DatasetDescriptor:
    source: str
    version: str
    manifest_sha256: str
    variables: tuple[VariableDescriptor, ...]
    spatial_coverage: Mapping[str, Any]
    temporal_coverage: Mapping[str, Any]
    coordinate_reference_system: str
    longitude_convention: str
    grid: Mapping[str, Any]
    cadence: str | None = None
    calendar: str = "proleptic_gregorian"
    missing_rules: tuple[str, ...] = ()
    quality_rules: tuple[str, ...] = ()
    path_template: str | None = None
    processing_level: str | None = None
    field_profile_identity: str | None = None

    def __post_init__(self) -> None:
        if not self.source or not self.version:
            raise ValueError("dataset source and version are required")
        _validate_sha256(self.manifest_sha256, "manifest_sha256")
        names = [variable.standard_name for variable in self.variables]
        if not names or len(names) != len(set(names)):
            raise ValueError("dataset variables must be non-empty and unique")
        if self.longitude_convention not in {"0_360", "-180_180", "projected"}:
            raise ValueError("unsupported longitude convention")
        if self.field_profile_identity is not None:
            _validate_sha256(self.field_profile_identity, "field_profile_identity")
            if not self.processing_level:
                raise ValueError("profiled datasets require a processing level")

    @property
    def descriptor_identity(self) -> str:
        return identity(self)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DatasetDescriptor":
        return cls(
            source=str(value["source"]),
            version=str(value["version"]),
            manifest_sha256=str(value["manifest_sha256"]),
            variables=tuple(VariableDescriptor.from_dict(item) for item in value["variables"]),
            spatial_coverage=dict(value["spatial_coverage"]),
            temporal_coverage=dict(value["temporal_coverage"]),
            coordinate_reference_system=str(value["coordinate_reference_system"]),
            longitude_convention=str(value["longitude_convention"]),
            grid=dict(value["grid"]),
            cadence=value.get("cadence"),
            calendar=str(value.get("calendar", "proleptic_gregorian")),
            missing_rules=tuple(value.get("missing_rules", ())),
            quality_rules=tuple(value.get("quality_rules", ())),
            path_template=value.get("path_template"),
            processing_level=value.get("processing_level"),
            field_profile_identity=value.get("field_profile_identity"),
        )


@dataclass(frozen=True)
class SpatialSelection:
    kind: SpatialKind
    latitude: float | None = None
    longitude: float | None = None
    bounding_box: tuple[float, float, float, float] | None = None
    named_region: str | None = None
    geometry: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.kind is SpatialKind.POINT:
            if self.latitude is None or self.longitude is None:
                raise ValueError("point selection requires latitude and longitude")
            if not -90 <= self.latitude <= 90 or not -360 <= self.longitude <= 360:
                raise ValueError("point coordinates are out of range")
        elif self.kind is SpatialKind.BOUNDING_BOX:
            if self.bounding_box is None or len(self.bounding_box) != 4:
                raise ValueError("bounding-box selection requires four bounds")
            west, south, east, north = self.bounding_box
            if south > north or not -90 <= south <= 90 or not -90 <= north <= 90:
                raise ValueError("invalid bounding-box latitude bounds")
            if not -360 <= west <= 360 or not -360 <= east <= 360:
                raise ValueError("invalid bounding-box longitude bounds")
        elif self.kind is SpatialKind.NAMED_REGION and not self.named_region:
            raise ValueError("named-region selection requires a name")
        elif self.kind is SpatialKind.GEOMETRY and not self.geometry:
            raise ValueError("geometry selection requires GeoJSON-like geometry")

    @classmethod
    def point(cls, latitude: float, longitude: float) -> "SpatialSelection":
        return cls(SpatialKind.POINT, latitude=latitude, longitude=longitude)

    @classmethod
    def box(cls, west: float, south: float, east: float, north: float) -> "SpatialSelection":
        return cls(SpatialKind.BOUNDING_BOX, bounding_box=(west, south, east, north))

    @classmethod
    def region(cls, name: str) -> "SpatialSelection":
        return cls(SpatialKind.NAMED_REGION, named_region=name)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SpatialSelection":
        box = value.get("bounding_box")
        return cls(
            kind=SpatialKind(value["kind"]),
            latitude=value.get("latitude"),
            longitude=value.get("longitude"),
            bounding_box=tuple(box) if box is not None else None,
            named_region=value.get("named_region"),
            geometry=value.get("geometry"),
        )


def canonical_spatial_selection_contract() -> Mapping[str, Any]:
    """Expose the single canonical LLM-facing SpatialSelection representation."""

    variants = {
        SpatialKind.POINT.value: {
            "required_fields": ("kind", "latitude", "longitude"),
            "optional_fields": (),
            "field_schema": {
                "kind": {"const": SpatialKind.POINT.value},
                "latitude": {"type": "number", "minimum": -90, "maximum": 90},
                "longitude": {
                    "type": "number",
                    "minimum": -360,
                    "maximum": 360,
                },
            },
        },
        SpatialKind.BOUNDING_BOX.value: {
            "required_fields": ("kind", "bounding_box"),
            "optional_fields": (),
            "field_schema": {
                "kind": {"const": SpatialKind.BOUNDING_BOX.value},
                "bounding_box": {
                    "type": "number[4]",
                    "item_order": ("west", "south", "east", "north"),
                    "latitude_range": (-90, 90),
                    "longitude_range": (-360, 360),
                    "constraint": "south_lte_north",
                },
            },
        },
        SpatialKind.NAMED_REGION.value: {
            "required_fields": ("kind", "named_region"),
            "optional_fields": (),
            "field_schema": {
                "kind": {"const": SpatialKind.NAMED_REGION.value},
                "named_region": {"type": "non-empty string"},
            },
        },
        SpatialKind.GEOMETRY.value: {
            "required_fields": ("kind", "geometry"),
            "optional_fields": (),
            "field_schema": {
                "kind": {"const": SpatialKind.GEOMETRY.value},
                "geometry": {"type": "non-empty GeoJSON-like object"},
            },
        },
    }
    return {
        "contract_schema_version": SPATIAL_SELECTION_CONTRACT_SCHEMA_VERSION,
        "type": "discriminated object",
        "discriminator": "kind",
        "allowed_values": tuple(variants),
        "variants": variants,
        "canonical_representation_only": True,
    }


@dataclass(frozen=True)
class TemporalSelection:
    start: datetime | None = None
    end: datetime | None = None
    reference_time: datetime | None = None
    lead_time: timedelta | None = None
    valid_time: datetime | None = None
    index_slice: str | None = None

    def __post_init__(self) -> None:
        values = (self.start, self.end, self.reference_time, self.valid_time)
        if not any(values) and self.index_slice is None:
            raise ValueError("temporal selection requires time values or an index slice")
        for value in values:
            if value is not None and value.utcoffset() is None:
                raise ValueError("all datetimes must be timezone-aware")
        if self.start and self.end and self.start > self.end:
            raise ValueError("temporal start cannot be after end")
        if self.lead_time is not None and self.lead_time.total_seconds() < 0:
            raise ValueError("lead time cannot be negative")
        if self.reference_time is not None and self.lead_time is not None:
            expected = self.reference_time + self.lead_time
            if self.valid_time is None:
                object.__setattr__(self, "valid_time", expected)
            elif self.valid_time != expected:
                raise ValueError("valid_time must equal reference_time plus lead_time")

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TemporalSelection":
        lead = value.get("lead_time")
        return cls(
            start=parse_datetime(value.get("start")),
            end=parse_datetime(value.get("end")),
            reference_time=parse_datetime(value.get("reference_time")),
            lead_time=timedelta(seconds=float(lead)) if lead is not None else None,
            valid_time=parse_datetime(value.get("valid_time")),
            index_slice=value.get("index_slice"),
        )


@dataclass(frozen=True)
class DataQuery:
    variables: tuple[str, ...]
    spatial: SpatialSelection
    temporal: TemporalSelection
    role: EvidenceRole
    resolution: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.variables or len(set(self.variables)) != len(self.variables):
            raise ValueError("query variables must be non-empty and unique")
        if self.resolution is not None and not isinstance(self.resolution, Mapping):
            raise TypeError("query resolution must be a mapping or null")
        has_forecast_axis = self.temporal.reference_time is not None or self.temporal.lead_time is not None
        if self.role is EvidenceRole.FORECAST and not has_forecast_axis:
            raise ValueError("forecast query requires reference_time or lead_time")
        if self.role is not EvidenceRole.FORECAST and self.temporal.lead_time is not None:
            raise ValueError("lead_time is valid only for forecast evidence")

    @property
    def query_identity(self) -> str:
        return identity(self)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DataQuery":
        return cls(
            variables=tuple(value["variables"]),
            spatial=SpatialSelection.from_dict(value["spatial"]),
            temporal=TemporalSelection.from_dict(value["temporal"]),
            role=EvidenceRole(value["role"]),
            resolution=value.get("resolution"),
        )


@dataclass(frozen=True)
class EvidenceBundle:
    outcome: EvidenceOutcome
    descriptor_identity: str
    query_identity: str
    payload: Any = None
    conversions: tuple[Mapping[str, Any], ...] = ()
    quality_flags: tuple[str, ...] = ()
    provenance: tuple[Mapping[str, Any], ...] = ()
    message: str | None = None
    error_code: str | None = None

    def __post_init__(self) -> None:
        _validate_sha256(self.descriptor_identity, "descriptor_identity")
        _validate_sha256(self.query_identity, "query_identity")
        if self.outcome is EvidenceOutcome.AVAILABLE and self.payload is None:
            raise ValueError("available evidence requires a payload")
        if self.outcome is not EvidenceOutcome.AVAILABLE and self.payload is not None:
            raise ValueError("non-available evidence cannot carry a payload")

    @classmethod
    def available(
        cls,
        descriptor: DatasetDescriptor,
        query: DataQuery,
        payload: Any,
        *,
        conversions: tuple[Mapping[str, Any], ...] = (),
        quality_flags: tuple[str, ...] = (),
        provenance: tuple[Mapping[str, Any], ...] = (),
    ) -> "EvidenceBundle":
        return cls(
            EvidenceOutcome.AVAILABLE,
            descriptor.descriptor_identity,
            query.query_identity,
            payload,
            conversions,
            quality_flags,
            provenance,
        )

    @classmethod
    def unavailable(
        cls,
        outcome: EvidenceOutcome,
        descriptor: DatasetDescriptor,
        query: DataQuery,
        *,
        message: str,
        error_code: str | None = None,
        quality_flags: tuple[str, ...] = (),
    ) -> "EvidenceBundle":
        if outcome is EvidenceOutcome.AVAILABLE:
            raise ValueError("use EvidenceBundle.available for available evidence")
        return cls(
            outcome,
            descriptor.descriptor_identity,
            query.query_identity,
            quality_flags=quality_flags,
            message=message,
            error_code=error_code,
        )


@runtime_checkable
class EvidenceSource(Protocol):
    def describe(self) -> DatasetDescriptor: ...

    def query(self, query: DataQuery) -> EvidenceBundle: ...
