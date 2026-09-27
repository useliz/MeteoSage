from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from enum import Enum
import json
import math
from pathlib import Path
import re
import unicodedata
from typing import Any, Mapping

from ._serialization import canonical_json, identity


_TRUST = "advisory_memory"
_PACKAGE_SCHEMA_VERSION = "trajectory-memory-package-v3"
_TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)
_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "before",
        "by",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "should",
        "the",
        "their",
        "to",
        "when",
        "with",
    }
)


class AdvisoryKind(str, Enum):
    STRATEGY = "strategy"
    CAUTION = "caution"


class RecallStatus(str, Enum):
    RESULTS = "results"
    NO_MATCH = "no-match"
    BUDGET_NOOP = "budget-noop"
    INVALID_QUERY = "invalid-query"


@dataclass(frozen=True)
class AuthorizedContext:
    max_results: int
    max_output_bytes: int
    denied_isolation_refs: tuple[str, ...]
    delivered_entry_ids: tuple[str, ...]
    prior_query_digests: tuple[str, ...]
    remaining_recalls: int

    def __post_init__(self) -> None:
        for name, value in (
            ("max_results", self.max_results),
            ("max_output_bytes", self.max_output_bytes),
            ("remaining_recalls", self.remaining_recalls),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in (
            "denied_isolation_refs",
            "delivered_entry_ids",
            "prior_query_digests",
        ):
            values = getattr(self, name)
            if isinstance(values, str) or not isinstance(
                values, (tuple, set, frozenset)
            ):
                raise ValueError(f"{name} must be a collection of strings")
            if any(
                not isinstance(value, str) or not value.strip() for value in values
            ):
                raise ValueError(f"{name} must contain non-blank strings")
            object.__setattr__(self, name, tuple(sorted(set(values))))


@dataclass(frozen=True)
class PublicAdvisory:
    identity: str
    version: str
    kind: AdvisoryKind
    guidance: str
    applicability: tuple[str, ...]
    negative_conditions: tuple[str, ...]
    trust: str = _TRUST


@dataclass(frozen=True)
class _RecallControl:
    isolation_refs: tuple[str, ...]


@dataclass(frozen=True)
class HostOnlyProvenance:
    canonical_payload: str
    recall_control: _RecallControl


@dataclass(frozen=True)
class _MemoryEntry:
    public_advisory: PublicAdvisory
    host_only_provenance: HostOnlyProvenance


@dataclass(frozen=True)
class RecallTrace:
    package_digest: str
    query_digest: str
    context_digest: str
    candidate_count: int
    returned_entry_ids: tuple[str, ...]
    status: RecallStatus
    retrieval: Mapping[str, Any] = field(default_factory=lambda: {"backend": "lexical"})


@dataclass(frozen=True)
class RecallOutcome:
    status: RecallStatus
    results: tuple[PublicAdvisory, ...]
    trace: RecallTrace
    public_record: Mapping[str, Any] | None = None


def _required_string(value: Mapping[str, Any], name: str) -> str:
    result = value.get(name)
    if not isinstance(result, str) or not result.strip():
        raise ValueError(f"{name} must be a non-blank string")
    return result


def _string_tuple(value: Mapping[str, Any], name: str) -> tuple[str, ...]:
    items = value.get(name)
    if not isinstance(items, list) or any(
        not isinstance(item, str) or not item.strip() for item in items
    ):
        raise ValueError(f"{name} must be a list of non-blank strings")
    return tuple(items)


def _public_advisory(value: Mapping[str, Any]) -> PublicAdvisory:
    try:
        kind = AdvisoryKind(value.get("kind"))
    except ValueError as exc:
        raise ValueError("kind must be strategy or caution") from exc
    return PublicAdvisory(
        identity=_required_string(value, "identity"),
        version=_required_string(value, "version"),
        kind=kind,
        guidance=_required_string(value, "guidance"),
        applicability=_string_tuple(value, "applicability"),
        negative_conditions=_string_tuple(value, "negative_conditions"),
    )


def _recall_control(provenance: Mapping[str, Any]) -> _RecallControl:
    value = provenance.get("recall_control")
    if not isinstance(value, Mapping):
        raise ValueError("recall_control must be an object")
    isolation_refs = _string_tuple(value, "isolation_refs")
    return _RecallControl(
        isolation_refs=tuple(sorted(set(isolation_refs))),
    )


def _entry(value: Any) -> _MemoryEntry:
    if not isinstance(value, Mapping):
        raise ValueError("each package entry must be an object")
    public = value.get("public_advisory")
    provenance = value.get("host_only_provenance")
    if not isinstance(public, Mapping):
        raise ValueError("public_advisory must be an object")
    if not isinstance(provenance, Mapping):
        raise ValueError("host_only_provenance must be an object")
    return _MemoryEntry(
        public_advisory=_public_advisory(public),
        host_only_provenance=HostOnlyProvenance(
            canonical_payload=canonical_json(provenance),
            recall_control=_recall_control(provenance),
        ),
    )


def _stem(token: str) -> str:
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ied"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("ed"):
        stemmed = token[:-2]
        if len(stemmed) > 2 and stemmed[-1] == stemmed[-2]:
            stemmed = stemmed[:-1]
        return stemmed
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _canonical_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _is_cjk(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x3040 <= codepoint <= 0x30FF
        or 0xAC00 <= codepoint <= 0xD7AF
    )


def _runs(token: str) -> tuple[tuple[bool, str], ...]:
    runs: list[tuple[bool, str]] = []
    for character in token:
        is_cjk = _is_cjk(character)
        if runs and runs[-1][0] == is_cjk:
            runs[-1] = (is_cjk, runs[-1][1] + character)
        else:
            runs.append((is_cjk, character))
    return tuple(runs)


def _terms(text: str) -> tuple[str, ...]:
    terms: list[str] = []
    for token in _TOKEN_PATTERN.findall(_canonical_text(text)):
        for is_cjk, run in _runs(token):
            if is_cjk:
                if len(run) == 1:
                    terms.append(run)
                else:
                    terms.extend(
                        run[index : index + 2] for index in range(len(run) - 1)
                    )
                continue
            stemmed = _stem(run)
            if stemmed and stemmed not in _STOP_WORDS:
                terms.append(stemmed)
    return tuple(terms)


def _weighted_terms(advisory: PublicAdvisory) -> tuple[str, ...]:
    applicability = tuple(
        term for text in advisory.applicability for term in _terms(text)
    )
    return (
        *_terms(advisory.guidance),
        *applicability,
        *applicability,
        *(term for text in advisory.negative_conditions for term in _terms(text)),
    )


def _scientific_text(text: str) -> str:
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n")).strip()


def _content_digest(advisory: PublicAdvisory) -> str:
    return identity(
        {
            "kind": advisory.kind.value,
            "guidance": _scientific_text(advisory.guidance),
            "applicability": tuple(
                _scientific_text(value) for value in advisory.applicability
            ),
            "negative_conditions": tuple(
                _scientific_text(value) for value in advisory.negative_conditions
            ),
        }
    )


def _bm25_advice_scores(
    query_terms: tuple[str, ...],
    advice: tuple[PublicAdvisory, ...],
) -> tuple[float, ...]:
    if not advice:
        return ()
    documents = tuple(
        Counter(_weighted_terms(item)) for item in advice
    )
    lengths = tuple(sum(document.values()) for document in documents)
    average_length = sum(lengths) / len(lengths) or 1.0
    document_frequencies = Counter(
        term for document in documents for term in document
    )
    scored: list[float] = []
    for document, length in zip(documents, lengths, strict=True):
        score = 0.0
        for term in sorted(set(query_terms)):
            frequency = document.get(term, 0)
            if frequency == 0:
                continue
            document_frequency = document_frequencies[term]
            inverse_document_frequency = math.log(
                1.0
                + (len(advice) - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            normalization = frequency + 1.2 * (
                0.25 + 0.75 * length / average_length
            )
            score += inverse_document_frequency * frequency * 2.2 / normalization
        scored.append(score)
    return tuple(scored)


def _bm25_scores(
    query_terms: tuple[str, ...], entries: tuple[_MemoryEntry, ...],
) -> tuple[tuple[float, _MemoryEntry], ...]:
    scores = _bm25_advice_scores(query_terms, tuple(e.public_advisory for e in entries))
    return tuple(zip(scores, entries, strict=True))


class AdvisoryMemory:
    def __init__(
        self,
        *,
        entries: tuple[_MemoryEntry, ...],
        package_digest: str,
        retrieval=None,
    ) -> None:
        self._entries = entries
        self._package_digest = package_digest
        self._retrieval = retrieval

    @classmethod
    def from_file(cls, package_path: str | Path, *, retrieval=None) -> AdvisoryMemory:
        raw = json.loads(Path(package_path).read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError("memory package must be an object")
        schema_version = _required_string(raw, "schema_version")
        if schema_version == "trajectory-memory-package-v2":
            raise ValueError(
                "trajectory-memory-package-v2 requires its matching loader or an explicit upgrade"
            )
        if schema_version != _PACKAGE_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {_PACKAGE_SCHEMA_VERSION}")
        raw_entries = raw.get("entries")
        if not isinstance(raw_entries, list):
            raise ValueError("entries must be a list")
        entries = tuple(_entry(entry) for entry in raw_entries)
        entry_ids = tuple(entry.public_advisory.identity for entry in entries)
        if len(entry_ids) != len(set(entry_ids)):
            raise ValueError("public advisory identities must be unique")
        payload = {"schema_version": schema_version, "entries": raw_entries}
        return cls(
            entries=entries,
            package_digest=identity(payload), retrieval=retrieval,
        )

    def with_retrieval(self, retrieval):
        """Share the already pinned immutable entries; never reopen the Registry."""
        return AdvisoryMemory(entries=self._entries,package_digest=self._package_digest,retrieval=retrieval)

    @property
    def package_digest(self) -> str:
        return self._package_digest

    def recall(
        self,
        query: str,
        authorized_context: AuthorizedContext,
        *,
        retrieval_context: Mapping[str, str] | None = None,
    ) -> RecallOutcome:
        contextual = self._retrieval is not None and self._retrieval.metadata.get("backend") == "contextual"
        rerank = self._retrieval is not None and self._retrieval.metadata.get("backend") == "rerank"
        normalized_query = _scientific_text(query) if isinstance(query, str) else ""
        public_context = None
        if retrieval_context is not None:
            if (not isinstance(retrieval_context, Mapping)
                    or set(retrieval_context) - {"task", "notebook", "observation", "results", "materials"}
                    or any(not isinstance(v, str) for v in retrieval_context.values())):
                raise ValueError("retrieval_context must contain only public text fields")
            # Materials is an already serialized public payload: preserve its exact text.
            public_context = {
                k: v if k == "materials" else _scientific_text(v)
                for k, v in retrieval_context.items() if v.strip()
            }
        # Preserve the query-only lexical compatibility contract. Scientific
        # query/context inputs retain case; insignificant query spacing does not
        # create another question. Content identity uses the separate NFC rule.
        semantic_query = " ".join(normalized_query.split())
        if not contextual and not rerank and not public_context:
            semantic_query = _canonical_text(normalized_query)
        query_digest = identity({"query":semantic_query, "context":public_context or {},
            "package":self.package_digest,
            "mode":self._retrieval.metadata if self._retrieval else {"backend":"lexical"}})
        context_digest = identity(authorized_context)
        def terminal(status):
            outcome = self._outcome(status,query_digest,context_digest,candidate_count=0,
                retrieval=dict(self._retrieval.metadata,legal_count=0,checked_count=0) if contextual else None)
            if not contextual:
                return outcome
            from .trajectory_memory_selection import selection_record
            record = selection_record((),{},query_digest,identity(public_context or {}),status.value)
            if len(canonical_json(record).encode()) > authorized_context.max_output_bytes:
                record = None
            return replace(outcome,public_record=record)
        if not normalized_query:
            return terminal(RecallStatus.INVALID_QUERY)
        if (
            authorized_context.remaining_recalls == 0
            or authorized_context.max_results == 0
            or authorized_context.max_output_bytes == 0
            or query_digest in authorized_context.prior_query_digests
        ):
            return terminal(RecallStatus.BUDGET_NOOP)

        if contextual:
            minimum = terminal(RecallStatus.BUDGET_NOOP)
            if minimum.public_record is None:
                return minimum

        denied_refs = frozenset(authorized_context.denied_isolation_refs)
        content_by_identity = {
            entry.public_advisory.identity: _content_digest(entry.public_advisory)
            for entry in self._entries
        }
        delivered_content = frozenset(
            content_by_identity[entry_id]
            for entry_id in authorized_context.delivered_entry_ids
            if entry_id in content_by_identity
        )
        eligible: list[_MemoryEntry] = []
        for entry in self._entries:
            control = entry.host_only_provenance.recall_control
            if denied_refs.intersection(control.isolation_refs):
                continue
            if _content_digest(entry.public_advisory) in delivered_content:
                continue
            eligible.append(entry)

        unique: list[_MemoryEntry] = []
        seen_content: set[str] = set()
        for entry in sorted(eligible, key=lambda item: item.public_advisory.identity):
            content_digest = _content_digest(entry.public_advisory)
            if content_digest in seen_content:
                continue
            seen_content.add(content_digest)
            unique.append(entry)

        if contextual:
            from .trajectory_memory_selection import pack_selection
            if not unique:
                return terminal(RecallStatus.NO_MATCH)
            ordered, notes, retrieval = self._retrieval.select(normalized_query,
                tuple(e.public_advisory for e in unique), retrieval_context=public_context)
            results, record = pack_selection(ordered, notes, query_digest, identity(public_context or {}),
                authorized_context.max_results, authorized_context.max_output_bytes)
            retrieval.update(delivered_ids=[a.identity for a in results],
                capacity_undelivered_ids=[a.identity for a in ordered if a.identity not in {r.identity for r in results}])
            outcome = self._outcome(RecallStatus(record["status"]) if record is not None else RecallStatus.BUDGET_NOOP, query_digest, context_digest,
                candidate_count=len(ordered), retrieval=retrieval, results=results)
            return replace(outcome, public_record=record)

        retrieval = {"backend": "lexical"}
        if self._retrieval is None:
            scored = _bm25_scores(_terms(normalized_query), tuple(unique))
            threshold = 0.0
        elif unique:
            advice = tuple(e.public_advisory for e in unique)
            if rerank:
                scores, retrieval = self._retrieval.score(normalized_query, advice,
                    retrieval_context=public_context)
            else:
                scores, retrieval = self._retrieval.score(query, advice)
            if len(scores) != len(unique) or any(s is not None and
                    (isinstance(s, bool) or not isinstance(s, (int, float)) or not math.isfinite(s)) for s in scores):
                raise OSError("Memory backend returned invalid score coverage")
            if rerank:
                rows = retrieval.get("candidate_scores", [])
                if (len(rows) != len(advice) or any(
                        not isinstance(r, Mapping) or r.get("identity") != a.identity or r.get("version") != a.version
                        or r.get("status") not in {"scored", "not_selected", "unscored_input_too_long"}
                        or (r.get("status") == "scored") != (s is not None)
                        or (s is not None and r.get("logit") != s)
                        for r, a, s in zip(rows, advice, scores, strict=True))):
                    raise OSError("Memory backend returned invalid candidate identities or states")
            scored = tuple((s, e) for s, e in zip(scores, unique, strict=True) if s is not None)
            if not scored:
                error = OSError("Memory backend could not score any eligible candidate")
                error.retrieval = retrieval
                raise error
            threshold = self._retrieval.threshold
        else:
            scored, threshold = (), self._retrieval.threshold
            retrieval = dict(self._retrieval.metadata)
        ranked = sorted(scored, key=lambda pair: (-pair[0], pair[1].public_advisory.identity))
        if self._retrieval is not None:
            retrieval = retrieval | {"scores": {e.public_advisory.identity: score for score, e in ranked}}
        candidates = tuple(entry for score, entry in ranked if score > threshold)
        if not candidates:
            return self._outcome(
                RecallStatus.NO_MATCH,
                query_digest,
                context_digest,
                candidate_count=0, retrieval=retrieval,
            )

        results: tuple[PublicAdvisory, ...] = ()
        for entry in candidates:
            if len(results) >= authorized_context.max_results:
                break
            proposed = (*results, entry.public_advisory)
            if (
                len(canonical_json(proposed).encode("utf-8"))
                > authorized_context.max_output_bytes
            ):
                continue
            results = proposed
        if not results:
            return self._outcome(
                RecallStatus.BUDGET_NOOP,
                query_digest,
                context_digest,
                candidate_count=len(candidates), retrieval=retrieval,
            )
        return self._outcome(
            RecallStatus.RESULTS,
            query_digest,
            context_digest,
            candidate_count=len(candidates), retrieval=retrieval,
            results=results,
        )

    def _outcome(
        self,
        status: RecallStatus,
        query_digest: str,
        context_digest: str,
        *,
        candidate_count: int,
        results: tuple[PublicAdvisory, ...] = (),
        retrieval=None,
    ) -> RecallOutcome:
        returned_entry_ids = tuple(result.identity for result in results)
        return RecallOutcome(
            status=status,
            results=results,
            trace=RecallTrace(
                package_digest=self._package_digest,
                query_digest=query_digest,
                context_digest=context_digest,
                candidate_count=candidate_count,
                returned_entry_ids=returned_entry_ids,
                status=status,
                retrieval=retrieval or (dict(self._retrieval.metadata) if self._retrieval else {"backend": "lexical"}),
            ),
        )
