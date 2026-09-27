"""Offline trajectory-to-advice generation. Single writer; never promotes a package."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Callable

from ._serialization import canonical_json, identity, to_primitive
from .trajectory_memory import AdvisoryMemory, AuthorizedContext
from .trajectory_memory_registry import BootstrapReceipt, LeakageContext, MemoryRegistry, StageStatus
from .trajectory_memory_generation_inputs import (
    leakage_literals, load_runs, make_batches, nonblank, public_material,
    query_for, read_json, size, string_list,
)

from .trajectory_memory_prompts import effective_prompt, workflow_prompts
from .trajectory_memory_learning_notes import run_labels, extract_notes

PROMPT_VERSION = "memory-generation-rules-v2"
PROMPT = effective_prompt("generation")
REVIEW_PROMPT = PROMPT  # Historical compatibility symbol; v4 checks then repairs locally.


@dataclass(frozen=True)
class GenerationConfig:
    provider: str = "deepseek"
    model: str = "deepseek-v4.1-flash-expires-on-0910"
    base_url: str = "https://api.deepseek.com"
    thinking: str = "on"
    reasoning_effort: str = "high"
    max_tokens: int = 131072
    timeout_seconds: float = 1800
    input_capacity_bytes: int = 2097152
    old_memory_bytes: int = 131072
    max_old_entries: int = 32
    transport_retries: int = 1
    format_retries: int = 0
    max_api_calls: int | None = None
    review_reserve_bytes: int = 131072
    forbidden_literals: tuple[str, ...] = ()
    audit_old_entries: bool = False
    learning_profile: str = "legacy"
    workflow_prompt_manifest: str | None = None
    workflow_request_profile: str = "sol-chat-v1"

    def __post_init__(self):
        if self.workflow_request_profile not in ('sol-chat-v1', 'deepseek-main-agent-v1'):
            raise ValueError('unknown workflow request profile')
        if self.learning_profile not in {"legacy", "workflow-lessons-v1", "workflow-lessons-v2", "workflow-lessons-v3"}:
            raise ValueError("unknown learning profile")
        if self.learning_profile in ("workflow-lessons-v1", "workflow-lessons-v2", "workflow-lessons-v3"):
            workflow_prompts(self.workflow_prompt_manifest, expected_profile=self.learning_profile)
            if self.audit_old_entries or self.transport_retries or self.format_retries:
                raise ValueError("workflow profile disallows library audit and implicit retries")
        elif self.workflow_prompt_manifest is not None:
            raise ValueError("workflow prompts require workflow profile")
        for key in ("max_tokens", "input_capacity_bytes", "old_memory_bytes", "max_old_entries", "review_reserve_bytes"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if self.max_tokens > 393216 or self.reasoning_effort not in {'low', 'high', 'max'}:
            raise ValueError('invalid max_tokens or reasoning effort')
        if self.max_api_calls is not None and (isinstance(self.max_api_calls, bool) or not isinstance(self.max_api_calls, int) or self.max_api_calls <= 0):
            raise ValueError("max_api_calls must be None or a positive integer")
        if self.timeout_seconds <= 0 or self.thinking not in {"on", "off"}:
            raise ValueError("invalid timeout or thinking mode")
        for key in ("transport_retries", "format_retries"):
            if getattr(self, key) not in (0, 1):
                raise ValueError(f"{key} must be 0 or 1")
        for key in ("provider", "model", "base_url", "reasoning_effort"):
            nonblank(getattr(self, key), key)
        string_list(list(self.forbidden_literals), "forbidden_literals")


def _config_data(config):
    value = asdict(config)
    if config.workflow_request_profile == 'sol-chat-v1':
        value.pop('workflow_request_profile')
    if config.learning_profile == 'legacy':
        value.pop('learning_profile')
        value.pop('workflow_prompt_manifest')
    return value


def _workflow(config):
    return config.learning_profile in ('workflow-lessons-v1', 'workflow-lessons-v2', 'workflow-lessons-v3')


def _workflow_freeze(batch, package_path, runs, config):
    prompts = workflow_prompts(config.workflow_prompt_manifest, expected_profile=config.learning_profile)
    if len(runs) != 1 or batch['host_binding']['raw_run_identity'] != identity(runs[0]):
        raise ValueError('workflow profile requires one exact complete run')
    body = batch['model_view']
    messages = [{'role': 'system', 'content': prompts['generation']['text']},
                {'role': 'user', 'content': canonical_json(body)}]
    if size(messages) > config.input_capacity_bytes:
        raise ValueError('input_over_budget: actual workflow G messages exceed capacity')
    binding = batch['host_binding']
    references = {s['source']: {'run_ref': runs[0]['run_ref'],
        'projection_source': s['source'], 'content_identity': identity(s['content']),
        'origins': deepcopy(binding['sources'][s['source']]['origins'])} for s in body['sources']}
    return {'stage': 'propose', 'messages': messages, 'batch': batch,
        'learning_profile': config.learning_profile, 'workflow_prompts': prompts,
        'workflow_view_identity': identity(batch), 'sources': [binding['disclosure']],
        'prompt_version': prompts['generation']['version'],
        'entry_aliases': {}, 'reference_map': references, 'disclosed_old': [],
        'recall_query': '', 'base_digest': AdvisoryMemory.from_file(package_path).package_digest,
        'batch_identity': identity(body)}


class TransportFailure(Exception):
    """Retryable network failure; carries no provider text or credentials."""


def openai_transport(config: GenerationConfig, *, credential_file=None) -> Callable[[list[dict]], dict]:
    """SDK retries disabled: every SDK invocation has a generator attempt record."""
    from openai import APIConnectionError, APITimeoutError, OpenAI

    if credential_file is None:
        client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=config.base_url,
                        timeout=config.timeout_seconds, max_retries=0)
    else:
        from .hosted_client import build_client_from_environment
        client, _ = build_client_from_environment(endpoint=config.base_url,
                                                 credential_file=credential_file)
        client = client.with_options(timeout=config.timeout_seconds, max_retries=0)

    if _workflow(config):
        class WorkflowTransport:
            def __init__(self):
                self.wire = None

            def prepare_attempt(self, frozen, recovery_request):
                if recovery_request is not None:
                    raise ValueError('workflow recovery requires a separately bound client')
                self.wire = workflow_request(config, frozen['messages'])
                return {'wire': deepcopy(self.wire),
                        'transport': {'implementation': 'openai-sdk-workflow-v1',
                                      'timeout_seconds': config.timeout_seconds, 'retry': 0},
                        'recovery_request_identity': None}

            def __call__(self, messages):
                if self.wire is None or self.wire['messages'] != messages:
                    raise ValueError('workflow request differs from prepared messages')
                try:
                    response = client.chat.completions.create(**self.wire)
                except (APIConnectionError, APITimeoutError) as exc:
                    raise TransportFailure(type(exc).__name__) from None
                choice = response.choices[0] if response.choices else None
                valid = choice is not None and choice.message.role == 'assistant'
                return {'content': choice.message.content if valid else None,
                        'usage': response.usage.model_dump() if response.usage else None,
                        'response_id': response.id, 'returned_model': response.model,
                        'finish_reason': choice.finish_reason if choice else None,
                        'refusal': getattr(choice.message, 'refusal', None) if valid else None,
                        'provider_envelope_valid': valid}

            def close(self):
                client.close()

        return WorkflowTransport()

    def invoke(messages: list[dict]) -> dict:
        try:
            response = client.chat.completions.create(
                model=config.model, messages=messages, max_tokens=config.max_tokens,
                reasoning_effort=config.reasoning_effort,
                response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "enabled" if config.thinking == "on" else "disabled"}},
            )
        except (APIConnectionError, APITimeoutError) as exc:
            raise TransportFailure(type(exc).__name__) from None
        # No model_dump of the message: hidden reasoning never leaves this function.
        choice = response.choices[0] if response.choices else None
        return {"content": choice.message.content if choice else None,
                "usage": response.usage.model_dump() if response.usage else None,
                "response_id": response.id, "returned_model": response.model,
                "finish_reason": choice.finish_reason if choice else None}
    return invoke


def _write(path: Path, value) -> None:
    from .runtime_errors import AuditWriteError
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as file:
            temporary = Path(file.name)
            try:
                file.write(canonical_json(value))
                file.flush()
                os.fsync(file.fileno())
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        os.replace(temporary, path)
    except OSError as error:
        raise AuditWriteError(f"generation audit write failed: {path}") from error

def _save_learning_notes(directory, frozen, response, response_ref):
    from .trajectory_memory_content_check import json_object
    value = json_object(response.get('content'))
    main, notes = extract_notes(value, frozen.get('run_labels', {}))
    phase = frozen['stage']
    name = 'generation-learning-notes.json' if phase == 'propose' else phase + '/learning-notes.json'
    _write(Path(directory) / name, {'phase': phase, 'run_labels': frozen.get('run_labels', {}),
        'request_identity': identity(frozen['messages']), 'raw_response_ref': response_ref, **notes})
    return main


def _code_identity() -> dict:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
    return {"commit": result.stdout.strip() if result.returncode == 0 else "uncommitted",
            "implementation": identity([Path(__file__).read_text(),
                Path(__file__).with_name("trajectory_memory_generation_inputs.py").read_text(),
                Path(__file__).with_name("trajectory_memory_content_check.py").read_text(),
                Path(__file__).with_name("trajectory_memory_prompts.py").read_text(),
                Path(__file__).with_name("trajectory_memory_learning_notes.py").read_text()]),
            "lock": identity((root / "uv.lock").read_text()) if (root / "uv.lock").exists() else None}


def _context(config: GenerationConfig, runs: list[dict]) -> LeakageContext:
    return LeakageContext(forbidden_literals=leakage_literals(runs, config.forbidden_literals))


def _old_entries(package_path: Path, batch: dict, config: GenerationConfig) -> tuple[list[dict], str]:
    package = read_json(package_path)
    entries = package["entries"]
    query = query_for(batch)
    public = [e["public_advisory"] for e in entries]
    if len(public) <= config.max_old_entries and size(public) <= config.old_memory_bytes:
        return entries, query
    recalled = AdvisoryMemory.from_file(package_path).recall(query, AuthorizedContext(
        max_results=config.max_old_entries, max_output_bytes=config.old_memory_bytes,
        denied_isolation_refs=(), delivered_entry_ids=(), prior_query_digests=(), remaining_recalls=1))
    by_id = {e["public_advisory"]["identity"]: e for e in entries}
    return [by_id[p.identity] for p in recalled.results], query


def _freeze_batch(batch: dict, package_path: Path, runs: list[dict], config: GenerationConfig) -> dict:
    if _workflow(config):
        return _workflow_freeze(batch, package_path, runs, config)
    old, query = _old_entries(package_path, batch, config)
    shown_runs = {location["run_ref"] for location in batch["reference_map"].values()}
    sources = [{k: run.get(k) for k in ("run_ref", "source_refs", "isolation_refs", "group", "feedback_sources")}
               for run in runs if run["run_ref"] in shown_runs]
    disclosed = [{"identity": e["public_advisory"]["identity"], "entry_digest": identity(e),
                  "provenance": e["host_only_provenance"]} for e in old]
    aliases = {f"old/{i+1}": e["public_advisory"]["identity"] for i, e in enumerate(old)}
    body = {k: v for k, v in batch.items() if k not in {"reference_map", "identity_map", "original_fragments"}}
    if config.audit_old_entries:
        body['audit_old_entries'] = True
        body['audit_instruction'] = ('Inspect all shown old advice for useful consolidation or supported corrections. '
            'Return ordinary changes only. Unchanged old entries are not newly scientifically certified.')
    body["run_labels"] = run_labels(batch["reference_map"])
    body["allowed_evidence_refs"] = sorted(batch["reference_map"])
    body["old_entries"] = [{**e["public_advisory"], "identity": alias} for alias, e in zip(aliases, old)]
    messages = [{"role": "system", "content": PROMPT}, {"role": "user", "content": canonical_json(body)}]
    if size(messages) > config.input_capacity_bytes:
        raise ValueError("actual prompt exceeds input capacity")
    return {"stage": "propose", "messages": messages, "batch": batch, "sources": sources,
            "audit_old_entries": config.audit_old_entries, "run_labels": body["run_labels"],
            "prompt_version": PROMPT_VERSION,
            "entry_aliases": aliases, "reference_map": batch["reference_map"],
            "disclosed_old": disclosed, "recall_query": query,
            "base_digest": AdvisoryMemory.from_file(package_path).package_digest,
            "batch_identity": identity(body)}


def _source_groups(value):
    """Flatten event ancestry of all disclosed old entries, including deletions."""
    groups = set()
    if isinstance(value, dict):
        group = value.get('group')
        if isinstance(group, str) and group:
            groups.add(group)
        groups.update(value.get('inherited_source_groups', []))
        for child in value.values():
            groups.update(_source_groups(child))
    elif isinstance(value, list):
        for child in value:
            groups.update(_source_groups(child))
    return groups


def _assemble(package: dict, proposal: dict, frozen: dict, response_ref: str) -> dict:
    if not isinstance(proposal, dict) or set(proposal) != ({"changes", "old_entry_reviews"} if frozen.get("audit_old_entries") else {"changes"}) or not isinstance(proposal["changes"], list):
        raise ValueError("proposal must have exactly a changes list")
    if identity(package) != frozen["base_digest"]:
        raise ValueError("stale batch package digest")
    by_id = {e["public_advisory"]["identity"]: e for e in package["entries"]}
    disclosed = {e["identity"]: e for e in frozen["disclosed_old"]}
    for key, old in disclosed.items():
        if key not in by_id or identity(by_id[key]) != old["entry_digest"]:
            raise ValueError("stale target content")
    allowed_refs = set(frozen["reference_map"])
    edits, touched = [], set()
    for position, change in enumerate(proposal["changes"]):
        if not isinstance(change, dict):
            raise ValueError("change must be object")
        op = change.get("op")
        if frozen.get("learning_profile") in ("workflow-lessons-v1", "workflow-lessons-v2", "workflow-lessons-v3") and (op != "add" or "target_id" in change or "derived_from_ids" in change):
            raise ValueError("workflow profile permits add-only local advice")
        if op not in {"add", "revise", "delete"}:
            raise ValueError("op must be add, revise or delete")
        expected = {"op", "evidence_refs", "reason"}
        if op != "add":
            expected.add("target_id")
        if op != "delete":
            expected.add("advice")
        optional = {"derived_from_ids"} if op != "delete" else set()
        if not expected <= set(change) or set(change) - expected - optional:
            raise ValueError("invalid change fields for op")
        refs = string_list(change["evidence_refs"], "evidence_refs")
        if not refs or not set(refs) <= allowed_refs:
            raise ValueError("evidence_refs must name actually shown steps, results or feedback")
        nonblank(change["reason"], "reason")
        target = frozen["entry_aliases"].get(change.get("target_id"), change.get("target_id"))
        if op != "add":
            nonblank(target, "target_id")
            if target not in disclosed or target in touched:
                raise ValueError("unknown, undisclosed or duplicate target " + str(target) + "; allowed disclosed IDs: " + canonical_json(sorted(disclosed)))
            touched.add(target)
        derived = [frozen["entry_aliases"].get(v, v) for v in string_list(change.get("derived_from_ids", []), "derived_from_ids")]
        if not set(derived) <= set(disclosed):
            raise ValueError("derived_from_ids must have been disclosed; invalid IDs: " + canonical_json(sorted(set(derived)-set(disclosed))) + "; allowed disclosed IDs: " + canonical_json(sorted(disclosed)))
        if op == "delete":
            edits.append((op, target, None))
            continue
        advice = change["advice"]
        if not isinstance(advice, dict) or set(advice) != {"kind", "guidance", "applicability", "negative_conditions"}:
            raise ValueError("advice requires exactly kind/guidance/applicability/negative_conditions")
        if advice["kind"] not in {"strategy", "caution"}:
            raise ValueError("invalid advice kind")
        nonblank(advice["guidance"], "guidance")
        string_list(advice["applicability"], "applicability")
        string_list(advice["negative_conditions"], "negative_conditions")
        stable = frozen.get("change_ids", ["c" + identity([frozen["batch_identity"], i])[:24] for i in range(len(proposal["changes"]))])[position]
        key = target if op == "revise" else "generated-" + identity([frozen["batch_identity"], stable])[:32]
        if op == "add" and key in by_id:
            raise ValueError("generated identity already exists")
        original = by_id[target] if op == "revise" else None
        # Preserve arbitrary host fields; revision version is a stable content identity, not a model number.
        entry = deepcopy(original) if original else {}
        version = identity([original["public_advisory"]["version"] if original else None, stable, change])
        entry["public_advisory"] = {**deepcopy(advice), "identity": key, "version": version}
        provenance = deepcopy(original["host_only_provenance"]) if original else {}
        ancestors = ([target] if original else []) + derived
        inherited = []
        histories = {}
        for ancestor in dict.fromkeys(ancestors):
            ancestor_provenance = deepcopy(by_id[ancestor]["host_only_provenance"])
            for record in ancestor_provenance.pop("generation_history", []):
                histories[identity(record)] = record
            inherited.append({"identity": ancestor, "provenance": ancestor_provenance})
        isolation = set(provenance.get("recall_control", {}).get("isolation_refs", []))
        # All disclosed material contributes isolation, even if the model cites only one step.
        for source in frozen["sources"]:
            isolation.update(source["isolation_refs"])
        for old in frozen["disclosed_old"]:
            isolation.update(old["provenance"]["recall_control"]["isolation_refs"])
        provenance["recall_control"] = {**provenance.get("recall_control", {}), "isolation_refs": sorted(isolation)}
        provenance["generation_history"] = list(histories.values())
        provenance['inherited_source_groups'] = sorted(_source_groups(frozen['disclosed_old']) | _source_groups(frozen['sources']))
        provenance["generation_history"].append({
            "batch_identity": frozen["batch_identity"], "response_ref": response_ref.get(stable) if isinstance(response_ref, dict) else response_ref,
            "evidence_refs": refs, "evidence_locations": {r: frozen["reference_map"][r] for r in refs},
            "reason": change["reason"], "shown_runs": frozen["sources"],
            "shown_old_sources": deepcopy(frozen["disclosed_old"]),
            "disclosed_reference_map": deepcopy(frozen["reference_map"]),
            "disclosure_history": deepcopy(frozen.get("disclosure_history", [])),
            "inherited_sources": inherited,
        })
        entry["host_only_provenance"] = provenance
        edits.append((op, key, entry))
    if frozen.get("audit_old_entries"):
        reviews = proposal["old_entry_reviews"]
        reviewed = set()
        if not isinstance(reviews, list):
            raise ValueError("old_entry_reviews must be a list")
        for review in reviews:
            if not isinstance(review, dict) or set(review) != {"target_id", "disposition", "evidence_refs", "reason"}:
                raise ValueError("invalid old entry review fields")
            target = frozen["entry_aliases"].get(review["target_id"], review["target_id"])
            if target not in disclosed or target in reviewed:
                raise ValueError("review target must be unique and disclosed")
            reviewed.add(target)
            disposition = review["disposition"]
            if disposition not in {"keep", "revise", "merge", "delete", "insufficient_evidence"}:
                raise ValueError("invalid old entry disposition")
            refs = string_list(review["evidence_refs"], "review evidence_refs")
            if not set(refs) <= allowed_refs or (not refs and disposition != "insufficient_evidence"):
                raise ValueError("old entry review needs shown supporting evidence")
            nonblank(review["reason"], "review reason")
            op = next((op for op, key, _ in edits if key == target), None)
            if ((disposition in {"keep", "insufficient_evidence"} and op is not None)
                    or (disposition == "revise" and op != "revise")
                    or (disposition == "delete" and op != "delete")
                    or (disposition == "merge" and op not in {"delete", "revise"})):
                raise ValueError("old entry disposition contradicts actual changes")
        if reviewed != set(disclosed):
            raise ValueError("old entry review omitted a disclosed entry")
    candidate = deepcopy(package)
    # No mutation until all edits pass; derived provenance was read from the pre-batch package.
    replacements = {key: entry for op, key, entry in edits if op == "revise"}
    deleted = {key for op, key, _ in edits if op == "delete"}
    candidate["entries"] = [replacements.get(e["public_advisory"]["identity"], e)
                            for e in candidate["entries"] if e["public_advisory"]["identity"] not in deleted]
    candidate["entries"].extend(entry for op, _, entry in edits if op == "add")
    return candidate


def workflow_request(config, messages, effective_max_tokens=None):
    """Construct the sole recorded request for each explicit workflow route."""
    limit = config.max_tokens if effective_max_tokens is None else effective_max_tokens
    if config.workflow_request_profile == 'sol-chat-v1':
        return {'model': config.model, 'messages': deepcopy(messages),
                'reasoning_effort': config.reasoning_effort, 'max_completion_tokens': limit,
                'response_format': {'type': 'json_object'}, 'stream': False}
    if config.workflow_request_profile == 'deepseek-main-agent-v1':
        return {'model': config.model, 'messages': deepcopy(messages), 'temperature': 0,
                'max_tokens': limit, 'reasoning_effort': config.reasoning_effort,
                'response_format': {'type': 'json_object'},
                'extra_body': {'thinking': {'type': 'enabled'}}}
    raise ValueError('unknown workflow request profile')


def workflow_http_body(request):
    """The SDK extra_body merge, also used by recorded HTTP/curl transport."""
    body = deepcopy(request)
    extra = body.pop('extra_body', {})
    if set(extra) & set(body):
        raise ValueError('extra body overwrites request fields')
    body.update(extra)
    return body


def _workflow_output_field(config):
    return 'max_tokens' if config.workflow_request_profile == 'deepseek-main-agent-v1' else 'max_completion_tokens'


def _workflow_effective_limit(config, request, prior):
    field = _workflow_output_field(config)
    if request and field in request['wire_override']:
        return request['wire_override'][field]
    if request and prior:
        return prior[-1].get('effective_request_parameters', {}).get(field, config.max_tokens)
    return config.max_tokens


def _workflow_response_status(attempt, model, request_profile='sol-chat-v1'):
    if attempt.get('status') != 'received':
        if attempt.get('http_status') in {400, 401, 403, 404}:
            return 'provider_protocol_failed'
        return 'interrupted_unknown' if attempt.get('status') == 'started' else 'transport_failed' if attempt.get('status') == 'transport-failed' else 'provider_protocol_failed'
    raw = attempt.get('response') or {}
    if raw.get('provider_envelope_valid') is not True or not isinstance(raw.get('response_id'), str) or not raw['response_id']:
        return 'provider_protocol_failed'
    if not isinstance(raw.get('returned_model'), str) or not raw['returned_model'].strip():
        return 'provider_protocol_failed'
    if request_profile == 'sol-chat-v1' and raw['returned_model'] != model:
        return 'wrong_model'
    finish = raw.get('finish_reason')
    if finish == 'content_filter' or raw.get('refusal'):
        return 'content_terminal'
    if finish == 'length':
        return 'output_truncated'
    if finish == 'stop' and isinstance(raw.get('content'), str):
        return 'content_terminal'
    return 'provider_protocol_failed'


def _file_sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _recovery_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('recovery evidence must remain within this results root')
    return path


def _validate_workflow_recovery(request, frozen, attempts, paths, config, root):
    """A diagnostic ticket unlocks one exact unusable predecessor, never bad content."""
    if not attempts or not isinstance(request, dict):
        raise ValueError('recovery requires an existing unusable attempt')
    required = {'case_id', 'phase', 'call_identity', 'prior_native_attempt', 'prior_attempt_sha256',
                'diagnosis_path', 'diagnosis_sha256', 'reason', 'wire_override', 'transport_override'}
    if config.workflow_request_profile == 'deepseek-main-agent-v1':
        required |= {'arm', 'anchor_id'}
    if set(request) != required:
        raise ValueError('invalid recovery request fields')
    prior = _recovery_path(root, request['prior_native_attempt'])
    diagnosis_path = _recovery_path(root, request['diagnosis_path'])
    if (prior != paths[-1].resolve() or _file_sha(prior) != request['prior_attempt_sha256']
            or _file_sha(diagnosis_path) != request['diagnosis_sha256']
            or request['call_identity'] != frozen['call_identity'] or request['phase'] != frozen['stage']):
        raise ValueError('recovery input, phase or predecessor digest changed')
    status = _workflow_response_status(attempts[-1], config.model, config.workflow_request_profile)
    if status == 'content_terminal' or request['reason'] != status:
        raise ValueError('content results cannot be technically retried')
    if _workflow_child_active(attempts[-1]):
        raise ValueError('previous API child remains alive')
    pid = attempts[-1].get('pid')
    if status == 'interrupted_unknown' and pid:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise ValueError('previous request process remains alive or unverifiable')
    diagnosis = read_json(diagnosis_path)
    if (diagnosis.get('failed_attempt') != request['prior_native_attempt']
            or diagnosis.get('failure_class') != status or not diagnosis.get('messages_unchanged')
            or not diagnosis.get('repair_action') or not diagnosis.get('recovery_evidence')
            or diagnosis.get('recovery_permitted') is not True):
        raise ValueError('diagnosis does not establish this recovery')
    evidence = diagnosis.get('evidence_paths', [])
    hashes = diagnosis.get('evidence_hashes', {})
    if not evidence or any(_file_sha(_recovery_path(root, p)) != hashes.get(p) for p in evidence):
        raise ValueError('recovery diagnostic evidence missing or changed')
    overrides = request['wire_override']
    field = _workflow_output_field(config)
    previous_limit = attempts[-1].get('effective_request_parameters', {}).get(field, config.max_tokens)
    raised_limit = 65536 if config.workflow_request_profile == 'deepseek-main-agent-v1' else 32768
    if overrides:
        if overrides != {field: raised_limit} or previous_limit not in (config.max_tokens, raised_limit):
            raise ValueError('invalid workflow wire override')
        if status != 'output_truncated' and previous_limit != raised_limit:
            raise ValueError('output allowance needs a real length lineage')
        if status == 'output_truncated' and previous_limit == raised_limit:
            raise ValueError('second length exhaustion is terminal')
    elif status == 'output_truncated':
        raise ValueError('length recovery must change only its output allowance')
    allowed_transport = {'proxy', 'connect_to', 'timeout', 'implementation'}
    if config.workflow_request_profile == 'deepseek-main-agent-v1':
        allowed_transport = {'proxy', 'timeout', 'implementation'}
    if set(request['transport_override']) - allowed_transport:
        raise ValueError('invalid transport override')
    if diagnosis.get('transport_override', {}) != request['transport_override']:
        raise ValueError('transport differs from its diagnostic evidence')
    return request


def _workflow_process_identity(pid):
    """Linux process birth identity; a zombie can no longer issue a request."""
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': pid, 'start_ticks': fields[19], 'state': fields[0],
                'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    except FileNotFoundError:
        return None


def _workflow_child_active(attempt):
    envelope = attempt.get('provider_envelope_path')
    if not envelope:
        return False
    path = Path(envelope).parent / 'client-process.json'
    if not path.exists():
        return False  # The child cannot read credentials before this durable handshake.
    root = Path(attempt['recording_root']).resolve()
    if not path.resolve().is_relative_to(root):
        raise ValueError('child identity escaped recording root')
    child = read_json(path)
    if child['request_identity'] != attempt['wire_identity']:
        raise ValueError('child request identity changed')
    current = _workflow_process_identity(child['pid'])
    return bool(current and current['state'] != 'Z' and all(
        current[k] == child[k] for k in ('pid', 'start_ticks', 'boot_id')))


def workflow_effective_attempt(path):
    """Recover a durable public envelope without changing the original attempt."""
    path = Path(path).resolve()
    original = read_json(path)
    if original.get('offline_completion'):
        link = original['offline_completion']
        native = Path(link['native_path']).resolve()
        if path != native.with_name('received-' + native.name) or _file_sha(native) != link['native_sha256']:
            raise ValueError('offline completion original identity changed')
        effective, expected_path = workflow_effective_attempt(native)
        if expected_path != path or effective != original:
            raise ValueError('offline completion changed')
        return original, path
    envelope_name = original.get('provider_envelope_path')
    if original.get('status') == 'received' or not envelope_name:
        return original, path
    envelope_path = Path(envelope_name).resolve()
    if not envelope_path.exists():
        return original, path
    if _workflow_child_active(original):
        raise ValueError('original API child remains alive')
    if original.get('status') == 'started':
        pid = original.get('pid')
        if not pid:
            raise ValueError('Interrupted response has no process identity')
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise ValueError('Original request process remains alive or unverifiable')
    root = Path(original['recording_root']).resolve()
    if not path.is_relative_to(root) or not envelope_path.is_relative_to(root):
        raise ValueError('saved provider envelope escaped recording root')
    sidecar = envelope_path.parent
    binding = read_json(sidecar/'input.json')
    wire = read_json(sidecar/'wire-request.json')
    frozen = binding['frozen']
    native_ref = read_json(sidecar/'native-attempt-ref.json')
    if (Path(root/native_ref['path']).resolve() != path
            or frozen['call_identity'] != original['call_identity']
            or frozen['stage'] != original['stage']
            or frozen['messages'] != original['messages']
            or wire != read_json(original['wire_path'])
            or identity(wire) != original['wire_identity']
            or _file_sha(original['wire_path']) != original['wire_sha256']):
        raise ValueError('saved provider envelope input identity changed')
    envelope = read_json(envelope_path)
    if envelope.get('http_status') != 200 or not (envelope.get('transport_ok') is True or envelope.get('curl_returncode') == 0):
        return original, path
    response = envelope.get('response')
    if not isinstance(response, dict):
        return original, path
    completed = dict(original, status='received', response=response, usage=response.get('usage'),
                     elapsed_seconds=envelope['elapsed_seconds'],
                     offline_completion={'native_path':str(path),'native_sha256':_file_sha(path),
                         'envelope_path':str(envelope_path),'envelope_sha256':_file_sha(envelope_path)})
    derived = path.with_name('received-' + path.name)
    if derived.exists():
        if read_json(derived) != completed:
            raise ValueError('saved offline completion evidence changed')
    else:
        _write(derived, completed)
    return completed, derived


def _verify_workflow_wire(path, frozen, config):
    """Validate actual request and recovery lineage on both live reuse and replay."""
    path = Path(path).resolve()
    attempt = read_json(path)
    if attempt.get('offline_completion'):
        attempt, _ = workflow_effective_attempt(path)
        _verify_workflow_wire(attempt['offline_completion']['native_path'], frozen, config)
        return attempt
    wire_path = Path(attempt['wire_path']).resolve()
    root = Path(attempt['recording_root']).resolve()
    if not path.is_relative_to(root) or not wire_path.is_relative_to(root):
        raise ValueError('workflow wire escaped its recording root')
    wire = read_json(wire_path)
    if (_file_sha(wire_path) != attempt['wire_sha256'] or identity(wire) != attempt['wire_identity']
            or wire['messages'] != frozen['messages'] or attempt['messages'] != frozen['messages']
            or attempt['request_digest'] != identity(frozen['messages'])):
        raise ValueError('workflow wire/messages identity changed')
    request = attempt.get('recovery_request')
    prior_paths = [p for p in sorted(path.parent.glob('attempt-*.json')) if p.name < path.name
                   and read_json(p).get('call_identity') == frozen['call_identity']]
    prior = [read_json(p) for p in prior_paths]
    if request is not None:
        if attempt.get('recovery_request_identity') != identity(request):
            raise ValueError('recovery ticket identity changed')
        _validate_workflow_recovery(request, frozen, prior, prior_paths, config, root)
        for predecessor in prior_paths:
            _verify_workflow_wire(predecessor, frozen, config)
    elif prior:
        raise ValueError('extra workflow attempt lacks a bound recovery')
    expected = workflow_request(config, frozen['messages'], _workflow_effective_limit(config, request, prior))
    if wire != expected or attempt['effective_request_parameters'] != {k:v for k,v in wire.items() if k != 'messages'}:
        raise ValueError('workflow request parameters changed')
    return attempt


def _recorded_workflow_response(attempt_dir, frozen, provider, config, *, can_call=None,
                                caller_code=None, recovery_request=None):
    if can_call is not None and hasattr(can_call, 'refresh'):
        can_call.refresh()
    paths = [p for p in sorted(attempt_dir.glob('attempt-*.json'))
             if read_json(p).get('call_identity') == frozen['call_identity']]
    effective = [workflow_effective_attempt(p) for p in paths]
    attempts = [row[0] for row in effective]
    for attempt, path in effective:
        if _workflow_response_status(attempt, config.model, config.workflow_request_profile) == 'content_terminal':
            _verify_workflow_wire(path, frozen, config)
            if attempt['messages'] != frozen['messages'] or attempt['request_digest'] != identity(frozen['messages']):
                raise ValueError('saved workflow response changed input')
            return attempt['response'], str(path)
    if attempts and recovery_request is None:
        raise ValueError('needs_diagnosis: original response is not consumable')
    if provider is None:
        raise ValueError('saved response unavailable; offline replay cannot call a provider')
    if len(attempts) >= 3:
        raise ValueError('workflow phase attempt allowance exhausted')
    if (config.max_api_calls is not None
            and len(list(attempt_dir.parents[1].glob('batches/*/attempt-*.json'))) >= config.max_api_calls):
        raise ValueError('generation total API attempt budget exhausted')
    if can_call is None or not hasattr(can_call, 'reserve'):
        raise ValueError('workflow provider requires an atomic shared reservation')
    if recovery_request is not None:
        _validate_workflow_recovery(recovery_request, frozen, attempts, paths, config, Path(can_call.root))
    request = provider.prepare_attempt(frozen, recovery_request)
    wire = request['wire']
    expected = workflow_request(config, frozen['messages'], _workflow_effective_limit(config, recovery_request, attempts))
    if wire != expected or size(wire['messages']) > config.input_capacity_bytes:
        raise ValueError('actual workflow wire differs from frozen messages/model/high/capacity')
    if request.get('recovery_request_identity') != (identity(recovery_request) if recovery_request else None):
        raise ValueError('prepared request did not bind the recovery ticket')
    path = attempt_dir / f"attempt-{len([p for p in attempt_dir.glob('attempt-*.json') if p.stem.removeprefix('attempt-').isdigit()]):03d}.json"
    if not can_call.reserve(frozen, path, request=request):
        raise ValueError('shared offline API attempt budget exhausted')
    wire_path = attempt_dir / 'wire-requests' / path.name
    _write(wire_path, wire)
    attempt = {'status': 'started', 'stage': frozen['stage'], 'kind': 'recovery' if recovery_request else 'primary',
               'pid': os.getpid(), 'recording_root': str(Path(can_call.root).resolve()), 'messages': deepcopy(frozen['messages']),
               'request_digest': identity(frozen['messages']), 'call_identity': frozen['call_identity'],
               'wire_path': str(wire_path), 'wire_sha256': _file_sha(wire_path), 'wire_identity': identity(wire),
               'transport': request['transport'], 'effective_request_parameters': {k: v for k, v in wire.items() if k != 'messages'},
               'recovery_request': deepcopy(recovery_request),
               'recovery_request_identity': identity(recovery_request) if recovery_request else None,
               'recovery_request_path': request.get('recovery_request_path'),
               'recovery_request_sha256': request.get('recovery_request_sha256'),
               'started_at': datetime.now(timezone.utc).isoformat(), 'usage': None,
               'provider_envelope_path': str(Path(can_call.last_sidecar)/'provider-envelope.json') if hasattr(can_call, 'last_sidecar') else None,
               'generation_code': _code_identity(), **({'caller_code': caller_code} if caller_code else {})}
    _write(path, attempt)
    started = time.monotonic()
    try:
        raw = provider(wire['messages'])
    except Exception as exc:
        from .host_control import raise_control_error
        from .hosted_stream import deadline_error
        raise_control_error(exc)
        if (deadline := deadline_error(exc)) is not None:
            raise deadline
        attempt.update(status='transport-failed' if isinstance(exc, TransportFailure) else 'provider-failed',
                       error_type=type(exc).__name__, http_status=getattr(exc, 'status_code', None),
                       provider_request_id=getattr(exc, 'request_id', None), elapsed_seconds=time.monotonic() - started)
        _write(path, attempt)
        raise ValueError('needs_diagnosis: workflow provider attempt failed') from None
    response = {k: public_material(raw.get(k)) for k in
                ('content', 'usage', 'response_id', 'returned_model', 'finish_reason', 'refusal', 'provider_envelope_valid')}
    attempt.update(status='received', response=response, usage=response['usage'],
                   completed_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started)
    _write(path, attempt)
    status = _workflow_response_status(attempt, config.model, config.workflow_request_profile)
    if status != 'content_terminal':
        raise ValueError('needs_diagnosis: ' + status)
    return response, str(path)


def recorded_response(attempt_dir: Path, frozen: dict, provider, config: GenerationConfig,
              correction: str | None = None, *, can_call=None, caller_code=None, retry_failed=False, recovery_request=None) -> tuple[dict, str]:
    """Reuse the latest public response; callers validate it before requesting correction."""
    if _workflow(config):
        if correction or retry_failed:
            raise ValueError("workflow recovery requires a bound diagnostic request")
        return _recorded_workflow_response(attempt_dir, frozen, provider, config, can_call=can_call,
            caller_code=caller_code, recovery_request=recovery_request)
    stage = frozen["stage"]
    reuse = attempt_dir / "propose-reuse.json"
    if stage == "propose" and reuse.exists() and correction is None:
        saved = read_json(reuse)
        return saved["response"], saved["response_ref"]
    paths = [p for p in sorted(attempt_dir.glob("attempt-*.json")) if read_json(p)["stage"] == stage
             and read_json(p).get("call_identity") == frozen.get("call_identity")]
    attempts = [read_json(p) for p in paths]
    if attempts and attempts[-1]["status"] == "received" and correction is None:
        response_path = paths[-1]
        origin_path = attempt_dir.parents[1] / 'replay-origin.json'
        if origin_path.exists():
            if provider is not None:
                raise ValueError('a saved-response replay revision cannot call a provider')
            origin = read_json(origin_path)
            relative = str(response_path.relative_to(attempt_dir.parents[1]))
            original = Path(origin['source_root']) / relative
            if (origin['attempt_identities'].get(relative) != identity(attempts[-1])
                    or identity(read_json(original)) != identity(attempts[-1])):
                raise ValueError('original response identity changed')
            response_path = original
        return attempts[-1]["response"], str(response_path)
    window_key = frozen.get("call_identity", stage)
    window_path = attempt_dir / f"{window_key}-retry-window.json"
    if retry_failed and attempts and attempts[-1]['status'] in ('provider-failed','transport-failed','started'):
        _write(window_path, {'after_attempts':len(attempts), 'reason':'explicit diagnosed technical recovery'})
    offset = read_json(window_path)["after_attempts"] if window_path.exists() else 0
    active = attempts[offset:]
    failures = sum(a["status"] in {"transport-failed", "started"} for a in active)
    if active and active[-1]["status"] == "provider-failed":
        raise ValueError("provider failure: fix infrastructure then explicitly resume with retry_failed")
    if failures > config.transport_retries:
        raise ValueError("transport attempt allowance exhausted; unresolved started attempts may have been billed")
    messages = deepcopy(frozen["messages"])
    if correction:
        corrections = sum(a.get("kind") == "format-correction" for a in active)
        if corrections >= config.format_retries:
            raise ValueError(correction)
        previous = next((a for a in reversed(attempts) if a["status"] == "received"), None)
        if previous:
            messages.append({"role": "assistant", "content": previous["response"].get("content") or ""})
        messages.append({"role": "user", "content": "Fix this validation error using the same shown evidence: " + correction})
    if size(messages) > config.input_capacity_bytes:
        raise ValueError("correction prompt exceeds input capacity; start a run with more capacity")
    while True:
        if provider is None:
            raise ValueError("saved response unavailable; provider required to continue")
        if config.max_api_calls is not None and len(list(attempt_dir.parents[1].glob("batches/*/attempt-*.json"))) >= config.max_api_calls:
            raise ValueError("generation total API attempt budget exhausted")
        if can_call is not None:
            allowed = (can_call.reserve(frozen, attempt_dir / f"attempt-{len([p for p in attempt_dir.glob('attempt-*.json') if p.stem.removeprefix('attempt-').isdigit()]):03d}.json")
                       if hasattr(can_call, "reserve") else can_call())
            if not allowed:
                raise ValueError("shared offline API attempt budget exhausted")
        path = attempt_dir / f"attempt-{len([p for p in attempt_dir.glob('attempt-*.json') if p.stem.removeprefix('attempt-').isdigit()]):03d}.json"
        attempt = {"status": "started", "stage": stage, "kind": "format-correction" if correction else "primary",
                   "messages": messages, "request_digest": identity(messages),
                   "started_at": datetime.now(timezone.utc).isoformat(), "usage": None,
                   **({"call_identity": frozen["call_identity"]} if "call_identity" in frozen else {}),
                   "generation_code": _code_identity(), **({"caller_code": caller_code} if caller_code else {})}
        _write(path, attempt)
        started = time.monotonic()
        try:
            raw = provider(messages)
        except Exception as exc:
            attempt.update(status="transport-failed" if isinstance(exc, TransportFailure) else "provider-failed",
                           error_type=type(exc).__name__, http_status=getattr(exc,'status_code',None),
                           provider_request_id=getattr(exc,'request_id',None), elapsed_seconds=time.monotonic()-started)
            _write(path, attempt)
            attempts.append(attempt)
            failures = sum(a["status"] in {"transport-failed", "started"} for a in attempts[offset:])
            if isinstance(exc, TransportFailure) and failures <= config.transport_retries:
                continue
            raise ValueError("provider attempt failed; inspect saved attempt status") from None
        # The injection contract returns a public envelope. Whitelist before persistence.
        response = {k: public_material(raw.get(k)) for k in ("content", "usage", "response_id", "returned_model", "finish_reason")}
        attempt.update(status="received", response=response, usage=response["usage"],
                       completed_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-started)
        _write(path, attempt)  # Accounting is durable BEFORE JSON interpretation.
        return response, str(path)



def _review_input(frozen: dict, draft: dict, response_ref: str, config: GenerationConfig) -> dict:
    body = json.loads(frozen["messages"][1]["content"])
    refs = {r for change in draft["changes"] for r in change["evidence_refs"]}
    # Retention decisions need primary fragments too, not only proposed edits.
    refs.update(r for review in draft.get("old_entry_reviews", []) for r in review["evidence_refs"])
    # Empty proposals still receive primary source evidence and the complete learning view.
    originals = frozen["batch"]["original_fragments"]
    body.update(draft=draft, original_fragments={r: originals[r] for r in sorted(refs or originals)})
    messages = [{"role": "system", "content": REVIEW_PROMPT}, {"role": "user", "content": canonical_json(body)}]
    if size(messages) > config.input_capacity_bytes:
        raise ValueError("review input exceeds reserved capacity; start a smaller learning batch")
    return {**frozen, "stage": "review", "messages": messages,
            "propose_response_ref": response_ref, "review_identity": identity(messages)}


def _propose(directory: Path, frozen: dict, package: dict, provider, config: GenerationConfig, *, can_call=None) -> tuple[dict, dict, str]:
    """Validate structural edits per stage; downstream storage never invokes correction."""
    response, response_ref = recorded_response(directory, frozen, provider, config, can_call=can_call)
    while True:
        try:
            if response["finish_reason"] not in {None, "stop"}:
                raise ValueError("model response not complete")
            proposal = _save_learning_notes(directory, frozen, response, response_ref)
            candidate = _assemble(package, proposal, frozen, response_ref)
            _write(directory / (frozen["stage"] + "-proposal.json"), proposal)
            return proposal, candidate, response_ref
        except (ValueError, TypeError, KeyError) as exc:
            error = str(exc)
            _write(directory / (frozen["stage"] + "-validation-error.json"), {"error": error, "response_ref": response_ref})
            if (provider is None or config.format_retries == 0
                    or (frozen["stage"] == "propose" and (directory / "propose-reuse.json").exists())):
                raise ValueError(error) from None
            response, response_ref = recorded_response(directory, frozen, provider, config, error, can_call=can_call)


REPAIR_PROMPT_VERSION = "memory-repair-rules-v2"
REPAIR_PROMPT = effective_prompt("repair")


def _phase_input(frozen, messages, phase, config, **binding):
    if config.learning_profile != 'workflow-lessons-v3' and size(messages) > config.input_capacity_bytes:
        raise ValueError('complete phase request exceeds capacity')
    return {**frozen, 'stage': phase, 'messages': messages,
            'prompt_version': REPAIR_PROMPT_VERSION if phase == 'repair' else frozen.get('prompt_version', PROMPT_VERSION),
            'call_identity': identity([phase, messages, _config_data(config), binding])}


def _disclosed_old_bodies(package, frozen):
    """Expose only the old entries whose ancestry was frozen for this batch."""
    by_id = {e['public_advisory']['identity']: e for e in package['entries']}
    bodies = []
    for old in frozen['disclosed_old']:
        entry = by_id[old['identity']]
        if identity(entry) != old['entry_digest']:
            raise ValueError('stale disclosed old entry')
        bodies.append(entry['public_advisory'])
    return bodies


def _check_input(package, proposal, frozen, support, config, *, phase='check',
                 prompt=None, prompt_version=None, groups=None, disclosure_history=None,
                 source_reference_policy='canonical-only-v1'):
    from .trajectory_memory_content_check import prepare_check, PROMPT as CHECK_PROMPT, PROMPT_VERSION
    if _workflow(config):
        spec = frozen['workflow_prompts']['checker']
        if prompt is not None and prompt != spec['text']:
            raise ValueError('workflow checker prompt override forbidden')
        prompt, prompt_version = spec['text'], spec['version']
    check = prepare_check(proposal, frozen['batch_identity'], support,
                          _disclosed_old_bodies(package, frozen), frozen['reference_map'],
                          prompt=prompt or CHECK_PROMPT, prompt_version=prompt_version or PROMPT_VERSION,
                          capacity=None if config.learning_profile == 'workflow-lessons-v3' else config.input_capacity_bytes, change_ids=frozen.get('change_ids'),
                          groups=groups, disclosure_history=disclosure_history, source_reference_policy=source_reference_policy,
                          workflow_view=frozen["batch"] if _workflow(config) else None,
                          change_order=frozen.get("change_order"))
    check['stage'] = phase
    check['generation_input_identity'] = identity([package, frozen])
    check['call_identity'] = identity([check['check_input_identity'], _config_data(config), phase, phase == 'recheck', check['generation_input_identity']])
    return check


def check_proposal(directory, package, proposal, frozen, support, provider, config, *,
                   can_call=None, prompt=None, prompt_version=None, phase='check',
                   groups=None, disclosure_history=None, source_reference_policy='canonical-only-v1'):
    """The same checkpoint is used by imported/natural generation and control experiments."""
    from .trajectory_memory_content_check import parse_verdict
    directory = Path(directory)
    check = _check_input(package, proposal, frozen, support, config, phase=phase,
                         prompt=prompt, prompt_version=prompt_version, groups=groups,
                         disclosure_history=disclosure_history, source_reference_policy=source_reference_policy)
    path = directory / (phase + '-input.json')
    if path.exists() and read_json(path) != check:
        raise ValueError('check input changed; use a separately identified revision')
    _write(path, check)
    response, response_ref = recorded_response(directory, check, provider, config, can_call=can_call)
    try:
        _save_learning_notes(directory, check, response, response_ref)
        verdict = parse_verdict(response, check)
    except (ValueError, TypeError, KeyError) as error:
        _write(directory / (phase + '-error.json'), {'error': str(error), 'response_ref': response_ref,
                                                   'content_status': 'unresolved'})
        raise
    _write(directory / (phase + 's.json'), verdict)
    result = {'status': 'partial', 'stop_after': phase, 'usable_candidate': False,
              'check_input_identity': check['check_input_identity'], 'report_identity': identity(verdict),
              'response_ref': response_ref}
    _write(directory / (phase + '-result.json'), result)
    return result


def generate_memory(trajectory_paths, output_dir, *, current_package=None, feedback_path=None,
                    config=None, provider=None, resume=False, replay_from_batch=None,
                    recall_queries=(), retry_failed=False, review_revision_from=None, can_call=None,
                    proposal_import=None, stop_after=None, prepared_batch=None, prepare_only=False, recovery_requests=None):
    """One item-isolating generation pipeline; imported drafts use this same checker/consumer."""
    from .trajectory_memory_content_check import PROMPT as CHECK_PROMPT, CONTRACT, json_object
    root=Path(output_dir).resolve();manifest_path=root/'run.json'
    if manifest_path.exists() and read_json(manifest_path).get('schema_version')=='trajectory-memory-generation-v3':
        if not resume or provider is not None or retry_failed:raise ValueError('v3 is read-only; import its draft into a new v4 revision')
        result=read_json(root/'summary.json')
        if validate_generated_candidate(result)=='pending':raise ValueError('legacy candidate not qualified')
        return result
    config=config or GenerationConfig()
    prompts = workflow_prompts(config.workflow_prompt_manifest, expected_profile=config.learning_profile) if _workflow(config) else None
    generation_prompt = prompts["generation"]["text"] if prompts else PROMPT
    check_prompt = prompts["checker"]["text"] if prompts else CHECK_PROMPT
    repair_prompt = prompts["repair"]["text"] if prompts else REPAIR_PROMPT
    if _workflow(config) and (feedback_path or proposal_import or prepared_batch):
        raise ValueError("workflow profile requires a natural single-run input")
    if review_revision_from:raise ValueError('use proposal_import/prepared_batch for a new content revision')
    if retry_failed and not resume:raise ValueError('retry_failed requires explicit resume after technical diagnosis')
    if replay_from_batch is not None and (not resume or provider is not None or not isinstance(replay_from_batch,int) or replay_from_batch<0):
        raise ValueError('replay_from_batch requires offline resume and nonnegative checkpoint')
    if stop_after not in (None,'check'):raise ValueError('invalid stop_after')
    imported=read_json(proposal_import) if isinstance(proposal_import,(str,Path)) else deepcopy(proposal_import)
    input_issues=[]
    if prepared_batch is not None:
        prepared_batch=deepcopy(prepared_batch);runs=[];snapshots=[]
        sources=prepared_batch['frozen']['sources'];start=prepared_batch['package']
        batches=[prepared_batch['frozen'].get('batch',{})]
    else:
        runs,sources,snapshots=load_runs(trajectory_paths,feedback_path,issues=input_issues)
        start=read_json(current_package) if current_package else {'schema_version':'trajectory-memory-package-v3','entries':[]}
        start={k:start[k] for k in ('schema_version','entries')}
        if _workflow(config):
            from .trajectory_memory_workflow_view import build_learning_view
            if len(runs) != 1 or start['entries']:
                raise ValueError('workflow profile requires one run and an empty starting package')
            batches = [build_learning_view(runs[0], profile=config.learning_profile)]
        else:
            capacity=config.input_capacity_bytes-size(PROMPT)-2*config.old_memory_bytes-config.review_reserve_bytes-2000
            try:
                batches=make_batches(runs,capacity)
            except (ValueError,TypeError,KeyError,AttributeError,IndexError):
                batches=[]
                # Isolate only actual packing failures; normal inputs keep the existing packer.
                for run in runs:
                    try:batches.extend(make_batches([run],capacity))
                    except (ValueError,TypeError,KeyError,AttributeError,IndexError) as exc:
                        input_issues.append({'stage':'input','run_ref':run['run_ref'],'origin':'structure','message':str(exc)})
    if imported is not None and len(batches)!=1:raise ValueError('import requires one exact complete batch')
    manifest=to_primitive({'schema_version':'trajectory-memory-generation-v4','config':_config_data(config),
        'prompt_digest':identity(generation_prompt),'check_prompt_digest':identity(check_prompt),'repair_prompt_digest':identity(repair_prompt),
        'check_contract':CONTRACT,'inputs':sources,'input_issues':input_issues,'normalized_input_digest':identity(runs),
        'start_digest':identity(start),'batches_digest':identity(batches),'proposal_import':imported,
        'prepared_input_identity':identity(prepared_batch) if prepared_batch else None,
        'proposal_origin':prepared_batch.get('proposal_origin') if prepared_batch else None})
    if manifest_path.exists():
        saved=read_json(manifest_path)
        if resume and provider is None and not prepare_only and stop_after is None and saved!=manifest:
            # A terminal result can be verified read-only under its original prompts.
            # Changed input/model/packing still requires a separate revision.
            original_prompts={**manifest,**{key:saved[key] for key in
                ('prompt_digest','check_prompt_digest','repair_prompt_digest')}}
            if original_prompts==saved and (root/'summary.json').exists():
                previous=read_json(root/'summary.json')
                if validate_generated_candidate(previous) in ('usable_candidate','retain_base'):
                    return previous
        if not resume or saved!=manifest:raise ValueError('resume input/prompt/model changed; use new revision')
    else:
        if resume or (root.exists() and any(root.iterdir())):raise ValueError('fresh directory or existing run.json required')
        _write(root/'start.json',start);AdvisoryMemory.from_file(root/'start.json')
        for i,snapshot in enumerate(snapshots):_write(root/'inputs'/f'source-{i:03d}.json',snapshot)
        _write(root/'normalized-runs.json',runs);_write(manifest_path,manifest)
    processors=read_json(root/'processing.json') if (root/'processing.json').exists() else []
    processors.append({'code':_code_identity(),'resume':resume,'at':datetime.now(timezone.utc).isoformat()})
    _write(root/'processing.json',processors)
    context=_context(config,runs);_write(root/'leakage-context.json',context)
    registry=MemoryRegistry(root/'registry')
    if not (root/'registry/registry.json').exists():registry.bootstrap(root/'start.json',context,BootstrapReceipt(identity(start),'offline-generation:'+identity(manifest)))
    if (root/'summary.json').exists():
        previous=read_json(root/'summary.json')
        if previous.get('status') in ('completed','completed_with_issues') and validate_generated_candidate(previous)=='pending':
            raise ValueError('saved candidate or supporting identity damaged; recover from trusted base explicitly')
    results=[];package=start;issues=deepcopy(input_issues);partial=None;batch_records=[]
    for index,batch in enumerate(batches):
        directory=root/'batches'/f'{index:04d}'
        try:
            if (directory/'base.json').exists() and read_json(directory/'base.json')!=package:
                raise ValueError('saved batch base changed; preserve original evidence and continue affected operations in a new revision')
            _write(directory/'base.json',package)
            frozen=deepcopy(prepared_batch['frozen']) if prepared_batch else _freeze_batch(batch,directory/'base.json',runs,config)
            if (directory/'input.json').exists() and read_json(directory/'input.json')!=frozen:raise ValueError('frozen batch changed; downstream old evidence cannot bind new base')
            _write(directory/'input.json',frozen)
            support=deepcopy(prepared_batch['support']) if prepared_batch else {'batch':deepcopy(batch)}
            if imported is not None:
                proposal=deepcopy(imported);response_ref='import:'+identity(imported)
            else:
                phase=_phase_input(frozen,frozen['messages'],'propose',config,batch=frozen['batch_identity'])
                if _workflow(config):
                    phase_path = directory / 'propose-input.json'
                    if phase_path.exists() and read_json(phase_path) != phase:
                        raise ValueError('frozen G input changed')
                    _write(phase_path, phase)
                if prepare_only:
                    _write(directory/'propose-input.json',phase);partial={'status':'partial','prepared':True};break
                raw,response_ref=recorded_response(directory,phase,provider,config,can_call=can_call,retry_failed=retry_failed,
                    recovery_request=(recovery_requests or {}).get(phase["call_identity"]))
                if raw.get('finish_reason')!='stop':raise ValueError('natural proposal incomplete')
                proposal=json_object(raw.get('content')) if _workflow(config) else _save_learning_notes(directory, phase, raw, response_ref)
            _write(directory/'propose-proposal.json',proposal)
            if prepare_only or stop_after=='check':
                ids,groups,valid,_=_proposal_units(package,proposal,frozen)
                check=_check_input(package,_subset(proposal,ids,valid),{**frozen,'change_ids':valid},support,config,groups=groups)
                if prepare_only:_write(directory/'prepared-check.json',check)
                else:_write(directory/'check-record.json',_capture_phase(directory,check,provider,config,can_call,retry_failed=retry_failed))
                partial={'status':'partial','prepared':prepare_only,'stop_after':stop_after};break
            result=_complete_checked_batch(directory,package,proposal,frozen,support,response_ref,provider,config,registry,context,
                                           can_call=can_call,retry_failed=retry_failed,recovery_requests=recovery_requests,check_origin=prepared_batch.get('check_origin') if prepared_batch else None)
            results.append(result);package=read_json(result['candidate_path'])
            issues.extend({'batch':index,**issue} for issue in result['report']['issues'])
            batch_records.append({'batch':index,'base_digest':frozen['base_digest'],'output_digest':identity(package),'status':result['status']})
            _write(root/'progress.json',{'processed_batches':index+1,'completed_batches':len(results),'candidate_path':result['candidate_path'],'issues':issues})
        except (ValueError,TypeError,KeyError,OSError) as exc:
            from .host_control import raise_control_error
            from .hosted_stream import deadline_error
            raise_control_error(exc)
            if (deadline := deadline_error(exc)) is not None:
                raise deadline
            issue={'batch':index,'stage':'generation','origin':'implementation_or_input','message':str(exc),'base_digest':identity(package)}
            issues.append(issue);batch_records.append({**issue,'status':'unresolved','output_digest':identity(package)})
            _write(directory/'issue.json',issue)
            # No new candidate from the failed batch; next independent input sees the actual retained base.
            continue
    attempts=[workflow_effective_attempt(p)[0] if _workflow(config) else read_json(p)
              for p in sorted(root.glob('batches/*/attempt-*.json'))]
    if partial is None:
        report={'schema':'generated-candidate-report-v4','start_package':start,'batches':results,'batch_records':batch_records,
            'issues':issues,'candidate_digest':identity(package),'manifest':manifest,'manifest_identity':identity(manifest),'total_batches':len(batches)}
        _write(root/'content-report.json',report)
        counts={key:sum(r['content_check'][key] for r in results) for key in ('accepted','rejected','unresolved')}
        summary={'status':'completed_with_issues' if issues else 'completed',
            'candidate_path':results[-1]['candidate_path'] if results else str(root/'start.json'),
            'candidate_digest':identity(package),'changes':counts['accepted'],'issues':issues,
            'content_check':{'schema':'generated-candidate-v4',**counts,'report_path':str(root/'content-report.json'),
                'report_identity':identity(report),'no_change_reason':None if counts['accepted'] else 'issues' if issues else 'no_justified_edits'}}
        summary['qualification']=validate_generated_candidate(summary,package,report)
        if summary['qualification']=='pending':
            summary.update(status='interrupted',error='candidate qualification failed; trusted base retained')
    else:summary={**partial,'qualification':'pending','content_check':{'status':'unresolved'}}
    summary.update(generation_record=str(manifest_path),registry_path=str(root/'registry'),completed_batches=len(results),
        total_batches=len(batches),unprocessed_batches=len(batches)-len(batch_records),batch_records=batch_records,
        formal_feedback=[{'run_ref':r['run_ref'],'status':r['formal_feedback']['status']} for r in runs],
        api_attempts=len(attempts),uncertain_attempts=sum(a['status']=='started' for a in attempts),
        unknown_usage_attempts=sum(a.get('usage') is None for a in attempts),
        usage={k:sum((a.get('usage') or {}).get(k,0) or 0 for a in attempts) for k in ('prompt_tokens','completion_tokens','total_tokens')},
        learning_notes_paths=[str(p) for p in sorted(root.glob('batches/**/**learning-notes.json'))],
        stage_attempts={s:sum(a['stage']==s for a in attempts) for s in ('propose','check','repair','recheck')})
    if recall_queries and summary.get('candidate_path'):
        memory=AdvisoryMemory.from_file(summary['candidate_path'])
        _write(root/'recall.json',{'candidate_digest':memory.package_digest,'queries':[
            {'query':q,'outcome':to_primitive(memory.recall(q,AuthorizedContext(8,32768,(),(),(),1)))} for q in recall_queries]})
        summary['recall_path']=str(root/'recall.json')
    _write(root/'summary.json',summary)
    return summary


def _linked_groups_v3(check_input, verdict):
    """Close model-declared relations over host atomic units without allocating new repair credit."""
    groups = [set(c['change_id'] for c in u['changes']) for u in check_input['units']]
    for c in verdict['checks']:
        unit = next(u for u in check_input['units'] if u['unit_id'] == c['unit_id'])
        linked = {x['change_id'] for x in unit['changes']} | set(c['related_change_ids'])
        hits = [g for g in groups if g & linked]
        groups = [g for g in groups if g not in hits] + [set().union(linked, *hits)]
    return [sorted(g) for g in groups]


def _validate_generated_candidate_v3(result, candidate=None, report=None):
    """Shared consumer gate rechecks bodies, obligations, transaction sets and package assembly."""
    from .trajectory_memory_content_check_legacy import parse_verdict, edit_units
    try:
        cc=result['content_check']
        if (result['status'] not in ('completed','no-change') or cc['schema']!='generated-candidate-v3'
                or cc['status'] not in ('passed','passed_subset','all_rejected','not_applicable')
                or cc['unresolved'] or result.get('unprocessed_batches',0)):
            return 'pending'
        candidate=read_json(result['candidate_path']) if candidate is None else candidate
        report=read_json(cc['report_path']) if report is None else report
        if (identity(report)!=cc['report_identity'] or report['schema']!='generated-candidate-report-v3'
                or identity(candidate)!=result['candidate_digest'] or report['candidate_digest']!=identity(candidate)):
            return 'pending'
        if (report['manifest']['schema_version'] != 'trajectory-memory-generation-v3'
                or identity(report['manifest']) != report['manifest_identity']
                or len(report['batches']) != report['total_batches']
                or report['manifest']['start_digest'] != identity(report['start_package'])):
            return 'pending'
        package=report['start_package']
        config=GenerationConfig(**report['manifest']['config'])
        accepted_total=rejected_total=0
        for batch in report['batches']:
            r=batch['report']
            if batch['stage']['status']!='passed' or batch['stage']['record']['package_digest']!=batch['candidate_digest']:
                return 'pending'
            if r['schema']!='generated-edits-report-v3' or r['base_package']!=package:
                return 'pending'
            ids=['c'+identity([r['original_frozen']['batch_identity'],i])[:24] for i in range(len(r['original_proposal']['changes']))]
            if ids!=r['initial_change_ids'] or set(r['accepted_change_ids']) & set(r['rejected_change_ids']) or set(ids)!=set(r['accepted_change_ids'])|set(r['rejected_change_ids']):
                return 'pending'
            if any(v!=1 for v in r['repair_used'].values()) or not set(r['repair_used'])<=set(ids):
                return 'pending'
            if ids:
                if not r['checks']:
                    return 'pending'
                for checked in r['checks']:
                    attempt=read_json(checked['response_ref'])
                    if (identity(checked['input']['messages'][0]['content'])!=report['manifest']['check_prompt_digest']
                            or attempt['status']!='received' or attempt['response']!=checked['response']
                            or attempt['call_identity']!=checked['input']['call_identity']
                            or attempt['messages']!=checked['input']['messages']):
                        return 'pending'
                verdicts=[parse_verdict(c['response'],c['input']) for c in r['checks']]
                first=r['checks'][0]['input']
                if first['units']!=edit_units(r['original_proposal'],r['original_frozen']['batch_identity'],old_entries=_disclosed_old_bodies(package,r['original_frozen'])):
                    return 'pending'
                if first!=_check_input_v3(package,r['original_proposal'],r['original_frozen'],r['support'],config,
                        prompt=first['messages'][0]['content'],prompt_version=first['prompt_version']):
                    return 'pending'
                first_verdict=verdicts[0]
                groups=_linked_groups_v3(first,first_verdict)
                bad={c['change_id'] for u in first['units'] for c in u['changes']
                     if next(v for v in first_verdict['checks'] if v['unit_id']==u['unit_id'])['disposition']!='accept'}
                if groups!=r['groups']:
                    return 'pending'
                repair_ids=set().union(*(set(g) for g in groups if set(g)&bad)) if bad else set()
                if repair_ids:
                    if r['repair'] is None or set(r['repair_used']) != repair_ids:
                        return 'pending'
                    repair=r['repair'];attempt=read_json(repair['response_ref'])
                    if (attempt['status']!='received' or attempt['stage']!='repair'
                            or attempt['response']!=repair['response']
                            or attempt['call_identity']!=repair['input']['call_identity']
                            or attempt['messages']!=repair['input']['messages']
                            or identity(repair['input']['messages'][0]['content'])!=report['manifest']['repair_prompt_digest']):
                        return 'pending'
                    updates=_parse_repair_v3(repair['response'],r['original_proposal'],ids,repair_ids,groups)
                    before=dict(zip(ids,r['original_proposal']['changes']))
                    surviving=[cid for cid in ids if cid not in updates or updates[cid] is not None]
                    changes=[updates.get(cid,before[cid]) for cid in surviving]
                    if surviving!=r['surviving_change_ids'] or changes!=r['final_proposal']['changes']:
                        return 'pending'
                elif r['repair'] is not None or r['repair_used'] or r['surviving_change_ids']!=ids or r['final_proposal']!=r['original_proposal']:
                    return 'pending'
                withdrawn=[cid for cid in ids if cid not in r['surviving_change_ids']]
                history=([{'proposal':r['original_proposal'],'states':{
                    cid:'withdrawn' if cid in withdrawn else 'surviving' for cid in ids}}] if repair_ids else [])
                if withdrawn!=r['withdrawn'] or r['assembled_frozen']['disclosure_history']!=history:
                    return 'pending'
                final=r['checks'][-1]['input']
                if repair_ids and r['surviving_change_ids']:
                    expected=_check_input_v3(package,r['final_proposal'],
                        {**r['original_frozen'],'change_ids':r['surviving_change_ids']},r['support'],config,
                        phase='recheck',groups=groups,disclosure_history=history,
                        prompt=final['messages'][0]['content'],prompt_version=final['prompt_version'])
                    if final!=expected:
                        return 'pending'
                elif len(r['checks'])!=1:
                    return 'pending'
                accepted=set(c['change_id'] for u in final['units'] for c in u['changes']
                    if next(v for v in verdicts[-1]['checks'] if v['unit_id']==u['unit_id'])['disposition']=='accept')
                groups=_linked_groups_v3(final,verdicts[-1])
                for group in groups:
                    if not set(group)<=accepted:
                        accepted-=set(group)
                if not r['surviving_change_ids']:
                    accepted=set()
                if accepted!=set(r['accepted_change_ids']):
                    return 'pending'
                final_by_id={c['change_id']:c['change'] for u in final['units'] for c in u['changes']}
                if any(final_by_id[cid]!=change for cid,change in zip(r['assembled_frozen']['change_ids'],r['assembled_proposal']['changes'])):
                    return 'pending'
                if r['repair_used'] and r['surviving_change_ids'] and len(r['checks'])!=2:
                    return 'pending'
            elif r['checks'] or r['accepted_change_ids']:
                return 'pending'
            original_frozen={k:v for k,v in r['original_frozen'].items() if k not in ('change_ids','disclosure_history','audit_old_entries')}
            assembled_original={k:v for k,v in r['assembled_frozen'].items() if k not in ('change_ids','disclosure_history','audit_old_entries')}
            if original_frozen!=assembled_original:
                return 'pending'
            if r['assembled_frozen']['change_ids']!=r['accepted_change_ids']:
                return 'pending'
            package=_assemble(package,r['assembled_proposal'],r['assembled_frozen'],r['response_ref'])
            if identity(package)!=r['candidate_digest'] or identity(package)!=batch['candidate_digest']:
                return 'pending'
            accepted_total+=len(r['accepted_change_ids']);rejected_total+=len(r['rejected_change_ids'])
        if package!=candidate or accepted_total!=cc['accepted'] or rejected_total!=cc['rejected']:
            return 'pending'
        expected_status=('passed_subset' if accepted_total and rejected_total else 'passed' if accepted_total else 'all_rejected' if rejected_total else 'not_applicable')
        if cc['status']!=expected_status or result['status']!=('completed' if accepted_total else 'no-change'):
            return 'pending'
        if not accepted_total and cc['no_change_reason']!=('all_rejected' if rejected_total else 'no_justified_edits'):
            return 'pending'
        return 'usable_candidate' if accepted_total else 'retain_base'
    except (OSError,ValueError,TypeError,KeyError,IndexError):
        return 'pending'


def _parse_repair_v3(raw, proposal, ids, repair_ids, groups):
    from .trajectory_memory_content_check import strict_json
    if raw.get('finish_reason') != 'stop':
        raise ValueError('repair incomplete')
    parsed=strict_json(raw['content'])
    if not isinstance(parsed,dict) or set(parsed)!={'repairs'} or not isinstance(parsed['repairs'],list):
        raise ValueError('repair requires exact repairs list')
    before=dict(zip(ids,proposal['changes']));updates={}
    for item in parsed['repairs']:
        if not isinstance(item,dict) or set(item)!={'change_id','change'}:
            raise ValueError('invalid repair item')
        cid,change=item['change_id'],item['change']
        if not isinstance(cid,str) or cid not in repair_ids or cid in updates:
            raise ValueError('repair unknown/duplicate target')
        if change is not None:
            original=before[cid]
            if not isinstance(change,dict) or {k:v for k,v in change.items() if k not in ('advice','reason')}!={k:v for k,v in original.items() if k not in ('advice','reason')}:
                raise ValueError('repair changed original operation/target/evidence')
        updates[cid]=change
    if set(updates)!=set(repair_ids):
        raise ValueError('repair omitted change')
    for group in groups:
        nulls=[cid for cid in group if cid in updates and updates[cid] is None]
        if nulls and set(nulls)!=set(group):
            raise ValueError('atomic group partially withdrawn')
    return updates

def _check_input_v3(package, proposal, frozen, support, config, *, phase='check',
                 prompt=None, prompt_version=None, groups=None, disclosure_history=None):
    from .trajectory_memory_content_check_legacy import prepare_check, PROMPT as CHECK_PROMPT, PROMPT_VERSION
    check = prepare_check(proposal, frozen['batch_identity'], support,
                          _disclosed_old_bodies(package, frozen), frozen['reference_map'],
                          prompt=prompt or CHECK_PROMPT, prompt_version=prompt_version or PROMPT_VERSION,
                          capacity=None if config.learning_profile == 'workflow-lessons-v3' else config.input_capacity_bytes, change_ids=frozen.get('change_ids'),
                          groups=groups, disclosure_history=disclosure_history)
    check['stage'] = phase
    check['call_identity'] = identity([check['check_input_identity'], _config_data(config), phase, phase == 'recheck'])
    return check



def _proposal_units(package, proposal, frozen):
    """Keep raw positions and relations, then isolate structural problems by group."""
    from .trajectory_memory_content_check import edit_units
    if not isinstance(proposal,dict) or not isinstance(proposal.get('changes'),list):
        raise ValueError('proposal requires changes list')
    if frozen.get('learning_profile') in ('workflow-lessons-v1', 'workflow-lessons-v2', 'workflow-lessons-v3'):
        if set(proposal) != {'changes'}:
            raise ValueError('workflow proposal requires exactly changes; no learning_notes')
        if len(proposal['changes']) > 3:
            raise ValueError('scope_limit_exceeded: more than three proposals')
    ids=['c'+identity([frozen['batch_identity'],i])[:24] for i in range(len(proposal['changes']))]
    relations=deepcopy(proposal)
    for c in relations['changes']:
        if not isinstance(c,dict):continue
        if isinstance(c.get('target_id'),str):
            c['target_id']=frozen['entry_aliases'].get(c['target_id'],c['target_id'])
        if isinstance(c.get('derived_from_ids'),list):
            c['derived_from_ids']=[frozen['entry_aliases'].get(v,v) if isinstance(v,str) else v for v in c['derived_from_ids']]
    groups=[[c['change_id'] for c in u['changes']] for u in edit_units(relations,frozen['batch_identity'],change_ids=ids)]
    issues=[]
    for cid,c in zip(ids,proposal['changes']):
        try:
            _assemble(package,{'changes':[c]}, {**frozen,'audit_old_entries':False,'change_ids':[cid]},'structural-only')
        except (ValueError,TypeError,KeyError) as exc:
            issues.append({'change_id':cid,'stage':'proposal','origin':'structure','message':str(exc)})
    # Conflicting edits to a target are a group issue, even if each is valid alone.
    for group in groups:
        try:
            _assemble(package,{'changes':[c for cid,c in zip(ids,proposal['changes']) if cid in group]},
                      {**frozen,'audit_old_entries':False,'change_ids':group},'structural-only')
        except (ValueError,TypeError,KeyError) as exc:
            for cid in group:
                if not any(i['change_id']==cid for i in issues):
                    issues.append({'change_id':cid,'stage':'proposal','origin':'structure','message':str(exc)})
    bad={i['change_id'] for i in issues}
    valid=[cid for cid in ids if not any(cid in g and set(g)&bad for g in groups)]
    return ids,groups,valid,issues


def _subset(proposal, ids, selected):
    return {'changes':[deepcopy(c) for cid,c in zip(ids,proposal['changes']) if cid in selected]}


def _merge_groups(groups, linked):
    linked=set(linked)
    hits=[g for g in groups if set(g)&linked]
    return [g for g in groups if g not in hits]+[sorted(set().union(linked,*map(set,hits)))] if hits else groups


def _judged_state(check, states, groups, adopted, issues):
    from .trajectory_memory_content_check import parse_verdict
    units={u['unit_id']:u for u in check['input']['units']}
    if check.get('response') is None:
        for u in units.values():
            for c in u['changes']:
                states[c['change_id']]='unresolved'
                issues.append({'change_id':c['change_id'],'stage':check['input']['stage'],
                    'origin':'transport','message':check.get('error','response unavailable')})
        return groups
    verdict=parse_verdict(check['response'],check['input'])
    for issue in verdict['issues']:
        cids=[c['change_id'] for c in units[issue['unit']]['changes']] if issue['unit'] in units else []
        for cid in cids:states[cid]='unresolved'
        issues.append({**issue,'change_ids':cids,'stage':check['input']['stage']})
    for result in verdict['results']:
        unit=units[result['unit']]
        related=[c['change_id'] for uid in [result['unit'],*result['related_units']] for c in units[uid]['changes']]
        groups=_merge_groups(groups,related)
        for c in unit['changes']:
            cid=c['change_id']
            states[cid]={'accept':'accepted','revise':'revise','reject':'rejected','uncertain':'unresolved'}[result['decision']]
            if result['decision']=='accept':
                adopted[cid]={'result':result,'input':check['input'],'response_ref':check['response_ref']}
            else:
                adopted.pop(cid,None)
                issues.append({'change_id':cid,'stage':check['input']['stage'],'origin':'content',
                    'decision':result['decision'],'message':result['reason']})
    return groups


def _repair_input_v4(frozen, first, proposal, ids, repair_ids, groups, config, *, prompt=None, prompt_version=None, input_profile='legacy-v4'):
    if input_profile not in ('legacy-v4', 'affected-only-v1', 'workflow-lessons-v1', 'workflow-lessons-v2', 'workflow-lessons-v3'):
        raise ValueError('unknown repair input profile')
    aliases={c['change_id']:c['short_id'] for u in first['input']['units'] for c in u['changes']}
    payload={'response_format': 'Return one JSON object with a repairs list.', 'edits':[{'change':aliases[cid],**deepcopy(c)} for i,(cid,c) in enumerate(zip(ids,proposal['changes'])) if cid in repair_ids],
        'atomic_groups':[[aliases[cid] for cid in g] for g in groups if set(g)&set(repair_ids)],
        'original_check':json.loads(first['input']['messages'][1]['content']),
        'findings':first.get('verdict')}
    binding = {'repair_ids': repair_ids}
    if input_profile in ('affected-only-v1', 'workflow-lessons-v1', 'workflow-lessons-v2', 'workflow-lessons-v3'):
        selected = set(repair_ids)
        if any(set(group) & selected and not set(group) <= selected for group in groups):
            raise ValueError('repair scope lacks complete atomic group')
        selected_units = [u for u in first['input']['units']
                          if any(c['change_id'] in selected for c in u['changes'])]
        if any(not {c['change_id'] for c in u['changes']} <= selected for u in selected_units):
            raise ValueError('repair scope lacks complete unit')
        unit_ids = {u['unit_id'] for u in selected_units}
        payload['original_check'] = deepcopy(payload['original_check'])
        payload['original_check']['units'] = [u for u in payload['original_check']['units'] if u['unit'] in unit_ids]
        findings = deepcopy(payload['findings'])
        findings['results'] = [r for r in findings['results'] if r['unit'] in unit_ids]
        if any(not set(r.get('related_units', [])) <= unit_ids for r in findings['results']):
            raise ValueError('repair finding refers outside selected scope')
        findings['issues'] = [i for i in findings['issues'] if i.get('unit') is None or i['unit'] in unit_ids]
        if 'source_normalizations' in findings:
            findings['source_normalizations'] = [n for n in findings['source_normalizations'] if n['unit'] in unit_ids]
        payload['findings'] = findings
        binding['repair_input_profile'] = input_profile
    if input_profile in ('workflow-lessons-v1', 'workflow-lessons-v2', 'workflow-lessons-v3'):
        if not _workflow(config) or input_profile != config.learning_profile:
            raise ValueError('workflow repair requires matching learning profile')
        spec = frozen['workflow_prompts']['repair']
        if prompt is not None and prompt != spec['text']:
            raise ValueError('workflow repair prompt override forbidden')
        prompt, prompt_version = spec['text'], spec['version']
        changes_by_unit = {u['unit_id']: [c['short_id'] for c in u['changes']] for u in selected_units}
        payload = {k: payload[k] for k in ('edits', 'atomic_groups')}
        payload['findings'] = [{k: deepcopy(v) for k, v in finding.items()
                               if k in {'unit', 'decision', 'reason', 'sources', 'related_units'}} |
                              {'changes': changes_by_unit[finding['unit']]}
                              for finding in findings['results']]
        payload['sources'] = deepcopy(frozen['batch']['model_view']['sources'])
        payload['notice'] = frozen['batch']['model_view']['notice'] + ' Only the listed complete atomic groups are editable.'
        if {c for f in payload['findings'] for c in f['changes']} != {e['change'] for e in payload['edits']}:
            raise ValueError('repair findings lack complete change mapping')
    phase = _phase_input(frozen,[{'role':'system','content':prompt or REPAIR_PROMPT},
        {'role':'user','content':canonical_json(payload)}],'repair',config,**binding)
    if input_profile != 'legacy-v4':
        phase['repair_input_profile'] = input_profile
    if prompt_version is not None:
        phase['prompt_version'] = prompt_version
    return phase


def _repair_updates(raw, proposal, ids, repair_ids, groups, aliases=None, *, strict_top=False):
    from .trajectory_memory_content_check import json_object
    updates,issues={},[]
    try:
        if raw is None or raw.get('finish_reason')!='stop':raise ValueError('repair response unavailable/incomplete')
        value=json_object(raw.get('content'))
        if strict_top and (not isinstance(value,dict) or set(value) != {'repairs'}):
            raise ValueError('workflow repair requires exactly repairs')
        if not isinstance(value,dict) or not isinstance(value.get('repairs'),list):raise ValueError('repair requires repairs list')
        items=value['repairs']
    except (ValueError,TypeError) as exc:
        return {},[{'change_id':cid,'stage':'repair','origin':'structure','message':str(exc)} for cid in repair_ids]
    aliases=aliases or {'c'+str(i+1):cid for i,cid in enumerate(ids) if cid in repair_ids}
    seen=set();bad=set()
    for item in items:
        alias=item.get('change') if isinstance(item,dict) else None
        if not isinstance(alias,str) or alias not in aliases:
            issues.append({'change_id':None,'stage':'repair','origin':'structure','message':'unknown repair change'})
            continue
        cid=aliases[alias]
        try:
            if cid in seen:raise ValueError('duplicate repair change')
            seen.add(cid)
            if item.get('withdraw') is True:
                if set(item)!={'change','withdraw'}:raise ValueError('withdrawal has conflicting fields')
                updates[cid]=None
            else:
                original=proposal['changes'][ids.index(cid)]
                expected={'change','reason'} | ({'advice'} if original['op']!='delete' else set())
                if set(item)!=expected:raise ValueError('repair only changes advice/reason')
                nonblank(item['reason'],'repair reason')
                updates[cid]={**deepcopy(original),**{k:deepcopy(v) for k,v in item.items() if k!='change'}}
        except (ValueError,TypeError,KeyError) as exc:
            bad.add(cid);issues.append({'change_id':cid,'stage':'repair','origin':'structure','message':str(exc)})
    for cid in set(repair_ids)-seen:
        bad.add(cid);issues.append({'change_id':cid,'stage':'repair','origin':'structure','message':'missing repair change'})
    for group in groups:
        if not set(group)&set(repair_ids):continue
        nulls={cid for cid in group if cid in updates and updates[cid] is None}
        if nulls and nulls!=set(group):
            bad.update(group)
            issues.extend({'change_id':cid,'stage':'repair','origin':'structure','message':'partial atomic withdrawal'} for cid in group)
        if set(group)&bad:bad.update(group)
    return {cid:c for cid,c in updates.items() if cid not in bad},issues


def _phase_capacity_event(input_value, input_path, affected, groups, limit):
    """A host gate, never a model verdict or provider attempt."""
    messages_bytes = size(input_value['messages'])
    if messages_bytes <= limit:
        raise ValueError('capacity event requires an actually oversized request')
    selected = set(affected)
    atomic = [list(group) for group in groups if set(group) & selected]
    if any(not set(group) <= selected for group in atomic):
        raise ValueError('capacity scope must contain complete atomic groups')
    return {'kind': 'phase_capacity_blocked', 'phase': input_value['stage'],
            'call_identity': input_value['call_identity'],
            'blocked_input_identity': identity(input_value),
            'messages_sha256': identity(input_value['messages']),
            'request_bytes': messages_bytes, 'messages_utf8_bytes': messages_bytes,
            'limit': limit, 'affected_change_ids': list(affected), 'atomic_groups': atomic,
            'input_path': str(input_path), 'input_file_sha256': identity(input_value),
            'reason': 'input_over_budget', 'dispatched': False}


def _verify_phase_capacity(phase, expected_input, affected, groups, config):
    if config.learning_profile != 'workflow-lessons-v3':
        raise ValueError('capacity event requires v3')
    event = phase['capacity_event']
    if phase['input'] != expected_input or phase.get('response') is not None or phase.get('response_ref') is not None:
        raise ValueError('capacity result differs from reconstructed input')
    path = Path(event['input_path'])
    if path.name != expected_input['stage'] + '-input.json' or read_json(path) != expected_input:
        raise ValueError('blocked input file differs')
    if _file_sha(path) != event['input_file_sha256']:
        raise ValueError('blocked input file hash differs')
    if event != _phase_capacity_event(expected_input, path, affected, groups, config.input_capacity_bytes):
        raise ValueError('blocked input metadata differs')


def _batch_decisions(package, proposal, frozen, support, config, checks, repair, *, planning=False):
    """Pure replay of actual response decisions; used identically by writer and consumer."""
    from .trajectory_memory_content_check import parse_verdict
    ids,groups,valid,issues=_proposal_units(package,proposal,frozen)
    states={cid:'unresolved' for cid in ids};adopted={};current=deepcopy(proposal)
    repair_ids=[];changed=[]
    def finish():
        for cid in ids:
            if states[cid]=='revise':states[cid]='rejected'
        for group in groups:
            if any(states[cid]!='accepted' for cid in group):
                for cid in group:
                    if states[cid]=='accepted':
                        states[cid]='unresolved';issues.append({'change_id':cid,'stage':'assembly','origin':'dependency','message':'related edit not accepted'})
        return {'ids':ids,'groups':groups,'valid':valid,'states':states,'issues':issues,
                'proposal':current,'adopted':adopted,'repair_ids':repair_ids,'recheck_ids':changed}

    def verify_check(checked, selected, phase, current_groups):
        from .trajectory_memory_content_check import edit_units
        value=checked['input']
        expected_units=edit_units(_subset(current,ids,selected),frozen['batch_identity'],change_ids=selected,groups=current_groups,change_order=frozen.get('change_order'))
        expected_generation=identity([package,{**frozen,'change_ids':selected}])
        if checked.get('reuse_origin'):
            expected=_check_input(package,_subset(current,ids,selected),{**frozen,'change_ids':selected},support,config,phase=phase,groups=current_groups,
                source_reference_policy=value.get('source_reference_policy','canonical-only-v1'))
            reused=_reuse_checked_response(checked['reuse_origin'],expected,package,proposal,frozen,support,config)
            if reused!=checked:raise ValueError('saved check reuse changed')
            return
        if (value['units']!=expected_units or value['stage']!=phase
                or value['generation_input_identity']!=expected_generation
                or value['support_identity']!=identity([support,_disclosed_old_bodies(package,frozen)])
                or value['call_identity']!=identity([value['check_input_identity'],_config_data(config),phase,phase=='recheck',expected_generation])):
            raise ValueError('saved check differs from exact final edits/sources')
        # parse_verdict binds the saved table/projection. Replay keeps its original source ordering,
        # even when a later formatter changes display order or adds event-position annotations.
    if valid:
        if not checks:raise ValueError('missing saved check')
        if checks[0].get('capacity_event'):
            expected = _check_input(package, _subset(current, ids, valid), {**frozen, 'change_ids': valid},
                                    support, config, groups=groups)
            _verify_phase_capacity(checks[0], expected, valid, groups, config)
            issues.append({'stage':'check','origin':'capacity','message':'input_over_budget','change_ids':valid})
            return finish()
        verify_check(checks[0],valid,'check',groups)
        groups=_judged_state(checks[0],states,groups,adopted,issues)
    repair_ids=[cid for cid in ids if any(cid in g and any(states[x]=='revise' for x in g)
        and all(states[x] in ('accepted','revise') for x in g) for g in groups)]
    changed=[]
    if repair_ids:
        if repair is None:
            if planning:return {'ids':ids,'groups':groups,'valid':valid,'states':states,'issues':issues,'proposal':current,'adopted':adopted,'repair_ids':repair_ids,'recheck_ids':[]}
            raise ValueError('missing repair attempt record')
        first={**checks[0],'verdict':parse_verdict(checks[0]['response'],checks[0]['input'])}
        if repair['input']!=_repair_input_v4(frozen,first,proposal,ids,repair_ids,groups,config,prompt=repair['input']['messages'][0]['content'], prompt_version=repair['input'].get('prompt_version'),
                input_profile=repair['input'].get('repair_input_profile','legacy-v4')):
            raise ValueError('repair input differs from frozen groups/material')
        if repair.get('capacity_event'):
            expected = _repair_input_v4(frozen, first, proposal, ids, repair_ids, groups, config,
                                        input_profile=config.learning_profile)
            _verify_phase_capacity(repair, expected, repair_ids, groups, config)
            for cid in repair_ids:
                states[cid]='unresolved';adopted.pop(cid,None)
            issues.append({'stage':'repair','origin':'capacity','message':'input_over_budget','change_ids':repair_ids})
            return finish()
        updates,repair_issues=_repair_updates(repair.get('response'),proposal,ids,repair_ids,groups,
            {c['short_id']:c['change_id'] for u in checks[0]['input']['units'] for c in u['changes'] if c['change_id'] in repair_ids}, strict_top=_workflow(config))
        issues.extend(repair_issues)
        for cid in repair_ids:
            if cid not in updates:states[cid]='unresolved';adopted.pop(cid,None)
            elif updates[cid] is None:states[cid]='withdrawn';adopted.pop(cid,None)
            elif updates[cid]!=proposal['changes'][ids.index(cid)]:
                current['changes'][ids.index(cid)]=updates[cid];changed.append(cid)
        changed=[cid for cid in ids if any(cid in g and set(g)&set(changed) for g in groups)
                 and states[cid] not in ('withdrawn','unresolved')]
        # Validate repaired groups before checking. Invalid changes cannot remove old entries.
        _,_,legal,new_issues=_proposal_units(package,current,frozen)
        for cid in changed:
            if cid not in legal:states[cid]='unresolved';adopted.pop(cid,None)
        issues.extend(new_issues)
        changed=[cid for cid in changed if cid in legal]
        if changed:
            if len(checks)!=2:
                if planning:return {'ids':ids,'groups':groups,'valid':valid,'states':states,'issues':issues,'proposal':current,'adopted':adopted,'repair_ids':repair_ids,'recheck_ids':changed}
                raise ValueError('changed edits lack final recheck')
            if checks[1].get('capacity_event'):
                expected = _check_input(package, _subset(current, ids, changed), {**frozen, 'change_ids':changed},
                                        support, config, phase='recheck', groups=groups)
                _verify_phase_capacity(checks[1], expected, changed, groups, config)
                for cid in changed:
                    states[cid]='unresolved';adopted.pop(cid,None)
                issues.append({'stage':'recheck','origin':'capacity','message':'input_over_budget','change_ids':changed})
            else:
                verify_check(checks[1],changed,'recheck',groups)
                groups=_judged_state(checks[1],states,groups,adopted,issues)
        elif len(checks)!=1:raise ValueError('unexpected recheck of unchanged edits')
    elif repair is not None or len(checks)!=(1 if valid else 0):raise ValueError('unexpected repair/check')
    return finish()


def _capture_phase(directory, input_value, provider, config, can_call, *, retry_failed=False, recovery_requests=None, capacity_ids=(), capacity_groups=()):
    """Immutable requests plus durable failures. Model parsing happens after accounting."""
    name=input_value['stage']
    path=directory/(name+'-input.json')
    if path.exists():
        saved=read_json(path)
        if saved!=input_value:
            if name not in ('check','recheck') or saved.get('generation_input_identity')!=input_value.get('generation_input_identity') or saved.get('support_identity')!=input_value.get('support_identity') or saved.get('units')!=input_value.get('units'):
                raise ValueError('phase input changed; use new revision')
            input_value=saved
    _write(path,input_value)
    if config.learning_profile == 'workflow-lessons-v3' and size(input_value['messages']) > config.input_capacity_bytes:
        event = _phase_capacity_event(input_value, path, capacity_ids, capacity_groups, config.input_capacity_bytes)
        _write(directory/(name+'-capacity-blocked.json'), event)
        return {'input':input_value,'response':None,'response_ref':None,'capacity_event':event}
    try:
        raw,ref=recorded_response(directory,input_value,provider,config,can_call=can_call,retry_failed=retry_failed,
            recovery_request=(recovery_requests or {}).get(input_value["call_identity"]))
        if name in ('check', 'recheck') and not _workflow(config):
            _save_learning_notes(directory, input_value, raw, ref)
        return {'input':input_value,'response':raw,'response_ref':ref}
    except (ValueError,OSError,TypeError) as exc:
        from .host_control import raise_control_error
        from .hosted_stream import deadline_error
        raise_control_error(exc)
        if (deadline := deadline_error(exc)) is not None:
            raise deadline
        return {'input':input_value,'response':None,'response_ref':None,'error':str(exc)}


def _adopted_assembly(package, frozen, decision, excluded=()):
    accepted=[cid for cid in decision['ids'] if decision['states'][cid]=='accepted' and cid not in excluded]
    selected=_subset(decision['proposal'],decision['ids'],accepted)
    reference_map=deepcopy(frozen['reference_map']);response_refs={}
    for cid,change in zip(accepted,selected['changes']):
        adopted=decision['adopted'][cid]
        checked=adopted['input']
        by_source={d['source']:d for d in checked['documents']}
        # Scope aliases by immutable request, so a future request cannot reuse their meaning.
        refs=[]
        for source in adopted['result']['sources']:
            ref=checked['check_input_identity']+'/'+source
            reference_map[ref]={'check_input_identity':checked['check_input_identity'],
                'source':source,'document':deepcopy(by_source[source])}
            refs.append(ref)
        change['evidence_refs']=refs
        response_refs[cid]=adopted['response_ref']
    assembled_frozen={**frozen,'audit_old_entries':False,'change_ids':accepted,
        'reference_map':reference_map,'disclosure_history':[{'original_proposal':decision['proposal'],
            'notice':'Draft and reason are model-generated history, not verified external observations.'}]}
    candidate=_assemble(package,selected,assembled_frozen,response_refs)
    return candidate,selected,assembled_frozen,response_refs,accepted


def _stage_exclusions(candidate, frozen, decision, stage):
    if stage['status']=='passed':return []
    ids={}
    for cid,c in zip(decision['ids'],decision['proposal']['changes']):
        if not isinstance(c,dict) or decision['states'][cid]!='accepted':continue
        target=frozen['entry_aliases'].get(c.get('target_id'),c.get('target_id'))
        key=target if c.get('op')=='revise' else 'generated-'+identity([frozen['batch_identity'],cid])[:32]
        ids[key]=cid
    bad=set()
    for issue in stage['issues']:
        cid=ids.get(issue['entry_identity'])
        if cid is None or decision['states'][cid]!='accepted':raise ValueError('stage failure affects base or unknown entry')
        bad.add(cid)
    if not bad:raise ValueError('unattributed stage failure')
    return [cid for cid in decision['ids'] if any(cid in g and set(g)&bad for g in decision['groups'])]


def _complete_checked_batch(directory, package, proposal, frozen, support, proposal_ref,
                            provider, config, registry, context, *, can_call=None, retry_failed=False, check_origin=None,
                            source_reference_policy='canonical-only-v1', repair_input_profile='legacy-v4', recovery_requests=None):
    from .trajectory_memory_content_check import parse_verdict
    if _workflow(config):
        repair_input_profile = config.learning_profile
    ids,groups,valid,_=_proposal_units(package,proposal,frozen)
    if _workflow(config):
        frozen = {**frozen, 'change_order': ids}
    checks=[];repair=None
    if valid:
        first=_check_input(package,_subset(proposal,ids,valid),{**frozen,'change_ids':valid},support,config,groups=groups,source_reference_policy=source_reference_policy)
        if check_origin:
            checked=_reuse_checked_response(check_origin,first,package,proposal,frozen,support,config)
            _write(directory/'check-input.json',checked['input'])
            checks.append(checked)
        else:
            checks.append(_capture_phase(directory,first,provider,config,can_call,retry_failed=retry_failed,recovery_requests=recovery_requests,capacity_ids=valid,capacity_groups=groups))
    decision=_batch_decisions(package,proposal,frozen,support,config,checks,repair,planning=True)
    if decision['repair_ids']:
        first={**checks[0],'verdict':parse_verdict(checks[0]['response'],checks[0]['input'])}
        repair_input=_repair_input_v4(frozen,first,proposal,ids,decision['repair_ids'],decision['groups'],config,input_profile=repair_input_profile)
        repair=_capture_phase(directory,repair_input,provider,config,can_call,retry_failed=retry_failed,recovery_requests=recovery_requests,capacity_ids=decision['repair_ids'],capacity_groups=decision['groups'])
        decision=_batch_decisions(package,proposal,frozen,support,config,checks,repair,planning=True)
        changed=decision['recheck_ids']
        if changed:
            last=_check_input(package,_subset(decision['proposal'],ids,changed),{**frozen,'change_ids':changed},
                              support,config,phase='recheck',groups=decision['groups'],source_reference_policy=source_reference_policy)
            checks.append(_capture_phase(directory,last,provider,config,can_call,retry_failed=retry_failed,recovery_requests=recovery_requests,capacity_ids=changed,capacity_groups=decision['groups']))
    decision=_batch_decisions(package,proposal,frozen,support,config,checks,repair)
    candidate,selected,assembled,refs,accepted=_adopted_assembly(package,frozen,decision)
    _write(directory/'pre-stage-candidate.json',candidate)
    initial_stage=to_primitive(registry.stage(directory/'pre-stage-candidate.json',context))
    excluded=_stage_exclusions(candidate,frozen,decision,initial_stage)
    if excluded:
        candidate,selected,assembled,refs,accepted=_adopted_assembly(package,frozen,decision,excluded)
    _write(directory/'candidate.json',candidate)
    stage=to_primitive(registry.stage(directory/'candidate.json',context))
    if stage['status']!='passed':raise ValueError('retained candidate failed structural stage')
    states={**decision['states'],**{cid:'unresolved' for cid in excluded}}
    issues=decision['issues']+[{'change_id':cid,'stage':'stage','origin':'structure',
                              'message':'Registry rejected this atomic group'} for cid in excluded]
    report={'schema':'generated-edits-report-v4','base_package':package,'original_proposal':proposal,
        'original_frozen':frozen,'support':support,'proposal_response_ref':proposal_ref,
        'checks':checks,'repair':repair,'initial_change_ids':ids,'groups':decision['groups'],
        'final_proposal':decision['proposal'],'states':states,'issues':issues,'accepted_change_ids':accepted,
        'assembled_proposal':selected,'assembled_frozen':assembled,'response_refs':refs,
        'stage_excluded':excluded,'initial_stage':initial_stage,'leakage_context':to_primitive(context),
        'candidate_digest':identity(candidate)}
    _write(directory/'content-report.json',report)
    result={'status':'completed_with_issues' if issues else 'completed','candidate_path':str(directory/'candidate.json'),
        'candidate_digest':identity(candidate),'stage':stage,'report':report,'changes':len(accepted),
        'learning_diagnostics': {'generation':str(directory/'generation-learning-notes.json'),
            'checker_omissions': 'requested; inspect private notes and issues' if checks else 'checker did not check omissions'},
        'content_check':{'accepted':len(accepted),'rejected':sum(v in ('rejected','withdrawn') for v in states.values()),
                         'unresolved':sum(v=='unresolved' for v in states.values())}}
    _write(directory/'result.json',result)
    return result


def validate_generated_candidate(result, candidate=None, report=None):
    """Verify adopted edits through saved responses; unrelated issues do not veto them."""
    if not isinstance(result,dict):return 'pending'
    if result.get('content_check',{}).get('schema')=='generated-candidate-v3':
        return _validate_generated_candidate_v3(result,candidate,report)
    try:
        from .trajectory_memory_registry import _lint, _Package
        def lint_package(value, context):
            return _lint(_Package(schema_version=value["schema_version"], digest=identity(value),
                canonical_bytes=canonical_json(value).encode(),entries=tuple(value["entries"])), context)
        cc=result['content_check']
        if result['status'] not in ('completed','completed_with_issues') or cc['schema']!='generated-candidate-v4':return 'pending'
        candidate=read_json(result['candidate_path']) if candidate is None else candidate
        report=read_json(cc['report_path']) if report is None else report
        manifest=report['manifest']
        if (report['schema']!='generated-candidate-report-v4' or manifest['schema_version']!='trajectory-memory-generation-v4'
                or identity(report)!=cc['report_identity'] or identity(manifest)!=report['manifest_identity']
                or identity(candidate)!=result['candidate_digest'] or report['candidate_digest']!=identity(candidate)
                or identity(report['start_package'])!=manifest['start_digest']):return 'pending'
        origin=manifest.get('proposal_origin')
        if origin:
            original=read_json(origin['response_ref'])
            if origin.get('original_propose_ref') and identity(read_json(origin['original_propose_ref']))!=origin['original_propose_identity']:
                raise ValueError('original propose predecessor changed')
            if (identity(original)!=origin['attempt_identity'] or original['request_digest']!=origin['request_digest']
                    or identity(json.loads(original['response']['content']))!=origin['proposal_identity']
                    or identity(manifest['proposal_import'])!=origin['proposal_identity']):raise ValueError('original proposal changed')
        missing=report['total_batches']-len(report['batch_records'])
        if (result.get('unprocessed_batches',missing)!=missing or missing<0
                or result.get('total_batches',report['total_batches'])!=report['total_batches']):return 'pending'
        record_root=Path(cc['report_path']).parent
        if (read_json(record_root/'run.json')!=manifest or read_json(record_root/'start.json')!=report['start_package']
                or AdvisoryMemory.from_file(result['candidate_path']).package_digest!=identity(candidate)):return 'pending'
        config=GenerationConfig(**manifest['config']);package=report['start_package'];counts={'accepted':0,'rejected':0,'unresolved':0}
        for batch in report['batches']:
            r=batch['report']
            if r['schema']!='generated-edits-report-v4' or r['base_package']!=package:raise ValueError('batch base mismatch')
            if _workflow(config):
                from .trajectory_memory_workflow_view import build_learning_view
                frozen = r['original_frozen']
                normalized_runs = read_json(record_root/'normalized-runs.json')
                if len(normalized_runs) != 1 or build_learning_view(normalized_runs[0], profile=config.learning_profile) != frozen['batch']:
                    raise ValueError('workflow source projection changed')
                if identity(frozen['batch']) != frozen['workflow_view_identity']:
                    raise ValueError('workflow projection binding changed')
                propose_input = read_json(Path(r['proposal_response_ref']).parent/'propose-input.json')
                original = _verify_workflow_wire(r['proposal_response_ref'], propose_input, config)
                from .trajectory_memory_content_check import json_object
                if json_object(original['response']['content']) != r['original_proposal']:
                    raise ValueError('original workflow proposal changed')
                if identity(propose_input['messages'][0]['content']) != manifest['prompt_digest']:
                    raise ValueError('workflow generation prompt changed')

            for phase in [*r['checks'],*([r['repair']] if r['repair'] else [])]:
                if phase.get('response') is None:continue
                attempt=read_json(phase['response_ref'])
                if _workflow(config):
                    _verify_workflow_wire(phase['response_ref'], phase['input'], config)
                if (attempt['status']!='received' or attempt['response']!=phase['response']
                        or attempt['call_identity']!=phase['input']['call_identity']
                        or attempt['messages']!=phase['input']['messages']
                        or attempt['request_digest']!=identity(attempt['messages'])):raise ValueError('saved response changed')
                digest=manifest['repair_prompt_digest'] if phase is r['repair'] else manifest['check_prompt_digest']
                if identity(phase['input']['messages'][0]['content'])!=digest:raise ValueError('prompt changed')
            decision=_batch_decisions(package,r['original_proposal'],r['original_frozen'],r['support'],config,r['checks'],r['repair'])
            initial,*_=_adopted_assembly(package,r['original_frozen'],decision)
            lint=to_primitive(lint_package(initial,LeakageContext(**r['leakage_context'])))
            if lint!=r['initial_stage']['issues']:raise ValueError('stage issues changed')
            excluded=_stage_exclusions(initial,r['original_frozen'],decision,r['initial_stage'])
            expected,selected,assembled,refs,accepted=_adopted_assembly(package,r['original_frozen'],decision,excluded)
            if (excluded!=r['stage_excluded'] or selected!=r['assembled_proposal'] or assembled!=r['assembled_frozen']
                    or refs!=r['response_refs'] or accepted!=r['accepted_change_ids']
                    or decision['groups']!=r['groups'] or decision['proposal']!=r['final_proposal']
                    or identity(expected)!=r['candidate_digest'] or identity(expected)!=batch['candidate_digest']):raise ValueError('adopted edits changed')
            states={**decision['states'],**{cid:'unresolved' for cid in excluded}}
            if states!=r['states']:raise ValueError('unit states changed')
            if batch['stage']['status']!='passed' or batch['stage']['record']['package_digest']!=identity(expected):raise ValueError('final stage mismatch')
            if lint_package(expected,LeakageContext(**r['leakage_context'])):raise ValueError('candidate lint changed')
            counts['accepted']+=len(accepted)
            counts['rejected']+=sum(s in ('rejected','withdrawn') for s in states.values())
            counts['unresolved']+=sum(s=='unresolved' for s in states.values())
            package=expected
        if package!=candidate or any(cc[k]!=v for k,v in counts.items()):return 'pending'
        return 'usable_candidate' if counts['accepted'] else 'retain_base'
    except (OSError,ValueError,TypeError,KeyError,IndexError,AttributeError):
        return 'pending'


def _reuse_checked_response(origin, expected, package, proposal, frozen, support, config):
    """Reuse an identical early check, including its original host evidence and call identity."""
    original=read_json(origin['input_path']);experiment=read_json(origin['experiment_path'])
    attempt=read_json(origin['response_ref'])
    if (experiment['prepared']!={'package':package,'frozen':frozen,'support':support}
            or experiment['proposal']!=proposal or identity(experiment['config'])!=identity(_config_data(config))
            or original['messages']!=expected['messages'] or original['units']!=expected['units']
            or original['documents']!=expected['documents'] or original['support_identity']!=expected['support_identity']
            or attempt['messages']!=original['messages'] or attempt['status']!='received'
            or attempt['call_identity']!=original['call_identity']
            or attempt['request_digest']!=identity(attempt['messages'])):
        raise ValueError('saved check reuse differs from actual input, source mapping or configuration')
    return {'input':original,'response':attempt['response'],'response_ref':origin['response_ref'],'reuse_origin':origin}
