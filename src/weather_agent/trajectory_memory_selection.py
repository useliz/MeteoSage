"""One bounded selection of existing advice; no scientific execution or run state."""
from __future__ import annotations

import json
import math
import time
from uuid import uuid4
from contextvars import ContextVar
from pathlib import Path

from ._serialization import canonical_json, identity, to_primitive

from .trajectory_memory_prompts import effective_prompt

PROMPT_VERSION = 'memory-selection-rules-v2'
PROMPT = effective_prompt('selection')
selection_http_scope = ContextVar('memory_selection_http_scope', default=None)


def parse_selection(content, candidate_ids):
    """Only recommendations are authoritative; optional annotations cannot veto them."""
    value = json.loads(content)
    if not isinstance(value, dict):
        raise ValueError('selector requires an order object')
    fields = [key for key in ('recommended_order', 'required_order') if key in value]
    if len(fields) != 1:
        raise ValueError('selector requires one unambiguous order field')
    order = value[fields[0]]
    if (not isinstance(order, list)
            or any(not isinstance(key, str) or key not in candidate_ids for key in order)):
        raise ValueError('invalid selector recommended_order')
    diagnostics = ['required_order-alias-normalized'] if fields[0] == 'required_order' else []
    duplicates = [{'position': i, 'id': key} for i, key in enumerate(order) if key in order[:i]]
    order = list(dict.fromkeys(order))
    if duplicates:
        diagnostics.append('duplicate-legal-ids-removed')
    if set(value) - {'recommended_order', 'required_order', 'explanations'}:
        diagnostics.append('unknown-top-level-fields-ignored')
    explanations = value.get('explanations', {})
    notes = {}
    if not isinstance(explanations, dict):
        diagnostics.append('invalid-explanations-ignored')
    else:
        for key, explanation in explanations.items():
            try:
                valid = (key in order and isinstance(explanation, str)
                         and len(explanation.encode('utf-8')) <= 4096)
            except UnicodeEncodeError:
                valid = False
            if valid:
                notes[key] = explanation
            else:
                diagnostics.append('invalid-or-oversize-explanation-ignored')
    return {'explanations': notes, 'diagnostics': diagnostics, 'removed_duplicates': duplicates, 'raw_order': value[fields[0]]}, order


class ContextualSelection:
    """Host injects a transport and durable audit sink. No implicit retries/cache."""
    def __init__(self, transport, audit, *, model='deepseek-v4.1-flash-expires-on-0910',
                 provider='deepseek', timeout=1800, max_tokens=131072,
                 prompt=PROMPT, prompt_version=PROMPT_VERSION, thinking='on', reasoning_effort='high', candidates=None, candidate_endpoint=None, input_capacity_bytes=1048576, selection_context_profile="legacy"):
        if not callable(transport) or not callable(audit):
            raise ValueError('selection requires transport and durable audit')
        if (isinstance(timeout, bool) or not isinstance(timeout, (int,float))
                or not math.isfinite(timeout) or not 0 < timeout <= 1800
                or isinstance(max_tokens, bool) or not isinstance(max_tokens, int)
                or not 0 < max_tokens <= 393216):
            raise ValueError('selection requires positive timeout <=1800s and integer max_tokens <=393216')
        if not ((thinking == 'off' and reasoning_effort is None) or
                (thinking == 'on' and reasoning_effort in ('low', 'high', 'max'))):
            raise ValueError('selection thinking requires off/None or on/low or on/high')
        if isinstance(input_capacity_bytes, bool) or not isinstance(input_capacity_bytes, int) or input_capacity_bytes <= 0:
            raise ValueError('input_capacity_bytes must be a positive integer')
        self.input_capacity_bytes = {"workflow-lessons-v1":120000,"workflow-lessons-v2":180000,"workflow-lessons-v3":200000}.get(selection_context_profile,input_capacity_bytes)
        self.transport, self.audit = transport, audit
        self.candidates = candidates
        self.candidate_endpoint = candidate_endpoint
        self.deadline = None
        self.timeout, self.max_tokens, self.prompt = timeout, max_tokens, prompt
        self.metadata = {'backend':'contextual', 'provider':provider, 'model':model,
            'thinking':thinking, 'reasoning_effort':reasoning_effort,
            'max_tokens':max_tokens, 'timeout_seconds':timeout, 'input_capacity_bytes':self.input_capacity_bytes,
            'output_contract_version':'memory-selection-order-alias-v1', 'prompt_version':prompt_version, 'prompt_identity':identity(prompt)}
        if selection_context_profile not in {'legacy', 'workflow-lessons-v1', 'workflow-lessons-v2', 'workflow-lessons-v3'}:
            raise ValueError('unknown selection context profile')
        if selection_context_profile != 'legacy':
            self.metadata['selection_context_profile'] = selection_context_profile
        if candidates is not None:
            self.metadata['candidates'] = candidates.metadata
        if candidate_endpoint is not None:
            self.metadata['candidate_endpoint'] = str(candidate_endpoint)

    def remaining_timeout(self, limit):
        remaining = limit if self.deadline is None else min(limit,self.deadline-time.monotonic())
        if remaining <= 0:
            raise OSError('memory-run-budget-exhausted')
        return remaining

    def _request(self, payload):
        request = {'model':self.metadata['model'], 'messages':[
            {'role':'system', 'content':self.prompt},
            {'role':'user', 'content':canonical_json(payload)}],
            'max_tokens':self.max_tokens, 'timeout':self.timeout,
            'response_format':{'type':'json_object'}, 'extra_body':{'thinking':{'type':'enabled' if self.metadata['thinking']=='on' else 'disabled'}}}
        if self.metadata['reasoning_effort'] is not None:
            request['reasoning_effort'] = self.metadata['reasoning_effort']
        return request

    def select(self, query, advice, *, retrieval_context=None):
        context = retrieval_context or {}
        legal_count = len(advice)
        candidate_metadata = {'small_package':True, 'encoded_entries':0}
        self.remaining_timeout(self.timeout)
        if legal_count > 32:
            if self.candidates is None and self.candidate_endpoint is not None:
                from .trajectory_memory_retrieval import CandidateRPC
                try:
                    self.candidates = CandidateRPC(self.candidate_endpoint)
                except (OSError,ValueError) as error:
                    from .host_control import raise_control_error
                    raise_control_error(error)
                    raise OSError('candidate-resource-unavailable') from error
            if self.candidates is None:
                raise OSError('candidate-resource-unavailable')
            advice, candidate_metadata = self.candidates.candidates(query,advice,retrieval_context=context,timeout=self.remaining_timeout(30.0))
            candidate_metadata = dict(candidate_metadata,small_package=False)
        # Logical identity order keeps editorial children adjacent; independent of request/model.
        advice = sorted(advice, key=lambda a:a.identity)
        payload = {'query':query, 'public_working_context':context, 'candidates':[]}
        included = {}
        for a in advice:
            key = 'm' + str(len(included))
            included[key] = a
            payload['candidates'].append({'id':key, 'guidance':a.guidance,
                'applicability':list(a.applicability), 'negative_conditions':list(a.negative_conditions)})
        request = self._request(payload)
        request["timeout"] = self.remaining_timeout(self.timeout)
        request_bytes = len(canonical_json(request["messages"] if self.metadata.get("selection_context_profile") in ("workflow-lessons-v1", "workflow-lessons-v2", "workflow-lessons-v3") else request).encode("utf-8"))
        if request_bytes > self.input_capacity_bytes or not included:
            self.audit('memory_selection_capacity', {'status':'input-budget-unavailable',
                'legal_count':legal_count, 'candidate_count':len(included),
                'request_bytes':request_bytes, 'input_capacity_bytes':self.input_capacity_bytes})
            raise OSError('input-budget-unavailable')
        request_id = identity(request)
        attempt = {'attempt_id':uuid4().hex, 'phase':'memory_selection',
            'request_identity':request_id, 'request':request, 'selection':self.metadata,
            'bindings':{key:{'identity':a.identity,'version':a.version} for key,a in included.items()},
            'legal_count':legal_count, 'checked_count':len(included),
            'candidate_omitted_count':legal_count-len(advice), 'byte_omitted_count':len(advice)-len(included)}
        # A failed write prevents the API call; response and usage precede parsing.
        self.audit('memory_selection_attempt', attempt | {'status':'started'})
        started = time.monotonic()
        scope_token = selection_http_scope.set({'phase': 'memory_selection',
            'selector_attempt_id': attempt['attempt_id'], 'request_identity': request_id})
        try:
            try:
                response = self.transport(request)
            finally:
                selection_http_scope.reset(scope_token)
        except Exception as error:
            from .host_control import raise_control_error
            from .hosted_stream import deadline_error
            from .memory_prefinal_budget import (PreFinalBudgetError, PreFinalBudgetExhausted,
                                                  raise_local_dispatch_error)
            raise_control_error(error)
            if (deadline := deadline_error(error)) is not None:
                raise deadline
            raise_local_dispatch_error(error)
            if isinstance(error, PreFinalBudgetExhausted):
                self.audit('memory_selection_attempt', attempt | {'status':'local-budget-stopped',
                    'error_type':type(error).__name__, 'provider_dispatch':False,
                    'elapsed_seconds':time.monotonic()-started})
                raise OSError('selector-budget-unavailable') from error
            if isinstance(error, PreFinalBudgetError):
                raise
            from .hosted_failure import classify_provider_failure
            failure = classify_provider_failure(error)
            self.audit('memory_selection_attempt', attempt | {'status':'transport-failed',
                'error_type':type(error).__name__, 'failure': failure,
                'elapsed_seconds':time.monotonic()-started})
            if failure is None:
                raise RuntimeError('selector transport implementation failed') from error
            if failure['failure_scope'] == 'deployment':
                from .runtime_errors import SharedProviderFailure
                raise SharedProviderFailure(failure['code'], http_status=failure.get('http_status'),
                                            attempt_ref='memory-selection-' + attempt['attempt_id']) from error
            raise OSError('selector-transport-unavailable') from error
        elapsed = time.monotonic()-started
        self.audit('memory_selection_attempt', attempt | {'status':'response',
            'response':response, 'elapsed_seconds':elapsed})
        try:
            if response.get('finish_reason') != 'stop':
                raise ValueError('selector did not finish normally')
            annotations, order = parse_selection(response['content'], included)
        except (ValueError, TypeError, KeyError):
            self.audit('memory_selection_attempt', {'attempt_id':attempt['attempt_id'],
                'phase':'memory_selection', 'request_identity':request_id, 'status':'invalid-output'})
            raise OSError('selector-invalid-output') from None
        notes = {included[key].identity: explanation
                 for key, explanation in annotations['explanations'].items()}
        metadata = self.metadata | {k:attempt[k] for k in ('legal_count','checked_count',
            'candidate_omitted_count','byte_omitted_count','request_identity','attempt_id')}
        metadata.update(candidate_metadata=candidate_metadata, selection_seconds=elapsed,
            checked_ids=[a.identity for a in included.values()], explanations=notes,
            diagnostics=annotations['diagnostics'], removed_duplicates=annotations['removed_duplicates'],
            raw_recommended_order=annotations['raw_order'], deduplicated_order=order,
            recommended_ids=[included[key].identity for key in order],
            input_bytes=len(request['messages'][1]['content'].encode()),
            request_bytes=len(canonical_json(request).encode()))
        return tuple(included[key] for key in order), notes, metadata


def selection_record(advice, notes, query_digest, context_digest, status):
    """Immutable public K; explanations and runtime navigation live outside it."""
    return {'trust':'advisory_memory', 'status':status, 'advice':to_primitive(advice),
        'query_digest':query_digest, 'public_context_digest':context_digest}


def pack_selection(ordered, notes, query_digest, context_digest, max_results, max_bytes):
    """Pack full entries once; never truncate conditions or exceed the K budget."""
    limit = max_bytes
    result = ()
    for advice in ordered:
        if len(result) >= max_results:
            break
        proposed = (*result, advice)
        if len(canonical_json(selection_record(proposed, notes, query_digest, context_digest, 'results')).encode()) <= limit:
            result = proposed
    status = 'results' if result else ('budget-noop' if ordered else 'no-match')
    record = selection_record(result, notes, query_digest, context_digest, status)
    return result, record if len(canonical_json(record).encode()) <= limit else None


def openai_selection_transport(client=None):
    # Called only by host composition; worker/code execution never needs secrets.
    if client is None:
        import os
        from openai import OpenAI
        client = OpenAI(api_key=os.environ['DEEPSEEK_API_KEY'], base_url='https://api.deepseek.com',
                        timeout=120, max_retries=0)
    else:
        client = client.with_options(max_retries=0)
    def invoke(request):
        try:
            response = client.chat.completions.create(**request)
        except Exception as error:
            from .host_control import raise_control_error
            from .hosted_stream import deadline_error
            from .memory_prefinal_budget import raise_local_dispatch_error
            raise_control_error(error)
            if (deadline := deadline_error(error)) is not None:
                raise deadline
            raise_local_dispatch_error(error)
            raise
        choice = response.choices[0] if response.choices else None
        from .hosted_failure import current_http_attempt
        audit_path = current_http_attempt.get()
        http_binding = {}
        if audit_path is not None:
            path = Path(audit_path)
            try:
                http_binding['http_attempt_id'] = path.name
                http_binding['http_request_sha256'] = json.loads(
                    (path / 'request.json').read_bytes())['body_sha256']
            except (OSError, ValueError, KeyError) as error:
                from .runtime_errors import AuditWriteError
                raise AuditWriteError('Memory selector HTTP audit binding unavailable') from error
        return {'content':choice.message.content if choice else None,
                'usage':response.usage.model_dump() if response.usage else None,
                'response_id':response.id, 'returned_model':response.model,
                'finish_reason':choice.finish_reason if choice else None,
                **http_binding}
    return invoke


def selection_attempt_records(events):
    """Project durable selector stages once per attempt, including prelude/unknown."""
    attempts = {}
    for event in events:
        if event.get('kind') != 'memory_selection_attempt':
            continue
        payload = event['payload']
        attempt_id = payload['attempt_id']
        key = (payload.get('run_ref'), payload.get('package_digest'), payload.get('memory_call'), payload.get('retrieval_stage'), attempt_id)
        attempts.setdefault(key,{}).update(payload)
    records = []
    for key,payload in attempts.items():
        attempt_id = key[-1]
        metadata = payload['selection']
        response = payload.get('response') or {}
        usage = response.get('usage') or {}
        details = usage.get('completion_tokens_details') or {}
        records.append({'attempt_ref':'memory-selection-'+attempt_id,'phase':'memory_selection',
            **{key:payload.get(key) for key in ('run_ref','package_digest','memory_call','retrieval_stage')},
            'provider':metadata['provider'],'model':metadata['model'],
            'status':'received' if payload['status']=='response' else payload['status'],
            'request_digest':payload['request_identity'],'max_tokens':metadata['max_tokens'],
            'prompt_tokens':usage.get('prompt_tokens'),'completion_tokens':usage.get('completion_tokens'),
            'reasoning_tokens':details.get('reasoning_tokens'),
            'prompt_cache_hit_tokens':usage.get('prompt_cache_hit_tokens'),
            'prompt_cache_miss_tokens':usage.get('prompt_cache_miss_tokens'),
            'provider_response_id':response.get('response_id'),'returned_model':response.get('returned_model'),
            'http_attempt_id':response.get('http_attempt_id'),
            'http_request_sha256':response.get('http_request_sha256'),
            'finish_reason':response.get('finish_reason'),'elapsed_seconds':payload.get('elapsed_seconds'),
            'usage_status':'known' if usage.get('prompt_tokens') is not None and usage.get('completion_tokens') is not None else 'unknown'})
    return records
