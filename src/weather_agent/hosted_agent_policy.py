from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Callable, Mapping

from ._serialization import canonical_json
from .c1_records import MAX_RESPONSE_BYTES, _DeltaIssue
from .episode_runner import AdapterProtocolError, Submission, Turn
from .runtime_records import LLMAttemptRef
from .working_state import MAINTENANCE_PROMPT_VERSION, MAINTAINED_ACTION_PROMPT_VERSION, SINGLE_ACTION_PROMPT_VERSION, KNOWLEDGE_VALIDATION_PROMPT_VERSION, SINGLE_ACTION_PROMPT_VERSIONS
from .working_state_prompts import MAINTENANCE_PROMPT, MAINTAINED_ACTION_PROMPT, SINGLE_ACTION_PROMPT, SINGLE_ACTION_PROMPT_V9
from .working_state import SINGLE_ACTION_PROMPT_V9_VERSION, SINGLE_ACTION_PROMPT_V10_VERSION
from .working_state_prompts import SINGLE_ACTION_PROMPT_V10


KNOWLEDGE_VALIDATION_SUFFIX = '本轮要求验证知识库的实际使用。在提交最终答案前，请使用可发现的知识搜索和正文读取能力，核对本题涉及的变量、单位、阈值或统计定义，并引用实际读过且相关的内容。题目和授权数据合同优先；没有相关知识或知识与题目冲突时如实说明，按题目合同作答。知识不能补回缺失、受限或未来数据。仍由你选择检索词、条目与后续计算步骤。'

HOSTED_DECISION_ADAPTER_SCHEMA_VERSION = "weather-agent-hosted-decision-adapter-v4"
HOSTED_WORK_PROTOCOL_PROMPT_VERSION = "scientific-openness-work-protocol-v2"

_HOSTED_WORK_PROTOCOL = """## Task and workflow
Complete the user's weather-science task using available capabilities. No scientific route is prescribed. Read the original objective, scope, latest observation, evidence, notebook, recent history, budget, and current choices. Identify what new results establish and which requested outputs remain unfinished; then choose one action. Revise your questions, hypotheses, or method when observations warrant it. Obtaining files or metadata is not completing their analysis.

## Working notebook
Include record_delta in every non-final response. It updates your working notes, not execution permissions:
- plan_text: initialize a short plan retaining the remaining user-requested outputs and the next useful step. Usually two or three concise sentences suffice. Replace it when the plan changes; omit it when unchanged. Keep the plan revisable, not merely a list of navigation steps.
- notes: add only useful findings, open questions, hypotheses, conflicts, or uncertainties, usually zero or one per turn. Each has label, text, and evidence_refs. Labels are free text. Reference visible canonical evidence supporting an assessment; a question may have no refs. Do not manufacture a finding or new note each turn.
- close_refs: Notebook note IDs resolved or superseded, never evidence IDs. To revise a note, close it and add its replacement. Empty notes and close_refs are valid.
Write concise conclusions and remaining work, not a transcript or private reasoning. Record only what is known before this action; interpret its result next turn. Notebook interpretations remain revisable. Retrieved experience is advisory: mention it in note text if useful, never as task evidence or in evidence_refs.

## Tools and records
Use displayed capability descriptions, current schemas, and artifact contracts. Reuse bindings and artifacts; inspect missing details rather than guess. Search is optional when a suitable card is already visible. Keep large results in artifacts and return compact findings. Reopen public records when summaries omit needed details. Follow specific error feedback; do not repeat an unchanged failed attempt without addressing its cause or an indicated transient failure.

## Completion
Compare actual results with every requested output before final. Continue if relevant available work can complete an unfinished output; do not exhaust tools for their own sake. Provide supported findings, units, scope, citations, and material uncertainty. Distinguish source observations or forecasts, calculations, and interpretation. Never claim unperformed work or call an untried capability unavailable. If a real limitation prevents completion, clearly identify the missing result and limitation.

## Response format
Return one JSON object and no surrounding text. Match one current choice and its arguments schema:
{"choice":"<current choice>","arguments":{},"record_delta":{"plan_text":"<remaining outputs and next step>","notes":[],"close_refs":[]}}
Placeholders illustrate structure, not literal values. For final, record_delta may be omitted.
"""

def _require_positive_integer_max_tokens(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("max_tokens must be a positive integer")
    return value


def hosted_system_prompt(prompt_version, *, provider, model):
    prompts = {HOSTED_WORK_PROTOCOL_PROMPT_VERSION: _HOSTED_WORK_PROTOCOL,
        SINGLE_ACTION_PROMPT_VERSION: SINGLE_ACTION_PROMPT,
        SINGLE_ACTION_PROMPT_V9_VERSION: SINGLE_ACTION_PROMPT_V9,
        SINGLE_ACTION_PROMPT_V10_VERSION: SINGLE_ACTION_PROMPT_V10,
        KNOWLEDGE_VALIDATION_PROMPT_VERSION: SINGLE_ACTION_PROMPT + "\n\n" + KNOWLEDGE_VALIDATION_SUFFIX,
        MAINTENANCE_PROMPT_VERSION: MAINTENANCE_PROMPT,
        MAINTAINED_ACTION_PROMPT_VERSION: MAINTAINED_ACTION_PROMPT}
    if prompt_version not in prompts:
        raise ValueError("unknown hosted prompt version")
    result = prompts[prompt_version]
    if provider in {'teamorouter','zhipu'} and model in {'gemini-3.8-flash', 'glm-5.3-flash'}:
        result += '\n\nTransport requirement: do not use Markdown or code fences.'
    return result


def hosted_generation_options(*, provider, thinking_mode, reasoning_effort, max_tokens):
    if provider == 'teamorouter':
        return {'max_completion_tokens':max_tokens, 'reasoning_effort':reasoning_effort}
    if provider == 'zhipu':
        return {'max_tokens':max_tokens, 'reasoning_effort':reasoning_effort,
                'extra_body':{'thinking':{'type':{'off':'disabled','on':'enabled'}.get(thinking_mode,thinking_mode)}}}
    if provider == 'qwen':
        return {'temperature':0, 'max_tokens':max_tokens, 'reasoning_effort':reasoning_effort,
                'extra_body':{'enable_thinking':thinking_mode in {'on','enabled'}}}
    if provider != 'deepseek':
        raise ValueError('unsupported hosted provider dialect')
    return {'temperature':0, 'max_tokens':max_tokens, 'reasoning_effort':reasoning_effort,
            'extra_body':{'thinking':{'type':{'off':'disabled','on':'enabled'}.get(thinking_mode,thinking_mode)}}}


def _hosted_request_document(
    turn_document: Mapping[str, Any],
    *,
    model: str,
    thinking_mode: str,
    reasoning_effort: str,
    max_tokens: int,
    prompt_version: str = SINGLE_ACTION_PROMPT_VERSION,
    provider: str = "deepseek",
    transport: str = "buffered",
) -> Mapping[str, Any]:
    if transport not in {"buffered", "streamed"}:
        raise ValueError("unsupported hosted transport")
    if not isinstance(turn_document, Mapping):
        raise TypeError("hosted turn document must be an object")
    if not model or not thinking_mode or not reasoning_effort:
        raise ValueError("hosted request identity fields must be non-empty")
    if reasoning_effort not in {"low", "medium", "high", "max"}:
        raise ValueError("hosted reasoning_effort is invalid")
    max_tokens = _require_positive_integer_max_tokens(max_tokens)
    system_prompt = hosted_system_prompt(prompt_version, provider=provider, model=model)
    return {
        "model": model,
        "messages": (
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": canonical_json(turn_document)},
        ),
        "response_format": {"type": "json_object"},
        **({"stream":True,"stream_options":{"include_usage":True}} if transport=="streamed" else {}),
        **hosted_generation_options(provider=provider, thinking_mode=thinking_mode,
                                    reasoning_effort=reasoning_effort, max_tokens=max_tokens),
    }


def hosted_decision_request_digest(
    turn_document: Mapping[str, Any],
    *,
    model: str,
    thinking_mode: str,
    reasoning_effort: str = "high",
    max_tokens: int,
    prompt_version: str = SINGLE_ACTION_PROMPT_VERSION,
    provider: str = "deepseek",
    transport: str = "buffered",
) -> str:
    """Return the canonical request identity without exposing the prompt body."""

    request = _hosted_request_document(
        turn_document,
        model=model,
        thinking_mode=thinking_mode,
        reasoning_effort=reasoning_effort,
        max_tokens=max_tokens,
        prompt_version=prompt_version,
        provider=provider,
        transport=transport,
    )
    return hashlib.sha256(canonical_json(request).encode("utf-8")).hexdigest()


def hosted_work_protocol_digest() -> str:
    """Return the fixed system protocol identity without exposing its content."""

    return hashlib.sha256(_HOSTED_WORK_PROTOCOL.encode("utf-8")).hexdigest()


def hosted_decision_request_audit(
    turn_document: Mapping[str, Any],
    *,
    model: str,
    thinking_mode: str,
    reasoning_effort: str = "high",
    max_tokens: int,
    prompt_version: str = SINGLE_ACTION_PROMPT_VERSION,
    provider: str = "deepseek",
    transport: str = "buffered",
) -> Mapping[str, Any]:
    """Project exact request identities and sizes without exposing prompt content."""

    request = _hosted_request_document(
        turn_document,
        model=model,
        thinking_mode=thinking_mode,
        reasoning_effort=reasoning_effort,
        max_tokens=max_tokens,
        prompt_version=prompt_version,
        provider=provider,
        transport=transport,
    )
    serialized_request = canonical_json(request).encode("utf-8")
    serialized_messages = canonical_json(request["messages"]).encode("utf-8")
    return {
        "request_digest": hashlib.sha256(serialized_request).hexdigest(),
        "serialized_request_bytes": len(serialized_request),
        "serialized_messages_bytes": len(serialized_messages),
        "prompt_sha256": hashlib.sha256(request["messages"][0]["content"].encode("utf-8")).hexdigest(),
    }


class HostedDecisionAdapterError(RuntimeError):
    pass


class HostedTransportError(HostedDecisionAdapterError):
    def __init__(self, message: str, *, retryable: bool = False, failure=None) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.failure = failure


def _hosted_http_status(error: Exception) -> int | None:
    from openai import APIStatusError
    return error.status_code if isinstance(error, APIStatusError) else None


def _retryable_transport(error: Exception) -> bool:
    from .hosted_stream import deadline_error
    if deadline_error(error) is not None:return False
    from openai import APIConnectionError
    from httpx import TransportError
    status = getattr(error, "status_code", None)
    return (isinstance(error, (TimeoutError, ConnectionError, APIConnectionError, TransportError))
            or status in {408, 409, 429, 500, 502, 503, 504})


def _shared_provider_code(error):
    from .hosted_failure import classify_provider_failure
    failure = classify_provider_failure(error)
    return failure['code'] if failure and failure['failure_scope'] == 'deployment' else None


class HostedProviderResponseError(HostedTransportError):
    """A durably received response that the provider did not permit us to use."""


class HostedResponseError(HostedDecisionAdapterError, AdapterProtocolError):
    pass


class HostedDecisionAdapter:
    """OpenAI-compatible transport for the single Submission wire."""

    def __init__(
        self,
        client: Any,
        *,
        model: str = "deepseek-v4-flash",
        provider: str = "deepseek",
        thinking_mode: str = "off",
        prompt_version: str = SINGLE_ACTION_PROMPT_VERSION,
        reasoning_effort: str = "high",
        max_tokens: int = 800,
        clock: Callable[[], datetime] | None = None,
        transport: str = "buffered",
        host_deadline: float | None = None,
        host_control: Any = None,
    ) -> None:
        if not all((model, provider, thinking_mode, prompt_version, reasoning_effort)):
            raise ValueError("hosted identity fields must be non-empty")
        if prompt_version not in {HOSTED_WORK_PROTOCOL_PROMPT_VERSION, *SINGLE_ACTION_PROMPT_VERSIONS, MAINTENANCE_PROMPT_VERSION, MAINTAINED_ACTION_PROMPT_VERSION}:
            raise ValueError(
                "prompt_version must identify the adapter's fixed hosted work protocol"
            )
        if reasoning_effort not in {"low", "medium", "high", "max"}:
            raise ValueError("reasoning_effort is invalid")
        if transport not in {"buffered", "streamed"}:
            raise ValueError("unsupported hosted transport")
        self.transport=transport
        self.host_deadline=host_deadline
        self.host_control=host_control
        max_tokens = _require_positive_integer_max_tokens(max_tokens)
        # The outer selection loop accounts for every network request.
        self._client = client.with_options(max_retries=0) if callable(getattr(client, "with_options", None)) else client
        self.model = model
        self.provider = provider
        self.thinking_mode = thinking_mode
        self.prompt_version = prompt_version
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    @property
    def manifest(self) -> Mapping[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "thinking_mode": self.thinking_mode,
            "prompt_version": self.prompt_version,
            "reasoning_effort": self.reasoning_effort,
            "max_tokens": self.max_tokens,
            "schema_version": HOSTED_DECISION_ADAPTER_SCHEMA_VERSION,
            **({"transport":self.transport} if self.transport!="buffered" else {}),
        }

    def _invoke_json(
        self,
        turn: Turn,
        *,
        record_attempt: Callable[[LLMAttemptRef], None],
    ) -> Submission:
        turn_document = turn.to_dict()
        if (turn.context_limits is not None
                and len(canonical_json(turn_document).encode('utf-8')) > turn.context_limits.turn_bytes):
            raise ValueError('Complete public Turn exceeds its configured UTF-8 JSON capacity')
        request = _hosted_request_document(
            turn_document,
            model=self.model,
            thinking_mode=self.thinking_mode,
            reasoning_effort=self.reasoning_effort,
            max_tokens=self.max_tokens,
            prompt_version=self.prompt_version,
            provider=self.provider,
            transport=self.transport,
        )
        request_digest = hosted_decision_request_digest(
            turn_document,
            model=self.model,
            thinking_mode=self.thinking_mode,
            reasoning_effort=self.reasoning_effort,
            max_tokens=self.max_tokens,
            prompt_version=self.prompt_version,
            provider=self.provider,
            transport=self.transport,
        )
        recorder = getattr(record_attempt, "record_public_request", None)
        if recorder is not None:
            recorder(request)
        started_at = self._clock()
        starter = getattr(record_attempt, "record_attempt_start", None)
        if starter is not None:
            starter({"request_digest": request_digest, "context_identity": turn.identity,
                "phase": self.phase, "started_at": started_at.isoformat(), "model": self.manifest})
        from .hosted_failure import current_http_attempt, http_facts, classify_provider_failure
        current_http_attempt.set(None)
        try:
            client=self._client
            if self.transport=="streamed":
                import time
                import httpx
                from .hosted_stream import check_runtime
                check_runtime(self.host_deadline,self.host_control)
                remaining=180.0 if self.host_deadline is None else min(180.0,self.host_deadline-time.monotonic())
                client=client.with_options(timeout=httpx.Timeout(max(0.001,remaining),connect=min(10.0,max(0.001,remaining))))
            response = client.chat.completions.create(**request)
            if self.transport=="streamed":
                from .hosted_stream import response_view
                response=response_view(response,check=lambda:check_runtime(self.host_deadline,self.host_control))
        except Exception as error:
            from .host_control import raise_control_error
            raise_control_error(error)
            facts = http_facts()
            failure = classify_provider_failure(error, facts=facts)
            if failure is None:
                raise
            attempt = LLMAttemptRef(
                    http_attempt_id=facts.get('http_attempt_id'),
                    http_request_sha256=facts.get('http_request_sha256'),
                    provider_failure=failure,
                    phase=self.phase,
                    decision_index=getattr(record_attempt, "decision_index", None),
                    provider=self.provider,
                    model=self.model,
                    thinking_mode=self.thinking_mode,
                    prompt_version=self.prompt_version,
                    context_identity=turn.identity,
                    request_digest=request_digest,
                    response_digest=None,
                    started_at=started_at,
                    completed_at=self._clock(),
                    status="transport_failure",
                    prompt_tokens=None,
                    completion_tokens=None,
                    max_tokens=self.max_tokens,
                    reasoning_effort=self.reasoning_effort,
                    finish_reason=None,
                    finish_reason_present=False,
                    reasoning_tokens=None,
                    visible_tokens=None,
                    reasoning_content_bytes=None,
                    content_bytes=None,
                    content_empty=None,
                    response_format="json_object",
                    response_classification="transport_failure",
                    sanitized_error_code=type(error).__name__,
                    http_status=failure['http_status'],
                    transport_retryable=failure['retry_class'] in {'transport', 'generation'},
                )
            record_attempt(attempt)
            shared_code = failure['code'] if failure['failure_scope'] == 'deployment' else None
            if shared_code is not None:
                from .runtime_errors import SharedProviderFailure
                raise SharedProviderFailure(shared_code, http_status=attempt.http_status,
                    attempt_ref=attempt.attempt_ref) from None
            raise HostedTransportError(failure['code'], retryable=failure['retry_class'] != 'none',
                failure=dict(failure, attempt_ref=attempt.attempt_ref)) from None

        completed_at = self._clock()
        usage = getattr(response, "usage", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        details = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens = getattr(details, "reasoning_tokens", None)
        visible_tokens = (
            completion_tokens - reasoning_tokens
            if isinstance(completion_tokens, int)
            and not isinstance(completion_tokens, bool)
            and isinstance(reasoning_tokens, int)
            and not isinstance(reasoning_tokens, bool)
            and completion_tokens >= reasoning_tokens
            else None
        )
        returned_model = getattr(response, "model", None)
        if not isinstance(returned_model, str) or not returned_model:
            returned_model = None
        system_fingerprint = getattr(response, "system_fingerprint", None)
        if not isinstance(system_fingerprint, str) or not system_fingerprint:
            system_fingerprint = None
        try:
            choice = response.choices[0]
            message = choice.message
        except (AttributeError, IndexError, KeyError, TypeError):
            choice = None
            message = None
        raw_content = getattr(message, "content", None)
        content = raw_content if isinstance(raw_content, str) else ""
        raw_reasoning = getattr(message, "reasoning_content", None)
        reasoning_content = raw_reasoning if isinstance(raw_reasoning, str) else ""
        finish_reason = getattr(choice, "finish_reason", None)
        finish_reason_present = isinstance(finish_reason, str) and bool(finish_reason)
        if not finish_reason_present:
            finish_reason = None
        content_empty = not bool(content.strip())
        response_classification = (
            "length_exhausted"
            if finish_reason == "length"
            or (content_empty and completion_tokens == self.max_tokens)
            else "empty_content"
            if content_empty
            else "received"
        )
        response_digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        facts = http_facts()
        native_output = (
            finish_reason in {"tool_calls", "function_call"}
            or bool(getattr(message, "tool_calls", None))
            or bool(getattr(message, "function_call", None))
            or getattr(response, "stream_protocol_error", None) == "tool_call_output_not_supported"
        )
        failure = None
        # A content_filter finish is handled like stop; keep the original flag
        # in audit and validate its text through the ordinary action contract.
        if finish_reason not in {None, "stop", "length", "content_filter"} and not native_output:
            failure = {
                "code": "provider_response_unsupported",
                "failure_scope": "node", "failure_kind": "provider_request",
                "retry_class": "none", "retryable": False,
                "provider_code": finish_reason, "provider_type": "finish_reason",
                "http_status": facts.get("http_status"),
                "response_started": facts.get("response_started"),
                "stream_complete": facts.get("stream_complete"),
                "message": "The provider returned an unsupported finish reason. No action from this response was executed.",
            }
        attempt = LLMAttemptRef(
            provider_failure=failure,
            http_status=facts.get("http_status"),
            transport_retryable=False if failure else None,
            http_attempt_id=facts.get('http_attempt_id'),
            http_request_sha256=facts.get('http_request_sha256'),
            phase=self.phase,
            decision_index=getattr(record_attempt, "decision_index", None),
            provider=self.provider,
            model=self.model,
            thinking_mode=self.thinking_mode,
            prompt_version=self.prompt_version,
            context_identity=turn.identity,
            request_digest=request_digest,
            response_digest=response_digest,
            started_at=started_at,
            completed_at=completed_at,
            status="received",
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=completion_tokens,
            max_tokens=self.max_tokens,
            reasoning_effort=self.reasoning_effort,
            finish_reason=finish_reason,
            finish_reason_present=finish_reason_present,
            reasoning_tokens=reasoning_tokens,
            visible_tokens=visible_tokens,
            reasoning_content_bytes=len(reasoning_content.encode("utf-8")),
            content_bytes=len(content.encode("utf-8")),
            content_empty=content_empty,
            response_format="json_object",
            response_classification=response_classification,
            returned_model=returned_model,
            system_fingerprint=system_fingerprint,
            provider_response_id=getattr(response, "id", None),
            prompt_cache_hit_tokens=getattr(usage, "prompt_cache_hit_tokens", None),
            prompt_cache_miss_tokens=getattr(usage, "prompt_cache_miss_tokens", None),
        )
        from .hosted_failure import current_http_attempt
        audit_directory = getattr(response, "audit_directory", None) or current_http_attempt.get()
        if audit_directory is not None and (self.transport == "streamed" or failure or native_output):
            from .t3_t4.provider_audit import persist_normalized_response
            persist_normalized_response(audit_directory, response, attempt,
                transport=self.transport, provider_failure=failure, native_output=native_output)
        record_attempt(attempt)
        response_recorder = getattr(record_attempt, "record_public_response", None)
        if response_recorder is not None:
            response_recorder({"content": content, "attempt_ref": attempt.attempt_ref,
                "response_digest": response_digest, "finish_reason": finish_reason})
        if failure is not None:
            raise HostedProviderResponseError(failure["message"], retryable=False,
                failure=dict(failure, attempt_ref=attempt.attempt_ref))
        if native_output:
            raise HostedResponseError('unsupported_stream_protocol', 'submission',
                'Return exactly one JSON object in assistant text, using a current choice and its argument schema. '
                'Native tool_calls and function_call are not supported. Use only fields allowed by the current action envelope.',
                attempt_ref=attempt.attempt_ref)
        short_action = (
            "Return one short JSON object matching a current choice and its "
            "arguments schema."
        )
        from .action_diagnostics import parse_action, ActionJSONError, diagnostic, copy_text
        parse_error = None
        try:
            value=parse_action(content);parsed=True
        except ActionJSONError as error:
            value=None;parsed=False;parse_error=error
        from .t3_t4.report_intent import parsed_report_final
        report_final=parsed and parsed_report_final(turn,value)
        if (self.transport=="streamed" or not report_final) and response_classification in {"length_exhausted", "empty_content"}:
            raise HostedResponseError(
                response_classification,
                "submission",
                copy_text('correction', 'empty_content') if content_empty else short_action,
                attempt_ref=attempt.attempt_ref,
                details=diagnostic('empty_content', finish_reason=finish_reason, actual_bytes=len(content.encode('utf-8'))) if content_empty else None,
            )
        content_bytes = len(content.encode("utf-8"))
        if not report_final and content_bytes > MAX_RESPONSE_BYTES:
            raise HostedResponseError(
                "response_oversize",
                "submission",
                copy_text("correction","response_too_large"),
                details=diagnostic("response_too_large",actual_bytes=content_bytes,allowed_bytes=MAX_RESPONSE_BYTES),
                attempt_ref=attempt.attempt_ref,
                update_issue=_DeltaIssue("response_oversize", kind="length", actual=content_bytes,
                    limit=MAX_RESPONSE_BYTES, unit="utf8_bytes").issue,
            )
        if not parsed:
            raise HostedResponseError(
                "invalid_json",
                parse_error.path,
                copy_text("correction", parse_error.details["error_kind"]),
                attempt_ref=attempt.attempt_ref, details=dict(parse_error.details,attempt_ref=attempt.attempt_ref),
            ) from None
        if not report_final:
            try: json.dumps(value,allow_nan=False)
            except ValueError:
                raise HostedResponseError('invalid_json','submission',copy_text('correction','field_value'),
                    attempt_ref=attempt.attempt_ref,details=diagnostic('field_value',expected_type='finite JSON values')) from None
        return value, attempt

    @property
    def phase(self):
        return "maintenance" if self.prompt_version == MAINTENANCE_PROMPT_VERSION else "action"

    def maintain(self, view, *, record_attempt):
        if self.phase != "maintenance":
            raise ValueError("maintenance requires its own prompt identity")
        value, attempt = self._invoke_json(view, record_attempt=record_attempt)
        if not isinstance(value, dict):
            raise HostedResponseError("invalid_shape", "notebook", "Return {} or a local notebook update with optional plan_text and changes.",
                attempt_ref=attempt.attempt_ref,
                update_issue=_DeltaIssue("invalid_shape", field_path="notebook", kind="type").issue)
        return value

    def choose(self, turn, *, record_attempt):
        if self.phase != "action":
            raise ValueError("action requires an action prompt identity")
        value, attempt = self._invoke_json(turn, record_attempt=record_attempt)
        short_action = "Return one JSON object matching a current choice and arguments schema."
        allowed = {"choice", "arguments"}
        if self.prompt_version in SINGLE_ACTION_PROMPT_VERSIONS:
            allowed.add("notebook_update")
        elif self.prompt_version in {HOSTED_WORK_PROTOCOL_PROMPT_VERSION, MAINTAINED_ACTION_PROMPT_VERSION}:
            allowed.add("record_delta")
        from .action_diagnostics import envelope_issue, diagnostic, candidate_details, copy_text
        schema={'type':'object','properties':{name:{} for name in allowed},
            'required':['choice','arguments'],'additionalProperties':False}
        issue=envelope_issue(value,schema,'submission')
        if issue:
            path,details=issue
            if not isinstance(value,dict): details=diagnostic('root_type',expected_type='object',actual_type=details['actual_type'])
            details.update(candidate_details(value),attempt_ref=attempt.attempt_ref)
            raise HostedResponseError('invalid_shape',path,copy_text('correction',details['error_kind']),
                attempt_ref=attempt.attempt_ref,details=details)
        schema={'type':'object','properties':{'choice':{'type':'string','minLength':1},'arguments':{'type':'object'}}}
        issue=envelope_issue(value,schema,'submission')
        if issue:
            path,details=issue
            details.update(candidate_details(value),attempt_ref=attempt.attempt_ref)
            raise HostedResponseError('invalid_action',path,copy_text('correction',details['error_kind']),
                attempt_ref=attempt.attempt_ref,details=details)
        return Submission(
            choice=value["choice"],
            arguments=value["arguments"],
            record_delta=value.get("record_delta"),
            **({"notebook_update": value["notebook_update"]} if "notebook_update" in value else {}),
        )


__all__ = [
    "HOSTED_DECISION_ADAPTER_SCHEMA_VERSION",
    "HOSTED_WORK_PROTOCOL_PROMPT_VERSION",
    "HostedDecisionAdapter",
    "HostedDecisionAdapterError",
    "HostedResponseError",
    "HostedTransportError",
    "hosted_decision_request_audit",
    "hosted_decision_request_digest",
    "hosted_work_protocol_digest",
]
