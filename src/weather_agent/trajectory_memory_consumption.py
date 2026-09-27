"""First-final advisory projection; never edits scientific state or a submission."""
from copy import deepcopy
from dataclasses import dataclass, field, replace
from importlib.resources import files

from ._serialization import canonical_json, identity, to_primitive
from .c1_records import json_bytes


PRE_FINAL_PROMPT_VERSION = 'memory-pre-final-inputfix-v1'


class PreFinalContractUnavailable(ValueError):
    """The current public Turn has no unique usable final schema."""


class PreFinalUnprojectable(ValueError):
    """The original final cannot be represented in a strict JSON selector input."""


def prompt(name):
    if name not in {'active-search-purpose', 'pre-final-notice', 'pre-final-selection',
                    'report-final-purpose', 'report-pre-final-notice'}:
        raise ValueError('unknown pre-final prompt')
    return files('weather_agent').joinpath('prompts', 'memory_prefinal', name + '.txt').read_text(encoding='utf-8')


@dataclass
class PreFinalState:
    status: str = 'pending'
    trigger_attempt_ref: str | None = None
    trigger_turn_identity: str | None = None
    trigger_state_revision: int | None = None
    pending_final: dict | None = None
    selector_status: str | None = None
    decision: str | None = None
    read_ref: str | None = None
    selected_entry_ids: list[str] = field(default_factory=list)
    prepared_entry_ids: list[str] = field(default_factory=list)
    reselected_entry_ids: list[str] = field(default_factory=list)
    record: dict | None = None
    draft_status: str | None = None
    draft_original_ref: str | None = None
    draft_original_sha256: str | None = None
    submission_attempt_ref: str | None = None


def pending_context(turn, arguments, *, preserve_source_record_ref=False):
    from .trajectory_memory_workflow_view import build_selection_view
    try:
        canonical_json(to_primitive(arguments))
    except (TypeError, ValueError) as error:
        raise PreFinalUnprojectable('original report arguments are not strict JSON') from error
    visible = deepcopy(to_primitive(turn.to_dict() if hasattr(turn, 'to_dict') else turn))
    result = build_selection_view(visible, profile='workflow-lessons-v3',
                                  pending_final_arguments=arguments,
                                  preserve_source_record_ref=preserve_source_record_ref)
    binding = result['host_binding']['projection_binding']
    pending = {'semantic_key': 'final:submit-answer', 'arguments': deepcopy(to_primitive(arguments))}
    return {'context': result['model_view'], 'pending_final': pending,
            'query': visible['objective'] + '\nHelp with the first proposed final submission using the displayed current state.',
            'turn_identity': visible.get('turn_identity'),
            'projection_identity': binding['projection_identity'],
            'context_identity': binding['context_identity'], 'projection_binding': deepcopy(binding)}


def deliver(turn, record, pending, *, remaining_budget, previous_update_result, turn_bytes,
            consumption_policy='pre-final-v1'):
    """Pack the whole draft and every selected entry on the next sent Turn."""
    index = deepcopy(dict(turn.records_index or {}))
    for name in ('initial_advice', 'persistent_advice', 'recent_advice'):
        index.pop(name, None)
    if consumption_policy == 'pre-final-v1':
        block = {'trust': 'advisory_memory', 'read_ref': record['read_ref'],
                 'notice': prompt('pre-final-notice'), 'pending_final': deepcopy(pending),
                 'advice': [], 'display': 'whole', 'omitted_count': len(record['advice'])}
        foundation = replace(turn, records_index=index, remaining_budget=remaining_budget,
                             previous_update_result=previous_update_result)
        for advice in record['advice']:
            trial = dict(block, advice=[*block['advice'], deepcopy(advice)],
                         omitted_count=block['omitted_count'] - 1)
            trial['display'] = 'partial' if trial['omitted_count'] else 'whole'
            candidate = replace(foundation, records_index={**index, 'pre_final_advice': trial})
            if json_bytes(trial) <= 12288 and json_bytes(candidate.to_dict()) + 1000 <= turn_bytes:
                block = trial
        return (replace(foundation, records_index={**index, 'pre_final_advice': block})
                if block['advice'] else None)
    if consumption_policy != 'report-pre-final-v1':
        raise ValueError('unknown pre-final consumption policy')
    notice = ('report-pre-final-notice' if consumption_policy == 'report-pre-final-v1'
              else 'pre-final-notice')
    block = {'trust': 'advisory_memory', 'read_ref': record['read_ref'],
             'notice': prompt(notice), 'advice': deepcopy(record['advice']),
             'display': 'whole', 'omitted_count': 0}
    foundation = replace(turn, records_index=index, remaining_budget=remaining_budget,
                         previous_update_result=previous_update_result)
    if not block['advice'] or json_bytes(block) > 12288:
        return None
    candidate = replace(foundation, records_index={**index,
        'pending_final': {'trust': 'unexecuted_model_proposal', **deepcopy(pending)},
        'pre_final_advice': block})
    if json_bytes(candidate.to_dict()) + 1000 > turn_bytes:
        return None
    return candidate
