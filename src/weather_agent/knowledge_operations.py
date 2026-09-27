"""Common public lexical search/read operations; never executes scientific instructions."""
from __future__ import annotations
import re
from typing import Mapping
from .c1_records import json_bytes
from .executable_operations import (ArgumentValidationError, ChoiceSpec, C1OperationProvider,
    ExecutableOperation, MappingArgumentContract, OperationObservation, TransitionResult)
from .knowledge_access import resolve_knowledge
from .material_reading import MaterialCursorError
from .knowledge_read_transaction import read_disclosed_knowledge, KnowledgeReadFailure
from .reliability import deterministic_identity
from .t3_t4.public_copy import copy_text
from .t3_t4.capability_copy import text as capability_text

FILTERS = ('content_kind', 'package_ref', 'track', 'variable', 'language')
KINDS = ('threshold_rule', 'channel_description', 'method_rule', 'report_example', 'reference')



def contract(operation):
    properties = {'cursor': {'type': 'string', 'maxLength': 100}}
    if operation == 'knowledge.search':
        properties.update(query={'type': 'string', 'maxLength': 1024}, page_size={'type': 'integer', 'minimum': 1, 'maximum': 32},
            filters={'type': 'object', 'properties': {k: {'type': 'array', 'minItems': 1, 'items': {'type': 'string', 'minLength': 1, **({'enum': KINDS} if k == 'content_kind' else {})}} for k in FILTERS}, 'additionalProperties': False})
    else:
        properties.update(ref={'type': 'string', 'minLength': 1, 'maxLength': 1000}, section={'type': 'string', 'maxLength': 1000}, page_bytes={'type': 'integer', 'minimum': 1})
    for name, specification in properties.items():
        specification['description'] = copy_text('operations', operation, 'properties', name)
    schema = {'type': 'object', 'properties': properties, 'required': ['ref'] if operation == 'knowledge.read' else [], 'additionalProperties': False}
    def validate(args):
        from .action_diagnostics import validate_envelope
        validate_envelope(args, schema)
        return dict(args)
    return MappingArgumentContract(schema=schema, validator=validate)


def _disclose(resources, entry):
    ref = entry.to_ref().entry_ref
    resources.issued_record_refs[ref] = {'record_type': 'knowledge_entry', 'trust': 'reference-knowledge-not-weather-evidence',
        'access_policy_identity': deterministic_identity(resources.state.task.access_policy),
        'resolved_scope_identity': resources.state.task.resolved_scope_identity}
    return ref


def search_page(resources, args):
    access = resources.knowledge_access
    query = args.get('query', '')
    filters = args.get('filters', {})
    size = args.get('page_size', 16)
    groups = []
    for token in re.findall(r'\w+', query.casefold()):
        groups.append(next((group for group in resources.knowledge_registry.aliases if token in group), (token,)))
    matches = []
    for entry in resources.knowledge_registry.entries:
        if not access.permits(resources, entry): continue
        metadata = dict(getattr(entry, 'applicability', {})) | entry.summary()
        if any(not any(item in values for item in (actual if isinstance(actual, (list, tuple)) else (actual,)))
               for key, values in filters.items() for actual in (metadata.get(key),)): continue
        title = getattr(entry, 'title', entry.entry_id)
        structured = ' '.join([title, entry.entry_id, getattr(entry, 'source_record_id', ''), *getattr(entry, 'keywords', ()), *getattr(entry, 'aliases', ())]).casefold()
        from ._serialization import canonical_json
        body = (' '.join(s['text'] for s in entry.sections) if hasattr(entry, 'sections') else canonical_json(entry.to_dict())).casefold()
        exact = query in (entry.to_ref().entry_ref, entry.entry_id, getattr(entry, 'source_record_id', ''))
        if not exact and groups and not all(any(term in structured or term in body for term in group) for group in groups): continue
        score = 0 if exact else 1 if all(any(term in structured for term in g) for g in groups) else 2
        matches.append((score, entry.entry_id, entry.version, entry))
    matches.sort(key=lambda row: row[:3])
    from .reader_recovery import resolve_task_resource
    hint=resolve_task_resource(resources,query,allow_prefix=True)
    # A visible real knowledge identifier keeps its own meaning, even when the
    # selected filters hide its card; filters never authorize a task resource.
    if query and any(access.permits(resources,entry) and query in (entry.entry_id,entry.to_ref().entry_ref,getattr(entry,'source_record_id',''))
            for entry in resources.knowledge_registry.entries):hint=None
    if hint is not None:
        hint={**hint,'description':copy_text('errors','task_resource_key','summary')}
        if json_bytes(hint)>768:
            hint['recovery']['arguments']['section']='/resources'
        if json_bytes(hint)>768:hint.pop('description',None)
    token = deterministic_identity(('task-resource-hint-v1',hint,access.identity,
        resources.knowledge_registry.package_manifests, resources.knowledge_registry.aliases_identity,
        query, filters, size, resources.context_limits.navigation_bytes))
    offset = 0
    if args.get('cursor'):
        try:
            prefix, raw = args['cursor'].split(':')
            offset = int(raw)
        except (ValueError, TypeError) as error:
            raise MaterialCursorError('Invalid search cursor') from error
        if prefix != token or not 0 <= offset < len(matches): raise MaterialCursorError('stale search cursor')
    value = {'status': 'ok' if matches else 'no_matches', 'query': query, 'filters': filters,
             'visibility_identity': access.identity, 'package_identities': access.package_identities,
             'total_matches': len(matches), 'offset': offset, 'cards': [], 'has_more': False}
    if hint is not None:value['task_resource_hint']=hint
    if json_bytes(value)>resources.context_limits.navigation_bytes:
        if hint is not None:
            hint.pop('description',None)
            hint['recovery']['arguments']['section']='/resources'
        if json_bytes(value)>resources.context_limits.navigation_bytes:
            raise ValueError('search envelope exceeds context capacity')
    for index in range(offset, min(len(matches), offset + size)):
        score, _, _, entry = matches[index]
        ref = entry.to_ref().entry_ref
        card = {'ref': ref, 'title': getattr(entry, 'title', entry.entry_id),
            'content_kind': getattr(entry, 'content_kind', entry.entry_kind), 'applicability': access.applicability(entry),
            'scope': getattr(entry, 'applicability', {}), 'match_reason': ('exact_id', 'structured_text', 'body_text')[score],
            'body_complete': False, 'read_arguments': {'ref': ref}}
        rule = getattr(entry, 'rule', None)
        if rule is not None and json_bytes(rule) < resources.context_limits.navigation_bytes // 4:
            card.update(rule=rule, body_complete=True)
        def project(candidate):
            trial=dict(value,cards=[*value['cards'],candidate],page_count=len(value['cards'])+1,
                remaining=len(matches)-index-1,has_more=index+1<len(matches))
            if trial['has_more']:
                trial['continuation']={'operation':'knowledge.search','arguments':{
                    'query':query,'filters':filters,'page_size':size,'cursor':f'{token}:{index+1}'}}
            else:trial.pop('continuation',None)
            return trial
        trial=project(card)
        if json_bytes(trial)>resources.context_limits.navigation_bytes:
            # A preview rule is atomic: remove it whole, never truncate it into
            # an apparent complete scientific rule to make navigation fit.
            compact={k:v for k,v in card.items() if k not in ('rule','scope','title')}
            compact['body_complete']=False
            trial=project(compact)
            if json_bytes(trial)>resources.context_limits.navigation_bytes and not value['cards'] and hint:
                hint.pop('description',None)
                hint['recovery']['arguments']['section']='/resources'
                trial=project(compact)
            if json_bytes(trial)>resources.context_limits.navigation_bytes:
                if not value['cards']:raise ValueError('minimum search card and continuation exceed context capacity')
                break
        _disclose(resources, entry)
        value = trial
    return value


class KnowledgeOperationProvider(C1OperationProvider):
    from .material_readers import KNOWLEDGE_READER as material_reader
    def offer(self, context):
        resources=context.resources
        if (getattr(resources,'knowledge_error',None)
                or getattr(resources,'knowledge_registry',None) is None
                or getattr(resources,'knowledge_access',None) is None):
            return ()
        result = []
        for operation in ('knowledge.search', 'knowledge.read'):
            arguments = contract(operation)
            def invoke(execution, args, operation=operation):
                resources = execution.resources
                error = getattr(resources, 'knowledge_error', None)
                if error or getattr(resources, 'knowledge_registry', None) is None or getattr(resources,'knowledge_access',None) is None:
                    return TransitionResult(OperationObservation(status='failed', code=error or 'not_available', summary=capability_text('knowledge_unavailable')))
                try:
                    if operation == 'knowledge.search':
                        page = search_page(resources, args)
                    else:
                        page = read_disclosed_knowledge(resources, args['ref'], section=args.get('section'),
                            cursor=args.get('cursor'), page_bytes=args.get('page_bytes'))
                except KnowledgeReadFailure as error:
                    return TransitionResult(OperationObservation(status='failed', code=error.code,
                        summary=str(error), details=error.details))
                except MaterialCursorError:
                    return TransitionResult(OperationObservation(status='failed', code='reader_cursor_invalid',
                        summary=copy_text('errors', 'reader_cursor_invalid', 'summary'),
                        details={'recovery': {'operation': operation,
                            'arguments': {key: value for key, value in args.items() if key != 'cursor'}}}))
                except (ValueError, TypeError, KeyError) as error:
                    from .action_diagnostics import copy_text as recovery_copy
                    from .material_reading import audit_reader_error
                    return TransitionResult(OperationObservation(status='failed', code='reader_internal_error',
                        summary=recovery_copy('errors','reader_internal_error'),details=audit_reader_error(resources,error,operation)))
                return TransitionResult(OperationObservation(status='succeeded', code='knowledge_read_succeeded' if operation.endswith('read') else 'knowledge_search_succeeded',
                    summary=copy_text('operations', operation, 'success_summary'), details=page))
            result.append(ExecutableOperation(public=ChoiceSpec(semantic_key=operation,
                purpose=copy_text('operations', operation, 'purpose'),
                arguments_schema=arguments.schema, expected_observation=copy_text('operations', operation, 'expected_observation')),
                argument_contract=arguments, invoke=invoke, audit_identity=deterministic_identity((operation, context.task.task_identity, context.state_revision))))
        return tuple(result)
