from .material_readers import is_material_reader, is_material_code
"""Pure final packing for maintained candidate validation and actual delivery."""
from dataclasses import replace
import json

from ._serialization import canonical_json
from .context_limits import ContextLimits
from .c1_records import json_bytes, _DeltaIssue
from .notebook_display import notebook_window
from .material_reading import material_reading_arguments, material_locations
from .material_dedup import project_fragment_page, page_covers_location


class WorkingStateDeliveryUnavailable(_DeltaIssue):
    """Expected projection capacity failure, never an audit failure."""


from .action_diagnostics import CORRECTION_BYTES, bound as bounded_correction


def page_shows_note(page, note):
    """Recognize a complete current note only in its own public reader page."""
    if page.get('ref') not in {note['record_ref'], note['note_handle']}:
        return False
    for fragment in page.get('fragments', ()):
        content = fragment.get('content')
        if fragment.get('complete') and isinstance(content, dict):
            if content.get('record_ref') == note['record_ref'] and content.get('text') == note['text']:
                return True
        if (fragment.get('section') == '/text' and fragment.get('complete')
                and content == note['text']):
            return True
    return False


def soft_notebook_window(notebook,attachments,pinned,limits):
    """Drop whole ordinary notes with their now-unused attachments; never edit storage."""
    from .t3_t4.public_copy import copy_text
    notebook=dict(notebook,active_records=list(notebook['active_records']))
    pinned_handles={h for note in pinned for h in note.get('basis',())}
    def selected():
        used=pinned_handles|{h for note in notebook['active_records'] for h in note.get('basis',())}
        return [item for item in attachments if item['handle'] in used]
    while notebook['active_records']:
        ordinary=[item for item in selected() if item['handle'] not in pinned_handles]
        trial=dict(notebook,display_notice=copy_text('context_projection','notebook_reopen'))
        if json_bytes(ordinary)<=limits.attachment_display_bytes and json_bytes(trial)<=limits.notebook_display_bytes:
            break
        notebook['active_records'].pop()
        notebook['hidden_active_record_count']+=1
    if notebook['hidden_active_record_count']:
        notebook['display_notice']=copy_text('context_projection','notebook_reopen')
    return notebook,selected()


def pack_working_action(foundation, control, view, *, preferred_refs=(),
                           remaining_budget=None, status=None, correction=None):
    """No readers, menu materialization, Memory queries, or persistence here.

    One correction slot is reserved on the initial candidate. The caller keeps
    the resulting payload across persistence and only refreshes real accounting
    and bounded error fields for action correction.
    """
    compact=view.config.mode=="single" and view.context_limits.automatic_material_bytes is not None
    aliases = {canonical_json(location): handle for handle, location in view.materials.items()}
    # Only the frozen maintenance window and accepted changes participate.
    # Removing a note must not pull a previously hidden note (and unprojected
    # attachment aliases) out of the store between validation and delivery.
    read_notes = tuple(note for note in view.document.get('pinned_notes', ())
        if any(page_shows_note(page, note) for page in view.document['materials']))
    read_refs = {note['record_ref'] for note in read_notes}
    pinned_refs = tuple(note['record_ref'] for note in view.document.get('pinned_notes', ())
        if note['record_ref'] in control._records and control._records[note['record_ref']].active)
    preferred=tuple(dict.fromkeys((*preferred_refs,*pinned_refs)))
    notebook = notebook_window(control, preferred_refs=() if compact else preferred,
        pinned_refs=preferred if compact else (), material_handles=aliases,
        budget=min(view.config.notebook_bytes,view.context_limits.notebook_display_bytes) if compact else view.config.notebook_bytes, config=view.config, available_refs=(set(view.visible_notes.values()) - read_refs) | set(preferred_refs),
        related_refs=tuple(page['ref'] for page in view.document['materials']))
    basis_notes = [*notebook['active_records'], *read_notes]
    if compact:
        basis_notes.extend(view.document.get('pinned_notes', ()))
    shown_basis = {handle for note in basis_notes for handle in note.get('basis', ())}
    # Every retained attachment has an exact original-version location even if
    # its body yields to a current explicit read in this final view.
    attachment_status = {item['handle']: item['display'] for item in view.document.get('material_attachments', ())}
    attachments = [{'handle': handle, **view.materials[handle], 'display': attachment_status.get(handle, 'not_expanded'),
        'reading_arguments': material_reading_arguments(view.materials[handle])}
        for handle in sorted(shown_basis) if handle in view.materials]
    if compact:
        notebook,attachments=soft_notebook_window(notebook,attachments,
            view.document.get('pinned_notes',()),view.context_limits)
    observation = foundation.latest_observation
    index = foundation.records_index
    event = control._trajectory[-1] if control._trajectory else None
    if (index and event and event['kind'] == 'notebook_local_update_applied'
            and event['payload']['notebook_before'] == view.document['notebook_identity']):
        payload = event['payload']
        added = len(payload['created']) + sum(row['target'] is None for row in payload['reasons'])
        added += payload.get('plan_text') is not None
        added -= sum(row['withdrawn'] for row in payload['reasons'])
        collections = dict(index['collections'])
        collections['working_notes'] = dict(collections['working_notes'],
            count=collections['working_notes']['count'] + added)
        index = dict(index, collections=collections, record_count=index['record_count'] + added)
    hidden = dict(foundation.hidden_counts)
    hidden.pop('notebook_records', None)
    if notebook['hidden_active_record_count']:
        hidden['notebook_records'] = notebook['hidden_active_record_count']
    turn = replace(foundation, notebook=notebook, working_state_mode=view.config.mode,
        notebook_identity=control.notebook_identity, update_limits=view.config.update_limits(),
        previous_update_result=status if view.config.mode == 'single' else None,
        latest_observation=observation, materials=tuple(view.document['materials']),
        material_attachments=tuple(attachments), work_facts=view.document['work_facts'],
        pinned_notes=tuple(n for n in view.document.get("pinned_notes",()) if n not in read_notes) if compact else foundation.pinned_notes,
        records_index=index, hidden_counts=hidden,
        remaining_budget=remaining_budget or foundation.remaining_budget,
        maintenance_status=status, correction=bounded_correction(correction))
    return _fit_action(turn, view.context_limits.turn_bytes - (CORRECTION_BYTES if correction is None else 0),
        read_operation=view.document.get("latest_operation"))


pack_maintained_action = pack_working_action


def fallback_maintained_action(foundation, *, remaining_budget, status, config=None):
    """Preserve the original usable action if added working-view packaging fails."""
    from .working_state import WorkingStateConfig
    config = config or WorkingStateConfig()
    return _fit_action(replace(foundation, remaining_budget=remaining_budget,
        maintenance_status=status), (foundation.context_limits or ContextLimits()).turn_bytes)


def _same_body(left, right, read_operation=None):
    """Compare one public reader page, never arbitrary nested scientific JSON."""
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    if not all(left.get(key) is not None and left.get(key) == right.get(key)
               for key in ('ref', 'content_identity', 'trust', 'record_type', 'content_format')):
        return False
    if left.get('reading_arguments') and right.get('reading_arguments') and left['reading_arguments'] != right['reading_arguments']:
        return False
    if left['content_format'] not in {'material-sections-v1', 'json-text-fragment'}:
        return False
    if not left.get('fragments') and 'content' not in left:
        return False
    if left.get('source_record_ref') != right.get('source_record_ref'):
        # The current successful public read supplies the missing association.
        if left.get('source_record_ref') and right.get('source_record_ref'):
            return False
        operation = read_operation or {}
        if (not is_material_reader(operation.get('operation')) or operation.get('status') != 'succeeded'
                or operation.get('arguments', {}).get('ref') != left['ref']):
            return False
    def comparable(page):
        value = {k:v for k,v in page.items() if k not in {'source_record_ref', 'reading_arguments'}}
        if 'fragments' in value:
            value['fragments'] = [{k:v for k,v in fragment.items() if k != 'material_handle'}
                for fragment in value['fragments']]
        return value
    return canonical_json(comparable(left)) == canonical_json(comparable(right))


def _unexpanded(page):
    """Keep public reentry for this exact page; directories are not bodies."""
    result = {key:value for key,value in page.items() if key not in {'content', 'fragments', 'body_location'}}
    directory = list(page.get('directory', ()))
    for location in material_locations(page):
        entry = {'section':location['section'], 'reading_arguments':material_reading_arguments(location)}
        if entry not in directory:
            directory.append(entry)
    result.update(fragments=[], display='not_expanded')
    if directory:
        result['directory'] = directory
    # A large directory can be reopened through the actual reader page entry.
    if json_bytes(result) > 1600 and result.get('reading_arguments'):
        result.pop('directory', None)
        result['directory_omitted'] = True
    return result


def _project_bodies(turn, original_observation, read_operation):
    pages = []
    for page in turn.materials:
        if page.get('content_format') == 'material-sections-v1':
            pages.append(project_fragment_page(page, pages, read_operation))
            continue
        duplicate = next((index for index, previous in enumerate(pages)
            if _same_body(page, previous, read_operation)), None)
        pages.append(dict(_unexpanded(page), body_location=f'/materials/{duplicate}')
            if duplicate is not None else dict(page))
    observation = original_observation
    if (observation is not None and observation.get('status') == 'succeeded'
            and is_material_code(observation.get('code'))
            and is_material_reader((read_operation or {}).get('operation'))
            and (read_operation or {}).get('status') == 'succeeded'):
        details = observation.get('details')
        duplicate = next((index for index, page in enumerate(pages)
            if _same_body(details, page, read_operation)), None)
        if isinstance(details, dict) and details.get('content_format') == 'material-sections-v1':
            observation = dict(observation, details=project_fragment_page(details, pages, read_operation))
        elif duplicate is not None:
            arguments = details.get('reading_arguments') or (read_operation or {}).get('arguments')
            observation = dict(observation, details=dict(_unexpanded(dict(details,
                **({'reading_arguments':arguments} if arguments else {}))),
                body_location=f'/materials/{duplicate}'))
    attachments = []
    for attachment in turn.material_attachments:
        item = {k:v for k,v in attachment.items() if k != 'body_location'}
        location = {k:v for k,v in item.items() if k not in {'handle', 'display', 'reading_arguments'}}
        if item.get('display') == 'unavailable':
            item.pop('reading_arguments', None)
        elif (covered := next((index for index, page in enumerate(pages)
                if page_covers_location(page, location, read_operation)), None)) is not None:
            item.update(display='expanded', body_location=f'/materials/{covered}')
        else:
            item['display'] = 'not_expanded'
        attachments.append(item)
    changes = {}
    work = getattr(turn, 'last_work_result', None)
    if (work and original_observation and read_operation
        and work.get('read_ref') == read_operation.get('ref')
        and 'details' in work and 'details' in original_observation
        and canonical_json(work['details']) == canonical_json(original_observation['details'])):
        changes['last_work_result'] = {key: value for key, value in work.items() if key != 'details'} | {'display': 'reference-only'}
    return replace(turn, materials=tuple(pages), latest_observation=observation,
        material_attachments=tuple(attachments), **changes)


def _yield_advisory_body(turn, excess):
    """Advice yields as whole entries before scientific evidence or notebook pages."""
    index = dict(turn.records_index or {})
    for key, field in (("persistent_advice", "items"), ("initial_advice", "advice")):
        record = index.get(key)
        if record and record.get(field):
            record = dict(record)
            record[field] = record[field][:-1]
            record.update(display="partial", omitted_count=record.get("omitted_count", 0) + 1)
            index[key] = record
            return replace(turn, records_index=index)
    observation = turn.latest_observation
    if observation is not None and not is_material_code(observation.get("code")):
        from .record_inspection import project_memory_observation
        projected = project_memory_observation(observation,
            max_bytes=max(0, json_bytes(observation) - excess),
            read_ref=(turn.last_work_result or {}).get("read_ref"))
        if projected != observation:
            return replace(turn, latest_observation=projected)
    return None


def _fit_action(turn, limit, *, read_operation=None):
    original_observation = turn.latest_observation
    # Reproject from bodies still present after each fit, so addresses cannot
    # survive a removed or moved page. The explicit current read stays first.
    while True:
        projected = _project_bodies(turn, original_observation, read_operation)
        if json_bytes(projected.to_dict()) <= limit:
            return projected
        if turn.source_coverage is not None:
            turn = replace(turn, source_coverage=None)
        elif turn.recent_history:
            turn = replace(turn, recent_history=())
        elif turn.records_index and turn.records_index.get('recent_results'):
            turn = replace(turn, records_index={k: v for k, v in turn.records_index.items() if k != 'recent_results'})
        elif turn.catalog_navigation is not None or turn.catalog_overview is not None:
            turn = replace(turn, catalog_navigation=None, catalog_overview=None)
        elif (advisory_turn := _yield_advisory_body(turn, json_bytes(projected.to_dict()) - limit)) is not None:
            turn = advisory_turn
            original_observation = turn.latest_observation
        elif turn.evidence:
            turn = replace(turn, evidence=turn.evidence[:-1])
        else:
            # The first saved page can also exceed the final delivery budget.
            # Its source and exact reader arguments remain available after yielding
            # the body; an explicit current read is still present in the observation.
            index = next((i for i in range(len(turn.materials)-1, -1, -1)
                if turn.materials[i].get('fragments') or 'content' in turn.materials[i]), None)
            if index is None:
                raise WorkingStateDeliveryUnavailable('notebook_update_delivery_oversize', field_path='$', kind='delivery',
                    actual=json_bytes(projected.to_dict()), limit=limit, unit='utf8_bytes')
            pages = list(turn.materials)
            pages[index] = _unexpanded(pages[index])
            turn = replace(turn, materials=tuple(pages))


def mark_omitted_body_locations(visible):
    """Offline fitting may omit a target; never borrow it from another phase."""
    if not isinstance(visible, dict):
        return visible
    pages = visible.get('materials', ())
    if not isinstance(pages, (list, tuple)):
        pages = ()
    def omitted(value):
        if isinstance(value, dict):
            return 'omitted_field' in value or any(omitted(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return any(omitted(v) for v in value)
        return False
    def usable(page):
        if not isinstance(page, dict) or omitted(page):
            return False
        if isinstance(page.get('content'), str):
            return True
        fragments = page.get('fragments', ())
        return isinstance(fragments, (list, tuple)) and any(isinstance(f, dict)
            and 'content' in f and not (isinstance(f['content'], dict) and 'omitted_field' in f['content'])
            for f in fragments)
    def clean(item):
        if not isinstance(item, dict) or 'body_location' not in item:
            return item
        pointer = item['body_location']
        parts = pointer.split('/') if isinstance(pointer, str) else ()
        valid = (len(parts) == 3 and parts[1] == 'materials' and parts[2].isdigit()
            and int(parts[2]) < len(pages) and usable(pages[int(parts[2])]))
        if valid:
            target = pages[int(parts[2])]
            if target.get('content_format') == 'material-sections-v1':
                if 'section' in item and 'content_format' not in item:
                    valid = page_covers_location(target, item)
                elif item.get('directory'):
                    duplicates = [entry for entry in item['directory'] if entry.get('display') == 'duplicate_body']
                    valid = bool(duplicates) and all(page_covers_location(target, entry) for entry in duplicates)
                else:
                    valid = False
            if valid:
                return item
        return {k:v for k,v in item.items() if k != 'body_location'} | {
            'display':'not_expanded', 'body_omitted_from_learning':True}
    def clean_page(page):
        if not isinstance(page, dict):
            return page
        result = clean(page)
        if isinstance(page.get('directory'), (list, tuple)):
            result = dict(result, directory=[clean(entry) for entry in page['directory']])
        return result
    result = dict(visible)
    observation = result.get('latest_observation')
    if isinstance(observation, dict) and isinstance(observation.get('details'), dict):
        result['latest_observation'] = dict(observation, details=clean_page(observation['details']))
    for key in ('materials', 'material_attachments'):
        if isinstance(result.get(key), (list, tuple)):
            result[key] = [clean_page(item) if key == 'materials' else clean(item) for item in result[key]]
    return result


def direct_source_directory(page, contract):
    """Do not automatically deliver two copies declared by the source owner."""
    from collections.abc import Mapping
    contract = contract.get('owner_declared') if isinstance(contract, Mapping) else None
    if not isinstance(contract, Mapping) or (
            contract.get('primary_data_pointer') != '/collections/result/0'
            or contract.get('audit_duplicate_pointer') != '/source_query_return/result'):
        return page
    if not page.get('fragments'):
        return page
    result = _unexpanded(page)
    result['primary_reading_arguments'] = {'ref': page['ref'], 'section': contract['primary_data_pointer']}
    result['audit_reading_arguments'] = {'ref': page['ref'], 'section': contract['audit_duplicate_pointer']}
    return result
