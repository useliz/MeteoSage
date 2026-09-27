"""Optional notebook maintenance: bounded public view and phase configuration."""
from __future__ import annotations
from .material_readers import is_material_reader, is_material_code

from dataclasses import dataclass, field
from typing import Any, Mapping

from .c1_records import json_bytes
from .public_payload_identity import public_payload_identity
from .reliability import deterministic_identity
from .record_inspection import notebook_working_view
from .context_limits import ContextLimits

SINGLE_ACTION_PROMPT_VERSION = "scientific-openness-single-action-v8"
KNOWLEDGE_VALIDATION_PROMPT_VERSION = "scientific-openness-single-action-v8-knowledge-validation-v1"
SINGLE_ACTION_PROMPT_V10_VERSION = "scientific-openness-single-action-v10"
SINGLE_ACTION_PROMPT_V9_VERSION = "scientific-openness-single-action-v9"
SINGLE_ACTION_PROMPT_VERSIONS = frozenset((SINGLE_ACTION_PROMPT_VERSION, KNOWLEDGE_VALIDATION_PROMPT_VERSION, SINGLE_ACTION_PROMPT_V9_VERSION, SINGLE_ACTION_PROMPT_V10_VERSION))

MAINTENANCE_PROMPT_VERSION = "scientific-working-state-material-delivery-v5"
MAINTAINED_ACTION_PROMPT_VERSION = "scientific-openness-maintained-action-v5"


@dataclass(frozen=True)
class WorkingStateConfig:
    # Default to single: update the notebook in the action response.
    # Keep all notebook features available without a maintenance model call.
    # maintained is explicit opt-in only; never enable it as a default or fallback.
    mode: str = "single"
    max_action_calls: int | None = None
    max_maintenance_calls: int | None = None
    notebook_bytes: int = 65536
    note_characters: int = 4096
    update_bytes: int = 65536
    plan_characters: int = 4096
    max_changes: int = 4
    max_basis_per_note: int = 8
    maintenance_max_tokens: int | None = None

    def __post_init__(self):
        if self.mode not in {"single", "maintained"}:
            raise ValueError("working state mode must be single or maintained")
        for name in ('notebook_bytes', 'note_characters', 'update_bytes', 'plan_characters', 'max_changes', 'max_basis_per_note'):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,int) or value<=0:
                raise ValueError(name+' must be a positive integer')
        for name in ('max_action_calls','max_maintenance_calls','maintenance_max_tokens'):
            value=getattr(self,name)
            minimum=1 if name=='maintenance_max_tokens' else 0
            if value is not None and (isinstance(value,bool) or not isinstance(value,int) or value<minimum):
                raise ValueError(name+' must be None or a valid integer limit')
        if self.mode=='single' and any(getattr(self,n) is not None for n in
                ('max_action_calls','max_maintenance_calls','maintenance_max_tokens')):
            raise ValueError('single uses the existing total call and model limits')

    def update_limits(self):
        return {name: getattr(self, name) for name in
            ('notebook_bytes', 'note_characters', 'plan_characters', 'update_bytes',
             'max_changes', 'max_basis_per_note')}

    @classmethod
    def from_public(cls, mode='single', options=None):
        if options is not None and not isinstance(options,Mapping):
            raise ValueError('working_state must be an object of maintenance limits')
        options=dict(options or {})
        allowed={'notebook_bytes','note_characters','update_bytes','max_action_calls',
            'max_maintenance_calls','maintenance_max_tokens','plan_characters','max_changes','max_basis_per_note'}
        if set(options)-allowed:
            raise ValueError('working_state contains an unsupported or duplicate configuration field')
        if mode=='single' and set(options) & {'max_action_calls', 'max_maintenance_calls', 'maintenance_max_tokens'}:
            raise ValueError('single forbids maintenance phase options: max_action_calls, max_maintenance_calls, maintenance_max_tokens')
        return cls(mode=mode,**options)


def phase_model_configuration(model, working_state, phase='action'):
    value=dict(model)
    if working_state.mode=='maintained':
        if model.get('prompt_version')!=MAINTAINED_ACTION_PROMPT_VERSION:
            raise ValueError('maintained model.prompt_version must identify the maintained action v5 prompt')
        if phase=='maintenance':
            value['prompt_version']=MAINTENANCE_PROMPT_VERSION
            if working_state.maintenance_max_tokens is not None:
                value['max_tokens']=working_state.maintenance_max_tokens
    else:
        if phase != 'action':
            raise ValueError('single mode has no maintenance model phase')
        if model.get('prompt_version') not in SINGLE_ACTION_PROMPT_VERSIONS:
            raise ValueError('single model.prompt_version must identify scientific-openness-single-action-v8')
    return value


def model_identity(model, working_state):
    from .reliability import ModelIdentity
    from ._serialization import to_primitive
    action=phase_model_configuration(model,working_state)
    sampling={'temperature':0,'prompt_version':action['prompt_version'],
        'reasoning_effort':action['reasoning_effort'],'max_output_tokens':action['max_tokens']}
    if action['provider'] in {'teamorouter','zhipu'}:sampling.pop('temperature')
    if 'transport' in action:sampling['transport']=action['transport']
    sampling['working_state'] = to_primitive(working_state)
    if working_state.mode == 'single':
        import hashlib
        from .hosted_agent_policy import hosted_system_prompt
        sampling['notebook_update_protocol'] = 'local-update-v1'
        sampling['prompt_sha256'] = hashlib.sha256(hosted_system_prompt(action['prompt_version'], provider=action['provider'], model=action['model']).encode('utf-8')).hexdigest()
    if working_state.mode=='maintained':
        sampling.update(maintenance_model=phase_model_configuration(model,working_state,'maintenance'),
            working_state=to_primitive(working_state))
    return ModelIdentity(provider=action['provider'],requested_model=action['model'],
        thinking_mode=action['thinking_mode'],sampling=sampling)


from .c1_records import _DeltaIssue


class WorkingStateInputUnavailable(_DeltaIssue):
    """Expected inability to supply a complete authorized reconciliation input."""


@dataclass(frozen=True)
class WorkingStateView:
    document: Mapping[str, Any]
    evidence_refs: tuple[str, ...]
    close_refs: tuple[str, ...]
    materials: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    visible_notes: Mapping[str, str] = field(default_factory=dict)
    config: WorkingStateConfig = field(default_factory=WorkingStateConfig)
    context_limits: ContextLimits = field(default_factory=ContextLimits)

    @property
    def identity(self):
        return public_payload_identity(self.document)

    def to_dict(self):
        return dict(self.document, turn_identity=self.identity)


def working_state_view(turn, resources, *, config=None, previous_update_result=None, preferred_refs=()) -> WorkingStateView:
    """Use the same public sections as action reads, with a larger material slot."""
    from .record_inspection import (_public_content, _artifact_inputs, _result_ref,
        publish_current_turn_record_refs, public_record_index, public_read_page, material_location_available)
    from .material_reading import material_locations, material_reading_arguments
    from .notebook_display import notebook_window
    from .executable_operations import OperationContext
    from .c1_final import eligible_evidence_refs
    from ._serialization import canonical_json
    config = config or WorkingStateConfig()
    limits = resources.context_limits
    compact = config.mode == "single" and limits.automatic_material_bytes is not None
    task, control = resources.state.task, resources.control_records
    legal = set(eligible_evidence_refs(OperationContext(task=task, state_revision=0,
        observations=(), evidence=(), terminal_output=None, resources=resources, c1_open_control=True)))
    legal -= set(resources.protected_record_refs)
    operations = control.public_operations()
    operation = operations[-1] if operations else None
    publish_current_turn_record_refs(resources, latest_observation=operation['observation'] if operation else None,
        evidence=tuple(item for item in resources.ledger.context_view().records if item['record_ref'] in legal),
        notebook=control.current_view(limit=len(control._records)), catalog_navigation=None)
    inputs = _artifact_inputs(resources)
    registry, aliases = {}, {}
    def register(location):
        key = canonical_json(location)
        if key not in aliases:
            handle = 'M' + str(len(aliases) + 1)
            aliases[key] = handle
            registry[handle] = dict(location)
        return aliases[key]
    retained_locations = []
    for note in control.current_view(limit=len(control._records))['active_records']:
        for location in note.get('basis', ()):
            handle = register(location)
            if handle not in {item['handle'] for item in retained_locations}:
                retained_locations.append({'handle': handle, **location, 'display': 'not_expanded',
                    'reading_arguments': material_reading_arguments(location)})
    pages, queue, direct, pinned = [], [], [], []
    explicit_count = 0
    if operation:
        obs = operation['observation']
        details = obs.get('details', {})
        if is_material_reader(operation['operation']) and obs['status'] == 'succeeded':
            args = dict(operation['arguments'])
            queue.append(args)
            body = _public_content(resources, args['ref'])
            if isinstance(body, Mapping) and body.get('note_handle') and body.get('record_ref') in control._records and control._records[body['record_ref']].active:
                # An explicit read pins the exact current version independently
                # of the 3 KiB notebook window; historical reads do not.
                pinned.append(body)
            if 'cursor' in args and not args['cursor'].startswith('s'):
                queue.append({'ref': args['ref']})
        else:
            for ref in obs.get('canonical_refs', ()):
                if ref['identity'] in legal:
                    record = resources.ledger.get(ref['identity'])
                    queue.append({'ref': _result_ref(record) or ref['identity']})
                    direct.extend(record.envelope.lineage_refs)
            if not queue and details.get('trust') == 'advisory_memory':
                queue.append({'ref': operation['ref']})
        direct.extend(operation['arguments'].get('artifact_refs', ()))
    # Public input support is acquired by its actual evidence/artifact reference;
    # no synthetic field-name inference or private ledger body is substituted.
    for args in tuple(queue):
        item = inputs.get(args['ref'])
        if item:
            direct.extend(item.record.envelope.lineage_refs)
    support_queue = [{'ref': ref} for ref in dict.fromkeys(direct) if ref in legal or ref in inputs]
    seen = set()
    page_identities = set()
    def append_page(args, *, expand=False, limit=10*1024, expected=None, automatic=False):
        key = canonical_json(args)
        if key in seen:
            return
        seen.add(key)
        try:
            page = public_read_page(resources, **args)
        except ValueError:
            if expected is None:
                raise
            return
        if page is None:
            return
        page = dict(page, reading_arguments=dict(args))
        # A historical read never grants access or silently follows a new version.
        if expected is not None and any(page.get(k) != expected.get(k)
                for k in ('ref', 'content_identity', 'source_record_ref', 'fragments', 'content')):
            return
        identity = canonical_json({k:page.get(k) for k in
            ('ref', 'content_identity', 'source_record_ref', 'fragments', 'content')})
        if identity in page_identities:
            return
        if compact and automatic and not args.get('section') and not args.get('cursor'):
            item = inputs.get(args['ref'])
            if item is not None:
                from .runtime_records import artifact_contract
                from .working_state_delivery import direct_source_directory
                page = direct_source_directory(page, artifact_contract(item.record, item.artifact))
        prior_handles = set(registry)
        locations = material_locations(page)
        if locations:
            page['fragments'] = [dict(fragment, material_handle=register(location))
                for fragment, location in zip(page['fragments'], locations)]
        over = (json_bytes([*pages[explicit_count:],page]) > limits.automatic_material_bytes
            if compact and automatic else json_bytes([*pages,page]) > limit)
        if over:
            for handle in set(registry) - prior_handles:
                aliases.pop(canonical_json(registry.pop(handle)))
            if compact and automatic:
                from .working_state_delivery import _unexpanded
                pages.append(_unexpanded(page))
                page_identities.add(identity)
            return
        pages.append(page)
        page_identities.add(identity)
        if expand:
            # Traverse only returned directories. No paths or scientific fields
            # are invented. Already displayed subtrees are not repeated. Long
            # strings keep their public entries; an explicit string read always
            # occupies the first slot instead of this automatic expansion.
            shown = {fragment['section'] for fragment in page.get('fragments', ())}
            queue.extend(entry['reading_arguments'] for entry in page.get('directory', ())
                if entry['section'] not in shown and entry['kind'] != 'string')
    # Current read first, direct input support next, then additional legal
    # sections. This replaces the old triple observation/result/actual_read.
    if queue:
        first = queue.pop(0)
        explicit = bool(operation and is_material_reader(operation['operation']) and operation['observation']['status']=='succeeded')
        append_page(first, expand=not compact, limit=limits.turn_bytes, automatic=compact and not explicit)
        if compact and explicit:explicit_count=len(pages)
    # Reopen at most four distinct previous pages through the current reader.
    # Current explicit material stays first and outside the old-page count.
    old_keys = set()
    for previous in reversed(operations[:-1] if operation else operations):
        if not is_material_reader(previous['operation']) or previous['observation']['status'] != 'succeeded':
            continue
        old = previous['observation'].get('details', {})
        if old.get('content_format') not in {'material-sections-v1', 'json-text-fragment'}:
            continue
        key = canonical_json({k:old.get(k) for k in
            ('ref', 'content_identity', 'source_record_ref', 'fragments', 'content')})
        if key in old_keys or key in page_identities:
            continue
        old_keys.add(key)
        append_page(dict(previous['arguments']), expected=old, limit=2*limits.record_page_bytes, automatic=compact)
        if len(old_keys) == (limits.automatic_previous_pages if compact else 4):
            break
    for args in support_queue:
        append_page(args, limit=2*limits.record_page_bytes, automatic=compact)
    for args in queue:
        append_page(args, expand=not compact, limit=2*limits.record_page_bytes, automatic=compact)
        if len(seen) >= 32 or json_bytes(pages) > 2*limits.record_page_bytes:
            break
    shown_locations = {f.get('material_handle') for p in pages for f in p.get('fragments', ())}
    attachments = [item for item in retained_locations if item['handle'] not in shown_locations]
    pinned_refs = {note['record_ref'] for note in pinned}
    notebook = notebook_window(control, preferred_refs=() if compact else preferred_refs, pinned_refs=preferred_refs if compact else (), related_refs=tuple(direct), material_handles=aliases,
        budget=min(config.notebook_bytes,limits.notebook_display_bytes) if compact else config.notebook_bytes, config=config)
    # The exact note is present only in the explicit read slot, never duplicated.
    notebook['active_records'] = [note for note in notebook['active_records'] if note['record_ref'] not in pinned_refs]
    visible = {note['note_handle']: note['record_ref'] for note in notebook['active_records']}
    # A legacy legal note can exceed a reader's display budget. Fix its complete
    # text as current read material, replacing that note's ordinary page.
    for note in pinned:
        fixed = {k: note[k] for k in ('record_ref', 'note_handle', 'text', 'evidence_refs')}
        fixed.update({k:v for k,v in note.items() if k in {'basis_origin_labels','basis_origin_notice'}})
        fixed['basis'] = [register(item) for item in note.get('basis', ())]
        matching = next((p for p in pages if p['ref'] in {note['record_ref'], note['note_handle']}), None)
        from .working_state_delivery import page_shows_note
        if matching is not None and not page_shows_note(matching, fixed):
            pages.remove(matching)
        pinned[pinned.index(note)] = fixed
        visible[fixed['note_handle']] = fixed['record_ref']
    used_handles = {h for n in [*notebook['active_records'], *pinned] for h in n.get('basis', ())}
    attachments = [item for item in attachments if item['handle'] in used_handles]
    if compact:
        from .working_state_delivery import soft_notebook_window
        notebook,attachments=soft_notebook_window(notebook,attachments,pinned,limits)
        visible={note['note_handle']:note['record_ref'] for note in [*notebook['active_records'],*pinned]}
        used_handles={h for n in [*notebook['active_records'],*pinned] for h in n.get('basis',())}
    for item in attachments:
        if not material_location_available(resources, registry[item['handle']]):
            item['display'] = 'unavailable'
    exposed = used_handles | {f['material_handle'] for page in pages for f in page.get('fragments', ()) if 'material_handle' in f}
    registry = {handle: location for handle, location in registry.items() if handle in exposed}
    scientific_refs = {ref for note in [*notebook['active_records'], *pinned] for ref in note.get('evidence_refs', ())}
    scientific_refs.update(page['ref'] for page in pages)
    scientific_refs.update(page['source_record_ref'] for page in pages if 'source_record_ref' in page)
    visible_evidence = tuple(sorted(scientific_refs & legal))
    raw_turn = turn.to_dict() if hasattr(turn, 'to_dict') else {
        'remaining_budget': turn.remaining_budget, 'state_revision': turn.state_revision}
    document = {'phase': 'maintenance' if config.mode == 'maintained' else 'action', 'objective': task.question,
        'previous_update_result': previous_update_result, 'update_limits': config.update_limits(),
        'fixed_scope': {'decision_time': task.decision_time.isoformat(),
            'target': task.target_scope.to_dict() if task.target_scope else None},
        'open_requirements': tuple(task.answer_contract.required_fields), 'task_context': task.disclosed_context,
        'notebook': notebook, 'notebook_identity': control.notebook_identity,
        'remaining_budget': raw_turn.get('remaining_budget',{}), 'state_revision': raw_turn['state_revision'],
        'materials': pages, 'material_attachments': attachments, 'pinned_notes': pinned,
        'records_index': public_record_index(resources),
        'allowed_scientific_evidence_refs': visible_evidence}
    if operation:
        observation = operation['observation']
        document['latest_operation'] = {'ref': operation['ref'], 'operation': operation['operation'],
            'arguments': {key: value for key, value in operation['arguments'].items() if key != 'code'},
            'status': observation['status'], 'code': observation['code'], 'summary': observation['summary'][:500]}
    from .material_reading import work_facts
    document['work_facts'] = work_facts(control, tuple(p['ref'] for p in pages))
    if compact and any(p.get('display')=='not_expanded' for p in pages):
        from .t3_t4.public_copy import copy_text
        document['work_facts']={**document['work_facts'],'material_reopen_notice':copy_text('context_projection','material_reopen')}
    if json_bytes(dict(document, turn_identity='0'*64)) > limits.turn_bytes:
        raise WorkingStateInputUnavailable('working_state_input_oversize', field_path='$', kind='display',
            actual=json_bytes(dict(document, turn_identity='0'*64)), limit=limits.turn_bytes, unit='utf8_bytes',
            recovery='The complete maintenance input cannot fit. Previous notes remain readable; use the current public reader.')
    return WorkingStateView(document, visible_evidence, tuple(visible.values()), registry, visible, config, limits)


def freeze_action_update_context(turn, view):
    """Bind update handles to complete bodies and attachments in the final Turn."""
    from dataclasses import replace
    notes = [*(turn.notebook or {}).get('active_records', ()), *turn.pinned_notes]
    from .working_state_delivery import page_shows_note
    notes.extend(note for note in view.document.get('pinned_notes', ())
        if any(page_shows_note(page, note) for page in turn.materials))
    visible = {note['note_handle']: note['record_ref'] for note in notes if 'text' in note}
    exposed = {fragment['material_handle'] for page in turn.materials
        for fragment in page.get('fragments', ()) if 'material_handle' in fragment}
    exposed.update(entry['material_handle'] for page in turn.materials for entry in page.get('directory', ())
        if entry.get('display') == 'duplicate_body' and 'material_handle' in entry)
    exposed.update(item['handle'] for item in turn.material_attachments)
    materials = {handle: location for handle, location in view.materials.items() if handle in exposed}
    document = dict(view.document, notebook_identity=turn.notebook_identity,
        notebook=turn.notebook, materials=list(turn.materials),
        material_attachments=list(turn.material_attachments), pinned_notes=list(turn.pinned_notes))
    return replace(view, document=document, materials=materials, visible_notes=visible,
        close_refs=tuple(visible.values()))
