"""Frozen host binding for T3/T4 Memory; never exposed as task content."""
from __future__ import annotations

import json
from pathlib import Path

from .identity import digest, file_sha

SCHEMA = 'weather-agent-t3-t4-memory-binding-v1'
POLICY = 'committed-after-memory-v1'
_FIELDS = {'schema_version', 'memory', 'selector_client_binding',
           'evaluation_final_policy', 'split_index', 'node_bindings'}
_NODE_FIELDS = {'execution_unit_id', 'source_refs', 'isolation_refs', 'group',
                'upstream_instance_ids', 'group_resolution'}


def _file(path):
    if not isinstance(path, str) or not path:
        raise ValueError('Memory asset path must be a nonempty string')
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_file():
        raise ValueError('Memory asset must be a regular file')
    return {'path': str(resolved), 'sha256': file_sha(resolved)}


def _strings(values, label):
    if (not isinstance(values, list) or any(not isinstance(x, str) or not x for x in values)
            or len(values) != len(set(values))):
        raise ValueError(label + ' must contain distinct nonempty strings')
    return values


def _source_identity_refs(refs):
    return (any(ref.startswith('run:') and len(ref) > 4 for ref in refs)
            and all(any(ref.startswith(prefix) and len(ref) == len(prefix) + 64
                        and all(ch in '0123456789abcdef' for ch in ref[len(prefix):])
                        for ref in refs)
                    for prefix in ('bundle-sha256:', 'public-input:')))


def normalize_binding(value, *, installation, units, model):
    """Validate one explicit binding and bind current asset bytes and selected nodes."""
    from ..trajectory_memory import AdvisoryMemory
    from ..trajectory_memory_operations import MemoryConfig
    from ..trajectory_memory_prompts import workflow_prompts
    from .client_config import load_client_config, selected_client

    if not isinstance(value, dict) or set(value) != _FIELDS or value['schema_version'] != SCHEMA:
        raise ValueError('Memory binding schema mismatch')
    if value['evaluation_final_policy'] != POLICY:
        raise ValueError('Memory-on requires committed-after-memory-v1')
    if (model['provider'] != 'deepseek' or model['model'] != 'deepseek-v4.1-flash-expires-on-0910'
            or model['reasoning_effort'] != 'high' or model['max_tokens'] != 131072):
        raise ValueError('Memory-on is limited to the frozen DS/high main route')
    spec = value['memory']
    if not isinstance(spec, dict) or spec.get('mode') != 'package':
        raise ValueError('Memory-on requires one explicit package')
    config = MemoryConfig.from_public(spec)
    if (config.consumption_policy != 'report-pre-final-v1'
            or config.selector_model != model['model'] or config.selector_thinking != 'on'
            or config.selector_reasoning_effort != 'high' or config.query_max_chars != 4096
            or config.recent_cards_max_results != 3 or config.recent_cards_max_bytes != 8192
            or config.recent_card_label_max_bytes != 160 or not config.candidate_endpoint):
        raise ValueError('Memory-on DS selector or limits differ from frozen binding')
    registry_ref = installation['client_config']
    registry = load_client_config(registry_ref['path'], expected_sha256=registry_ref['sha256'],
                                  expected_identity=registry_ref['identity'])
    _, client_identity = selected_client(registry, 'deepseek')
    expected_client = {'provider': 'deepseek', 'registry_path': registry_ref['path'],
                       'registry_identity': registry_ref['identity'],
                       'client_identity': client_identity}
    if value['selector_client_binding'] != expected_client:
        raise ValueError('Memory selector client binding changed')

    split_ref = value['split_index']
    if not isinstance(split_ref, dict) or set(split_ref) != {'path', 'sha256'}:
        raise ValueError('Memory split index reference is incomplete')
    split_asset = _file(split_ref['path'])
    if split_asset != split_ref:
        raise ValueError('Memory split index bytes changed')
    split = json.loads(Path(split_ref['path']).read_bytes())
    if (not isinstance(split, dict) or split.get('schema_version') != 'weather-agent-t3-t4-split-index-v1'
            or not isinstance(split.get('nodes'), dict)):
        raise ValueError('Memory split index schema mismatch')
    selected = {instance_id: unit for unit in units for instance_id in unit['instance_ids']}
    official = {row['instance_id']: row for row in installation['host_index']['tasks']}
    bindings = value['node_bindings']
    if not isinstance(bindings, dict) or set(bindings) != set(selected):
        raise ValueError('Memory node binding must cover exactly the selected nodes')
    for instance_id, node in bindings.items():
        unit = selected[instance_id]
        row = split['nodes'].get(instance_id)
        frozen = official.get(instance_id)
        if (not isinstance(node, dict) or set(node) != _NODE_FIELDS
                or not isinstance(row, dict) or not isinstance(frozen, dict)
                or node['execution_unit_id'] != unit['unit_id']
                or row.get('execution_unit_id') != unit['unit_id']
                or node['group_resolution'] != 'resolved'
                or not isinstance(node['group'], str) or not node['group']
                or node['group'] != row.get('statistical_group_id')
                or not isinstance(row.get('event_family_id'), str) or not row['event_family_id']
                or row.get('source_id') != unit.get('source_id')
                or row.get('split') not in {'train_dev', 'validation'}
                or any(row.get(key) != frozen.get(key) for key in
                       ('split', 'statistical_group_id', 'event_family_id'))):
            raise ValueError('Memory node group, event, source, or split is unresolved')
        upstream = _strings(node['upstream_instance_ids'], 'upstream_instance_ids')
        sources = _strings(node['source_refs'], 'source_refs')
        isolation = _strings(node['isolation_refs'], 'isolation_refs')
        expected_upstream = unit['instance_ids'][:unit['instance_ids'].index(instance_id)]
        if (not _source_identity_refs(sources)
                or upstream != expected_upstream or upstream != row.get('upstream_instance_ids', [])):
            raise ValueError('Memory node lineage/source binding changed')
        required = {f't3t4-instance:{name}' for name in unit['instance_ids']}
        required.update(f't3t4-instance:{name}' for name in upstream)
        required.update({f"t3t4-unit:{unit.get('source_id') or 'single'}:{unit['unit_id']}",
                         f"t3t4-statistical-group:{node['group']}",
                         f"t3t4-event-family:{row['event_family_id']}"})
        if not required <= set(isolation):
            raise ValueError('Memory node isolation omits instance, unit, group, event, or lineage')

    assets = {'package': _file(spec['package_path']),
              'workflow_prompt_manifest': _file(spec['workflow_prompt_manifest']),
              'candidate_endpoint': _file(spec['candidate_endpoint']),
              'split_index': split_asset}
    workflow_prompts(spec['workflow_prompt_manifest'], expected_profile='workflow-lessons-v3')
    package_digest = AdvisoryMemory.from_file(spec['package_path']).package_digest
    result = {'source_config': value, 'assets': assets,
              'memory_identity': {'package_digest': package_digest}}
    result['identity'] = digest(result)
    return result


def validate_binding(value, *, installation, units, model):
    if not isinstance(value, dict) or value.get('identity') != digest({k: v for k, v in value.items() if k != 'identity'}):
        raise ValueError('Memory manifest binding identity mismatch')
    expected = normalize_binding(value['source_config'], installation=installation,
                                 units=units, model=model)
    if value != expected:
        raise ValueError('Memory manifest binding or asset bytes changed')
    return value


def configuration_for_node(run_config, binding, instance_id):
    """Derive a fresh node config without sharing recall state across I and W."""
    from copy import deepcopy
    node = binding['source_config']['node_bindings'][instance_id]
    config = deepcopy(run_config)
    memory = config['memory']
    memory['denied_isolation_refs'] = list(dict.fromkeys(
        [*memory.get('denied_isolation_refs', ()), *node['isolation_refs']]))
    config['memory_identity'] = deepcopy(binding['memory_identity'])
    config['memory_node_binding_identity'] = digest(node)
    return config
