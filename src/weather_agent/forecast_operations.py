"""Public forecast owners sharing current evidence, executor and artifact store."""
from __future__ import annotations
import json
import numpy as np
import xarray as xr

from ._serialization import parse_datetime, to_primitive
from .action_contracts import ActionOwnerPreparation
from .action_operations import ActionOperationRegistration
from .capability_catalog import CapabilityDescriptor, canonical_binding_request_contract
from .executable_operations import ArgumentValidationError
from .field_inputs import field_inputs, field_support, validate_field_binding
from .forecast_members import validate_member_payload, member_from_dict, target_from_dict, ensemble_kind
from .forecast_numerics import combine, expand_weights
from .reliability import deterministic_identity
from .runtime import ToolResult, ToolStatus
from .runtime_records import EvidenceEnvelope, derived_evidence_times
from .workspace import ScientificWorkspace, load_verified_context_field, context_evidence_carries_artifact

PROFILE = 'agent-discrete-mixture-all-positive-v1'
TOOL_ID = 'weather-agent-forecast-combine'
DESCRIPTION = ('Combine explicitly selected forecast candidates or members already aligned to one target. '
    'Choose component weights; internal member weights remain source-declared. Compute exact weighted '
    'member distribution, mean, threshold probabilities or discrete inverse-CDF quantiles. Zero weights '
    'stay in the audit; every positive member must be valid at each output cell. This internal numerical '
    'profile is not a Benchmark registry identity and does not by itself authorize a formal recipe.')


def source_member_contract(envelope):
    raw = to_primitive(envelope)
    if raw.get('producer_id') not in {'manifest-evidence', 'forecast.align-target', 'forecast.radar-lk', 'forecast.model-run'}:
        raise ValueError('input needs an authenticated forecast producer, not arbitrary array attributes')
    if raw.get('producer_id')=='forecast.model-run':
        # Current input/card/target and sealed receipt bytes are checked by the
        # common evidence eligibility gate before records reach this consumer.
        proofs=[p for p in raw.get('provenance',()) if 'model_run_receipt_ref' in p]
        shared=[p for p in raw.get('provenance',()) if 'forecast_publication_receipt_ref' in p]
        refs={a.get('artifact_ref',deterministic_identity(a)) for a in raw.get('artifacts',())}
        if shared:
            authenticated=(not proofs and len(shared)==1 and all(shared[0].get(key) in refs for key in
                ('forecast_publication_receipt_ref','forecast_computation_receipt_ref','forecast_manifest_ref')))
        else:
            authenticated=len(proofs)==1 and proofs[0]['model_run_receipt_ref'] in refs
        if raw.get('producer_kind')!='forecast_model' or not authenticated:
            raise ValueError('model members require a sealed actual-run receipt')
    matches = [item['forecast_contract'] for item in raw.get('provenance', ()) if 'forecast_contract' in item]
    if len(matches) != 1:
        raise ValueError('input lacks one authenticated member contract')
    return matches[0]


def combine_descriptor(variables):
    return CapabilityDescriptor(capability_id='forecast.combine',version='1.0.0',namespace='science.forecast',
        description=DESCRIPTION,kind='diagnostic_operator',variables=tuple(variables),products=('forecast_result',),evidence_roles=('analysis',),
        spatial_coverage={'global':True},temporal_coverage={'timeless':True},lead_time_coverage={},
        schema_identity=deterministic_identity(('forecast.combine',PROFILE)),time_semantics={'kind':'derived_input_valid_interval'},
        access={'enabled':True,'requires_generated_code':False},resource={'kind':'cpu','max_resource_units':1,'max_timeout_seconds':300},
        limitations=(DESCRIPTION,),execution_binding={'action_kind':'tool_call','tool_id':TOOL_ID,
            'argument_contract':'forecast_combine_v1','required_selectors':('valid_time','spatial'),
            'binding_request_contract':canonical_binding_request_contract('forecast_combine_v1')},
        frozen_parameters={'operator_profile_ref':PROFILE})


def canonical_components(arguments, contracts):
    components = arguments['components']
    if not components or len(components)>128:
        raise ValueError('select between one and 128 components')
    mode=arguments['weight_mode']
    if mode=='source_declared' and (len(components)!=1 or components[0].get('member_ref') is not None):
        raise ValueError('source_declared requires one complete candidate')
    internal=[];members=[];selections=[];targets=[]
    for component in components:
        if set(component)-{'input_ref','evidence_ref','member_ref','weight'}:
            raise ValueError('unknown component argument')
        entry=contracts[component['input_ref']]
        if entry['input_evidence_ref']!=component['evidence_ref']:
            raise ValueError('component artifact and evidence differ')
        contract=entry['forecast_contract']
        targets.append(target_from_dict(contract['target']))
        selected=list(range(len(contract['members'])))
        if component.get('member_ref') is not None:
            selected=[i for i,m in enumerate(contract['members']) if m['member_ref']==component['member_ref']]
            if len(selected)!=1:
                raise ValueError('member selection is absent or ambiguous')
        internal.append([1.] if component.get('member_ref') is not None else contract['member_weights'])
        members.extend(member_from_dict(contract['members'][i]) for i in selected)
        selections.append({'input_ref':component['input_ref'],'evidence_ref':component['evidence_ref'],'indices':selected})
    if len({m.native_key for m in members})!=len(members):
        raise ValueError('duplicate ForecastRun/member across components')
    if len({target.identity for target in targets})!=1 or targets[0].target_ref!=arguments['target_ref']:
        raise ValueError('components are not aligned to exactly the same target')
    has_weights=['weight' in c for c in components]
    if (mode=='explicit' and not all(has_weights)) or (mode!='explicit' and any(has_weights)):
        raise ValueError('component weights conflict with weight mode')
    expanded=expand_weights(internal,mode,[c['weight'] for c in components] if mode=='explicit' else None)
    positive=[m for m,w in zip(members,expanded,strict=True) if w>0]
    return targets[0],members,expanded,selections,ensemble_kind(positive)


class ForecastCombineOwner:
    def __init__(self,descriptor):
        self.descriptor=descriptor
        self.contract_identity=deterministic_identity(('forecast-combine-owner-v1',descriptor.descriptor_identity))

    def _inputs(self,context,binding=None):
        candidates,rejected=field_inputs(context,binding)
        accepted={}
        for ref,entry in candidates.items():
            try:
                record=context.resources.ledger.get(entry['input_evidence_ref'])
                contract=source_member_contract(record.envelope)
                artifact=next(a for a in record.envelope.artifacts if a.artifact_ref==ref)
                with xr.open_dataset(context.resources.scientific_workspace.artifacts.resolve(artifact)) as data:
                    validate_member_payload(data,contract,check_values=False)
                accepted[ref]={**entry,'forecast_contract':contract}
            except (KeyError,ValueError,TypeError,OSError) as error:
                rejected.append({'artifact_ref':ref,'reason':str(error)[:200]})
        return accepted,rejected

    def input_support(self,context):
        candidates,rejected=self._inputs(context)
        return {'status':'ready' if candidates else 'missing_compatible_input','required_input':'Current authorized aligned forecast member artifacts.',
            'candidates':[{'artifact_ref':ref,'evidence_ref':entry['input_evidence_ref'],
                'candidate_ref':entry['forecast_contract']['candidate_ref'],'target_ref':entry['forecast_contract']['target']['target_ref'],
                'member_refs':[m['member_ref'] for m in entry['forecast_contract']['members']]} for ref,entry in candidates.items()],
            'reasons':rejected[:4]}

    def prepare(self,bound_arguments,binding,context):
        if bound_arguments or binding.descriptor_identity!=self.descriptor.descriptor_identity:
            return ()
        candidates,_=self._inputs(context,binding)
        if not candidates:
            return ()
        all_fields,_=field_inputs(context,binding)
        references={}
        for ref,entry in all_fields.items():
            envelope=context.resources.ledger.get(entry['input_evidence_ref']).envelope
            if envelope.producer_id=='manifest-evidence' and envelope.evidence_role=='analysis':
                references[ref]={**entry,'envelope':to_primitive(envelope)}
        ref={'type':'string','minLength':1,'maxLength':300}
        schema={'type':'object','properties':{'target_ref':ref,
            'components':{'type':'array','minItems':1,'maxItems':128,'items':{'type':'object','properties':{
                'input_ref':{'enum':list(candidates)},'evidence_ref':ref,'member_ref':ref,'weight':{'type':'number','minimum':0}},
                'required':['input_ref','evidence_ref'],'additionalProperties':False}},
            'weight_mode':{'enum':['explicit','equal_components','source_declared']},
            'outputs':{'type':'array','minItems':1,'uniqueItems':True,'items':{'enum':['distribution','mean','probability','quantiles']}},
            'operator_profile_ref':{'const':PROFILE},'threshold_refs':{'type':'array','uniqueItems':True,'items':ref},
            'reference_inputs':{'type':'object','additionalProperties':{'type':'object','properties':{'input_ref':ref,'evidence_ref':ref},'required':['input_ref','evidence_ref'],'additionalProperties':False}},
            'quantile_levels':{'type':'array','maxItems':128,'uniqueItems':True,'items':{'type':'number','minimum':0,'maximum':1}}},
            'required':['target_ref','components','weight_mode','outputs','operator_profile_ref'],'additionalProperties':False}
        return (ActionOwnerPreparation(identity_inputs={'binding':binding.binding_identity,'candidates':candidates},arguments_schema=schema,
            frozen_arguments={'candidates':candidates,'references':references,'targets':context.task.disclosed_context.get('target_contracts',{})},
            input_artifact_refs=tuple(dict.fromkeys((*candidates,*references))),selected_artifact_refs_path=('request','input_artifact_refs'),owner_summary=DESCRIPTION),)

    def canonicalize(self,arguments,preparation,binding):
        try:
            from .submission_contract import schema_validator
            error=next(schema_validator(preparation.arguments_schema).iter_errors(arguments),None)
            if error:
                raise ValueError(error.message)
            target,_,_,_,_=canonical_components(arguments,preparation.frozen_arguments['candidates'])
            thresholds=arguments.get('threshold_refs',[])
            if ('probability' in arguments['outputs'])!=bool(thresholds):
                raise ValueError('probability requires explicit published threshold refs')
            if ('quantiles' in arguments['outputs'])!=bool(arguments.get('quantile_levels',[])):
                raise ValueError('quantiles require explicit levels')
            settings=preparation.frozen_arguments['targets'].get(target.target_ref,{})
            if any(ref not in settings.get('thresholds',{}) for ref in thresholds):
                raise ValueError('threshold profile unavailable for this target')
            from .forecast_event_references import reference_profile,reference_provenance
            required=set()
            for threshold in thresholds:
                profile=settings['thresholds'][threshold];object_ref=reference_profile(profile,target.unit)
                if object_ref is None:continue
                required.add(object_ref);choice=arguments.get('reference_inputs',{}).get(object_ref,{})
                entry=preparation.frozen_arguments['references'].get(choice.get('input_ref'))
                if entry is None or entry['input_evidence_ref']!=choice.get('evidence_ref'):raise ValueError('select an available public reference slice')
                reference_provenance(entry['envelope'],profile)
            if set(arguments.get('reference_inputs',{}))!=required:raise ValueError('reference selection differs from requested events')
            inputs=[c['input_ref'] for c in arguments['components']]+[c['input_ref'] for c in arguments.get('reference_inputs',{}).values()]
            return {'request':{'selection':to_primitive(arguments),'input_artifact_refs':list(dict.fromkeys(inputs)),
                'target_identity':target.identity,'target_settings':to_primitive(settings),'binding_identity':binding.binding_identity}}
        except (KeyError,ValueError,TypeError) as error:
            raise ArgumentValidationError('invalid_forecast_selection','arguments',str(error)) from error

    def binding_error(self,action,preparation,binding,state,context):
        try:
            if to_primitive(action.arguments)!=self.canonicalize(action.arguments['request']['selection'],preparation,binding):
                return 'forecast selection differs from its frozen owner request'
            if binding.resolved_scope_identity!=state.task.resolved_scope_identity or binding.decision_time!=state.task.decision_time:
                return 'forecast authority changed'
            from .field_inputs import current_field_error
            components=action.arguments['request']['selection']['components']
            for component in components:
                record=context.runtime_documents['evidence_records'][component['evidence_ref']]
                contract=source_member_contract(record.get('envelope',record))
                if to_primitive(contract)!=to_primitive(preparation.frozen_arguments['candidates'][component['input_ref']]['forecast_contract']):
                    return 'forecast input member contract changed'
            inputs=[*components,*action.arguments['request']['selection'].get('reference_inputs',{}).values()]
            return current_field_error({'input_artifact_refs':[c['input_ref'] for c in inputs],
                'input_evidence_refs':[c['evidence_ref'] for c in inputs]},state,context)
        except (KeyError,ValueError,TypeError):
            return 'invalid forecast binding'
        return None

    def project_result(self,result):
        return result.value['summary'] if result.status is ToolStatus.SUCCESS else {'status':result.status.value,'reason':result.message}


class ForecastCombineAdapter:
    requires_bounded_cpu=True

    def execute(self,invocation,context):
        try:
            request=invocation.arguments['request'];selection=request['selection']
            authority=context.runtime_documents.get('task_authority',{})
            if invocation.tool_id!=TOOL_ID or invocation.task_identity!=context.task_identity or request['binding_identity']!=authority.get('binding_identity') or invocation.bound_capability_identity!=request['binding_identity']:
                raise ValueError('forecast invocation authority differs')
            if selection['operator_profile_ref']!=PROFILE:
                raise ValueError('unsupported distribution profile')
            settings=context.runtime_documents.get('forecast_contracts',{}).get(selection['target_ref'],{})
            if to_primitive(settings)!=request['target_settings']:
                raise ValueError('target profile differs from current public target')
            contracts={};datasets={};artifacts={};estimate=0
            for component in selection['components']:
                ref=component['input_ref'];evidence=component['evidence_ref']
                if not context_evidence_carries_artifact(context,evidence,ref):
                    raise ValueError('current evidence does not carry component artifact')
                record=context.runtime_documents['evidence_records'][evidence]
                contract=source_member_contract(record.get('envelope',record))
                if ref in contracts:
                    continue
                # Check dimensions before the common verified loader materializes arrays.
                with xr.open_dataset(context.artifact_paths[ref]) as opened:
                    estimate+=opened[contract['value_variable']].size*80
                if estimate>512*1024*1024:
                    raise ValueError('resource_limit: select smaller spatial blocks, retaining exact members')
                artifact,data=load_verified_context_field(context,ref)
                validate_member_payload(data,contract)
                contracts[ref]={'input_evidence_ref':evidence,'forecast_contract':contract}
                datasets[ref]=data;artifacts[ref]=artifact
            target,members,weights,selections,kind=canonical_components(selection,contracts)
            if target.identity!=request['target_identity']:
                raise ValueError('target identity differs')
            first=datasets[selections[0]['input_ref']]
            from .forecast_members import field_support_variables
            first_variable=contracts[selections[0]['input_ref']]['forecast_contract']['value_variable']
            support_names=field_support_variables(first,first_variable)
            mapping=first[first_variable].attrs.get('grid_mapping')
            for ref,data in datasets.items():
                variable=contracts[ref]['forecast_contract']['value_variable']
                if field_support_variables(data,variable)!=support_names or data[variable].attrs.get('grid_mapping')!=mapping or any(not first[name].identical(data[name]) for name in support_names):
                    raise ValueError('actual field support differs despite declared target identity')
                for name,coordinate in first.coords.items():
                    if 'member' not in coordinate.dims and (name not in data.coords or not coordinate.identical(data[name])):
                        raise ValueError('actual target coordinates differ despite declared identity')
            values=np.concatenate([datasets[s['input_ref']][contracts[s['input_ref']]['forecast_contract']['value_variable']].values[s['indices']] for s in selections])
            validity=np.concatenate([datasets[s['input_ref']][contracts[s['input_ref']]['forecast_contract']['validity_variable']].values[s['indices']] for s in selections])
            requested=list(selection['outputs']);base_outputs=[v for v in requested if v!='probability']
            result=combine(values,validity,weights,outputs=base_outputs,quantile_levels=selection.get('quantile_levels',()),missing_policy='all_positive_valid') if base_outputs else None
            fields=dict(result.values) if result else {};masks=dict(result.validity) if result else {}
            reference_audits={}
            from .forecast_event_references import prepare_event
            variable=contracts[selections[0]['input_ref']]['forecast_contract']['value_variable']
            for ref in selection.get('threshold_refs',()):
                profile=settings['thresholds'][ref]
                ex,em,threshold,reference_artifact,reference_audit=prepare_event(context,profile,selection,target,first,variable,values,validity)
                probability=combine(values,validity,weights,outputs=['probability'],threshold=threshold,operator=profile['operator'],missing_policy='all_positive_valid',event_values=ex,event_validity=em)
                fields['probability:'+ref]=probability.values['probability'];masks['probability:'+ref]=probability.validity['probability']
                if reference_artifact is not None:
                    artifacts[reference_artifact.artifact_ref]=reference_artifact;reference_audits[ref]=reference_audit
            original=datasets[selections[0]['input_ref']]
            variable=contracts[selections[0]['input_ref']]['forecast_contract']['value_variable']
            dims=original[variable].dims[1:]
            output=xr.Dataset(coords={name:coord for name,coord in original.coords.items() if 'member' not in coord.dims},
                attrs={key:original.attrs[key] for key in ('grid_identity','region_identity','coordinate_reference_system','support_kind') if key in original.attrs})
            for name in support_names:output[name]=original[name]
            output_map={}
            for index,(name,array) in enumerate(fields.items()):
                field=target.variable if name=='mean' else 'forecast_output_'+str(index)
                unit='1' if name.startswith('probability:') else target.unit
                output[field]=(dims,array,{'units':unit,'level':target.level,'temporal_support':target.temporal_kind,**({'grid_mapping':mapping} if mapping else {})})
                output[field+'_validity']=(dims,masks[name])
                output_map[name]={'variable':field,'validity_variable':field+'_validity','unit':unit}
            audit={'target':to_primitive(target),'original_selection':selection,'members':[
                {'member':to_primitive(member),'weight':weight} for member,weight in zip(members,weights,strict=True) if weight>0],
                'ensemble_kind':kind,'operator_profile_ref':PROFILE,'profile_origin':'agent-internal',
                'threshold_profiles':{ref:to_primitive(settings['thresholds'][ref]) for ref in selection.get('threshold_refs',())},
                'event_references':reference_audits,'distribution_inputs':selections,'outputs':output_map,'missing_policy':'all_positive_valid'}
            output.attrs['forecast_result_identity']=deterministic_identity(audit)
            store=ScientificWorkspace(context.run_scope_root,str(context.run_identity)).artifacts
            producer=deterministic_identity(('forecast.combine',PROFILE))
            if fields:
                stored=store.put_xarray(output,producer_identity=producer,parent_refs=tuple(a.sha256 for a in artifacts.values()))
            else:
                stored=store.put_json(audit,producer_identity=producer,payload_kind='forecast_distribution_v1',
                    schema_identity=deterministic_identity('forecast_distribution_v1'),summary={'target_ref':target.target_ref,'kind':'member_distribution'},
                    parent_refs=tuple(a.sha256 for a in artifacts.values()))
            evidence_refs=tuple(dict.fromkeys([c['evidence_ref'] for c in selection['components']]+[a['evidence_ref'] for a in reference_audits.values()]))
            available,retrieved=derived_evidence_times(context.runtime_documents.get('evidence_records',{}),evidence_refs)
            support=field_support(original)
            envelope=EvidenceEnvelope(producer_kind='forecast_computation',producer_id='forecast.combine',producer_version='1.0.0',
                evidence_role='analysis',query_identity=deterministic_identity(request),binding_identity=str(invocation.bound_capability_identity),
                invocation_identity=invocation.invocation_identity,observed_time=None,reference_time=None,issue_time=None,
                available_time=available,retrieved_time=retrieved,valid_start=target.supports[0][0],valid_end=target.supports[-1][1],
                lead_hours=(),spatial=support['spatial'],grid={'identity':target.grid_identity,'coordinate_reference_system':support['coordinate_reference_system']},
                variables=tuple(v['variable'] for v in output_map.values()) or (target.variable,),
                units={v['variable']:v['unit'] for v in output_map.values()} or {target.variable:target.unit},
                artifacts=(stored.artifact,),provenance=({'forecast_result':audit},),transformations=({'operation':'forecast.combine'},),lineage_refs=evidence_refs)
            return ToolResult(ToolStatus.SUCCESS,TOOL_ID,value={'artifact':stored.artifact,'payload_ref':stored.payload,'envelope':envelope,
                'summary':{'target_ref':target.target_ref,'ensemble_kind':kind,'outputs':list(output_map),'distribution_available':'distribution' in requested,
                    'result_identity':output.attrs['forecast_result_identity']}},origin=TOOL_ID,invocation_identity=invocation.invocation_identity,artifact_identity=stored.artifact.artifact_ref)
        except (ValueError,KeyError,TypeError,OSError) as error:
            return ToolResult(ToolStatus.EXECUTION_ERROR,TOOL_ID,message='forecast_combine_rejected: '+str(error)[:250],origin=TOOL_ID,invocation_identity=invocation.invocation_identity)

    def restart(self,context):
        return None


def combine_registration(descriptor):
    return ActionOperationRegistration(semantic_key='action:forecast:combine',capability_ref=descriptor.capability_ref,
        action_id='forecast.combine',purpose=DESCRIPTION,expected_observation='A computed forecast artifact with exact member selection and lineage.',owner=ForecastCombineOwner(descriptor))
