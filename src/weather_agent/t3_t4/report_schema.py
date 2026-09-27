"""Compact public final contract; no dynamic reference enumeration."""
from .reports import report_template

from .public_copy import copy_text

REF_GUIDANCE=copy_text('final','reference_guidance')

def report_schema(view, contract):
    if report_template(contract).get('probability_adjustments')!=[]:
        raise ValueError('unsupported probability adjustment public contract')
    text={'type':'string','minLength':1,'pattern':r'\S'}
    impact={'type':'null'};focus={'type':'null'}
    if view['track']=='T4':
        names=('hazard_and_evolution','area_and_critical_window','severity_probability_uncertainty',
            'public_objects_and_qualitative_exposure','threshold_interpretation','potential_impacts')
        impact={'type':'object','properties':{k:dict(text) for k in names},'required':list(names),'additionalProperties':False}
        subtype=view['profile_or_subtype'].removeprefix('T4-')
        if subtype not in ('I','W','S','P'):raise ValueError('unsupported report subtype')
        focus={'type':'object','properties':{'subtype':{'const':subtype},'content':{'type':'null'} if subtype=='I' else dict(text,description=copy_text('final','partner_focus_description')
                if report_template(contract)['focus']['content']=='nonempty partner decision briefing' else 'English text.')},
            'required':['subtype','content'],'additionalProperties':False}
    properties={'report_schema_version':{'const':'report-object-v1'},'report_text':dict(text,description=copy_text('final','report_text_description')),
        'impact_basis':impact,'focus':focus,'probability_adjustments':{'type':'array','maxItems':0}}
    summary={'type':'object','properties':{'query_id':text,'result_sha256':{'type':'string','pattern':'^[0-9a-f]{64}$'},'used_for':text},
        'required':['query_id','result_sha256','used_for'],'additionalProperties':False}
    return {'type':'object','x-report-format':'report-object-v1','x-reference-selection':'exact-node-labels-v1',
        'description':REF_GUIDANCE+' '+copy_text('final','shape_guidance'),'properties':{
        'report':{'type':'object','properties':properties,'required':list(properties),'additionalProperties':False},
        'evidence_refs':{'type':'array','items':{'type':'string'},'uniqueItems':True,'description':copy_text('final','evidence_refs_description')},
        'source_use_summary':{'type':'array','minItems':1,'items':summary,'description':copy_text('final','source_use_summary_description')},
        'knowledge_refs':{'type':'array','items':{'type':'string'},'uniqueItems':True,'maxItems':32,'description':copy_text('final','knowledge_refs_description')}},
        'required':['report','evidence_refs'],'additionalProperties':False}
