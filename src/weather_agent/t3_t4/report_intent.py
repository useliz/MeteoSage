"""Preserve invalid numeric first-final intent without admitting nonfinite evidence."""
from collections.abc import Mapping
import json
from .._serialization import to_primitive


def original_value(value):
    if isinstance(value,Mapping):return {key:original_value(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [original_value(item) for item in value]
    if value is None or type(value) in (str,int,float,bool):return value
    raise TypeError('report intent must contain JSON values')


def audit_intent(submission):
    original=original_value(submission.arguments)
    raw=json.dumps(original,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=True)
    try:audited=to_primitive(original)
    except ValueError:
        audited={'invalid_numeric_report_intent':True,'original_arguments_json':raw}
    # Structured audit stays canonical. Original invalid numbers remain in the
    # dedicated raw string and subsequently in immutable report/wire archives.
    response=dict(choice=submission.choice,arguments=original)
    from ..episode_runner import _OMITTED_UPDATE
    if submission.record_delta is not None:response['record_delta']=original_value(submission.record_delta)
    if submission.notebook_update is not _OMITTED_UPDATE:response['notebook_update']=original_value(submission.notebook_update)
    return audited,len(raw.encode('utf-8')),len(json.dumps(response,ensure_ascii=False,allow_nan=True).encode('utf-8'))


def parsed_report_final(turn,value):
    if not isinstance(value,dict) or not isinstance(value.get('arguments'),dict) or 'report' not in value['arguments']:
        return False
    return any(choice.choice==value.get('choice') and choice.semantic_key=='final:submit-answer'
        and choice.arguments_schema.get('x-report-format')=='report-object-v1' for choice in getattr(turn,'choices',()))
