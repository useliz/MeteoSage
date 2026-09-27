"""Explicit ReportObject branch of the existing final provider."""
from .public_copy import copy_text
from pathlib import Path
from .._serialization import to_primitive
from ..executable_operations import (ChoiceSpec,ExecutableOperation,MappingArgumentContract,
    ArgumentValidationError,OperationObservation,TransitionResult)
from ..runtime_contracts import TypedOutput,EvidenceSufficiency
from ..reliability import deterministic_identity
from .reports import seal_report, output_contract


def offer_report(context):
    from ..c1_final import eligible_evidence_refs
    node=getattr(context.resources,'t3_t4_node',None)
    if node is None or getattr(context.resources,'report_delivery',None) is not None:return ()
    # Inner ReportObject validity is deliberately checked after the first final is sealed.
    from .report_schema import report_schema
    schema=report_schema(node.public_view,output_contract(node))
    def validate(arguments):
        if 'report' not in arguments:raise ArgumentValidationError('missing_report','report','Submit the complete ReportObject.')
        from .report_intent import original_value
        return original_value(arguments)
    def invoke(execution,arguments):
        from .provider_audit import response_reference
        attempts=execution.resources.state.llm_attempt_refs
        raw=response_reference(getattr(execution.resources,'provider_audit_root',None),attempts[-1].response_digest if attempts else None,
            attempt_ref=attempts[-1].attempt_ref if attempts else None)
        delivery=seal_report(arguments,node,execution.resources.scientific_workspace.artifacts.run_root/'report-delivery',
                             eligible_refs=eligible_evidence_refs(execution),raw_response_ref=raw,resources=execution.resources,selection_version='exact-node-labels-v1')
        execution.resources.report_delivery=delivery
        valid=all(delivery[key]['status']=='valid' for key in ('local_validation','envelope_validation','reference_validation','appendix_validation','knowledge_validation'))
        if not valid:
            return TransitionResult(OperationObservation(status='failed',code=delivery['terminal_code'],summary='Original ReportObject sealed; no automatic rewrite.'))
        output=TypedOutput(kind=execution.task.answer_contract.output_kind,payload={'report':arguments['report'],'knowledge_citations':delivery['knowledge_citations'],
            'citation_manifest':{'path':delivery['citation_manifest_ref'],'sha256':delivery['citation_manifest_sha256']},
            'presentation':dict(delivery['presentation'],status=delivery['presentation_validation']['status'])},
            artifact_refs=(),claim_refs=(),uncertainty={},evidence_sufficiency=EvidenceSufficiency('unrated',tuple(delivery['evidence_refs'])),
            provenance_refs=tuple(delivery['evidence_refs']),report_ref=delivery['report_ref'])
        execution.resources.state.complete(output)
        return TransitionResult(OperationObservation(status='succeeded',code='report_object_sealed',
            summary='ReportObject sealed; host acceptance remains pending.'),terminal_output=output)
    purpose=copy_text('final','purpose')
    if getattr(context.resources,'evaluation_final_policy','first-explicit-final-v1')=='committed-after-memory-v1':
        from ..trajectory_memory_consumption import prompt
        purpose=prompt('report-final-purpose')
    return (ExecutableOperation(public=ChoiceSpec(semantic_key='final:submit-answer',
        purpose=purpose,
        arguments_schema=schema,expected_observation='Immutable report and separate local/host validation.'),
        argument_contract=MappingArgumentContract(schema,validate),invoke=invoke,
        audit_identity=deterministic_identity(('report-object-final-exact-labels-v1',context.task.task_identity)),action_budget_cost=0),)
