import json
from pathlib import Path
import pytest
from causalquest.framework.artifacts import read
from causalquest.framework.config import DEFAULTS
from causalquest.framework.environments import CausalQuestEnvironment
from causalquest.tasks.design_case_builder import build_design_case

ROOT=Path(__file__).parents[1]


@pytest.fixture
def domain(tmp_path):
    raw=read(ROOT/'examples/tutorial_design.json')
    built=build_design_case(raw,tmp_path)
    case={'case_id':raw['case_id'],'split':'train','source_group':raw['source_group_id'],
          'task_type':'experiment_design_review','case_dir':str(tmp_path/raw['case_id']),
          'case_snapshot_id':built['case_snapshot_id'],'suite_snapshot_id':'tutorial-v1'}
    cfg={**DEFAULTS,'metric_catalog':str(ROOT/'examples/metric_catalog.yaml')}
    return CausalQuestEnvironment(case,cfg)


def call(env,name,arguments):
    env.step(json.dumps({'name':name,'arguments':arguments}))


def test_real_environment_exposes_all_five_calculations_and_keeps_agent_control(domain):
    request=domain.request();names={t['function']['name'] for t in request['tools']}
    assert {'calculate','calculate_change','convert_rate','compare_numbers','compare_interval'}<=names
    assert 'cq_submit_review' not in names
    assert 'oracle' not in json.dumps(request) and 'case_dir' not in json.dumps(request)
    call(domain,'get_experiment',{})
    call(domain,'compare_interval',{'lower':0.01,'upper':0.02,'threshold':0.015})
    assert domain.events[-1]['outcome']=='ok'
    assert len(domain.session.environment.trace)==2
    # Arithmetic does not select an action, register a finding or create investigation state.
    assert domain.report is None and domain.session.state_snapshot()=={}
    ev=domain.events[0]['observation']['evidence_id']
    call(domain,'cq_update_investigation_state',{'hypothesis_id':'readiness','status':'open',
         'summary':'Evidence has been read; further checks remain.','evidence_ids':[ev]})
    assert domain.events[-1]['outcome']=='ok'
    assert 'cq_submit_review' in {t['function']['name'] for t in domain.request()['tools']}
    call(domain,'cq_submit_review',{'task_type':'experiment_design_review','decision':'insufficient_design_evidence',
         'confidence':'low','estimand':'triggered_population_ate','validated_findings':['Experiment configuration was read.'],
         'blocking_issues':[],'guardrail_findings':[],'uncertainties':['Tutorial has not completed a semantic review.'],
         'next_actions':['Complete the evidence review.'],'claim_evidence':[{'claim':'Experiment configuration was read.',
         'evidence_ids':[ev]}],'requires_human_approval':True})
    assert domain.submitted and domain.done and domain.termination=='final_report'
    domain.close()


def test_hidden_document_and_invented_state_reference_rejected(domain):
    call(domain,'get_document',{'document_id':'../hidden/oracle.json'})
    assert domain.events[-1]['outcome']=='error'
    call(domain,'cq_update_investigation_state',{'hypothesis_id':'readiness','status':'supported',
         'summary':'unsupported','evidence_ids':['ev_not_observed']})
    assert domain.events[-1]['outcome']=='error' and domain.session.state_snapshot()=={}
    domain.close()


def test_calculator_budget_and_nonfinite_input_use_real_environment_boundary(domain):
    domain.session.environment.max_tool_calls=1
    call(domain,'calculate',{'operation':'add','left':True,'right':1})
    assert domain.events[-1]['outcome']=='error'
    call(domain,'calculate',{'operation':'add','left':1,'right':1})
    assert domain.events[-1]['outcome']=='ok'
    call(domain,'compare_numbers',{'left':1,'right':2})
    assert domain.done and domain.termination=='tool_budget_exhausted'
    domain.close()


def test_completed_experiment_adapter_uses_real_readonly_tools_and_all_numeric_interfaces(tmp_path):
    from domain_fixtures import _source_database, _raw_case
    from causalquest.tasks.post_experiment_case_builder import build_post_experiment_case
    source=tmp_path/'authored_source.duckdb';_source_database(source)
    raw=_raw_case();built=build_post_experiment_case(raw,source,tmp_path/'cases')
    case={'case_id':raw['case_id'],'split':'train','source_group':'tutorial_post',
          'task_type':'post_experiment_decision','case_dir':str(tmp_path/'cases'/raw['case_id']),
          'case_snapshot_id':built['case_snapshot_id'],'suite_snapshot_id':'tutorial-post-v1'}
    env=CausalQuestEnvironment(case,{**DEFAULTS,'metric_catalog':str(ROOT/'examples/metric_catalog.yaml')})
    call(env,'get_experiment',{});call(env,'get_arm_results',{})
    call(env,'estimate_ab_effect',{'candidate_arm_id':'candidate','reference_arm_id':'ref'})
    assert all(e['outcome']=='ok' for e in env.events)
    call(env,'compare_numbers',{'left':0.04,'right':0.01})
    assert env.events[-1]['observation']['result']['relation']=='>'
    assert not env.submitted and env.session.state_snapshot()=={}
    assert 'eligible_for_positive_causal_case' not in str(env.request().get('hidden', ''))
    env.close()


def test_health_adapter_keeps_fault_reference_out_of_model_request(tmp_path):
    from causalquest.data.controlled_lab import ControlledLabConfig,generate_clean_experiment
    from causalquest.data.faults import FaultSpec,FaultType
    from causalquest.tasks.health_case_builder import build_health_case
    clean=generate_clean_experiment(ControlledLabConfig(experiment_id='tutorial_health',template_id='feed_card_layout',
          seed=71,population_size=100,start_time_utc='2026-08-01T00:00:00Z',duration_days=14,
          randomization_unit='user',analysis_unit='user',control_variant='control',treatment_variant='compact_card',
          treatment_share=.5,exposure_rate=.8,baseline_ctr=.12,treatment_ctr_absolute_lift=.015,
          baseline_conversion_rate=.035,treatment_conversion_absolute_lift=.003,baseline_crash_rate=.01,
          treatment_crash_absolute_lift=0,baseline_latency_ms=220,treatment_latency_delta_ms=5))
    fault=FaultSpec(fault_id='tutorial_click_loss',fault_type=FaultType.TELEMETRY_MISSINGNESS,seed=73,fraction=.75,
                    target_variant='compact_card',target_platform='android',target_app_version='9.3.0',target_event_name='click_observation')
    raw={'case_id':'tutorial_health_case','title':'Authored health tutorial','business_context':'Investigate observed logging.',
         'fault_id':fault.fault_id,'monitored_metric':'click_through_rate','template_family_id':'tutorial_health',
         'source_group_id':'tutorial_health_source'}
    built=build_health_case(raw,clean,{fault.fault_id:fault},tmp_path/'cases')
    case={'case_id':raw['case_id'],'split':'train','source_group':'tutorial_health_source','task_type':'experiment_health_diagnosis',
          'case_dir':str(tmp_path/'cases'/raw['case_id']),'case_snapshot_id':built['case_snapshot_id'],'suite_snapshot_id':'tutorial-health-v1'}
    env=CausalQuestEnvironment(case,{**DEFAULTS,'metric_catalog':str(ROOT/'examples/metric_catalog.yaml')})
    request=json.dumps(env.request())
    assert fault.fault_id not in request and 'hidden/' not in request
    call(env,'get_experiment',{});call(env,'check_sample_ratio',{});call(env,'check_metric_completeness',{'metric_id':'click_through_rate'})
    assert all(e['outcome']=='ok' for e in env.events)
    env.close()

