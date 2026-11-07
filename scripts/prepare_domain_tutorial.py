"""Build three authored TRAIN cases and replay 13 demonstration actions; no model calls."""
import copy
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from causalquest.framework.artifacts import read, require, reserve, save, write_jsonl, digest
from causalquest.framework.config import DEFAULTS
from causalquest.framework.environments import CausalQuestEnvironment



def build(root, output):
    from causalquest.tasks.design_case_builder import build_design_case
    from causalquest.tasks.health_case_builder import build_health_case
    from causalquest.tasks.post_experiment_case_builder import build_post_experiment_case
    from causalquest.data.controlled_lab import ControlledLabConfig, generate_clean_experiment
    from causalquest.data.faults import FaultSpec, FaultType
    output = reserve(output)
    cases = output / 'cases'
    raw = read(root / 'examples/tutorial_design.json')
    raw.update(case_id='accept_design', source_group_id='accept_design_source')
    raw['design'].update(industrial_review=True, daily_eligible_units=1000, max_duration_days=14)
    built = build_design_case(raw, cases)
    catalog = [entry(raw['case_id'], raw['source_group_id'], 'experiment_design_review', built)]
    clean = generate_clean_experiment(ControlledLabConfig(
        experiment_id='accept_health', template_id='feed_card_layout', seed=71, population_size=100,
        start_time_utc='2026-08-01T00:00:00Z', duration_days=14, randomization_unit='user',
        analysis_unit='user', control_variant='control', treatment_variant='compact_card',
        treatment_share=.5, exposure_rate=.8, baseline_ctr=.12, treatment_ctr_absolute_lift=.015,
        baseline_conversion_rate=.035, treatment_conversion_absolute_lift=.003,
        baseline_crash_rate=.01, treatment_crash_absolute_lift=0,
        baseline_latency_ms=220, treatment_latency_delta_ms=5))
    fault = FaultSpec(fault_id='authored_click_loss', fault_type=FaultType.TELEMETRY_MISSINGNESS,
        seed=73, fraction=.75, target_variant='compact_card', target_platform='android',
        target_app_version='9.3.0', target_event_name='click_observation')
    health = dict(case_id='accept_health', title='Authored health acceptance',
        business_context='Investigate observed experiment logging before interpreting effects.',
        fault_id=fault.fault_id, monitored_metric='click_through_rate',
        template_family_id='accept_health', source_group_id='accept_health_source')
    built = build_health_case(health, clean, {fault.fault_id: fault}, cases)
    catalog.append(entry(health['case_id'], health['source_group_id'], 'experiment_health_diagnosis', built))
    source = output / 'authored_source.duckdb'
    _source_database(source)
    post = _raw_case()
    post.update(case_id='accept_post', source_group_id='accept_post_source')
    built = build_post_experiment_case(post, source, cases)
    catalog.append(entry(post['case_id'], post['source_group_id'], 'post_experiment_decision', built))
    save(output / 'case_catalog.json', catalog)
    cfg = {**DEFAULTS, 'metric_catalog': str(root / 'examples/metric_catalog.yaml')}
    rows = []
    for case in catalog:
        actual = {**case, 'case_dir': str(output / case['case_dir'])}
        env = CausalQuestEnvironment(actual, cfg)
        try:
            # These explicit tutorial demonstrations are never injected into student rollouts.
            replay(rows, env, 'get_experiment', {})
            document = {'experiment_design_review': 'experiment_brief',
                        'experiment_health_diagnosis': 'incident_ticket',
                        'post_experiment_decision': 'decision_brief'}[case['task_type']]
            replay(rows, env, 'get_document', {'document_id': document})
            if case['task_type'] != 'experiment_design_review':
                continue
            replay(rows, env, 'get_document', {'document_id': 'metric_plan'})
            replay(rows, env, 'get_metric_definition', {'metric_id': 'click_through_rate'})
            replay(rows, env, 'get_metric_definition', {'metric_id': 'crash_free_session_rate'})
            replay(rows, env, 'estimate_power', {})
            replay(rows, env, 'get_document', {'document_id': 'decision_charter'})
            ids = [e['observation']['evidence_id'] for e in env.events]
            findings = ['The guardrail has no registered numeric threshold.',
                        'The session guardrail lacks aggregation to the user analysis unit.',
                        'The multi-metric decision policy is not registered.']
            replay(rows, env, 'cq_update_investigation_state', {
                'hypothesis_id': 'registered_design', 'status': 'supported',
                'summary': 'Revise the guardrail and analysis contract before launch: ' + ' '.join(findings),
                'evidence_ids': ids})
            replay(rows, env, 'cq_submit_review', {
                'task_type': 'experiment_design_review', 'decision': 'revise_design',
                'confidence': 'high', 'estimand': 'triggered_population_ate',
                'validated_findings': findings, 'blocking_issues': findings,
                'guardrail_findings': [findings[0]],
                'uncertainties': ['No realized treatment effect is estimated by this pre-launch review.'],
                'next_actions': ['Register a numeric guardrail threshold and session-to-user aggregation.',
                                 'Register the multi-metric decision policy, then obtain human review.'],
                'claim_evidence': [{'claim': f, 'evidence_ids': ids} for f in findings],
                'requires_human_approval': True})
            require(env.submitted, 'authored design tail did not submit')
        finally:
            env.close()
    write_jsonl(output / 'train_steps.jsonl', rows)
    summary = {'status': 'authored_TRAIN_inputs_replayed', 'case_count': len(catalog),
        'row_count': len(rows), 'dataset_sha256': digest(rows), 'protocol': DEFAULTS['protocol'],
        'source': 'authored synthetic fixtures', 'original_research_data_read': False,
        'oracle_read_for_targets': False, 'student_starts_from_normal_initial_state': True,
        'demonstration_scope': 'tool acquisition plus one documented design-revision tail; integration acceptance'}
    save(output / 'summary.json', summary)
    return summary


def entry(case_id, source_group, task, built):
    return {'case_id': case_id, 'source_group': source_group, 'split': 'train',
            'task_type': task, 'case_dir': 'cases/' + case_id,
            'case_snapshot_id': built['case_snapshot_id'], 'suite_snapshot_id': 'acceptance-v1-authored'}


def replay(rows, env, name, arguments):
    request = env.request()
    available = {t['function']['name'] for t in request['tools']}
    require(name in available, 'demonstration action is unavailable: ' + name)
    action = {'name': name, 'arguments': arguments}
    env.step(json.dumps(action, ensure_ascii=False))
    require(env.events[-1]['outcome'] == 'ok', 'demonstration failed: ' + str(env.events[-1]))
    rows.append({'id': env.case['case_id'] + ':' + str(len(env.events)),
        'case_id': env.case['case_id'], 'source_group': env.case['source_group'], 'split': 'train',
        'protocol': DEFAULTS['protocol'], 'review_status': 'accepted',
        'review_basis': 'authored public facts and successful replay; not a population-level quality label',
        'messages': copy.deepcopy(request['messages']) + [{'role': 'assistant', 'content': None,
                    'tool_calls': [{'type': 'function', 'function': action}]}],
        'tools': copy.deepcopy(request['tools']), 'target_kind': env.events[-1]['action']['kind'],
        'successor_observation_sha256': digest(env.events[-1]['observation'])})


import duckdb


def _source_database(path):
    connection = duckdb.connect(str(path))
    try:
        connection.execute(
            "CREATE TABLE snapshot_metadata AS SELECT 'upworthy-canonical-v1' schema_version, "
            "'source-snapshot' snapshot_id"
        )
        connection.execute(
            """
            CREATE TABLE upworthy_tests AS SELECT
                'confirmatory'::VARCHAR source_partition,
                'test-1'::VARCHAR test_id,
                'component-1'::VARCHAR component_id,
                'candidate_train'::VARCHAR assigned_pool,
                false::BOOLEAN randomization_risk,
                NULL::VARCHAR quarantine_reasons,
                true::BOOLEAN schema_eligible,
                true::BOOLEAN eligible_for_positive_causal_case
            """
        )
        connection.execute(
            """
            CREATE TABLE upworthy_arms AS
            SELECT * FROM (VALUES
              ('confirmatory', 'test-1', 'ref', 1, 'reference headline', 'ref-slug',
               'image-a', 100000, 10000,
               0.10, '0.0', false, false),
              ('confirmatory', 'test-1', 'candidate', 2, 'candidate headline', 'candidate-slug',
               'image-b', 100000, 14000,
               0.14, '999.0', true, true)
            ) AS t(source_partition, test_id, arm_id, source_row_index, headline, slug,
                   eyecatcher_id, impressions, clicks, ctr, source_significance_undocumented,
                   editor_first_place, editor_winner)
            """
        )
        connection.execute("CHECKPOINT")
    finally:
        connection.close()

def _raw_case():
    return {
        "case_id": "post_test_001",
        "test_id": "test-1",
        "title": "标题方案实验后评审",
        "business_context": "编辑团队需要判断新标题是否达到登记收益。",
        "scenario_family": "content_package_selection",
        "template_family_id": "post_clear_practical_win",
        "decision_policy": {
            "reference_arm_id": "ref",
            "candidate_arm_id": "candidate",
            "mde_absolute": 0.01,
            "alpha": 0.05,
            "target_power": 0.80,
        },
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(Path(__file__).resolve().parents[1], args.output.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
