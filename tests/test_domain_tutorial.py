"""The portable domain tutorial builds/replays without the historical workspace."""
import importlib.util
from pathlib import Path

from causalquest.framework.artifacts import read,jsonl

ROOT=Path(__file__).resolve().parents[1]


def test_portable_three_case_thirteen_action_tutorial(tmp_path):
    spec=importlib.util.spec_from_file_location('domain_tutorial',ROOT/'scripts/prepare_domain_tutorial.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=module.build(ROOT,tmp_path/'tutorial')
    catalog=read(tmp_path/'tutorial/case_catalog.json');rows=jsonl(tmp_path/'tutorial/train_steps.jsonl')
    assert result['case_count']==3 and result['row_count']==len(rows)==13
    assert result['original_research_data_read'] is False and result['oracle_read_for_targets'] is False
    assert {c['task_type'] for c in catalog}=={'experiment_design_review','experiment_health_diagnosis','post_experiment_decision'}
    assert {kind:sum(r['target_kind']==kind for r in rows) for kind in ['tool_call','state_update','final_report']}=={
        'tool_call':11,'state_update':1,'final_report':1}
    assert all(r['messages'][-1]['role']=='assistant' and r['successor_observation_sha256'] for r in rows)
    for case in catalog:
        first=next(r for r in rows if r['case_id']==case['case_id'])
        assert '"public_history": []' in first['messages'][1]['content']
        assert first['messages'][-1]['tool_calls'][0]['function']['name']=='get_experiment'
    assert rows[-1]['case_id']=='accept_post'
