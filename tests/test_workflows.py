import copy
import json
from pathlib import Path
import pytest
import torch

from causalquest.framework.artifacts import read, save, digest
from causalquest.framework.config import DEFAULTS, load_config
from causalquest.framework.workflows import sft, sample, rl
from causalquest.framework.reviews import validate, toy_reviews, evaluate
from causalquest.framework.cli import demo
from causalquest.framework.backends import create_backend, fingerprint


@pytest.fixture
def prepared(tmp_path):
    cfg = {**DEFAULTS, 'max_steps': 1, 'batch_rows': 6, 'learning_rate': 0.02}
    sft(cfg, tmp_path/'sft')
    actor = {**cfg, 'adapter': str(tmp_path/'sft/model')}
    sample(actor, tmp_path/'run', 'train')
    return tmp_path, actor


def test_complete_cli_workflow_changes_real_tensors_and_retains_denominator(tmp_path):
    result = demo(tmp_path/'demo')
    assert result['test_fixture_only'] and result['sft_updates'] == 1
    assert result['rl_status'] == 'one_group_relative_update_saved_reloaded'
    assert result['before']['episodes'] == result['after']['episodes'] == 8
    before = read(tmp_path/'demo/sft/checkpoint.json')
    after = read(tmp_path/'demo/rl/checkpoint.json')
    assert before['parameter_sha256'] != after['parameter_sha256']
    assert read(tmp_path/'demo/rl/reload.json')['fresh_reload_verified']


def test_pending_or_changed_review_cannot_update_or_create_output(prepared):
    root, cfg = prepared
    template = read(root/'run/review_template.json')
    save(root/'pending.json', template)
    with pytest.raises(ValueError, match='pending'):
        validate(root/'run', template)
    assert not (root/'rl').exists()
    template['run_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='bind'):
        validate(root/'run', template)


def test_missing_duplicate_or_invented_evidence_rejected(prepared):
    root, cfg = prepared
    toy_reviews(root/'run')
    original = read(root/'run/reviews.json')
    missing = copy.deepcopy(original); missing['items'].pop()
    with pytest.raises(ValueError, match='population'):
        validate(root/'run', missing)
    duplicate = copy.deepcopy(original); duplicate['items'][1] = duplicate['items'][0]
    with pytest.raises(ValueError, match='population'):
        validate(root/'run', duplicate)
    fake = copy.deepcopy(original); fake['items'][0]['evidence_ids'] = ['ev_never_observed']
    with pytest.raises(ValueError, match='observations'):
        validate(root/'run', fake)
    bad = copy.deepcopy(original); bad['items'][0]['scores']['validity'] = 0
    with pytest.raises(ValueError, match='inapplicable'):
        validate(root/'run', bad)


def test_stale_actor_and_validation_rollouts_rejected(prepared):
    root, cfg = prepared; toy_reviews(root/'run')
    changed = {**cfg, 'adapter': None, 'reference': cfg['adapter']}
    with pytest.raises(ValueError, match='stale'):
        rl(changed, root/'run', root/'stale')
    run = read(root/'run/run.json')
    for e in run['episodes']: e['split'] = 'validation'
    # Re-exporting a malicious but internally consistent validation file must still not authorize training.
    other = root/'validation'; other.mkdir()
    save(other/'run.json', run)
    from causalquest.framework.reviews import export_reviews
    export_reviews(other); toy_reviews(other)
    with pytest.raises(ValueError, match='validation/test'):
        rl({**cfg, 'reference': cfg['adapter']}, other, root/'forbidden')
    assert not (root/'forbidden').exists()


def test_update_not_repeated_and_reference_checkpoint_preserved(prepared):
    root, cfg = prepared; toy_reviews(root/'run')
    before = (root/'sft/model/model.pt').read_bytes()
    rl({**cfg, 'reference': cfg['adapter']}, root/'run', root/'rl')
    with pytest.raises(FileExistsError):
        rl({**cfg, 'reference': cfg['adapter']}, root/'run', root/'rl')
    assert before == (root/'sft/model/model.pt').read_bytes()


def test_config_paths_are_relative_to_config_not_working_directory(tmp_path):
    cfg_path = tmp_path/'config.json'
    save(cfg_path, {'cases': 'local/catalog.json'})
    assert load_config(cfg_path)['cases'] == str((tmp_path/'local/catalog.json').resolve())
    cfg_path2 = tmp_path/'bad.json'; save(cfg_path2, {'draws_per_case': True})
    with pytest.raises(ValueError): load_config(cfg_path2)


def test_source_group_split_overlap_rejected(tmp_path):
    from causalquest.framework.environments import load_cases
    path = tmp_path/'cases.json'
    save(path, [{'case_id': 'a', 'source_group': 'same', 'split': 'train', 'task_type': 'toy_decision', 'margin': 1},
                {'case_id': 'b', 'source_group': 'same', 'split': 'test', 'task_type': 'toy_decision', 'margin': -1}])
    with pytest.raises(ValueError, match='crosses'):
        load_cases({**DEFAULTS, 'cases': str(path)})

