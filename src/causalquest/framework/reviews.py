"""Human review exchange: frozen packets, explicit pending status, no inferred semantics."""
from .artifacts import read, save, digest, require
from .objectives import DIMENSIONS, APPLICABLE

RUBRIC = {'version': 'cq-seven-dimension-v1', 'dimensions': list(DIMENSIONS),
          'applicable': APPLICABLE, 'scale': {'0': 'central failure', '1': 'supported partial progress', '2': 'fully supported obligations'},
          'core': 'Object, decisive relation, main decision and action must agree with observed evidence.',
          'hard_boundary': 'A correct enum cannot override unsupported effect interpretation or unauthorized action.',
          'format': 'A repaired earlier syntax error does not permanently reduce final format.',
          'scope': 'Human semantic assessment; ID and range checks do not prove entailment.'}
from pathlib import Path
RUBRIC['criteria'] = read(Path(__file__).with_name('rubric.json'))['criteria']


def known_evidence(episode):
    return {value for event in episode['events'] if event.get('outcome') == 'ok'
            for value in [event.get('observation', {}).get('evidence_id') or event.get('evidence_id')]
            if value}


def terminal_review(episode):
    return {'episode_id': episode['episode_id'], 'episode_sha256': digest(episode), 'status': 'terminal',
            'reviewer_id': 'runtime', 'tier': 0, 'core_supported': False, 'hard_violations': [],
            'scores': {d: None for d in DIMENSIONS}, 'reasons': {}, 'evidence_ids': [],
            'note': 'Observed model non-submission: ' + episode['termination_reason']}


def export_reviews(run_dir):
    from pathlib import Path
    run_dir = Path(run_dir); run = read(run_dir/'run.json')
    require(run['status'] == 'complete', 'sampling not complete')
    packet = {'schema': 'cq-review-packet-v1', 'run_sha256': digest(run),
              'actor_sha256': run['actor_sha256'], 'rubric': RUBRIC,
              'episodes': [{k: episode[k] for k in ('episode_id', 'case_id', 'task_type', 'source_group', 'seed',
                           'submitted', 'termination_reason', 'events', 'report')} for episode in run['episodes']]}
    binding = digest(packet)
    template = {'schema': 'cq-review-v1', 'run_sha256': digest(run), 'packet_sha256': binding,
                'rubric_sha256': digest(RUBRIC), 'items': []}
    for episode in run['episodes']:
        if not episode['submitted']:
            item = terminal_review(episode)
        else:
            item = {'episode_id': episode['episode_id'], 'episode_sha256': digest(episode),
                    'status': 'pending', 'reviewer_id': '', 'tier': None, 'core_supported': None,
                    'hard_violations': [], 'scores': {d: None for d in DIMENSIONS},
                    'reasons': {}, 'evidence_ids': [], 'note': ''}
        template['items'].append(item)
    save(run_dir/'review_packet.json', packet)
    save(run_dir/'review_template.json', template)
    return {'episodes': len(template['items']), 'pending': sum(x['status']=='pending' for x in template['items'])}


def validate(run_dir, supplied):
    from pathlib import Path
    run_dir = Path(run_dir)
    run, packet = read(run_dir/'run.json'), read(run_dir/'review_packet.json')
    require(supplied['schema'] == 'cq-review-v1', 'unknown review schema')
    require(supplied['run_sha256'] == digest(run) and supplied['packet_sha256'] == digest(packet)
            and supplied['rubric_sha256'] == digest(RUBRIC), 'review does not bind this run/packet/rubric')
    items = supplied['items']; require(type(items) is list, 'review list required')
    by_id = {item['episode_id']: item for item in items}
    require(len(items) == len(by_id) == len(run['episodes']), 'incomplete or duplicated review population')
    require(set(by_id) == {x['episode_id'] for x in run['episodes']}, 'unexpected review IDs')
    for episode in run['episodes']:
        row = by_id[episode['episode_id']]
        require(row['episode_sha256'] == digest(episode), 'review trajectory changed')
        if not episode['submitted']:
            require(row == terminal_review(episode), 'observed terminal receipt must remain unchanged')
            continue
        require(row['status'] == 'scored', 'pending/unassessed review; no update allowed')
        require(type(row['tier']) is int and row['tier'] in range(4), 'ordinal tier 0..3 required')
        require(type(row['core_supported']) is bool, 'core judgment must be explicit')
        require(type(row['reviewer_id']) is str and row['reviewer_id'].strip() and
                type(row['note']) is str and row['note'].strip(), 'reviewer and explanation required')
        require(set(row['scores']) == set(DIMENSIONS), 'all dimension slots required')
        applicable = APPLICABLE[episode['task_type']]
        for key in DIMENSIONS:
            value = row['scores'][key]
            if key in applicable:
                require(type(value) is int and value in (0, 1, 2), 'applicable score missing/invalid')
                require(type(row['reasons'].get(key)) is str and row['reasons'][key].strip(), 'dimension rationale required')
            else:
                require(value is None, 'inapplicable dimension must stay null')
        require(type(row['hard_violations']) is list and all(type(x) is str and x.strip() for x in row['hard_violations']), 'invalid boundary assessment')
        require(type(row['evidence_ids']) is list and row['evidence_ids'] and
                set(row['evidence_ids']) <= known_evidence(episode), 'review must cite actual prior observations')
        expected = 0 if row['hard_violations'] else (1 if not row['core_supported'] else
                   (3 if all(row['scores'][d] == 2 for d in applicable) else 2))
        require(row['tier'] == expected and (not row['hard_violations'] or not row['core_supported']), 'tier/core/boundary contradiction')
        require(not all(row['scores'][d] == 2 for d in applicable) or row['core_supported'], 'fully supported vector contradicts unsupported core')
    return run, [by_id[x['episode_id']] for x in run['episodes']]


def import_reviews(run_dir, path):
    from pathlib import Path
    supplied = read(path)
    _, items = validate(run_dir, supplied)
    save(Path(run_dir)/'reviews.json', supplied)
    return {'accepted': len(items), 'pending': 0, 'semantic_entailment_checked_by_program': False}


def toy_reviews(run_dir):
    """Deterministic test scorer registered only for the separate toy environment."""
    from pathlib import Path
    run_dir = Path(run_dir)
    run = read(run_dir/'run.json')
    require(run['config']['environment'] == 'toy' and all(x['task_type']=='toy_decision' for x in run['episodes']),
            'toy scoring cannot certify a real experiment task')
    supplied = read(run_dir/'review_template.json')
    for episode, row in zip(run['episodes'], supplied['items']):
        if not episode['submitted']:
            continue
        margin = next(e['observation']['margin'] for e in episode['events'] if 'margin' in e.get('observation', {}))
        correct = episode['report']['decision'] == ('accept' if margin > 0 else 'reject')
        row.update(status='scored', reviewer_id='toy-rules-fixture-only', tier=3 if correct else 1,
                   core_supported=correct, evidence_ids=sorted(known_evidence(episode)), note='Toy public contract checked.')
        row['scores'].update(evidence=2, decision=2 if correct else 0, format=2)
        row['reasons'] = {d: 'Toy fixture check; not experiment semantics.' for d in APPLICABLE['toy_decision']}
    validate(run_dir, supplied)
    save(run_dir/'toy_scored_reviews.json', supplied)
    return import_reviews(run_dir, run_dir/'toy_scored_reviews.json')


def evaluate(run_dir):
    from pathlib import Path
    run_dir = Path(run_dir); run, items = validate(run_dir, read(run_dir/'reviews.json'))
    n = len(items)
    result = {'schema': 'cq-evaluation-v1', 'run_sha256': digest(run), 'review_sha256': digest(items),
              'episodes': n, 'sources': len({x['source_group'] for x in run['episodes']}),
              'submitted': sum(x['submitted'] for x in run['episodes']),
              'core_supported': sum(x['core_supported'] for x in items),
              'complete_success': sum(x['tier']==3 for x in items),
              'hard_violations': sum(bool(x['hard_violations']) for x in items), 'pending': 0,
              'completion_tokens': sum(len(s['completion_ids']) for e in run['episodes'] for s in e['generations']),
              'test_fixture_only': run['config']['environment'] == 'toy',
              'by_task': {}}
    for task in sorted({x['task_type'] for x in run['episodes']}):
        selected = [(e, r) for e, r in zip(run['episodes'], items) if e['task_type']==task]
        result['by_task'][task] = {'episodes': len(selected), 'submitted': sum(e['submitted'] for e, _ in selected),
                                 'core_supported': sum(r['core_supported'] for _, r in selected),
                                 'complete_success': sum(r['tier']==3 for _, r in selected)}
    save(run_dir/'evaluation.json', result)
    return result

