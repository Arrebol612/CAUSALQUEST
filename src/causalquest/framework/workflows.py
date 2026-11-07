"""Explicit SFT, sampling, review and one-update RL stages; no scheduler side effects."""
import copy
import math
import time
from pathlib import Path
from .artifacts import read, save, sha, digest, require, jsonl, reserve, atomic, implementation_digest
from .backends import create_backend, fingerprint, describe_backend
from .environments import load_cases, create_environment
from .objectives import advantages, policy_loss


def checkpoint(out, backend, optimizer, receipt):
    import torch
    backend.save(out/'model')
    atomic(out/'optimizer.pt', lambda stream: torch.save(optimizer.state_dict(), stream))
    atomic(out/'rng.pt', lambda stream: torch.save({'torch': torch.get_rng_state(),
             'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}, stream))
    files = {p.relative_to(out).as_posix(): sha(p) for p in (out/'model').rglob('*') if p.is_file()}
    files.update({n: sha(out/n) for n in ('optimizer.pt', 'rng.pt')})
    identity = fingerprint(backend)
    save(out/'checkpoint.json', {**receipt, 'parameter_sha256': identity, 'files_sha256': files})
    # Fresh model loading must reproduce parameters, not merely save readable files.
    clone_cfg = {**backend.cfg, 'adapter': str(out/'model')}
    clone = create_backend(clone_cfg)
    require(fingerprint(clone) == identity, 'fresh checkpoint reload differs')
    save(out/'reload.json', {'parameter_sha256': identity, 'fresh_reload_verified': True})
    return identity


def checked_parent(cfg):
    if not cfg.get('adapter'):
        return None
    parent = Path(cfg['adapter']).parent
    if not (parent/'checkpoint.json').is_file():
        require(not (parent/'started.json').exists(), 'incomplete framework checkpoint; do not treat as external adapter')
        return None  # external adapter: an explicit fresh optimizer boundary
    receipt = read(parent/'checkpoint.json')
    require(read(parent/'reload.json')['fresh_reload_verified'] is True, 'parent was not reloaded')
    for name, expected in receipt['files_sha256'].items():
        require((parent/name).stat().st_size > 0 and sha(parent/name) == expected, 'parent checkpoint file differs: ' + name)
    if receipt['stage'] == 'rl':
        require((parent/'summary.json').is_file() and read(parent/'summary.json')['status'] ==
                'one_group_relative_update_saved_reloaded', 'RL checkpoint diagnostics not completed')
    return parent, receipt


def restore_optimizer(parent, optimizer, backend, stage):
    import torch
    if not parent or parent[1]['stage'] != stage:
        return 0
    path, receipt = parent
    require(receipt['parameter_sha256'] == fingerprint(backend), 'optimizer parent does not match actor')
    optimizer.load_state_dict(torch.load(path/'optimizer.pt', map_location=backend.cfg['device'], weights_only=True))
    rng = torch.load(path/'rng.pt', map_location='cpu', weights_only=True)
    torch.set_rng_state(rng['torch'])
    if rng['cuda']:
        require(torch.cuda.is_available() and len(rng['cuda']) == torch.cuda.device_count(), 'CUDA RNG topology changed')
        torch.cuda.set_rng_state_all(rng['cuda'])
    return receipt['optimizer_step']


def toy_rows(cases):
    rows = []
    for case in cases:
        require(case['split'] == 'train', 'SFT includes non-TRAIN case')
        relation = 4 if case['margin'] > 0 else 5
        for phase, target in [(0, 1), (1, 2), (2, 3 if relation == 4 else 4)]:
            rows.append({'id': case['case_id'] + ':' + str(phase), 'split': 'train',
                         'source_group': case['source_group'], 'case_id': case['case_id'],
                         'prompt_ids': [phase, 3 if phase == 0 else relation], 'completion_ids': [target]})
    return rows


def sft(cfg, output):
    import torch
    require(cfg['backend'] == 'toy' or cfg['dataset'] is not None, 'HF SFT requires reviewed native examples')
    started = time.monotonic()
    out = reserve(output)
    save(out/'config.json', cfg)
    rows = jsonl(cfg['dataset']) if cfg['dataset'] else toy_rows(load_cases(cfg))
    require(rows and all(r['split'] == 'train' and r.get('source_group') for r in rows), 'TRAIN rows with source identity required')
    if cfg['backend'] != 'toy':
        require(all(r.get('review_status') == 'accepted' for r in rows), 'expert content review required')
    require(len({r['id'] for r in rows}) == len(rows), 'duplicate SFT action IDs')
    if cfg['cases']:
        eligible = {c['case_id']: c['source_group'] for c in load_cases(cfg) if c['split'] == 'train'}
        require(all(r['case_id'] in eligible and r['source_group']==eligible[r['case_id']] for r in rows), 'SFT source not in TRAIN catalog')
    data_digest = digest(rows)
    parent = checked_parent(cfg)
    backend = create_backend(cfg)
    save(out/'model_profile.json', describe_backend(backend))
    encoded = [backend.encode_example(row) for row in rows]
    require(all(p and c for p, c in encoded), 'empty assistant target')
    optimizer = torch.optim.AdamW([p for p in backend.model.parameters() if p.requires_grad], lr=cfg['learning_rate'], weight_decay=0)
    step = restore_optimizer(parent, optimizer, backend, 'sft')
    cursor = parent[1]['next_row'] if parent and parent[1]['stage']=='sft' else 0
    if parent and parent[1]['stage']=='sft':
        require(parent[1]['dataset_sha256'] == data_digest, 'resume dataset changed')
    require(cursor < len(rows) and step < cfg['max_steps'], 'SFT budget already complete')
    before = fingerprint(backend); metrics = []; timed_out = False
    while cursor < len(rows) and step < cfg['max_steps']:
        batch = encoded[cursor:cursor+cfg['batch_rows']]
        denominator = sum(len(c) for _, c in batch)
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for prompt, completion in batch:
            if time.monotonic()-started > cfg['wall_time_seconds']-cfg['checkpoint_reserve_seconds']:
                optimizer.zero_grad(set_to_none=True); timed_out = True; break
            loss = -backend.score(prompt, completion).sum()/denominator
            require(torch.isfinite(loss).item(), 'nonfinite SFT objective')
            total += float(loss.detach()); loss.backward()
        if timed_out: break
        norm = torch.nn.utils.clip_grad_norm_(backend.model.parameters(), cfg['max_grad_norm'])
        require(torch.isfinite(norm).item(), 'nonfinite gradient')
        optimizer.step(); step += 1; cursor += len(batch)
        metric = {'step': step, 'target_token_mean_CE': total, 'target_tokens': denominator,
                  'gradient_norm_pre_clip': float(norm), 'learning_rate': optimizer.param_groups[0]['lr'],
                  'actual_checkpoint_forward_calls': getattr(backend, 'checkpoint_forward_calls', [0])[0]}
        save(out/f'step_{step:04d}.json', metric); metrics.append(metric)
    receipt = {'stage': 'sft', 'optimizer_step': step, 'next_row': cursor, 'dataset_sha256': data_digest,
               'config_sha256': digest(cfg), 'parameter_sha256_before': before}
    after = checkpoint(out, backend, optimizer, receipt)
    save(out/'summary.json', {**receipt, 'parameter_sha256_after': after, 'metrics': metrics,
               'rows_consumed': cursor, 'total_rows': len(rows), 'stopped_before_partial_batch_update': timed_out,
               'elapsed_seconds': time.monotonic()-started, 'test_fixture_only': cfg['backend']=='toy'})
    return read(out/'summary.json')


def sample(cfg, output, split):
    started = time.monotonic()
    out = reserve(output); save(out/'config.json', cfg)
    parent = checked_parent(cfg)
    backend = create_backend(cfg)
    save(out/'model_profile.json', describe_backend(backend))
    actor = fingerprint(backend)
    cases = [c for c in load_cases(cfg) if c['split'] == split]
    require(cases, 'no cases in requested split')
    planned = len(cases)*cfg['draws_per_case']
    episodes = []
    save(out/'plan.json', {'actor_sha256': actor, 'case_count': len(cases), 'episodes': planned,
                           'draws_per_case': cfg['draws_per_case'], 'split': split})
    try:
        for case in cases:
            for draw in range(cfg['draws_per_case']):
                seed = cfg['seed'] + draw
                episode_id = digest([case['case_id'], seed, actor])[:24]
                env = create_environment(case, cfg); generations = []
                try:
                    for index in range(cfg['max_actions']):
                        require(time.monotonic()-started < cfg['wall_time_seconds']-cfg['checkpoint_reserve_seconds'],
                                'stage time budget reached; retain partial receipts, do not resample')
                        request = env.request()
                        receipt_dir = out/'generations'/episode_id
                        save(receipt_dir/f'{index+1:04d}_intent.json',
                             {'actor_sha256': actor, 'seed': seed+index*100003, 'request_sha256': digest(request)})
                        try:
                            generated = backend.generate(request, seed + index*100003)
                        except ValueError as exc:
                            if str(exc) != 'context_budget_exhausted':
                                raise
                            env.termination = 'context_budget_exhausted'; break
                        save(receipt_dir/f'{index+1:04d}_completed.json', {'request': request, **generated})
                        generations.append({'request': request, **generated})
                        env.step(generated['response'])
                        if env.done:
                            break
                    episode = {'episode_id': episode_id, 'case_id': case['case_id'], 'task_type': case['task_type'],
                               'source_group': case['source_group'], 'split': split, 'seed': seed,
                               'submitted': env.submitted, 'termination_reason': env.termination or 'max_actions',
                               'events': copy.deepcopy(env.events), 'report': env.report, 'generations': generations}
                    save(out/(episode_id+'.json'), episode); episodes.append(episode)
                finally:
                    env.close()
        require(len(episodes) == planned and fingerprint(backend) == actor, 'incomplete sampling or actor changed')
        save(out/'run.json', {'schema': 'cq-run-v1', 'status': 'complete', 'actor_sha256': actor,
                              'implementation_sha256': implementation_digest(),
                              'config': cfg, 'planned_episodes': planned, 'episodes': episodes})
    except Exception as exc:
        save(out/'failure.json', {'type': type(exc).__name__, 'message': str(exc), 'completed': len(episodes),
                                 'planned': planned, 'automatic_retry': False})
        raise
    from .reviews import export_reviews
    exported = export_reviews(out)
    return {'episodes': len(episodes), 'submitted': sum(e['submitted'] for e in episodes), **exported}


def rl(cfg, run_dir, output):
    import torch
    from .reviews import validate
    run_dir = Path(run_dir)
    run, reviews = validate(run_dir, read(run_dir/'reviews.json'))
    require(run['implementation_sha256'] == implementation_digest(), 'runtime implementation changed since sampling')
    require(run['config']['protocol'] == cfg['protocol'] and run['config']['backend'] == cfg['backend'], 'sampling/training profile changed')
    require(all(e['split'] == 'train' for e in run['episodes']), 'validation/test rollouts cannot update parameters')
    require(cfg['reference'] is not None, 'fixed reference checkpoint required')
    # Reserve before model creation: an interrupted update is never implicitly retried.
    started = time.monotonic()
    out = reserve(output); save(out/'config.json', cfg)
    parent = checked_parent(cfg)
    backend = create_backend(cfg)
    require(fingerprint(backend) == run['actor_sha256'], 'rollouts are stale or actor differs')
    reference = create_backend({**cfg, 'adapter': cfg['reference']})
    save(out/'model_profile.json', {'actor': describe_backend(backend), 'reference': describe_backend(reference)})
    reference_id = fingerprint(reference)
    if parent and parent[1]['stage'] == 'rl':
        require(parent[1]['reference_sha256'] == reference_id, 'fixed RL reference changed')
    for p in reference.model.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in backend.model.parameters() if p.requires_grad],
                                 lr=cfg['learning_rate'], weight_decay=0)
    prior_steps = restore_optimizer(parent, optimizer, backend, 'rl')
    grouped = {}
    for episode, review in zip(run['episodes'], reviews):
        grouped.setdefault(episode['case_id'], []).append((episode, review))
    require(all(len(g) == run['config']['draws_per_case'] for g in grouped.values()), 'incomplete rollout group')
    work, group_rows = [], []
    for case_id, group in grouped.items():
        a, raw = advantages([r for _, r in group])
        group_rows.append({'case_id': case_id, 'advantages': a, 'diagnostic_advantages': raw})
        for (episode, review), advantage in zip(group, a):
            require(episode['generations'], 'a zero-action episode blocks full-batch training')
            work.append((episode, advantage))
    save(out/'advantages.json', group_rows)
    # Re-score every old cache first, without updating or retaining autograd graphs.
    with torch.no_grad():
        worst = 0.0
        for episode, _ in work:
            for generation in episode['generations']:
                require(time.monotonic()-started < cfg['wall_time_seconds']-cfg['checkpoint_reserve_seconds'], 'time limit during old-cache validation; no update')
                score = backend.score(generation['prompt_ids'], generation['completion_ids'])
                old = torch.tensor(generation['old_logprobs'], device=score.device)
                require(score.shape == old.shape, 'old probability vector length differs')
                worst = max(worst, float((score-old).abs().max()))
                require(torch.isfinite(score).all().item(), 'nonfinite before-update probabilities')
        require(worst <= cfg['before_update_logprob_tolerance'], 'old/current probabilities differ before update; stop')
    save(out/'before_update.json', {'max_abs_logprob_change': worst, 'actor_sha256': run['actor_sha256']})
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0; target_count = 0; kl_sum = 0.0; clipped = 0
    for episode, advantage in work:
        n = sum(len(g['completion_ids']) for g in episode['generations'])
        denominator = len(work)*n
        for generation in episode['generations']:
            if time.monotonic()-started > cfg['wall_time_seconds']-cfg['checkpoint_reserve_seconds']:
                optimizer.zero_grad(set_to_none=True)
                result = {'status': 'time_limit_before_full_batch_no_update', 'optimizer_steps': 0}
                save(out/'summary.json', result); return result
            p, c = generation['prompt_ids'], generation['completion_ids']
            current = backend.score(p, c)
            with torch.no_grad():
                ref = reference.score(p, c)
            old = torch.tensor(generation['old_logprobs'], device=current.device)
            behavior = torch.tensor(generation['behavior_logprobs'], device=current.device)
            loss, metric = policy_loss(current, old, behavior, ref, advantage,
                                      epsilon=cfg['clip_epsilon'], beta=cfg['kl_beta'])
            reduced = loss/denominator
            reduced.backward(); loss_sum += float(reduced.detach())
            target_count += len(c); kl_sum += metric['k3_sum']; clipped += metric['clipped_tokens']
    norm = torch.nn.utils.clip_grad_norm_(backend.model.parameters(), cfg['max_grad_norm'])
    require(torch.isfinite(norm).item(), 'nonfinite full-batch gradient')
    if float(norm) == 0:
        result = {'status': 'zero_gradient_no_optimizer_step', 'optimizer_steps': 0,
                  'episodes': len(work), 'active_groups': sum(any(x['advantages']) for x in group_rows)}
        save(out/'summary.json', result); return result
    save(out/'update_intent.json', {'optimizer_step': prior_steps+1, 'actor_sha256': run['actor_sha256']})
    optimizer.step()
    receipt = {'stage': 'rl', 'optimizer_step': prior_steps+1, 'run_sha256': digest(run),
               'reviews_sha256': digest(reviews), 'reference_sha256': reference_id,
               'parameter_sha256_before': run['actor_sha256'], 'config_sha256': digest(cfg)}
    after = checkpoint(out, backend, optimizer, receipt)
    with torch.no_grad():
        post_kl = 0.0; post_clipped = 0; post_count = 0
        for episode, _ in work:
            for generation in episode['generations']:
                cur = backend.score(generation['prompt_ids'], generation['completion_ids'])
                old = torch.tensor(generation['old_logprobs'], device=cur.device)
                ref = reference.score(generation['prompt_ids'], generation['completion_ids'])
                ratio = (cur-old).exp(); delta = ref-cur
                require(torch.isfinite(ratio).all().item() and torch.isfinite(delta.exp()).all().item(), 'nonfinite post-update diagnostics')
                post_kl += float((delta.exp()-delta-1).sum())
                post_clipped += int(((ratio < 1-cfg['clip_epsilon']) | (ratio > 1+cfg['clip_epsilon'])).sum())
                post_count += cur.numel()
        save(out/'after_update.json', {'sample_KL_to_fixed_reference': post_kl/post_count,
                                      'clip_fraction_vs_old': post_clipped/post_count, 'target_tokens': post_count})
    result = {**receipt, 'status': 'one_group_relative_update_saved_reloaded',
              'parameter_sha256_after': after, 'gradient_norm_pre_clip': float(norm),
              'loss': loss_sum, 'target_tokens': target_count, 'episodes': len(work),
              'active_groups': sum(any(x['advantages']) for x in group_rows),
              'sample_k3_token_mean': kl_sum/target_count, 'clip_fraction': clipped/target_count,
              'test_fixture_only': cfg['backend']=='toy'}
    save(out/'summary.json', result)
    return result

