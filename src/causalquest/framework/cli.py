"""Portable commands: each expensive stage is explicitly invoked."""
import argparse
import importlib.util
import json
from pathlib import Path
from .artifacts import read, save, require, jsonl
from .config import load_config, DEFAULTS


def demo(output, manual=False):
    from .workflows import sft, sample, rl
    from .reviews import toy_reviews, evaluate
    out = Path(output).resolve(); out.mkdir(parents=True, exist_ok=False)
    cfg = {**DEFAULTS, 'max_steps': 1, 'batch_rows': 6, 'learning_rate': 0.02, 'scorer': 'toy_rules'}
    first = sft(cfg, out/'sft')
    actor = {**cfg, 'adapter': str(out/'sft/model')}
    sampling = sample(actor, out/'rollouts', 'train')
    if manual:
        save(out/'summary.json', {'status': 'awaiting_manual_review', 'test_fixture_only': True,
                                 'sft_updates': first['optimizer_step'], **sampling})
        return read(out/'summary.json')
    toy_reviews(out/'rollouts'); before = evaluate(out/'rollouts')
    update = rl({**actor, 'reference': actor['adapter']}, out/'rollouts', out/'rl')
    if update['status'] != 'zero_gradient_no_optimizer_step':
        after_cfg = {**actor, 'adapter': str(out/'rl/model')}
        sample(after_cfg, out/'after', 'train'); toy_reviews(out/'after'); after = evaluate(out/'after')
    else:
        after = None
    summary = {'status': 'toy_integration_complete', 'test_fixture_only': True,
               'sft_updates': first['optimizer_step'], 'rl_status': update['status'],
               'before': before, 'after': after,
               'scope': 'Workflow/gradient smoke only, not an LLM or CausalQuest capability result.'}
    save(out/'summary.json', summary)
    return summary


def doctor():
    from importlib import metadata
    from causalquest import __version__
    result = {'version': __version__, 'packages': {}, 'remote_calls': 0}
    for name in ('torch', 'transformers', 'peft', 'accelerate', 'duckdb', 'PyYAML'):
        try:
            result['packages'][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result['packages'][name] = None
    return result


def main(argv=None):
    p = argparse.ArgumentParser(prog='cq', description='CausalQuest reproducible research workflows')
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('doctor')
    d = sub.add_parser('demo'); d.add_argument('--output', required=True); d.add_argument('--manual', action='store_true')
    t = sub.add_parser('train'); ts = t.add_subparsers(dest='stage', required=True)
    for stage in ('sft', 'rl'):
        parser = ts.add_parser(stage); parser.add_argument('--config', required=True); parser.add_argument('--output', required=True)
        if stage == 'rl': parser.add_argument('--run', required=True)
    s = sub.add_parser('sample'); s.add_argument('--config', required=True); s.add_argument('--output', required=True)
    s.add_argument('--split', choices=['train', 'validation', 'test'], default='train')
    r = sub.add_parser('reviews'); rs = r.add_subparsers(dest='review_command', required=True)
    for name in ('export', 'import', 'check', 'toy-score'):
        parser = rs.add_parser(name); parser.add_argument('--run', required=True)
        if name in ('import', 'check'): parser.add_argument('--input', required=True)
    e = sub.add_parser('evaluate'); e.add_argument('--run', required=True)
    data = sub.add_parser('data'); ds = data.add_subparsers(dest='data_command', required=True)
    b = ds.add_parser('build-design'); b.add_argument('--input', required=True); b.add_argument('--output', required=True)
    check = ds.add_parser('check'); check.add_argument('--input', required=True)
    args = p.parse_args(argv)
    if args.command == 'doctor': result = doctor()
    elif args.command == 'demo': result = demo(args.output, args.manual)
    elif args.command == 'train':
        from .workflows import sft, rl
        cfg = load_config(args.config)
        result = sft(cfg, args.output) if args.stage == 'sft' else rl(cfg, args.run, args.output)
    elif args.command == 'sample':
        from .workflows import sample
        result = sample(load_config(args.config), args.output, args.split)
    elif args.command == 'reviews':
        from .reviews import export_reviews, import_reviews, validate, toy_reviews
        if args.review_command == 'export': result = export_reviews(args.run)
        elif args.review_command == 'import': result = import_reviews(args.run, args.input)
        elif args.review_command == 'toy-score': result = toy_reviews(args.run)
        else:
            _, items = validate(args.run, read(args.input)); result = {'valid': True, 'reviewed': len(items)}
    elif args.command == 'evaluate':
        from .reviews import evaluate
        result = evaluate(args.run)
    elif args.data_command == 'build-design':
        from causalquest.tasks.design_case_builder import build_design_case
        require(not Path(args.output).exists(), 'build output already exists')
        result = build_design_case(read(args.input), args.output)
    else:
        rows = jsonl(args.input)
        require(rows and len({r['id'] for r in rows})==len(rows), 'duplicate/empty dataset')
        require(all(r.get('split')=='train' and r.get('source_group') and r.get('review_status')=='accepted'
                    and r.get('protocol')=='cq-public-history-v1' and r.get('messages', [{}])[-1].get('role')=='assistant'
                    for r in rows), 'accepted TRAIN next-assistant rows required')
        result = {'rows': len(rows), 'sources': len({r['source_group'] for r in rows}),
                  'semantic_quality_verified_by_this_command': False}
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()

