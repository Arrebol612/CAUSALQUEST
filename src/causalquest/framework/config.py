"""A portable JSON configuration; paths are relative to the config file."""
import math
from pathlib import Path
from .artifacts import read, require

DEFAULTS = dict(backend='toy', environment='toy', seed=17, cases=None, metric_catalog=None,
                model_path=None, adapter=None, reference=None, device='cpu', dtype='float32',
                learning_rate=0.01, max_steps=8, batch_rows=8, draws_per_case=4,
                max_actions=16, max_new_tokens=256, max_context=16384, sampling_temperature=1.0,
                clip_epsilon=0.2, kl_beta=0.02, max_grad_norm=1.0,
                before_update_logprob_tolerance=0.001, lora_rank=16, lora_alpha=32,
                protocol='cq-public-history-v1', scorer='manual', dataset=None,
                wall_time_seconds=18000, checkpoint_reserve_seconds=900)
PATHS = {'cases', 'metric_catalog', 'model_path', 'adapter', 'reference', 'dataset'}


def load_config(path):
    path = Path(path).resolve()
    raw = read(path)
    require(type(raw) is dict and not set(raw) - set(DEFAULTS), 'unknown configuration fields')
    cfg = {**DEFAULTS, **raw}
    require(cfg['backend'] in {'toy', 'hf'}, 'backend must be toy or hf')
    require(cfg['environment'] in {'toy', 'causalquest'}, 'unknown environment')
    require(cfg['scorer'] in {'manual', 'toy_rules'}, 'unknown scorer')
    require(cfg['protocol'] == 'cq-public-history-v1', 'unsupported request profile; do not relabel legacy data')
    require(cfg['dtype'] in {'float32', 'bfloat16'}, 'unsupported dtype')
    for key in ['seed', 'max_steps', 'batch_rows', 'draws_per_case', 'max_actions',
                'max_new_tokens', 'max_context', 'lora_rank', 'lora_alpha', 'wall_time_seconds', 'checkpoint_reserve_seconds']:
        require(type(cfg[key]) is int and cfg[key] > 0, key + ' must be a positive integer')
    require(cfg['draws_per_case'] >= 2, 'group sampling needs at least two draws')
    for key in ['learning_rate', 'clip_epsilon', 'kl_beta', 'max_grad_norm', 'before_update_logprob_tolerance']:
        value = cfg[key]
        require(type(value) in (int, float) and math.isfinite(value) and value >= 0, key + ' invalid')
    require(0 < cfg['learning_rate'] <= 0.1 and 0 < cfg['clip_epsilon'] < 1 and cfg['max_grad_norm'] > 0,
            'invalid optimizer bounds')
    require(type(cfg['sampling_temperature']) in (int,float) and
            math.isfinite(cfg['sampling_temperature']) and cfg['sampling_temperature']>0,
            'sampling_temperature must be finite and positive')
    require(cfg['max_new_tokens'] < cfg['max_context'], 'generation reserve exceeds context')
    require(cfg['checkpoint_reserve_seconds'] < cfg['wall_time_seconds'], 'checkpoint reserve exceeds stage time')
    require((cfg['backend']=='toy') == (cfg['environment']=='toy'), 'toy backend and environment must be used together')
    if cfg['environment'] != 'toy':
        require(cfg['backend'] != 'toy' and cfg['scorer'] == 'manual', 'toy backend/scoring only supports toy fixtures')
    if cfg['backend'] == 'hf':
        require(cfg['model_path'] is not None, 'local HF model snapshot required')
    for key in PATHS:
        if cfg[key] is not None:
            require(type(cfg[key]) is str and cfg[key], key + ' must be a path')
            cfg[key] = str((path.parent / cfg[key]).resolve())
    return cfg

