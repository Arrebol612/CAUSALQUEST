"""Ordinal group advantages and separated sampling/policy probabilities."""
from .artifacts import require

DIMENSIONS = ('validity', 'effect', 'diagnosis', 'evidence', 'decision', 'uncertainty', 'format')
APPLICABLE = {
    'experiment_design_review': ('validity', 'evidence', 'decision', 'uncertainty', 'format'),
    'experiment_health_diagnosis': ('validity', 'diagnosis', 'evidence', 'decision', 'uncertainty', 'format'),
    'post_experiment_decision': ('validity', 'effect', 'evidence', 'decision', 'uncertainty', 'format'),
    'toy_decision': ('evidence', 'decision', 'format'),
}


def preference(a, b):
    if a['tier'] != b['tier']:
        return 1 if a['tier'] > b['tier'] else -1
    if a['tier'] == 0:
        return 0
    keys = [key for key in DIMENSIONS if key != 'format' and a['scores'][key] is not None]
    signs = [(a['scores'][k] > b['scores'][k]) - (a['scores'][k] < b['scores'][k]) for k in keys]
    if all(s >= 0 for s in signs) and any(s > 0 for s in signs):
        return 1
    if all(s <= 0 for s in signs) and any(s < 0 for s in signs):
        return -1
    if all(s == 0 for s in signs):
        return (a['scores']['format'] > b['scores']['format']) - (a['scores']['format'] < b['scores']['format'])
    return 0


def advantages(reviews):
    require(len(reviews) >= 2, 'complete group required')
    raw = [sum(preference(a, b) for b in reviews) / (len(reviews) - 1) for a in reviews]
    active = any(a['core_supported'] for a in reviews)
    return ([x if active else 0.0 for x in raw], raw)


def policy_loss(current, old, behavior, reference, advantage, *, epsilon, beta):
    import torch
    require(current.ndim == 1 and current.numel() > 0, 'nonempty target vector required')
    require(all(x.shape == current.shape for x in (old, behavior, reference)), 'probability shape mismatch')
    require(all(torch.isfinite(x).all().item() for x in (current, old, behavior, reference)), 'nonfinite probabilities')
    old, behavior, reference = (x.detach() for x in (old, behavior, reference))
    ratio = torch.exp(current - old)
    correction = torch.exp(old - behavior).clamp(max=2.0).detach()
    delta = reference - current
    k3 = torch.exp(delta) - delta - 1
    unclipped = ratio * advantage
    clipped = ratio.clamp(1 - epsilon, 1 + epsilon) * advantage
    loss = (correction * (-torch.minimum(unclipped, clipped) + beta * k3)).sum()
    require(torch.isfinite(loss).item(), 'nonfinite policy objective')
    return loss, {'ratio_min': float(ratio.detach().min()), 'ratio_max': float(ratio.detach().max()),
                  'clipped_tokens': int(((ratio < 1-epsilon) | (ratio > 1+epsilon)).sum().detach()),
                  'k3_sum': float((correction * k3).detach().sum()),
                  'correction_capped_tokens': int(((old-behavior).exp() > 2).sum())}

