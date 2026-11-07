import math
import pytest
import torch
from causalquest.framework.objectives import policy_loss, advantages


def test_autograd_matches_finite_difference_and_caches_detached():
    x = torch.tensor([-0.4, -1.0], dtype=torch.float64, requires_grad=True)
    old = torch.tensor([-0.45, -0.96], dtype=torch.float64, requires_grad=True)
    behavior = torch.tensor([-0.5, -0.9], dtype=torch.float64, requires_grad=True)
    ref = torch.tensor([-0.7, -1.2], dtype=torch.float64, requires_grad=True)
    loss, metric = policy_loss(x, old, behavior, ref, .6, epsilon=.2, beta=.03)
    loss.backward()
    assert old.grad is behavior.grad is ref.grad is None
    def scalar(values):
        result = 0
        for current, o, b, r in zip(values, old.tolist(), behavior.tolist(), ref.tolist()):
            ratio = math.exp(current-o); correction = min(math.exp(o-b), 2)
            result += correction*(-min(ratio*.6, min(max(ratio,.8),1.2)*.6)+.03*(math.exp(r-current)-(r-current)-1))
        return result
    for i in range(2):
        plus=x.detach().tolist();minus=plus.copy();plus[i]+=1e-6;minus[i]-=1e-6
        assert x.grad[i].item() == pytest.approx((scalar(plus)-scalar(minus))/2e-6, abs=1e-8)


def test_policy_clipping_is_separate_from_sampler_gap_and_flat_keeps_kl():
    cur=torch.tensor([0.],requires_grad=True);old=torch.tensor([0.]);beh=torch.tensor([-2.]);ref=torch.tensor([-.5])
    loss,m=policy_loss(cur,old,beh,ref,0,epsilon=.2,beta=.1);loss.backward()
    assert m['ratio_min']==m['ratio_max']==1 and m['clipped_tokens']==0
    assert m['correction_capped_tokens']==1 and cur.grad.abs().item()>0
    cur2=torch.tensor([math.log(2)],requires_grad=True)
    loss,_=policy_loss(cur2,old,old,old,1,epsilon=.2,beta=0);loss.backward()
    assert cur2.grad.item()==0
    cur3=torch.tensor([math.log(.5)],requires_grad=True)
    loss,_=policy_loss(cur3,old,old,old,-1,epsilon=.2,beta=0);loss.backward()
    assert cur3.grad.item()==0


def test_ordinals_ties_and_all_wrong_group_gate():
    def row(tier, core):
        return {'tier':tier,'core_supported':core,'scores':{'evidence':1,'decision':1,'format':2}}
    # Use full dimension slots, with unrelated dimensions inapplicable.
    from causalquest.framework.objectives import DIMENSIONS
    group=[row(1,False),row(2,True),row(2,True),row(3,True)]
    for r in group:r['scores']={d:r['scores'].get(d) for d in DIMENSIONS}
    a,raw=advantages(group)
    assert a==pytest.approx([-1,0,0,1]) and sum(a)==0
    bad=[row(0,False),row(1,False),row(1,False),row(1,False)]
    for r in bad:r['scores']={d:r['scores'].get(d) for d in DIMENSIONS}
    a,raw=advantages(bad)
    assert a==[0]*4 and any(raw)


def test_response_accumulation_matches_whole_trajectory_reduction():
    # One trajectory has responses of unequal lengths. Compare accumulated gradients to
    # one concatenated objective with equal trajectory, not equal response, weighting.
    x=torch.tensor([-.5,-.8,-1.,-.3,-.9],dtype=torch.float64,requires_grad=True)
    old=x.detach().clone();ref=old-.2;beh=old+.03
    loss1,_=policy_loss(x[:1],old[:1],beh[:1],ref[:1],1,epsilon=.2,beta=.03)
    loss2,_=policy_loss(x[1:4],old[1:4],beh[1:4],ref[1:4],1,epsilon=.2,beta=.03)
    loss3,_=policy_loss(x[4:],old[4:],beh[4:],ref[4:],-1,epsilon=.2,beta=.03)
    micro=(loss1+loss2)/8+loss3/2
    micro.backward();g=x.grad.clone();x.grad=None
    full_a,_=policy_loss(x[:4],old[:4],beh[:4],ref[:4],1,epsilon=.2,beta=.03)
    full_b,_=policy_loss(x[4:],old[4:],beh[4:],ref[4:],-1,epsilon=.2,beta=.03)
    (full_a/8+full_b/2).backward()
    assert torch.allclose(g,x.grad,atol=1e-12)

