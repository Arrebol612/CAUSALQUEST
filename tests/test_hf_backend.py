"""Real HF causal LM + PEFT tests, initialized locally with random tiny weights."""
import json
from pathlib import Path
import pytest
import torch
from causalquest.framework.artifacts import write_jsonl, read
from causalquest.framework.config import DEFAULTS
from causalquest.framework.backends import HFBackend, fingerprint, describe_backend
from causalquest.framework.workflows import sft


@pytest.fixture(params=['gpt2', 'qwen3'])
def hf(tmp_path, request):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, GPT2Config, GPT2LMHeadModel, Qwen3Config, Qwen3ForCausalLM
    root=tmp_path/'base';root.mkdir()
    vocab={word:i for i,word in enumerate(['<pad>','<unk>','<eos>','<bos>','user','assistant',':','question','answer'])}
    token=Tokenizer(WordLevel(vocab,unk_token='<unk>'));token.pre_tokenizer=Whitespace()
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=token,unk_token='<unk>',pad_token='<pad>',eos_token='<eos>',bos_token='<bos>')
    tokenizer.chat_template="{{bos_token}}{% for m in messages %}{{m['role']}} : {{m['content']}}{{eos_token}}{% endfor %}{% if add_generation_prompt %}assistant : {% endif %}"
    tokenizer.save_pretrained(root)
    torch.manual_seed(4)
    if request.param == 'gpt2':
        model=GPT2LMHeadModel(GPT2Config(vocab_size=len(vocab),n_positions=128,n_embd=16,n_layer=1,n_head=2,
                            resid_pdrop=0,embd_pdrop=0,attn_pdrop=0,eos_token_id=2,pad_token_id=0,bos_token_id=3))
    else:
        model=Qwen3ForCausalLM(Qwen3Config(vocab_size=len(vocab),hidden_size=16,intermediate_size=32,
              num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8,
              max_position_embeddings=128,attention_dropout=0,eos_token_id=2,pad_token_id=0,bos_token_id=3))
    model.save_pretrained(root)
    cfg={**DEFAULTS,'backend':'hf','environment':'causalquest','model_path':str(root),'device':'cpu','dtype':'float32',
         'max_context':128,'max_new_tokens':8,'lora_rank':2,'lora_alpha':4,
         'learning_rate':.01,'max_steps':1,'batch_rows':1}
    rows=[{'id':'example-'+str(i),'case_id':'tutorial','source_group':'tutorial','split':'train',
           'review_status':'accepted','protocol':cfg['protocol'],'tools':[],
           'messages':[{'role':'user','content':'question'},{'role':'assistant','content':'answer'}]} for i in range(2)]
    dataset=tmp_path/'rows.jsonl';write_jsonl(dataset,rows);cfg['dataset']=str(dataset)
    return cfg,rows


def test_exact_hf_completion_logprobs_and_assistant_mask(hf):
    cfg,rows=hf;backend=HFBackend(cfg)
    profile=describe_backend(backend)
    assert profile['all_trainable_are_lora'] and profile['trainable_tensor_count'] > 0
    assert set(profile['parameter_counts_by_dtype']['trainable']) == {'torch.float32'}
    p,c=backend.encode_example(rows[0])
    ids=torch.tensor([p+c])
    with torch.no_grad():
        logits=backend.model(ids,use_cache=False).logits[0,len(p)-1:-1].float()
        independent=logits.log_softmax(-1).gather(1,torch.tensor(c)[:,None]).squeeze(1)
        actual=backend.score(p,c)
    assert len(actual)==len(c) and c[-1]==2
    assert torch.allclose(actual,independent,atol=1e-6)
    bad={**rows[0],'messages':[{'role':'user','content':'question'}]}
    with pytest.raises(ValueError):backend.encode_example(bad)


def test_real_lora_sft_resume_equals_uninterrupted_and_base_unchanged(hf,tmp_path):
    cfg,rows=hf
    full={**cfg,'max_steps':2}
    sft(full,tmp_path/'full')
    sft(cfg,tmp_path/'first')
    resumed={**full,'adapter':str(tmp_path/'first/model')}
    sft(resumed,tmp_path/'second')
    assert read(tmp_path/'full/checkpoint.json')['parameter_sha256']==read(tmp_path/'second/checkpoint.json')['parameter_sha256']
    b=HFBackend({**cfg,'adapter':str(tmp_path/'full/model')})
    untouched=HFBackend(cfg)
    frozen_before={n:p.detach().clone() for n,p in untouched.model.named_parameters() if not p.requires_grad}
    assert all(torch.equal(p,frozen_before[n]) for n,p in b.model.named_parameters() if not p.requires_grad)
    assert fingerprint(b)!=fingerprint(untouched)


def test_actual_hf_sampling_receipts_match_teacher_forced_target_positions(hf):
    cfg,rows=hf;b=HFBackend(cfg)
    request={'messages':rows[0]['messages'][:-1],'tools':[]}
    generated=b.generate(request,17)
    assert len(generated['completion_ids'])==len(generated['old_logprobs'])==len(generated['behavior_logprobs'])
    with torch.no_grad():cur=b.score(generated['prompt_ids'],generated['completion_ids'])
    assert torch.allclose(cur,torch.tensor(generated['old_logprobs']),atol=1e-6)
    assert torch.isfinite(torch.tensor(generated['behavior_logprobs'])).all()


def test_gradient_forward_really_calls_checkpoint_and_matches_no_grad_scores(hf, monkeypatch):
    cfg,rows=hf;b=HFBackend(cfg)
    p,c=b.encode_example(rows[0]);calls=[]
    checkpointed=[m for m in b.model.modules() if hasattr(m,'_gradient_checkpointing_func')]
    assert checkpointed
    for module in checkpointed:
        original=module._gradient_checkpointing_func
        def counted(*args,_original=original,**kwargs):
            calls.append(True)
            return _original(*args,**kwargs)
        monkeypatch.setattr(module,'_gradient_checkpointing_func',counted)
    with torch.no_grad():
        expected=b.score(p,c)
    assert not calls and not b.model.training
    actual=b.score(p,c)
    assert calls and b.model.training
    assert all(not m.training for m in b.model.modules() if isinstance(m,torch.nn.Dropout))
    assert torch.allclose(actual.detach(),expected,atol=1e-6)
    (-actual.mean()).backward()
    grads=[v.grad for v in b.model.parameters() if v.requires_grad and v.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads) and any(g.abs().sum()>0 for g in grads)
    b.generate({'messages':rows[0]['messages'][:-1],'tools':[]},17)
    assert not b.model.training


def test_checkpointed_gradients_match_uncheckpointed_same_weights(hf):
    cfg,rows=hf;b=HFBackend(cfg)
    p,c=b.encode_example(rows[0]);loss=-b.score(p,c).mean();loss.backward()
    actual={n:v.grad.detach().clone() for n,v in b.model.named_parameters() if v.requires_grad and v.grad is not None}
    b.model.zero_grad(set_to_none=True)
    b.model.gradient_checkpointing_disable()
    reference=-b.score(p,c).mean();reference.backward()
    assert torch.allclose(loss.detach(),reference.detach(),atol=1e-6)
    for name,parameter in b.model.named_parameters():
        if name in actual:
            assert torch.allclose(parameter.grad,actual[name],atol=1e-6,rtol=1e-5),name


@pytest.mark.parametrize('temperature',[1.0,.6])
def test_real_generate_ignores_model_specific_temperature_and_nucleus_defaults(hf,monkeypatch,temperature):
    cfg,rows=hf;cfg={**cfg,'sampling_temperature':temperature};b=HFBackend(cfg);base=b.model.get_base_model()
    base.generation_config.temperature=.6;base.generation_config.top_p=.95
    base.generation_config.transformers_version='4.51.3'
    seen=[];original=base._get_logits_processor
    def capture(*args,**kwargs):
        actual=kwargs['generation_config'];seen.append((actual.temperature,actual.top_p,actual.top_k))
        return original(*args,**kwargs)
    monkeypatch.setattr(base,'_get_logits_processor',capture)
    receipt=b.generate({'messages':rows[0]['messages'][:-1],'tools':[]},17)
    assert seen and all(v==(temperature,1.0,0) for v in seen)
    effective=receipt['effective_generation_config']
    assert effective['temperature']==temperature and effective['top_p']==1.0
    assert effective['top_k']==0 and effective['use_model_defaults'] is False
    assert effective['pad_token_id']==0


