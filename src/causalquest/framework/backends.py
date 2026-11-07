"""One scoring boundary shared by sampling, SFT and RL; local models only."""
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace
from .artifacts import canonical, digest, read, require, save, sha


class ToyBackend:
    """Small learned categorical policy for workflow tests, not an LLM benchmark."""
    def __init__(self, cfg):
        import torch
        torch.manual_seed(cfg['seed'])
        self.cfg = cfg
        self.model = torch.nn.Sequential(torch.nn.Embedding(16, 8), torch.nn.Flatten(),
                                         torch.nn.Linear(16, 5))
        self.base_identity = 'toy-policy-v1'
        if cfg.get('adapter'):
            self.model.load_state_dict(torch.load(Path(cfg['adapter'])/'model.pt', map_location='cpu', weights_only=True))
        self.model.eval()

    def encode(self, request):
        toy = request['toy']
        return [toy['phase'], toy['relation_token']]

    def score(self, prompt, completion):
        import torch
        require(len(prompt) == 2 and len(completion) == 1 and completion[0] in (1, 2, 3, 4), 'bad toy tokens')
        logits = self.model(torch.tensor([prompt], dtype=torch.long))[0]
        allowed = {0: [1], 1: [2], 2: [3, 4]}[prompt[0]]
        mask = torch.full_like(logits, -float('inf')); mask[allowed] = 0
        require(completion[0] in allowed, 'toy action unavailable')
        return (logits + mask).log_softmax(-1)[completion]

    def generate(self, request, seed):
        import torch
        prompt = self.encode(request)
        choices = {0: [1], 1: [2], 2: [3, 4]}[prompt[0]]
        with torch.no_grad():
            logp = torch.stack([self.score(prompt, [a])[0] for a in choices])
            index = torch.multinomial(logp.exp(), 1, generator=torch.Generator().manual_seed(seed)).item()
        token = choices[index]
        return {'prompt_ids': prompt, 'completion_ids': [token], 'behavior_logprobs': [float(logp[index])],
                'old_logprobs': [float(logp[index])],
                'response': json.dumps({'toy_action': {1: 'inspect', 2: 'state', 3: 'accept', 4: 'reject'}[token]}),
                'finish_reason': 'stop'}

    def encode_example(self, row):
        return row['prompt_ids'], row['completion_ids']

    def save(self, path):
        import torch
        path = Path(path); path.mkdir()
        from .artifacts import atomic
        atomic(path/'model.pt', lambda stream: torch.save(self.model.state_dict(), stream))


class HFBackend:
    """HF causal LM + trainable LoRA. Native completion tokens are never retokenized."""
    def __init__(self, cfg):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, PeftModel, get_peft_model
        require(Path(cfg['model_path']).is_dir(), 'local model directory required; no automatic download')
        self.cfg = cfg
        snapshot = Path(cfg['model_path'])
        weights = sorted(set(snapshot.glob('*.safetensors')) | set(snapshot.glob('pytorch_model*.bin')))
        require(weights, 'no local model weights found')
        self.base_identity = digest({p.name: sha(p) for p in [snapshot/'config.json', *weights]})
        torch.manual_seed(cfg['seed'])
        dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16}[cfg['dtype']]
        self.tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        require(self.tokenizer.chat_template is not None, 'a pinned chat template is required')
        base = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True,
                    trust_remote_code=False, torch_dtype=dtype, attn_implementation='sdpa').to(cfg['device'])
        if cfg.get('adapter'):
            self.model = PeftModel.from_pretrained(base, cfg['adapter'], is_trainable=True,
                                                  local_files_only=True)
        else:
            self.model = get_peft_model(base, LoraConfig(r=cfg['lora_rank'], lora_alpha=cfg['lora_alpha'],
                              lora_dropout=0.0, target_modules='all-linear', task_type='CAUSAL_LM'))
        # Eval is only the idle/generation mode. Grad-enabled scoring switches to train:
        # Transformers checkpointed decoder layers require self.training == True.
        self.model.eval()
        self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        require(getattr(self.model.config, 'attention_dropout', 0) == 0,
                'functional attention dropout must be zero for identical scoring/generation probabilities')
        self.checkpoint_forward_calls = [0]
        for module in self.model.modules():
            if hasattr(module, '_gradient_checkpointing_func'):
                original = module._gradient_checkpointing_func
                def counted_checkpoint(*args, _original=original, _counter=self.checkpoint_forward_calls, **kwargs):
                    _counter[0] += 1
                    return _original(*args, **kwargs)
                module._gradient_checkpointing_func = counted_checkpoint
        self.tokenizer_identity = digest({'vocab': self.tokenizer.get_vocab(), 'template': self.tokenizer.chat_template})

    def encode(self, request):
        return self.tokenizer.apply_chat_template(request['messages'], tools=self.canonical_tools(request['tools']),
                   tokenize=True, add_generation_prompt=True, enable_thinking=False)

    @staticmethod
    def canonical_tools(tools):
        # Artifact JSON is canonical; render live schemas in that same order as SFT.
        return json.loads(canonical(tools))

    def encode_example(self, row):
        require(row['protocol'] == self.cfg['protocol'], 'SFT protocol mismatch')
        messages = row['messages']; require(messages[-1]['role'] == 'assistant', 'new assistant target required')
        prompt = self.encode({'messages': messages[:-1], 'tools': row['tools']})
        full = self.tokenizer.apply_chat_template(messages, tools=self.canonical_tools(row['tools']), tokenize=True,
                    add_generation_prompt=False, enable_thinking=False)
        require(full[:len(prompt)] == prompt and len(full) > len(prompt), 'chat-template prefix mismatch')
        require(len(full) <= self.cfg['max_context'], 'SFT sequence too long; no truncation')
        require(len(prompt)+self.cfg['max_new_tokens'] <= self.cfg['max_context'], 'insufficient serving reserve')
        require(self.tokenizer.eos_token_id in full[len(prompt):], 'assistant target lacks an end token')
        return prompt, full[len(prompt):]

    def score(self, prompt, completion):
        import torch
        import torch.nn.functional as F
        require(prompt and completion and len(prompt)+len(completion) <= self.cfg['max_context'], 'invalid target length')
        self.model.train(torch.is_grad_enabled())
        # Disable stochastic dropout without disabling decoder training/checkpointing.
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.eval()
        ids = torch.tensor([prompt+completion], dtype=torch.long, device=self.cfg['device'])
        base = self.model.get_base_model()
        # Project only completion positions through the vocabulary head.
        backbone = getattr(base, base.base_model_prefix)
        checkpoint_before = self.checkpoint_forward_calls[0]
        hidden = backbone(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state
        if torch.is_grad_enabled() and self.model.is_gradient_checkpointing:
            require(self.checkpoint_forward_calls[0] > checkpoint_before,
                    'gradient checkpointing enabled but not executed during scoring')
        selected = hidden[:, len(prompt)-1:-1, :]
        logits = base.get_output_embeddings()(selected).float()[0]
        target = torch.tensor(completion, device=logits.device)
        return -F.cross_entropy(logits, target, reduction='none')

    def generate(self, request, seed):
        import torch
        from transformers import GenerationConfig
        self.model.eval()
        prompt = self.encode(request)
        require(len(prompt)+self.cfg['max_new_tokens'] <= self.cfg['max_context'], 'context_budget_exhausted')
        torch.manual_seed(seed)
        ids = torch.tensor([prompt], dtype=torch.long, device=self.cfg['device'])
        temperature = self.cfg['sampling_temperature']
        gc = GenerationConfig(do_sample=True, temperature=temperature, top_k=0, top_p=1.0,
                  max_new_tokens=self.cfg['max_new_tokens'], repetition_penalty=1.0,
                  eos_token_id=self.tokenizer.eos_token_id,
                  pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
                  return_dict_in_generate=True, output_scores=True, use_cache=True)
        effective, unused = self.model.get_base_model()._prepare_generation_config(gc, use_model_defaults=False)
        require(not unused and effective.temperature == temperature and effective.top_p == 1.0 and effective.top_k == 0,
                'effective sampler differs from the declared full-support profile')
        receipt = {key: getattr(effective, key) for key in ('do_sample','temperature','top_p','top_k',
                   'max_new_tokens','repetition_penalty','eos_token_id','pad_token_id','bos_token_id')}
        receipt.update(use_model_defaults=False, token_serialization='cq-canonical-tools-v1')
        with torch.no_grad():
            result = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                         generation_config=effective, use_model_defaults=False)
            tokens = result.sequences[0, len(prompt):].tolist()
            require(tokens and len(result.scores) == len(tokens), 'incomplete native generation receipt')
            behavior = [float(logits.float().log_softmax(-1)[0, token]) for logits, token in zip(result.scores, tokens)]
            old = self.score(prompt, tokens).tolist()
        return {'prompt_ids': prompt, 'completion_ids': tokens, 'behavior_logprobs': behavior,
                'old_logprobs': old, 'response': self.tokenizer.decode(tokens, skip_special_tokens=True),
                'effective_generation_config': receipt,
                'finish_reason': 'stop' if tokens[-1] == self.tokenizer.eos_token_id else 'length'}

    def save(self, path):
        path = Path(path); path.mkdir()
        self.model.save_pretrained(path, safe_serialization=True)
        self.tokenizer.save_pretrained(path)


def fingerprint(backend):
    import torch
    h = hashlib.sha256(backend.base_identity.encode())
    h.update(canonical({key: backend.cfg[key] for key in ('backend', 'dtype')}))
    h.update(backend.cfg['device'].split(':')[0].encode())
    if hasattr(backend, 'tokenizer_identity'):
        h.update(backend.tokenizer_identity.encode())
    for name, value in backend.model.named_parameters():
        if value.requires_grad:
            h.update(canonical([name, list(value.shape), str(value.dtype)]))
            h.update(value.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def create_backend(cfg):
    return {'toy': ToyBackend, 'hf': HFBackend}[cfg['backend']](cfg)


def describe_backend(backend):
    """Measured parameter counts and dtypes, independent of the requested config."""
    groups = {'trainable': {}, 'frozen': {}}
    trainable_names = []
    for name, value in backend.model.named_parameters():
        group = 'trainable' if value.requires_grad else 'frozen'
        key = str(value.dtype)
        groups[group][key] = groups[group].get(key, 0) + value.numel()
        if value.requires_grad:
            trainable_names.append(name)
    return {'backend': backend.cfg['backend'], 'parameter_counts_by_dtype': groups,
            'trainable_tensor_count': len(trainable_names),
            'all_trainable_are_lora': bool(trainable_names) and all('lora_' in n for n in trainable_names),
            'gradient_checkpointing_enabled': bool(getattr(backend.model, 'is_gradient_checkpointing', False)),
            'actual_checkpoint_forward_calls': getattr(backend, 'checkpoint_forward_calls', [0])[0],
            'device': str(next(backend.model.parameters()).device)}

