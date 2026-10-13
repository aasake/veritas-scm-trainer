import os, json, time, random, traceback, urllib.request

JOB = os.environ.get('JOB_ID', '')
TOK = os.environ.get('JOB_TOKEN', '')
CB = os.environ.get('CALLBACK_URL', '')
HF = os.environ.get('HF_TOKEN', '')
BASE = os.environ.get('BASE_MODEL', 'Qwen/Qwen2.5-0.5B-Instruct')
DATASET = os.environ.get('DATASET', 'veritas-steward/veritas-covenant-corpus')
MODEL_REPO = os.environ.get('MODEL_REPO', 'veritas-steward/veritas-scm-fifth-face')
EPOCHS = int(os.environ.get('EPOCHS', '2'))
LORA_R = int(os.environ.get('LORA_R', '16'))
LORA_ALPHA = int(os.environ.get('LORA_ALPHA', '32'))
MAXLEN = int(os.environ.get('MAXLEN', '512'))
MAX_MIN = float(os.environ.get('MAX_MINUTES', '300'))

# This log is public on GitHub: print metrics only, never training text.

def cb(phase, **metrics):
    try:
        data = json.dumps({'action': 'callback', 'job_id': JOB, 'token': TOK, 'phase': phase, 'metrics': metrics}).encode()
        req = urllib.request.Request(CB, data=data, headers={'Content-Type': 'application/json'})
        urllib.request.urlopen(req, timeout=30).read()
    except Exception as e:
        print('callback failed', type(e).__name__, flush=True)

def run():
    t0 = time.time()
    import torch
    from huggingface_hub import hf_hub_download, HfApi
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    torch.set_num_threads(os.cpu_count() or 2)
    path = hf_hub_download(DATASET, 'train.jsonl', repo_type='dataset', token=HF)
    rows = [json.loads(l) for l in open(path, encoding='utf-8') if l.strip()]
    cb('STARTED', base_model=BASE, pairs=len(rows), cpus=os.cpu_count(), runner='github-actions')
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE).float()
    cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05, task_type='CAUSAL_LM',
                     target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
    model = get_peft_model(model, cfg)
    model.train()
    examples = []
    for r in rows:
        msgs = r.get('messages')
        if not msgs:
            continue
        text = tok.apply_chat_template(msgs, tokenize=False)
        ids = tok(text, truncation=True, max_length=MAXLEN)['input_ids']
        if len(ids) > 8:
            examples.append(ids)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=2e-4)
    accum = 4
    total_steps = max(1, (len(examples) * EPOCHS) // accum)
    step = 0; micro = 0; losses = []; early = False; tokens = 0
    for ep in range(EPOCHS):
        random.shuffle(examples)
        for ids in examples:
            x = torch.tensor([ids])
            out = model(input_ids=x, labels=x)
            (out.loss / accum).backward()
            losses.append(out.loss.item())
            tokens += len(ids)
            micro += 1
            if micro % accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); opt.zero_grad(); step += 1
                if step % 5 == 0:
                    recent = losses[-accum * 5:]
                    mins = (time.time() - t0) / 60
                    msg = dict(step=step, total_steps=total_steps, epoch=ep + 1, loss=round(sum(recent) / len(recent), 4),
                               minutes=round(mins, 1), tokens_per_sec=round(tokens / max(1.0, time.time() - t0), 2))
                    print(msg, flush=True)
                    cb('PROGRESS', **msg)
            if (time.time() - t0) / 60 > MAX_MIN:
                early = True
                break
        if early:
            break
    out_dir = '/tmp/adapter'
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    card = ('---\nbase_model: ' + BASE + '\nlibrary_name: peft\nlicense: other\ntags:\n- lora\n- covenant-intelligence\n---\n\n'
            '# VERITAS SCM Fifth Face (LoRA adapter)\n\nTrained on ' + str(len(examples)) + ' privacy-screened instruction pairs from ' + DATASET +
            ' with base model ' + BASE + ': ' + str(step) + ' optimizer steps, job ' + JOB + '.\n')
    open(out_dir + '/README.md', 'w', encoding='utf-8').write(card)
    HfApi(token=HF).upload_folder(folder_path=out_dir, repo_id=MODEL_REPO, repo_type='model', commit_message='Fifth Face LoRA ' + JOB)
    model.eval()
    samples = []
    for q in ['What is a covenant?', 'Why does the framework require zero fabrication?']:
        msgs = [{'role': 'system', 'content': 'You are the VERITAS-Asake Covenant Intelligence.'}, {'role': 'user', 'content': q}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        x = tok(text, return_tensors='pt')
        with torch.no_grad():
            g = model.generate(**x, max_new_tokens=70, do_sample=False)
        samples.append({'q': q, 'a': tok.decode(g[0][x['input_ids'].shape[1]:], skip_special_tokens=True)[:300]})
    final = round(sum(losses[-accum * 5:]) / max(1, len(losses[-accum * 5:])), 4) if losses else None
    first = round(sum(losses[:accum * 5]) / max(1, len(losses[:accum * 5])), 4) if losses else None
    cb('DONE', steps=step, pairs=len(examples), first_loss=first, final_loss=final, minutes=round((time.time() - t0) / 60, 1),
       stopped_early=early, adapter_repo=MODEL_REPO, base_model=BASE, tokens_per_sec=round(tokens / max(1.0, time.time() - t0), 2), samples=samples)

if __name__ == '__main__':
    try:
        run()
    except Exception:
        cb('FAILED', error=traceback.format_exc()[-1500:])
        raise
