"""Generate one fresh continuation per prompt family from frozen, unmodified Qwen3-8B.

Decoding policy (fixed, recorded in results/gen_config.json):
  non-thinking chat template, temperature 0.7, top_p 0.8, top_k 20, max_new_tokens 768,
  stop at <|im_end|>/EOS. finish_reason = 'stop' (ended normally) or 'length' (cap -> censoring).
Usage: python src/generate.py [families.json] [out.jsonl] [n_samples_per_prompt] [max_new]
"""
import json, sys, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "/data/smallyan/hf_home/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"  # Qwen/Qwen3-8B @ b968826
fam_path = sys.argv[1] if len(sys.argv) > 1 else "results/families.json"
out_path = sys.argv[2] if len(sys.argv) > 2 else "results/generations.jsonl"
NS = int(sys.argv[3]) if len(sys.argv) > 3 else 1
MAX_NEW = int(sys.argv[4]) if len(sys.argv) > 4 else 768
CFG = dict(model=MODEL, temperature=0.7, top_p=0.8, top_k=20, max_new_tokens=MAX_NEW, enable_thinking=False,
           seed=0, dtype="bfloat16", stop="eos(<|im_end|>)", n_samples=NS)
BS = 160

tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
data = json.load(open(fam_path))
items = data["families"] if isinstance(data, dict) else data
# items may carry a 'prefix_ids' (response prefix for repeated-rollout eval)
jobs = []
for f in items:
    chat = tok.apply_chat_template([{"role": "user", "content": f["prompt"]}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    pids = tok(chat, add_special_tokens=False).input_ids
    pre = f.get("prefix_ids", [])
    for s in range(NS):
        jobs.append((f, pids, pre, s))
jobs.sort(key=lambda j: len(j[1]) + len(j[2]))
eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id]
torch.manual_seed(0)
t0 = time.time()
with open(out_path, "w") as fo:
    for b in range(0, len(jobs), BS):
        batch = jobs[b:b + BS]
        seqs = [j[1] + j[2] for j in batch]
        L = max(map(len, seqs))
        ids = torch.full((len(seqs), L), tok.pad_token_id); att = torch.zeros_like(ids)
        for i, s in enumerate(seqs):
            ids[i, L - len(s):] = torch.tensor(s); att[i, L - len(s):] = 1
        with torch.no_grad():
            out = model.generate(input_ids=ids.cuda(), attention_mask=att.cuda(), do_sample=True, temperature=0.7,
                                 top_p=0.8, top_k=20, max_new_tokens=MAX_NEW, eos_token_id=eos, pad_token_id=tok.pad_token_id)
        for (f, pids, pre, s), row in zip(batch, out[:, L:].tolist()):
            fin = "length"
            for k, t in enumerate(row):
                if t in eos:
                    row = row[:k]; fin = "stop"; break
            row = [t for t in row if t != tok.pad_token_id] if fin == "length" else row
            rec = dict(family_id=f["family_id"], sample=s, prompt_ids=pids, prefix_ids=pre, response_ids=pre + row,
                       response=tok.decode(pre + row), finish_reason=fin)
            for k in ("split", "source", "setting", "cutoff", "prefix_key"):
                if k in f: rec[k] = f[k]
            fo.write(json.dumps(rec) + "\n")
        fo.flush()
        print(f"{b + len(batch)}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
json.dump(CFG, open(out_path.replace(".jsonl", "_config.json"), "w"), indent=1)
