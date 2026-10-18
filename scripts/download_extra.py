from datasets import load_dataset
import json, os, sys
for hf, cfg, name in [("liyang-ict/FineHarm", None, "fineharm"), ("Qwen/Qwen3GuardTest", None, "qwen3guardtest"),
                      ("abhayesian/circuit-breakers-dataset", None, "circuit_breakers"),
                      ("Unispac/shallow-vs-deep-safety-alignment-dataset", None, "harmful_hexphi")]:
    out = f"datasets/{name}"
    if os.path.exists(out): continue
    try:
        ds = load_dataset(hf, cfg) if cfg else load_dataset(hf)
        ds.save_to_disk(out); os.makedirs(f"{out}/samples", exist_ok=True)
        for s in ds:
            json.dump([ds[s][i] for i in range(min(2, len(ds[s])))], open(f"{out}/samples/{s}_samples.json", "w"), indent=1, default=str)
        print("OK", name, {s: (len(ds[s]), ds[s].column_names) for s in ds})
    except Exception as e:
        print("FAIL", name, repr(e)[:300])
