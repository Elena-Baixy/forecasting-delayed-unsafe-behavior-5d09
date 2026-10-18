from datasets import load_dataset
import sys, json, os
specs = [
  ("JailbreakBench/JBB-Behaviors", "behaviors", "jbb_behaviors"),
  ("PKU-Alignment/BeaverTails", None, "beavertails"),
  ("PKU-Alignment/PKU-SafeRLHF", None, "pku_saferlhf"),
  ("LLM-LAT/harmful-dataset", None, "llm_lat_harmful"),
  ("mlabonne/harmful_behaviors", None, "harmful_behaviors"),
  ("mlabonne/harmless_alpaca", None, "harmless_alpaca"),
  ("databricks/databricks-dolly-15k", None, "dolly15k"),
  ("walledai/HarmBench", "standard", "harmbench_standard"),
  ("walledai/AdvBench", None, "advbench"),
  ("walledai/StrongREJECT", None, "strongreject"),
]
only = sys.argv[1:]
for hf, cfg, name in specs:
    if only and name not in only: continue
    out = f"datasets/{name}"
    if os.path.exists(out): print("skip", name); continue
    try:
        ds = load_dataset(hf, cfg) if cfg else load_dataset(hf)
        ds.save_to_disk(out)
        os.makedirs(f"{out}/samples", exist_ok=True)
        for split in ds:
            rows = [ds[split][i] for i in range(min(3, len(ds[split])))]
            json.dump(rows, open(f"{out}/samples/{split}_samples.json","w"), indent=1, default=str)
        print("OK", name, {s: (len(ds[s]), ds[s].column_names) for s in ds})
    except Exception as e:
        print("FAIL", name, repr(e)[:300])
