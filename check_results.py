import glob, json, numpy as np
R = "results_case118"
bad = []
for f in glob.glob(R + "/runs/*.json") + glob.glob(R + "/tuning_*.json"):
    try:
        json.load(open(f))
    except Exception:
        bad.append(f)
for f in glob.glob(R + "/preds/*.npz"):
    try:
        z = np.load(f)
        [z[k] for k in z.files]
    except Exception:
        bad.append(f)
print("finished runs:", len(glob.glob(R + "/runs/*.json")), "(120 when complete)")
print("damaged files:", bad if bad else "none")
