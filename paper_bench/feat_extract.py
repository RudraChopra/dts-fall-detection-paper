"""Extract DTS-128 features for all 5,572 manifest clips from raw CSVs using
the repo's own extract_dts128 (tau=0.30). Resumable chunks.
Output: /tmp/seqtf/F.npy (5572,128). Then validates DTS+HGB on the main split.
Run: python3 feat_extract.py <budget_s>
"""
import csv, json, os, sys, time
import numpy as np

REPO = "/sessions/tender-kind-dijkstra/mnt/dts-fall-detection"
WORK_OLD = "/Users/rudrachopra/Documents/Codex/2026-06-10/files-mentioned-by-the-user-paper/work"
WORK_NEW = "/sessions/tender-kind-dijkstra/mnt/work"
D = "/tmp/seqtf"
import importlib.util as ilu
spec = ilu.spec_from_file_location("dts_features", f"{REPO}/dts/features.py")
feat = ilu.module_from_spec(spec); spec.loader.exec_module(feat)
sys.path.insert(0, "/sessions/tender-kind-dijkstra/mnt/outputs/experiments")
from parse_seq import parse_csv_flexible

budget = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
t0 = time.time()
man = list(csv.DictReader(open(f"{REPO}/results/twinfree/ninefive_core_full/split_manifest_twinfree.csv")))
n = len(man)
F = np.lib.format.open_memmap(f"{D}/F.npy", mode=("r+" if os.path.exists(f"{D}/F.npy") else "w+"),
                              dtype=np.float64, shape=(n, 128))
done = np.load(f"{D}/fdone.npy") if os.path.exists(f"{D}/fdone.npy") else np.zeros(n, bool)
did = 0
for i, r in enumerate(man):
    if done[i]: continue
    if time.time() - t0 > budget: break
    p = r['path'].replace(WORK_OLD, WORK_NEW)
    seq, conf = parse_csv_flexible(open(p, errors='ignore').read())
    F[i] = feat.extract_dts128(seq, conf, tau=0.30)
    done[i] = True; did += 1
np.save(f"{D}/fdone.npy", done)
print(f"chunk {did}, total {int(done.sum())}/{n}")
if not done.all():
    sys.exit(3)

# validation: main-split DTS+HGB with the paper's selected config
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
meta = json.load(open(f"{D}/meta.json"))
split = np.array(meta["split"]); y = np.array(meta["label"])
tr, te = np.where(split == "train")[0], np.where(split == "test")[0]
clf = HistGradientBoostingClassifier(max_iter=600, learning_rate=0.1,
                                     l2_regularization=0.1, random_state=0)
clf.fit(F[tr], y[tr])
au = roc_auc_score(y[te], clf.predict_proba(F[te])[:, 1])
print(f"VALIDATION DTS+HGB main split AUROC = {au:.4f} (paper 0.9895)")
json.dump(dict(val_auroc=float(au)), open(f"{D}/feat_validation.json", "w"))
