"""Rebuild the strict twin-free sequence tensor from the raw FallVision CSVs,
using the repo's own manifest and normalisation code (exact fidelity).

Resumable: processes clips in chunks, writes to a memmap; state in done.npy.
Run: python3 parse_seq.py <budget_seconds>
Output: /tmp/seqtf/{X.npy (memmap 5572x150x34 f32), L.npy, meta.json}
"""
import csv, io, json, os, sys, time
from collections import defaultdict
import numpy as np

REPO = "/sessions/tender-kind-dijkstra/mnt/dts-fall-detection"
WORK_OLD = "/Users/rudrachopra/Documents/Codex/2026-06-10/files-mentioned-by-the-user-paper/work"
WORK_NEW = "/sessions/tender-kind-dijkstra/mnt/work"
OUT = "/tmp/seqtf"
MAX_T = 150
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("dts_features", f"{REPO}/dts/features.py")
_feat = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_feat)
normalise = _feat.normalise

COCO17 = {"nose":0,"left eye":1,"right eye":2,"left ear":3,"right ear":4,
 "left shoulder":5,"right shoulder":6,"left elbow":7,"right elbow":8,
 "left wrist":9,"right wrist":10,"left hip":11,"right hip":12,
 "left knee":13,"right knee":14,"left ankle":15,"right ankle":16}

def parse_csv_flexible(text):
    try: rows = list(csv.DictReader(io.StringIO(text)))
    except Exception: return None, None
    if not rows: return None, None
    cm = {}
    for col in rows[0]:
        cl = col.strip().lower()
        if cl == 'frame': cm['frame'] = col
        elif cl == 'keypoint': cm['keypoint'] = col
        elif cl == 'x': cm['x'] = col
        elif cl == 'y': cm['y'] = col
        elif 'conf' in cl: cm['confidence'] = col
    if {'frame','keypoint','x','y','confidence'} - set(cm): return None, None
    frames = defaultdict(lambda: {'xy': np.zeros((17,2)), 'cf': np.zeros(17), 'cnt': np.zeros(17)})
    for row in rows:
        try:
            fr = int(float(row[cm['frame']])); raw = row[cm['keypoint']]
            try: kp = int(float(raw))
            except Exception: kp = COCO17.get(str(raw).strip().lower(), -1)
            if 0 <= kp < 17:
                frames[fr]['xy'][kp] += [float(row[cm['x']]), float(row[cm['y']])]
                frames[fr]['cf'][kp] = max(frames[fr]['cf'][kp], float(row[cm['confidence']]))
                frames[fr]['cnt'][kp] += 1
        except Exception: continue
    if not frames: return None, None
    sf = sorted(frames); seq = np.zeros((len(sf),17,2), np.float32); conf = np.zeros((len(sf),17), np.float32)
    for i, fr in enumerate(sf):
        cnt = np.maximum(frames[fr]['cnt'], 1)[:, None]
        seq[i] = frames[fr]['xy'] / cnt; conf[i] = frames[fr]['cf']
    return seq, conf

def prep(seq, conf):
    n, _ = normalise(seq); n = np.asarray(n, np.float32)
    n[np.asarray(conf) < 0.10] = 0.0
    flat = n.reshape(len(n), -1)
    L = min(len(flat), MAX_T)
    out = np.zeros((MAX_T, 34), np.float32); out[:L] = flat[:L]
    return out, L

def main(budget):
    os.makedirs(OUT, exist_ok=True)
    man = list(csv.DictReader(open(f"{REPO}/results/twinfree/ninefive_core_full/split_manifest_twinfree.csv")))
    n = len(man)
    X = np.lib.format.open_memmap(f"{OUT}/X.npy", mode=("r+" if os.path.exists(f"{OUT}/X.npy") else "w+"),
                                  dtype=np.float32, shape=(n, MAX_T, 34))
    L = np.lib.format.open_memmap(f"{OUT}/L.npy", mode=("r+" if os.path.exists(f"{OUT}/L.npy") else "w+"),
                                  dtype=np.int64, shape=(n,))
    done = np.load(f"{OUT}/done.npy") if os.path.exists(f"{OUT}/done.npy") else np.zeros(n, bool)
    t0 = time.time(); did = 0; mismatch = 0
    for i, r in enumerate(man):
        if done[i]: continue
        if time.time() - t0 > budget: break
        p = r['path'].replace(WORK_OLD, WORK_NEW)
        seq, conf = parse_csv_flexible(open(p, errors='ignore').read())
        if seq is None:
            print("PARSE FAIL", i, p); done[i] = True; continue
        if len(seq) != int(r['n_frames']): mismatch += 1
        x, l = prep(seq, conf)
        X[i] = x; L[i] = l; done[i] = True; did += 1
    np.save(f"{OUT}/done.npy", done)
    if done.all():
        meta = dict(
            split=[r['split'] for r in man], label=[int(r['label']) for r in man],
            scenario=[r['scenario'] for r in man], session=[r['session_id'] for r in man],
            clip_id=[r['clip_id'] for r in man])
        json.dump(meta, open(f"{OUT}/meta.json", "w"))
    print(f"chunk: parsed {did}, total {int(done.sum())}/{n}, frame-count mismatches so far {mismatch}")
    return 0 if done.all() else 3

if __name__ == "__main__":
    sys.exit(main(float(sys.argv[1]) if len(sys.argv) > 1 else 30.0))
