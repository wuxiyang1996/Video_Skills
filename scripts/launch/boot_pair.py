"""Paired bootstrap between any two answer-chain sidecars, by question type.
Usage: boot_pair.py <A.rows.jsonl> <B.rows.jsonl> [labelA] [labelB]
Reports B - A accuracy with 95% CI overall and per type; pairs by example_id;
drops error / planner-fallback rows. 10k draws, seed 20260903."""
_IDX = 'dataset_clip_wrapper/output/vh_full_questions_v1/example_index.json'
def _qtypes():
    import json, os, glob
    if os.path.exists(_IDX):
        return {k: v["question_type"] for k, v in json.load(open(_IDX)).items()}
    out = {}
    for p in glob.glob(f'dataset_clip_wrapper/output/vh_full_questions_v1/**/04_l1_example.json', recursive=True):
        try:
            e = json.load(open(p)); out[str(e.get("example_id"))] = (e.get("question") or {}).get("question_type") or "?"
        except Exception:
            pass
    return out
import json, glob, sys, random, collections, os
A, B = sys.argv[1], sys.argv[2]; la = sys.argv[3] if len(sys.argv) > 3 else 'A'; lb = sys.argv[4] if len(sys.argv) > 4 else 'B'
L = '/fs/gamma-projects/vlm-robot/Video_Skills/dataset_clip_wrapper/output/vh_full_questions_v1'
qt = _qtypes()
def rows(path):
    out = {}
    for l in open(path):
        if not l.strip(): continue
        r = json.loads(l)
        if 'error' in r or r.get('planner_fell_back'): continue
        out[r['example_id']] = r
    return out
X, Y = rows(A), rows(B); ids = sorted(set(X) & set(Y))
def boot(sub, n=10000, seed=20260903):
    d = [float(Y[e]['correct']) - float(X[e]['correct']) for e in sub]
    if not d: return None
    rng = random.Random(seed); N = len(d); m = sum(d) / N
    s = sorted(sum(d[rng.randrange(N)] for _ in range(N)) / N for _ in range(n))
    return 100 * m, 100 * s[int(.025 * n)], 100 * s[int(.975 * n)], N
acc = lambda sub, Z: 100 * sum(Z[e]['correct'] for e in sub) / max(1, len(sub))
print(f'paired n={len(ids)}   ({la}: {len(X)} rows, {lb}: {len(Y)} rows)')
print(f'{"type":6s} {"n":>5} {la:>8} {lb:>8}   {lb}-{la} 95% CI')
by = collections.defaultdict(list)
for e in ids: by[qt.get(e, '?')].append(e)
for t, sub in sorted(by.items(), key=lambda kv: -len(kv[1])) + [('ALL', ids)]:
    b = boot(sub)
    if not b: continue
    m, lo, hi, N = b; sig = '*' if (lo > 0 or hi < 0) else ' '
    print(f'{t:6s} {N:5d} {acc(sub, X):7.1f}% {acc(sub, Y):7.1f}%   {m:+6.2f} [{lo:+6.2f}, {hi:+6.2f}] {sig}')
