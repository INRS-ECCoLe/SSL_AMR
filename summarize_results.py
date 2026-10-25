#!/usr/bin/env python3
"""Collect the JSON results written by train.py into CSV tables.

Writes to --out (default: results_summary/):
  supervised_snr_accuracy.csv   model, percent, snr_db, accuracy
  ssl_iterations.csv            model, percent, node_db, threshold, iteration,
                                accuracy at the node's own SNR
                                (iteration 0 = supervised seed)
  ssl_per_node.csv              per model / percent / node:
                                  seed accuracy,
                                  mean and std across thresholds of the
                                  final-iteration accuracy and of the
                                  best-iteration accuracy
  ssl_plateau.csv               per model / percent: mean of the per-node
                                values over SNR >= 10 dB and over -6..0 dB

The paper reports, per node, the mean across the swept thresholds. Select the
column that matches your reporting rule (final or best iteration per threshold).

USAGE
  python summarize_results.py                       # reads results/
  python summarize_results.py --root my_results --out tables
"""
import os
import re
import csv
import glob
import json
import argparse
from collections import defaultdict

import numpy as np


def rel_parts(path):
    return path.replace('\\', '/').split('/')


def collect_supervised(root):
    rows = []
    for p in glob.glob(os.path.join(root, '**', 'test_metrics.json'), recursive=True):
        m = re.search(r'supervised_(\d+)_percent', p.replace('\\', '/'))
        if not m:
            continue
        parts = rel_parts(p)
        model = parts[parts.index('Supervised') - 1]
        acc = json.load(open(p)).get('snr_accuracies', {})
        for s, v in acc.items():
            rows.append([model, int(m.group(1)), int(s), float(v)])
    return sorted(rows)


def collect_ssl(root):
    rows = []
    for nd in glob.glob(os.path.join(root, '**', 'Node_*dB'), recursive=True):
        u = nd.replace('\\', '/')
        if 'SSL_Softmax' not in u:
            continue
        pct = int(re.search(r'ssl_(\d+)_percent', u).group(1))
        node = int(re.search(r'Node_(-?\d+)dB', u).group(1))
        parts = rel_parts(nd)
        model = parts[parts.index('SSL_Softmax') - 1]
        key = str(node)
        bp = os.path.join(nd, 'base_metrics.json')
        if os.path.exists(bp):
            acc = json.load(open(bp)).get('snr_accuracies', {})
            if key in acc:
                rows.append([model, pct, node, '', 0, float(acc[key])])
        for tf in glob.glob(os.path.join(nd, 'chunks_*', 'threshold_*.json')):
            t = json.load(open(tf))
            for it in t.get('iterations_data', []):
                met = it.get('metrics')
                if not isinstance(met, dict):        # no pseudo-labels yet
                    continue
                acc = met.get('snr_accuracies', {})
                if key in acc:
                    rows.append([model, pct, node, float(t['threshold']),
                                 int(it['stats']['iteration']), float(acc[key])])
    return sorted(rows, key=lambda r: (r[0], -r[1], r[2], str(r[3]), r[4]))


def per_node(ssl_rows):
    seed = {}
    runs = defaultdict(lambda: defaultdict(list))      # (m,p,n) -> tau -> [(it, acc)]
    for m, p, n, tau, it, acc in ssl_rows:
        if it == 0:
            seed[(m, p, n)] = acc
        else:
            runs[(m, p, n)][tau].append((it, acc))
    out = []
    for key in sorted(set(seed) | set(runs), key=lambda k: (k[0], -k[1], k[2])):
        final = [max(v)[1] for v in runs[key].values()] if key in runs else []
        best = [max(a for _, a in v) for v in runs[key].values()] if key in runs else []
        f = lambda xs, fn: f"{fn(xs):.6f}" if xs else ''
        out.append([*key, f"{seed.get(key, float('nan')):.6f}", len(final),
                    f(final, np.mean), f(final, lambda x: np.std(x, ddof=1) if len(x) > 1 else 0.0),
                    f(best, np.mean), f(best, lambda x: np.std(x, ddof=1) if len(x) > 1 else 0.0)])
    return out


def plateau(node_rows):
    groups = defaultdict(lambda: {'hi': [], 'lo': []})
    for m, p, n, _, _, fin_mean, _, best_mean, _ in node_rows:
        if fin_mean == '':
            continue
        if n >= 10:
            groups[(m, p)]['hi'].append((float(fin_mean), float(best_mean)))
        if -6 <= n <= 0:
            groups[(m, p)]['lo'].append((float(fin_mean), float(best_mean)))
    out = []
    for (m, p), g in sorted(groups.items(), key=lambda kv: (kv[0][0], -kv[0][1])):
        hi, lo = np.array(g['hi']), np.array(g['lo'])
        mean = lambda a, c: f"{100 * a[:, c].mean():.2f}" if len(a) else ''
        out.append([m, p, mean(hi, 0), mean(lo, 0), mean(hi, 1), mean(lo, 1)])
    return out


def write(path, header, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"wrote {path}  ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser(description="Summarize train.py results as CSV")
    ap.add_argument('--root', default='results')
    ap.add_argument('--out', default='results_summary')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    sup = collect_supervised(a.root)
    write(os.path.join(a.out, 'supervised_snr_accuracy.csv'),
          ['model', 'percent', 'snr_db', 'accuracy'], sup)

    ssl = collect_ssl(a.root)
    write(os.path.join(a.out, 'ssl_iterations.csv'),
          ['model', 'percent', 'node_db', 'threshold', 'iteration', 'accuracy'], ssl)

    nodes = per_node(ssl)
    write(os.path.join(a.out, 'ssl_per_node.csv'),
          ['model', 'percent', 'node_db', 'seed_accuracy', 'n_thresholds',
           'final_iter_mean', 'final_iter_std', 'best_iter_mean', 'best_iter_std'], nodes)

    write(os.path.join(a.out, 'ssl_plateau.csv'),
          ['model', 'percent', 'plateau_pct_final', 'low_snr_pct_final',
           'plateau_pct_best', 'low_snr_pct_best'], plateau(nodes))


if __name__ == '__main__':
    main()
