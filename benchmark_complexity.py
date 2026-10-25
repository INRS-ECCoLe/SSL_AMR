#!/usr/bin/env python3
"""Table I -- computational complexity and CPU inference timing of all models.

For every model in models.MODEL_REGISTRY this reports:
  Params     trainable parameters
  Wts (MB)   FP32 parameter + buffer storage (1 MB = 1e6 bytes)
  Act (MB)   cumulative size of leaf-module output tensors in one forward pass
             at batch size 1 (an upper bound on activation memory)
  GFLOPs     2 x MACs + element-wise ops per frame, counted with forward hooks
  latency    best / average / worst per-frame time over timed passes, both
             single-sample (batch 1) and amortized over a batch

CPU only, by design. Latency depends on the machine: report numbers measured
on the CPU named in the paper. Params / Wts / Act / GFLOPs do not.

USAGE
  python benchmark_complexity.py                 # writes complexity.csv + .tex
  python benchmark_complexity.py --threads 1 --iters 50
"""
import argparse
import csv
import platform
import statistics
import time

import torch
import torch.nn as nn

from models import MODEL_REGISTRY

DEVICE = torch.device("cpu")
SEQ_LEN = 1024
MB, GIGA = 1e6, 1e9
NUM_CLASSES = 24
GROUP_TITLE = {"traditional": "Traditional Models",
               "physics-informed": "Physics-Informed Models",
               "recent": "Recent Methods"}


def param_stats(model):
    """Exact parameter count and fp32 weight+buffer footprint in decimal MB."""
    n = sum(p.numel() for p in model.parameters())
    b = (sum(p.numel() * p.element_size() for p in model.parameters())
         + sum(q.numel() * q.element_size() for q in model.buffers()))
    return n, b / MB


def count_macs(model):
    """Exact per-frame MACs / FLOPs / activation bytes (hook based, batch = 1).

    Counted analytically rather than with thop, so the result is exact and
    needs no third-party package. The recurrent hook counts gate GEMMs
    explicitly, which thop approximates and sometimes omits for bidirectional
    layers.
    """
    store = {"macs": 0.0, "ew": 0.0, "act_bytes": 0.0}
    handles = []

    def lin(m, i, o):
        store["macs"] += o.numel() * m.in_features
        if m.bias is not None:
            store["ew"] += o.numel()

    def conv(m, i, o):
        k = 1
        for kk in m.kernel_size:
            k *= kk
        store["macs"] += o.numel() * (m.in_channels // m.groups) * k
        if m.bias is not None:
            store["ew"] += o.numel()

    def bn(m, i, o):
        n = i[0].numel()
        store["macs"] += n
        store["ew"] += n

    def rnn(gates):
        def hook(m, i, o):
            x = i[0]
            B, T = x.shape[0], x.shape[1]
            H = m.hidden_size
            nd = 2 if m.bidirectional else 1
            for layer in range(m.num_layers):
                in_dim = m.input_size if layer == 0 else H * nd
                store["macs"] += B * T * nd * gates * (H * in_dim + H * H)
                store["ew"]   += B * T * nd * (gates + 3) * H
        return hook

    def act(m, i, o):
        def walk(t):
            if torch.is_tensor(t):
                store["act_bytes"] += t.numel() * t.element_size()
            elif isinstance(t, (list, tuple)):
                for e in t:
                    walk(e)
        walk(o)

    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            handles.append(mod.register_forward_hook(lin))
        elif isinstance(mod, (nn.Conv1d, nn.Conv2d)):
            handles.append(mod.register_forward_hook(conv))
        elif isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
            handles.append(mod.register_forward_hook(bn))
        elif isinstance(mod, nn.LSTM):
            handles.append(mod.register_forward_hook(rnn(4)))
        elif isinstance(mod, nn.GRU):
            handles.append(mod.register_forward_hook(rnn(3)))
        if len(list(mod.children())) == 0:
            handles.append(mod.register_forward_hook(act))

    model.eval()
    raw = torch.randn(1, SEQ_LEN, 2, device=DEVICE)
    with torch.no_grad():
        model(raw)
    for h in handles:
        h.remove()
    return store["macs"], 2.0 * store["macs"] + store["ew"], store["act_bytes"]


def bench_latency(model, batch, warmup, iters):
    """Per-frame latency in ms: returns (best, average, worst) over `iters`."""
    model.eval()
    x = torch.randn(batch, SEQ_LEN, 2, device=DEVICE)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        samples = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(x)
            samples.append((time.perf_counter() - t0) * 1e3 / batch)
    return min(samples), statistics.fmean(samples), max(samples)


def main():
    ap = argparse.ArgumentParser(description="Complexity / latency benchmark")
    ap.add_argument("--threads", type=int, default=None,
                    help="torch CPU threads (1 gives the most repeatable timing)")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=50, help="timed single-sample passes")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--batch-iters", type=int, default=20)
    ap.add_argument("--out", default="complexity")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    print(f"CPU: {platform.processor() or platform.machine()} | torch {torch.__version__}"
          f" | threads {torch.get_num_threads()}")

    order = sorted(MODEL_REGISTRY, key=lambda k: list(GROUP_TITLE).index(MODEL_REGISTRY[k][1]))
    rows = []
    for name in order:
        cls, group = MODEL_REGISTRY[name]
        model = cls(num_classes=NUM_CLASSES).to(DEVICE).eval()
        params, mem = param_stats(model)
        macs, flops, act = count_macs(model)
        b1 = bench_latency(model, 1, a.warmup, a.iters)
        bb = bench_latency(model, a.batch, a.warmup, a.batch_iters)
        rows.append(dict(group=group, model=name, params=params, wts_mb=mem,
                         act_mb=act / MB, gflops=flops / GIGA,
                         b1_best_ms=b1[0], b1_avg_ms=b1[1], b1_worst_ms=b1[2],
                         batch_best_ms=bb[0], batch_avg_ms=bb[1], batch_worst_ms=bb[2]))
        print(f"  {name:<24} {params:>10,}  {flops / GIGA:.4f} GFLOPs  {b1[1]:.3f} ms (batch 1)")

    with open(f"{a.out}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {a.out}.csv")

    with open(f"{a.out}.tex", "w") as f:
        f.write("\\begin{tabular}{lrrrrrrr}\n\\hline\\hline\n")
        f.write("\\textbf{Model} & \\textbf{Params} & \\textbf{Wts.} & \\textbf{Act.} & "
                "\\textbf{GFLOPs} & \\textbf{Best} & \\textbf{Avg} & \\textbf{Worst} \\\\\n")
        f.write(" & & \\textbf{(MB)} & \\textbf{(MB)} & & \\textbf{(ms)} & \\textbf{(ms)} & "
                "\\textbf{(ms)} \\\\\n\\hline\n")
        last = None
        for r in rows:
            if r["group"] != last:
                f.write(f"\\multicolumn{{8}}{{l}}{{\\textit{{{GROUP_TITLE[r['group']]}}}}} \\\\\n\\hline\n")
                last = r["group"]
            f.write(f"{r['model'].replace('_', '-')} & {r['params']:,} & {r['wts_mb']:.3f} & "
                    f"{r['act_mb']:.2f} & {r['gflops']:.4f} & {r['b1_best_ms']:.4f} & "
                    f"{r['b1_avg_ms']:.4f} & {r['b1_worst_ms']:.4f} \\\\\n")
        f.write("\\hline\\hline\n\\end{tabular}\n")
    print(f"wrote {a.out}.tex  (latency columns: single-sample, batch 1)")


if __name__ == "__main__":
    main()
