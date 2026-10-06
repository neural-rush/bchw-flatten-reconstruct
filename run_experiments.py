"""Run all experiments, print the table, save results/results.csv and results.md."""
import csv
import json
import os
import time

import numpy as np

from src.data import load_dataset_samples, synthetic_fmap
from src.flatten import flatten_bchw, reconstruct_bchw, flat_index
from src.metrics import errors
from src.reference import reference_check


def manual_example():
    B, C, H, W = 1, 2, 2, 3
    x = np.zeros((B, C, H, W), dtype=np.float32)
    v = 0
    for c in range(C):
        for h in range(H):
            for w in range(W):
                x[0, c, h, w] = v
                v += 1
    print("Manual example: B=1, C=2, H=2, W=3 (CHW=12)")
    for (c, h, w) in [(0, 0, 0), (0, 0, 2), (0, 1, 0), (1, 0, 0), (1, 1, 2)]:
        j = flat_index(c, h, w, H, W)
        print(f"  I[0,{c},{h},{w}] -> flat[0,{j}]  ({c}*{H*W} + {h}*{W} + {w} = {j})")
    flat = flatten_bchw(x, B, C, H, W)
    print("  flat row:", [int(t) for t in flat[0]], "\n")


def main():
    manual_example()
    B = 2
    cases = list(load_dataset_samples(B).items())
    for i, C in enumerate([8, 16, 32, 64, 128, 256, 500]):
        cases.append((f"Synthetic fmap{i+1}", synthetic_fmap(B, C, 32, 32, seed=i)))

    rows = []
    hdr = f"{'Dataset/Input':<24}{'B':>3}{'C':>5}{'H':>4}{'W':>4}{'Emax':>10}{'MAE':>10}{'ref ok':>8}{'time(s)':>9}"
    print(hdr)
    for name, x in cases:
        b, c, h, w = x.shape
        t0 = time.time()
        flat = flatten_bchw(x, b, c, h, w)
        back = reconstruct_bchw(flat, b, c, h, w)
        e_max, mae = errors(x, back, b, c, h, w)
        dt = time.time() - t0
        ok = reference_check(x, flat, back)
        rows.append([name, b, c, h, w, e_max, mae, ok, round(dt, 3)])
        print(f"{name:<24}{b:>3}{c:>5}{h:>4}{w:>4}{e_max:>10.1e}{mae:>10.1e}{str(ok):>8}{dt:>9.2f}")

    cols = ["Dataset/Input", "B", "C", "H", "W", "Emax", "MAE", "RefMatch", "Time_s"]
    with open("results/results.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(cols)
        wr.writerows(rows)
    with open("results/results.md", "w") as f:
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "---|" * len(cols) + "\n")
        for r in rows:
            f.write("| " + " | ".join(f"{v:.1e}" if isinstance(v, float) and k in (5, 6) else str(v)
                                      for k, v in enumerate(r)) + " |\n")

    # Data for the web app (loaded by docs/index.html via a <script> tag)
    os.makedirs("docs", exist_ok=True)
    with open("docs/results.js", "w") as f:
        f.write("window.RESULTS = " + json.dumps([dict(zip(cols, r)) for r in rows], indent=2) + ";\n")


if __name__ == "__main__":
    main()