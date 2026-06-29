import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


INPUT = Path("data/20260628-122442-bench_lora_op_impls.jsonl")
OUTPUT = Path("data/plots/20260628-122442-bench_lora_op_impls/lora_op_impls_by_popularity.png")
POPULARITIES = ("bgmv", "uniform", "zipf:1.5", "bmm", "Nx8")
KERNELS = ("gbmm", "gather", "bmm", "sgmv")
LABELS = {
    "gbmm": "gbmm",
    "gather": "gather",
    "bmm": "bmm",
    "sgmv": "sgmv",
}

POPULARITIES_LABELS = {
    "bmm" : "Identical", 
    "bgmv": "Distinct", 
    "uniform": "Uniform",
    "zipf:1.5": "Skewed",
}


def open_jsonl(path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open()


def load_rows(path):
    with open_jsonl(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def group_by_popularity(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["setup"]["popularity"]].append(row)
    for pop_rows in groups.values():
        pop_rows.sort(key=lambda row: row["setup"]["batch_size"])
    return groups


def latency_us(row, kernel):
    return row[kernel]["avg"] * 1_000_000.0


def plot(rows, output):
    groups = group_by_popularity(rows)
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.2), dpi=180, sharey=True)

    for ax, popularity in zip(axes, POPULARITIES):
        pop_rows = groups.get(popularity, [])
        if not pop_rows:
            ax.set_visible(False)
            continue

        xs = [row["setup"]["batch_size"] for row in pop_rows]
        for kernel in KERNELS:
            ys = [latency_us(row, kernel) for row in pop_rows]
            ax.plot(xs, ys, linewidth=1.2, label=LABELS[kernel])

        ax.set_title(POPULARITIES_LABELS.get(popularity, popularity))
        ax.set_xlabel("Batch size")
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("Latency (us)")
    axes[0].legend(frameon=False, fontsize=8)
    # fig.suptitle("LoRA op implementations by popularity")
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    print(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    plot(load_rows(args.input), args.output)


if __name__ == "__main__":
    main()
