import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# INPUT = Path("data/20260628-142321-with_sigma.jsonl")
INPUT = Path("data/20260629-103211-with_sigma.jsonl")
# OUTPUT_DIR = Path("data/plots/20260628-142321-with_sigma")
OUTPUT_DIR = Path("data/plots/20260629-103211-with_sigma")
# POPULARITIES = ("bgmv", "uniform", "zipf:1.5", "bmm", "Nx8")
POPULARITIES = ("bgmv", "uniform", "zipf:1.5", "bmm")
# KERNELS = ("gbmm", "sgmv", "sigma_bgmv")
KERNELS = ("bgmv", "sgmv", "sigma_bgmv")
LABELS = {
    "gbmm": "gbmm",
    "sgmv": "sgmv",
    "sigma_bgmv": "sigma_bgmv",
    "bgmv": "bgmv",
}
POPULARITY_LABELS = {
    "bmm": "Identical",
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


def average_duplicate_batches(rows):
    buckets = defaultdict(list)
    for row in rows:
        setup = row["setup"]
        key = (setup["sigma_popularity"], setup["batch_size"])
        buckets[key].append(row)

    averaged = []
    for (popularity, batch_size), bucket in buckets.items():
        setup = dict(bucket[0]["setup"])
        setup["sigma_popularity"] = popularity
        setup["batch_size"] = batch_size
        setup["num_sigma"] = ",".join(str(row["setup"]["num_sigma"]) for row in bucket)
        row = {"setup": setup}
        for kernel in KERNELS:
            if kernel in bucket[0]:
                row[kernel] = {
                    "avg": sum(item[kernel]["avg"] for item in bucket) / len(bucket),
                    "std": sum(item[kernel]["std"] for item in bucket) / len(bucket),
                }
        averaged.append(row)
    return averaged


def group_by_popularity(rows):
    groups = defaultdict(list)
    for row in average_duplicate_batches(rows):
        groups[row["setup"]["sigma_popularity"]].append(row)
    for pop_rows in groups.values():
        pop_rows.sort(key=lambda row: row["setup"]["batch_size"])
    return groups


def latency_us(row, kernel):
    return row[kernel]["avg"] * 1_000_000.0


def default_output_path(input_path, num_clusters, num_sigma, rank):
    stem = input_path.name
    for suffix in (".jsonl.gz", ".jsonl"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    sigma_part = "" if num_sigma is None else f"_sigma{num_sigma}"
    rank_part = "" if rank is None else f"_rank{rank}"
    return OUTPUT_DIR / f"{stem}_clusters{num_clusters}{sigma_part}{rank_part}.png"


def filter_rows(rows, num_clusters, num_sigma, rank):
    selected = [row for row in rows if row["setup"]["num_clusters"] == num_clusters]
    if num_sigma is not None:
        selected = [row for row in selected if row["setup"]["num_sigma"] == num_sigma]
    if rank is not None:
        selected = [row for row in selected if row["setup"].get("r") == rank]
    return selected


def plot(rows, output, num_clusters, num_sigma, rank):
    groups = group_by_popularity(rows)
    num_rows = 2
    num_cols = (len(POPULARITIES) + num_rows - 1) // num_rows
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(8, 6), dpi=180, sharey=True)
    axes = axes.ravel()

    for ax, popularity in zip(axes, POPULARITIES):
        pop_rows = groups.get(popularity, [])
        if not pop_rows:
            ax.set_visible(False)
            continue

        xs = [row["setup"]["batch_size"] for row in pop_rows]
        for kernel in KERNELS:
            if kernel not in pop_rows[0]:
                continue
            ys = [latency_us(row, kernel) for row in pop_rows]
            ax.plot(xs, ys, linewidth=1.2, label=LABELS[kernel])

        ax.set_title(POPULARITY_LABELS.get(popularity, popularity))
        ax.set_xlabel("Batch size")
        ax.grid(True, alpha=0.25)

    for ax in axes[len(POPULARITIES) :]:
        ax.set_visible(False)

    axes[0].set_ylabel("Latency (us)")
    axes[0].legend(frameon=False, fontsize=8)
    title = f"num_clusters={num_clusters}"
    if num_sigma is not None:
        title += f", num_sigma={num_sigma}"
    if rank is not None:
        title += f", r={rank}"
    fig.suptitle(title)
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    print(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--num-clusters", type=int, required=True)
    parser.add_argument(
        "--num-sigma",
        type=int,
        default=None,
        help="Optional extra filter. If omitted, duplicate batch-size rows are averaged.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=16,
        help="Optional LoRA rank filter.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = load_rows(args.input)
    selected = filter_rows(rows, args.num_clusters, args.num_sigma, args.rank)
    if not selected:
        clusters = sorted({row["setup"]["num_clusters"] for row in rows})
        sigmas = sorted({row["setup"]["num_sigma"] for row in rows})
        ranks = sorted({row["setup"].get("r") for row in rows if "r" in row["setup"]})
        raise SystemExit(
            f"No rows for num_clusters={args.num_clusters}, num_sigma={args.num_sigma}, "
            f"rank={args.rank}. Available num_clusters={clusters}, num_sigma={sigmas}, "
            f"ranks={ranks}"
        )

    output = args.output or default_output_path(
        args.input, args.num_clusters, args.num_sigma, args.rank
    )
    plot(selected, output, args.num_clusters, args.num_sigma, args.rank)


if __name__ == "__main__":
    main()
