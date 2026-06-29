import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


INPUT = Path("data/20260625-174851-bench_model_lora_decode.jsonl")
OUTPUT = Path("data/20260625-174851-bench_model_lora_decode_32layer_batch_by_seqlen.png")


def main():
    rows = [json.loads(line) for line in INPUT.open() if line.strip()]
    rows = [row for row in rows if row["setup"]["num_layers"] == 32]

    by_seq = {}
    for row in rows:
        seqlen = row["setup"]["seqlen"]
        batch_size = row["setup"]["batch_size"]
        latency_ms = row["latency"]["avg"] * 1000.0
        by_seq.setdefault(seqlen, []).append((batch_size, latency_ms))

    seqs = sorted(by_seq)
    norm = Normalize(vmin=min(seqs), vmax=max(seqs))
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(6, 4), dpi=180)
    for seqlen in seqs:
        points = sorted(by_seq[seqlen])
        xs = [batch_size for batch_size, _ in points]
        ys = [latency_ms for _, latency_ms in points]
        ax.plot(xs, ys, color=cmap(norm(seqlen)), linewidth=1.35, alpha=0.9)

    ax.set_title("LoRA Decode Latency, 32-layer Model")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Average latency (ms)")
    ax.set_xlim(1, 36)
    ax.grid(True, alpha=0.25)

    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cbar.set_label("Sequence length")

    fig.tight_layout()
    fig.savefig(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
