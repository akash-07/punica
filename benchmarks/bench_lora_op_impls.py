import argparse
import gzip
import itertools
import json
import pathlib
from datetime import datetime

import pytz
import torch
from tqdm import tqdm

import punica.ops

from .benchmark_utils import bench, get_lora_lens


def lora_loop(
    y: torch.Tensor,
    x: torch.Tensor,
    wa: torch.Tensor,
    wb: torch.Tensor,
    s: torch.IntTensor,
    layer_idx: int,
):
    for i in range(len(wa)):
        xi = x[s[i] : s[i + 1]]
        wai = wa[i][layer_idx, :, :]
        wbi = wb[i][layer_idx, :, :]
        y[s[i] : s[i + 1]] += xi @ wai @ wbi


def gather(
    wa_all: list[torch.Tensor],
    wb_all: list[torch.Tensor],
    problem_sizes: list[int],
    layer_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    wa, wb = [], []
    for i, l in enumerate(problem_sizes):
        wa.extend([wa_all[i][layer_idx, :, :]] * l)
        wb.extend([wb_all[i][layer_idx, :, :]] * l)
    wa = torch.stack(wa)
    wb = torch.stack(wb)
    return wa, wb


def bmm(
    y: torch.Tensor,
    x: torch.Tensor,
    wa: torch.Tensor,
    wb: torch.Tensor,
):
    y += (x.unsqueeze(1) @ wa @ wb).squeeze(1)


def lora_gbmm(
    y: torch.Tensor,
    x: torch.Tensor,
    wa_all: list[torch.Tensor],
    wb_all: list[torch.Tensor],
    problem_sizes: list[int],
    layer_idx: int,
):
    wa, wb = gather(wa_all, wb_all, problem_sizes, layer_idx)
    bmm(y, x, wa, wb)


def _repeat_indices(problem_sizes: list[int]) -> list[int]:
    return [i for i, size in enumerate(problem_sizes) for _ in range(size)]


def _make_adapter_to_cluster(
    num_adapters: int,
    num_clusters: int,
    assignment: str = "round_robin",
) -> tuple[list[int], int]:
    if num_clusters < 1:
        raise ValueError("num_clusters must be at least 1")

    if assignment == "round_robin":
        return [i % num_clusters for i in range(num_adapters)], min(num_adapters, num_clusters)

    if assignment == "single":
        return [0] * num_adapters, 1

    if assignment == "distinct":
        if num_clusters < num_adapters:
            raise ValueError("distinct assignment requires num_clusters >= num_adapters")
        return list(range(num_adapters)), num_adapters

    raise KeyError(assignment)

def _make_sigma_bgmv_layout(
    bs: int,
    sigma_popularity: str,
    num_clusters: int,
    cluster_assignment: str = "round_robin",
) -> tuple[
    list[int],              # cluster_indices, length bs
    list[int],              # sigma_indices, length bs
    list[int],              # pair_lens / problem_sizes
    list[tuple[int, int]],  # pairs: (cluster_idx, sigma_idx)
    list[int],              # sigma_lens / adapter_lens
    int,                    # clusters in use
]:
    if sigma_popularity == "Nx8":
        if bs % 8 != 0:
            raise ValueError("Nx8 popularity requires batch size divisible by 8")
        sigma_lens = [bs // 8] * 8
    else:
        sigma_lens = get_lora_lens(bs, sigma_popularity)

    num_sigma = len(sigma_lens)

    sigma_to_cluster, clusters_in_use = _make_adapter_to_cluster(
        num_adapters=num_sigma,
        num_clusters=num_clusters,
        assignment=cluster_assignment,
    )

    sigma_indices = _repeat_indices(sigma_lens)
    cluster_indices = [sigma_to_cluster[sigma_idx] for sigma_idx in sigma_indices]

    order = sorted(
        range(bs),
        key=lambda i: (cluster_indices[i], sigma_indices[i]),
    )

    cluster_indices = [cluster_indices[i] for i in order]
    sigma_indices = [sigma_indices[i] for i in order]

    pair_lens: list[int] = []
    pairs: list[tuple[int, int]] = []

    for cluster_idx, sigma_idx in zip(cluster_indices, sigma_indices):
        pair = (cluster_idx, sigma_idx)
        if pairs and pairs[-1] == pair:
            pair_lens[-1] += 1
        else:
            pairs.append(pair)
            pair_lens.append(1)

    return (
        cluster_indices,
        sigma_indices,
        pair_lens,
        pairs,
        sigma_lens,
        clusters_in_use,
    )

def _make_effective_lora_weights(
    wv: torch.Tensor,
    wsigma: torch.Tensor,
    wu: torch.Tensor,
    pairs: list[tuple[int, int]],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    wa = []
    wb = []

    for cluster_idx, sigma_idx in pairs:
        wa.append(wv[cluster_idx] @ wsigma[sigma_idx])
        wb.append(wu[cluster_idx])

    return wa, wb

@torch.inference_mode()
def bench_lora_with_sigma_op_impls(f, rank: int):
    bs_ = list(range(1, 65))
    sigma_pop_ = ["bmm", "bgmv", "uniform", "zipf:1.5"]
    num_clusters_ = [1, 8, 32, 64]
    h1 = 4096
    h2 = 11008
    r = rank
    num_layers = 1
    dtype = torch.float16
    device = torch.device("cuda:0")

    all_ = list(itertools.product(sigma_pop_, num_clusters_, bs_))
    for sigma_pop, num_clusters, bs in (pbar := tqdm(all_)):
        if sigma_pop == "Nx8" and bs % 8 != 0:
            continue

        cluster_indices, sigma_indices, problem_sizes, pairs, sigma_lens, clusters_in_use = (
            _make_sigma_bgmv_layout(
                bs=bs,
                sigma_popularity=sigma_pop,
                num_clusters=num_clusters,
                cluster_assignment="round_robin",
            )
        )

        num_sigma = len(sigma_lens)

        setup = dict(
            h1=h1,
            h2=h2,
            r=r,
            sigma_popularity=sigma_pop,
            cluster_assignment="round_robin",
            num_sigma=num_sigma,
            num_clusters=num_clusters,
            clusters_in_use=clusters_in_use,
            num_problems=len(problem_sizes),
            batch_size=bs,
        )

        pbar.set_postfix(setup)

        torch.manual_seed(0xABCDABCD987)
        wv = torch.rand(
            (num_clusters, num_layers, h1, r), dtype=dtype, device=device
        )
        wsigma = torch.rand((num_sigma, num_layers, r, r), dtype=dtype, device=device)
        wu = torch.rand((num_clusters, num_layers, r, h2), dtype=dtype, device=device)

        wa, wb = _make_effective_lora_weights(
            wv=wv,
            wsigma=wsigma,
            wu=wu,
            pairs=pairs,
        )
        wa_t = [t.transpose(-1, -2) for t in wa]
        wb_t = [t.transpose(-1, -2) for t in wb]
        wa_ptr = torch.tensor(
            [t.data_ptr() for t in wa_t], dtype=torch.int64, device=device
        )
        wb_ptr = torch.tensor(
            [t.data_ptr() for t in wb_t], dtype=torch.int64, device=device
        )

        wv_t = wv.transpose(-1, -2).contiguous()
        wsigma_t = wsigma.transpose(-1, -2).contiguous()
        wu_t = wu.transpose(-1, -2).contiguous()
        cluster_indices_t = torch.tensor(
            cluster_indices, dtype=torch.long, device=device
        )
        sigma_indices_t = torch.tensor(sigma_indices, dtype=torch.long, device=device)
        s = torch.cumsum(
            torch.tensor([0] + problem_sizes, device=device), dim=0, dtype=torch.int32
        )
        x = torch.rand((bs, h1), dtype=dtype, device=device)
        y = torch.rand((bs, h2), dtype=dtype, device=device)

        l_gbmm = bench(
            lambda: lora_gbmm(y, x, wa, wb, problem_sizes, layer_idx=0),
            warmup=200,
            repeat=1000,
        )
        l_sgmv = bench(
            lambda: punica.ops.add_lora_sgmv_custom_cutlass(
                y, x, wa_ptr, wb_ptr, s, layer_idx=0, lora_rank=r
            ),
            warmup=200,
            repeat=1000,
        )
        l_sigma_bgmv = bench(
            lambda: punica.ops.add_lora_with_sigma_bgmv(
                y,
                x,
                wv_t,
                wsigma_t,
                wu_t,
                cluster_indices_t,
                sigma_indices_t,
                layer_idx=0,
                scale=1.0,
            ),
            warmup=200,
            repeat=1000,
        )

        result = {
            "setup": setup,
            "gbmm": dict(avg=l_gbmm.avg(), std=l_gbmm.std()),
            "sgmv": dict(avg=l_sgmv.avg(), std=l_sgmv.std()),
            "sigma_bgmv": dict(avg=l_sigma_bgmv.avg(), std=l_sigma_bgmv.std()),
        }
        f.write(json.dumps(result) + "\n")
        f.flush()


@torch.inference_mode()
def bench_lora_op_impls(f, rank: int):
    bs_ = list(range(1, 65))
    pop_ = ["bmm", "bgmv", "uniform", "zipf:1.5", "Nx8"]
    h1 = 4096
    h2 = 11008
    r = rank
    num_layers = 1
    dtype = torch.float16
    device = torch.device("cuda:0")

    all_ = list(itertools.product(pop_, bs_))
    for pop, bs in (pbar := tqdm(all_)):
        if pop == "Nx8":
            if bs % 8 != 0:
                continue
            problem_sizes = [(bs // 8)] * 8
        else:
            problem_sizes = get_lora_lens(bs, pop)

        setup = dict(
            h1=h1,
            h2=h2,
            r=r,
            popularity=pop,
            num_problems=len(problem_sizes),
            batch_size=bs,
        )
        pbar.set_postfix(setup)

        torch.manual_seed(0xABCDABCD987)
        wa = [
            torch.rand((num_layers, h1, r), dtype=dtype, device=device)
            for _ in range(len(problem_sizes))
        ]
        wb = [
            torch.rand((num_layers, r, h2), dtype=dtype, device=device)
            for _ in range(len(problem_sizes))
        ]
        wa_t = [t.transpose(-1, -2) for t in wa]
        wb_t = [t.transpose(-1, -2) for t in wb]
        wa_ptr = torch.tensor(
            [t.data_ptr() for t in wa_t], dtype=torch.int64, device=device
        )
        wb_ptr = torch.tensor(
            [t.data_ptr() for t in wb_t], dtype=torch.int64, device=device
        )
        s = torch.cumsum(
            torch.tensor([0] + problem_sizes, device=device), dim=0, dtype=torch.int32
        )
        x = torch.rand((s[-1], h1), dtype=dtype, device=device)
        y = torch.rand((s[-1], h2), dtype=dtype, device=device)
        gwa, gwb = gather(wa, wb, problem_sizes, layer_idx=0)

        l_loop = bench(
            lambda: lora_loop(y, x, wa, wb, s, layer_idx=0), warmup=200, repeat=1000
        )
        l_gbmm = bench(
            lambda: lora_gbmm(y, x, wa, wb, problem_sizes, layer_idx=0),
            warmup=200,
            repeat=1000,
        )
        l_gather = bench(
            lambda: gather(wa, wb, problem_sizes, layer_idx=0), warmup=200, repeat=1000
        )
        l_bmm = bench(lambda: bmm(y, x, gwa, gwb), warmup=200, repeat=1000)
        l_sgmv = bench(
            lambda: punica.ops.add_lora_sgmv_custom_cutlass(
                y, x, wa_ptr, wb_ptr, s, layer_idx=0, lora_rank=r
            ),
            warmup=200,
            repeat=1000,
        )

        result = {
            "setup": setup,
            "loop": dict(avg=l_loop.avg(), std=l_loop.std()),
            "gbmm": dict(avg=l_gbmm.avg(), std=l_gbmm.std()),
            "gather": dict(avg=l_gather.avg(), std=l_gather.std()),
            "bmm": dict(avg=l_bmm.avg(), std=l_bmm.std()),
            "sgmv": dict(avg=l_sgmv.avg(), std=l_sgmv.std()),
        }
        f.write(json.dumps(result) + "\n")
        f.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--with-sigma",
        action="store_true",
        help="benchmark add_lora_with_sigma_bgmv against sgmv and lora_gbmm",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=16,
        help="LoRA rank to benchmark",
    )
    args = parser.parse_args()
    if args.rank < 1:
        raise SystemExit("--rank must be at least 1")

    this_file = pathlib.Path(__file__)
    project_root = this_file.parents[1]
    now = datetime.now(pytz.timezone("US/Pacific"))
    suffix = "with_sigma" if args.with_sigma else this_file.stem
    out_filename = f"{now:%Y%m%d-%H%M%S}-{suffix}.jsonl.gz"
    out_path = project_root / "data" / out_filename

    print(out_path)
    with gzip.open(out_path, "wt") as f:
        if args.with_sigma:
            bench_lora_with_sigma_op_impls(f, rank=args.rank)
        else:
            bench_lora_op_impls(f, rank=args.rank)


if __name__ == "__main__":
    main()
