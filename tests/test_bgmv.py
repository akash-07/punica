import pytest
import torch

import punica.ops


def assert_close(a, b, rtol=None, atol=None):
    if rtol is None or atol is None:
        rtol, atol = {
            torch.float16: (5e-3, 5e-3),
            torch.bfloat16: (3e-2, 2e-2),
        }[a.dtype]
        
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol)


def _lora_ref_impl(
    y: torch.Tensor,
    x: torch.Tensor,
    wa_T_all: torch.Tensor,
    wb_T_all: torch.Tensor,
    indicies: torch.LongTensor,
    layer_idx: int,
    scale: float,
):
    bs = x.shape[0]
    s = torch.tensor(scale, dtype=torch.float32, device=x.device)
    for i, lora_idx in zip(range(bs), indicies.cpu().tolist()):
        xi = x[i].unsqueeze(0).to(torch.float32)
        wa = wa_T_all[lora_idx, layer_idx].transpose(-1, -2).to(torch.float32)
        wb = wb_T_all[lora_idx, layer_idx].transpose(-1, -2).to(torch.float32)

        tmp = (xi @ wa).to(x.dtype).to(torch.float32)
        y[i] += (tmp @ wb).squeeze(0) * s


def _lora_with_sigma_ref_impl(
    y: torch.Tensor,
    x: torch.Tensor,
    wv_T_all: torch.Tensor,
    wsigma_T_all: torch.Tensor,
    wu_T_all: torch.Tensor,
    cluster_indicies: torch.LongTensor,
    lora_indicies: torch.LongTensor,
    layer_idx: int,
    scale: float,
):
    bs = x.shape[0]
    s = torch.tensor(scale, dtype=torch.float32, device=x.device)
    for i, cluster_idx, lora_idx in zip(
        range(bs), cluster_indicies.cpu().tolist(), lora_indicies.cpu().tolist()
    ):
        xi = x[i].unsqueeze(0).to(torch.float32)
        wv = wv_T_all[cluster_idx, layer_idx].transpose(-1, -2).to(torch.float32)
        wsigma = (
            wsigma_T_all[lora_idx, layer_idx].transpose(-1, -2).to(torch.float32)
        )
        wu = wu_T_all[cluster_idx, layer_idx].transpose(-1, -2).to(torch.float32)

        tmp = (xi @ wv).to(x.dtype).to(torch.float32)
        tmp_sigma = (tmp @ wsigma).to(x.dtype).to(torch.float32)
        y[i] += (tmp_sigma @ wu).squeeze(0) * s


@pytest.mark.parametrize("dtype_str", ["float16", "bfloat16"])
@pytest.mark.parametrize("r", [8, 16, 32, 64])
@torch.inference_mode()
def test_square_bgmv_correctness(dtype_str, r):
    torch.manual_seed(0xBEEFBEEF)
    num_loras = 4
    num_layers = 3
    bs = 11
    scale = 0.375
    dtype = getattr(torch, dtype_str)
    device = torch.device("cuda:0")

    w_T_all = torch.randn(num_loras, num_layers, r, r, dtype=dtype, device=device)
    indices = torch.randint(num_loras, (bs,), dtype=torch.long, device=device)

    for layer_idx in range(num_layers):
        x = torch.randn(bs, r, dtype=dtype, device=device)
        y = torch.randn(bs, r, dtype=dtype, device=device)

        y_ref = y.clone()
        for i, lora_idx in zip(range(bs), indices.cpu().tolist()):
            xi = x[i].unsqueeze(0).to(torch.float32)
            w = w_T_all[lora_idx, layer_idx].transpose(-1, -2).to(torch.float32)
            y_ref[i] += (xi @ w).squeeze(0) * scale

        y_our = y.clone()
        punica.ops.bgmv(y_our, x, w_T_all, indices, layer_idx, scale)

        assert_close(y_ref, y_our)


@pytest.mark.parametrize("dtype_str", ["float16", "bfloat16"])
@torch.inference_mode()
def test_lora_correctness(dtype_str):
    torch.manual_seed(0xABCDABCD987)
    num_loras = 4
    num_layers = 5
    h1 = 4096
    h2 = 11008
    r = 8
    bs = 32
    scale = 0.123
    dtype = getattr(torch, dtype_str)
    device = torch.device("cuda:0")

    wa_T_all = torch.randn(num_loras, num_layers, r, h1, dtype=dtype, device=device)
    wb_T_all = torch.randn(num_loras, num_layers, h2, r, dtype=dtype, device=device)
    indices = torch.randint(num_loras, (bs,), dtype=torch.long, device=device)

    for layer_idx in range(num_layers):
        x = torch.randn(bs, h1, dtype=dtype, device=device)
        y = torch.randn(bs, h2, dtype=dtype, device=device)

        y_ref = y.clone()
        _lora_ref_impl(y_ref, x, wa_T_all, wb_T_all, indices, layer_idx, scale)
        y_our = y.clone()
        punica.ops.add_lora_bgmv(
            y_our, x, wa_T_all, wb_T_all, indices, layer_idx, scale
        )

        assert_close(y_ref, y_our)


@pytest.mark.parametrize("dtype_str", ["float16", "bfloat16"])
@torch.inference_mode()
def test_lora_with_sigma_correctness(dtype_str):
    torch.manual_seed(0xABCDABCD987)
    num_clusters = 3
    num_loras = 7
    num_layers = 5
    h1 = 4096
    h2 = 11008
    r = 16
    bs = 32
    scale = 0.123
    dtype = getattr(torch, dtype_str)
    device = torch.device("cuda:0")

    wv_T_all = torch.randn(
        num_clusters, num_layers, r, h1, dtype=dtype, device=device
    )
    wsigma_T_all = torch.randn(
        num_loras, num_layers, r, r, dtype=dtype, device=device
    )
    wu_T_all = torch.randn(
        num_clusters, num_layers, h2, r, dtype=dtype, device=device
    )
    cluster_indices = torch.arange(bs, dtype=torch.long, device=device) % num_clusters
    lora_indices = (
        torch.arange(bs, dtype=torch.long, device=device) * 2 + 1
    ) % num_loras

    for layer_idx in range(num_layers):
        x = torch.randn(bs, h1, dtype=dtype, device=device)
        y = torch.randn(bs, h2, dtype=dtype, device=device)

        y_ref = y.clone()
        _lora_with_sigma_ref_impl(
            y_ref,
            x,
            wv_T_all,
            wsigma_T_all,
            wu_T_all,
            cluster_indices,
            lora_indices,
            layer_idx,
            scale,
        )
        y_our = y.clone()
        punica.ops.add_lora_with_sigma_bgmv(
            y_our,
            x,
            wv_T_all,
            wsigma_T_all,
            wu_T_all,
            cluster_indices,
            lora_indices,
            layer_idx,
            scale,
        )

        assert_close(y_ref, y_our)