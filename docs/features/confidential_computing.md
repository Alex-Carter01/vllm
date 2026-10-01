# Confidential Computing

vLLM detects NVIDIA GPU Confidential Computing (CC) and applies stock
defaults that match bounce-buffer copy semantics. Under bounce-buffer CC,
`cudaMemcpyAsync` / `non_blocking=True` host↔device copies become
**host-synchronous**: the issuing thread blocks until the copy and prior
work on that stream finish.

## Detection

Auto-detection (CUDA) prefers `nvmlSystemGetConfComputeSettings` and falls
back to `nvmlSystemGetConfComputeState`. Bounce-buffer mode is treated as
active when any of the following hold:

- `ccFeature` enabled (classic CC)
- `devToolsMode` on (DevTools CC workflows still use bounce buffers)
- `multiGpuMode == PROTECTED_PCIE` (Hopper PPCIe — often reported as
  `CC status: OFF` by `nvidia-smi conf-compute -f`)

Override with `VLLM_CONFIDENTIAL_COMPUTE=0` or `1` for testing / hosts
without NVML. The vendored NVML bindings also define
`NVML_CC_SYSTEM_MULTIGPU_NVLE` (Blackwell NVLink encryption).

Inspect the startup log line that reports the detection source and the
resolved `model_runner` / `async_scheduling` stock defaults.

## Stock defaults (when knobs are left unset)

| Knob | Under bounce-buffer CC | Notes |
| --- | --- | --- |
| Model runner | Prefer **V1** when V1 can serve the config | Staged H2D / deferred D2H copy opts are V1-only today ([#52226](https://github.com/vllm-project/vllm/pull/52226)). Set `VLLM_USE_V2_MODEL_RUNNER=1` to force V2. |
| `--async-scheduling` | **On** if V1 copy opts are active; **off** if V2 is selected | Deferred D2H only engages with async scheduling on V1. Explicit `--async-scheduling` / `--no-async-scheduling` always wins. |
| CUDA graphs | Unchanged (keep default on) | Warns if `--enforce-eager` is set under CC. |
| CPU offload | Unchanged | Warns if CPU offload is enabled under CC (bounce-buffer bandwidth is typically only a few GB/s). |
| Batch / `max_num_seqs` / chunked prefill | Unchanged | No stock changes until measured. |

## Serving tip

```bash
# Stock serve under CC (auto V1 + async on when copy opts apply):
vllm serve $MODEL

# Force the pre-opt / V2 policy-inversion comparison:
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve $MODEL
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve $MODEL --no-async-scheduling
```

## Validation matrix (CC GPU required)

Hold model / hardware / ISL-OSL fixed. Sweep:

| Axis | Values |
| --- | --- |
| CC | on vs off |
| Runner | V1 vs V2 (`VLLM_USE_V2_MODEL_RUNNER`) |
| Async scheduling | on vs off |
| Copy opts (#52226) | present on V1; absent on V2 |

Example `vllm bench serve` skeleton (async × fixes 2×2 under CC on):

```bash
MODEL=...   # same checkpoint for all cells
ISL=1024; OSL=1024; CONC=4

# A: V1 + async on  (stock CC default once this branch is used)
vllm serve $MODEL --port 8001
vllm bench serve --model $MODEL --dataset-name random \
  --random-input-len $ISL --random-output-len $OSL --max-concurrency $CONC \
  --port 8001

# B: V1 + async off
vllm serve $MODEL --no-async-scheduling --port 8002
vllm bench serve --model $MODEL --dataset-name random \
  --random-input-len $ISL --random-output-len $OSL --max-concurrency $CONC \
  --port 8002

# C: V2 + async on  (no V1 copy opts)
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve $MODEL --async-scheduling --port 8003
vllm bench serve --model $MODEL --dataset-name random \
  --random-input-len $ISL --random-output-len $OSL --max-concurrency $CONC \
  --port 8003

# D: V2 + async off
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve $MODEL --no-async-scheduling --port 8004
vllm bench serve --model $MODEL --dataset-name random \
  --random-input-len $ISL --random-output-len $OSL --max-concurrency $CONC \
  --port 8004
```

Record output tok/s and median TPOT for each cell. Repeat with CC off for
the policy-inversion baseline.

!!! warning "PPCIe validation gap"
    Published [#52226](https://github.com/vllm-project/vllm/pull/52226) numbers
    are **B300 / NVLE**. Hopper **Protected PCIe (PPCIe)** bounce-buffer
    behavior is detected by this code path but is **not** covered by those
    B300 results — schedule a separate PPCIe concurrency sweep before
    treating PPCIe stock defaults as validated.
