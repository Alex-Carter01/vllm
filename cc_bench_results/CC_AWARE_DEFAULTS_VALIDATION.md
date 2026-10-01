# CC-aware scheduler defaults — hardware validation (blocked on this host)

**Date:** 2026-09-30  
**Host:** Apple M2 Max (darwin arm64) — **no NVIDIA GPU / nvidia-smi**  
**Branch:** `cursor/cc-aware-scheduler-defaults`  
**Fork:** https://github.com/Alex-Carter01/vllm

This machine cannot run the async × copy-opts matrix. Mocked unit tests cover
detection + policy predicates; live numbers need a CC-capable GPU.

## Open validation item

**Hopper PPCIe is not covered by B300 / NVLE numbers from #52226.** Schedule
separate PPCIe runs before claiming PPCIe parity.

## Exact commands (CC GPU)

```bash
cd ~/Docs/programming/vllm   # or clone Alex-Carter01/vllm @ cursor/cc-aware-scheduler-defaults
pip install -e .   # or your usual vLLM install

export MODEL=<your-model>
export ISL=1024 OSL=1024
CONCURRENCIES="4 8 16 32"

# Ensure CC on (guest / nvidia-smi conf-compute). Confirm:
nvidia-smi conf-compute -f
nvidia-smi conf-compute -mgm 2>/dev/null || true

# 2×2 under CC: async × (V1 copy opts | forced V2 without opts)
for ASYNC in "--async-scheduling" "--no-async-scheduling"; do
  for RUNNER_ENV in "" "VLLM_USE_V2_MODEL_RUNNER=1"; do
    echo "=== CC-on $ASYNC runner_env='$RUNNER_ENV' ==="
    env $RUNNER_ENV vllm serve "$MODEL" $ASYNC &
    SERVER_PID=$!
    # wait until /health is ready, then:
    for C in $CONCURRENCIES; do
      vllm bench serve --model "$MODEL" --dataset-name random \
        --random-input-len "$ISL" --random-output-len "$OSL" \
        --max-concurrency "$C" \
        | tee "cc_matrix_${ASYNC##--}_runner-${RUNNER_ENV:-v1}_c${C}.log"
    done
    kill "$SERVER_PID"; wait "$SERVER_PID" 2>/dev/null || true
  done
done
```

Repeat with CC **off** for a baseline column. Optional bridge microbench:

```bash
bash benchmarks/confidential_compute/run.sh --out-dir cc_bench_results
```

## Unit tests (no GPU)

```bash
pytest tests/platforms/test_confidential_compute.py -q
```
