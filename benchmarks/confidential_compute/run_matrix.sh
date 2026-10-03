#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# End-to-end serving matrix for the CC-aware stock defaults.
#
# For each "arm" (a combination of env vars and flags) this starts `vllm serve`,
# records which config it resolved to, drives `vllm bench serve` at several
# concurrencies, and stops the server. Then summarize_matrix.py compares the
# arms. Run it on a machine where NVIDIA Confidential Computing is ON.
#
#   bash run_matrix.sh --dry-run                     # print every command, run nothing
#   bash run_matrix.sh --model Qwen/Qwen3-8B --out cc_matrix_results
#   bash run_matrix.sh --arms A,B,C --concurrency 4,16
#
# Arms (see the README for the reasoning):
#   A   VLLM_CONFIDENTIAL_COMPUTE=0                     what stock main did under CC
#   B   (nothing)                                       the new CC-aware stock defaults
#   C   --no-async-scheduling                           V1, copy opts on, async off
#   D1  CC=0 + V2 runner off, --async-scheduling        V1 with the copy opts off
#   D2  CC=0 + V2 runner off, --no-async-scheduling     V1 with the copy opts off
#   E1  VLLM_USE_V2_MODEL_RUNNER=1                      forced V2 under CC, stock async
#   E2  VLLM_USE_V2_MODEL_RUNNER=1, --async-scheduling  forced V2 under CC, async on
#
# The HF token (if the model is gated) is read from the environment (HF_TOKEN)
# and is never printed or written to the logs.
set -euo pipefail

MODEL="Qwen/Qwen3-8B"
OUT="cc_matrix_results"
ARMS="A,B,C,D1,D2,E1,E2"
CONCURRENCY="4,8,16,32"
INPUT_LEN=1024
OUTPUT_LEN=256
PROMPTS_PER_CONC=8   # num_prompts = concurrency * this, floor of MIN_PROMPTS
MIN_PROMPTS=64
PORT=8000
READY_TIMEOUT=1200   # seconds to wait for the server to come up
PYTHON="${PYTHON:-python3}"
DRY_RUN=0
ALLOW_NON_CC=0

usage() { sed -n '4,28p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --arms) ARMS="$2"; shift 2 ;;
        --concurrency) CONCURRENCY="$2"; shift 2 ;;
        --input-len) INPUT_LEN="$2"; shift 2 ;;
        --output-len) OUTPUT_LEN="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --ready-timeout) READY_TIMEOUT="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --allow-non-cc) ALLOW_NON_CC=1; shift ;;
        -h|--help) usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage 2 ;;
    esac
done

run() {  # echo the command; execute it unless --dry-run
    echo "+ $*"
    if [[ $DRY_RUN -eq 0 ]]; then "$@"; fi
}

# arm -> "ENV_ASSIGNMENTS|SERVE_FLAGS"
arm_spec() {
    case "$1" in
        A)  echo "VLLM_CONFIDENTIAL_COMPUTE=0|" ;;
        B)  echo "|" ;;
        C)  echo "|--no-async-scheduling" ;;
        D1) echo "VLLM_CONFIDENTIAL_COMPUTE=0 VLLM_USE_V2_MODEL_RUNNER=0|--async-scheduling" ;;
        D2) echo "VLLM_CONFIDENTIAL_COMPUTE=0 VLLM_USE_V2_MODEL_RUNNER=0|--no-async-scheduling" ;;
        E1) echo "VLLM_USE_V2_MODEL_RUNNER=1|" ;;
        E2) echo "VLLM_USE_V2_MODEL_RUNNER=1|--async-scheduling" ;;
        *)  echo "unknown arm: $1" >&2; return 1 ;;
    esac
}

SERVER_PID=""
stop_server() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill -TERM "-$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null || true
        for _ in $(seq 1 60); do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "-$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=""
}
trap stop_server EXIT INT TERM

[[ $DRY_RUN -eq 0 ]] && mkdir -p "$OUT"

# --- CC preflight: a matrix run without CC is meaningless ------------------
if [[ $DRY_RUN -eq 0 && $ALLOW_NON_CC -eq 0 ]]; then
    cc_state="$(nvidia-smi conf-compute -f 2>&1 || true)"
    echo "$cc_state" | tee "$OUT/cc_state.txt"
    if ! grep -qi "CC status: ON" <<<"$cc_state"; then
        echo "error: Confidential Computing is not ON (pass --allow-non-cc to override)." >&2
        exit 3
    fi
else
    echo "+ nvidia-smi conf-compute -f   # must report 'CC status: ON'"
fi

IFS=',' read -r -a ARM_LIST <<<"$ARMS"
IFS=',' read -r -a CONC_LIST <<<"$CONCURRENCY"

for arm in "${ARM_LIST[@]}"; do
    spec="$(arm_spec "$arm")"
    env_part="${spec%%|*}"
    flag_part="${spec#*|}"
    log="$OUT/arm_${arm}_server.log"
    echo
    echo "=== arm $arm: env=[${env_part:-none}] flags=[${flag_part:-none}] ==="

    env_words=()
    flag_words=()
    read -r -a env_words <<<"$env_part" || true
    read -r -a flag_words <<<"$flag_part" || true
    serve_cmd=(env ${env_words[@]+"${env_words[@]}"} vllm serve "$MODEL" --port "$PORT"
        ${flag_words[@]+"${flag_words[@]}"})
    echo "+ ${serve_cmd[*]} > $log 2>&1 &"
    if [[ $DRY_RUN -eq 0 ]]; then
        setsid "${serve_cmd[@]}" >"$log" 2>&1 &
        SERVER_PID=$!
        waited=0
        until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
            if ! kill -0 "$SERVER_PID" 2>/dev/null; then
                echo "error: server for arm $arm exited early; see $log" >&2
                tail -n 40 "$log" >&2 || true
                exit 4
            fi
            if [[ $waited -ge $READY_TIMEOUT ]]; then
                echo "error: server for arm $arm not ready after ${READY_TIMEOUT}s" >&2
                exit 4
            fi
            sleep 5
            waited=$((waited + 5))
        done
        # Record what the server actually resolved to (the CC startup line).
        grep -iE "Confidential Computing|async.scheduling|model runner" "$log" \
            >"$OUT/arm_${arm}_resolved.txt" || true
    else
        echo "+ wait for http://127.0.0.1:$PORT/health"
    fi

    for conc in "${CONC_LIST[@]}"; do
        n=$((conc * PROMPTS_PER_CONC))
        [[ $n -lt $MIN_PROMPTS ]] && n=$MIN_PROMPTS
        run vllm bench serve --backend vllm --model "$MODEL" --port "$PORT" \
            --dataset-name random --random-input-len "$INPUT_LEN" \
            --random-output-len "$OUTPUT_LEN" --ignore-eos --seed 0 \
            --num-prompts "$n" --max-concurrency "$conc" --num-warmups 4 \
            --save-result --result-dir "$OUT" \
            --result-filename "arm_${arm}_c${conc}.json" \
            --metadata "arm=$arm" "concurrency=$conc"
    done

    stop_server
done

echo
echo "+ $PYTHON $(dirname "${BASH_SOURCE[0]}")/summarize_matrix.py $OUT"
if [[ $DRY_RUN -eq 0 ]]; then
    "$PYTHON" "$(dirname "${BASH_SOURCE[0]}")/summarize_matrix.py" "$OUT" | tee "$OUT/summary.md"
fi
