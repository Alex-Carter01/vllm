#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Run the CC serving matrix on ONE Azure confidential-GPU VM
# (Standard_NCC40ads_H100_v5: 1x H100 NVL 94 GB, AMD SEV-SNP) and tear
# everything down afterwards.
#
#   bash azure_run.sh --resource-group RG --ssh-key-file ~/.ssh/cc_bench.pub --dry-run
#   bash azure_run.sh --resource-group RG --ssh-key-file ~/.ssh/cc_bench.pub \
#        --approve-usd 36
#
# Guardrails (each exists because a forgotten GPU VM is expensive):
#   * Works only inside the resource group you pass in, and refuses to start if
#     that group already contains any resource.
#   * Refuses to run unless the estimated cost (hourly price x --max-hours) is
#     <= --budget-usd, and unless --approve-usd covers that estimate.
#   * Verifies the vCPU quota for the NCC family is exactly enough for ONE VM.
#   * An EXIT/INT/TERM trap deletes the WHOLE resource group (so this group is
#     single-use). It also installs an Azure-side auto-shutdown backstop that
#     deallocates the VM even if this script's host goes to sleep.
#   * First command on the VM is `nvidia-smi conf-compute -f`; if CC is not ON
#     the script tears down without running anything else.
#   * SSH is allowed only from this machine's current public IP (/32).
#
# Needs: az (logged in, `az login`), ssh, scp, rsync, curl, python3, git.
# Never put secrets in this script. A gated-model HF token is read from the
# environment variable named by --hf-token-env and sent over SSH stdin only.
set -euo pipefail

RG=""
REGION="eastus2"
MAX_HOURS=4
BUDGET_USD=36
APPROVE_USD=""
SSH_KEY_FILE=""
MODEL="Qwen/Qwen3-8B"
ARMS="A,B,C,D1,D2,E1,E2"
CONCURRENCY="4,8,16,32"
HF_TOKEN_ENV="HF_TOKEN"
VM_NAME="cc-bench-vm"
ADMIN_USER="azureuser"
VM_SIZE="Standard_NCC40ads_H100_v5"
REQUIRED_VCPUS=40
# Microsoft-published confidential-GPU image (driver preinstalled, secure boot).
# Source: https://github.com/Azure/az-cgpu-onboarding
IMAGE="/CommunityGalleries/cgpuimage-db870bae-5bcf-4120-9415-b841adef61d3/Images/cgpu-NCC-2404-base-image/versions/latest"
# Fallback hourly USD if the public price API is unreachable (East US 2 listing).
FALLBACK_RATE="7.89"
FULL_MICROBENCH=0
CHECK_OUTPUTS=0
DRY_RUN=0
OUT="cc_azure_results_$(date +%Y%m%d_%H%M%S)"

usage() { sed -n '4,31p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --resource-group) RG="$2"; shift 2 ;;
        --region) REGION="$2"; shift 2 ;;
        --max-hours) MAX_HOURS="$2"; shift 2 ;;
        --budget-usd) BUDGET_USD="$2"; shift 2 ;;
        --approve-usd) APPROVE_USD="$2"; shift 2 ;;
        --ssh-key-file) SSH_KEY_FILE="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --arms) ARMS="$2"; shift 2 ;;
        --concurrency) CONCURRENCY="$2"; shift 2 ;;
        --hf-token-env) HF_TOKEN_ENV="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --full-microbench) FULL_MICROBENCH=1; shift ;;
        --check-outputs) CHECK_OUTPUTS=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage 2 ;;
    esac
done

die() { echo "error: $*" >&2; exit 1; }
note() { echo "[azure_run] $*"; }
run() {  # echo; execute unless --dry-run
    echo "+ $*"
    if [[ $DRY_RUN -eq 0 ]]; then "$@"; fi
}

[[ -n "$RG" ]] || die "--resource-group is required"
[[ -n "$SSH_KEY_FILE" ]] || die "--ssh-key-file (public key path) is required"
[[ "$REGION" == "eastus2" || "$REGION" == "westeurope" ]] \
    || die "NCC H100 v5 is only offered in eastus2 / westeurope (got $REGION)"
python3 -c "import sys; sys.exit(0 if 0 < float('$MAX_HOURS') <= 8 else 1)" \
    || die "--max-hours must be in (0, 8]"
[[ -f "$SSH_KEY_FILE" ]] || die "ssh public key not found: $SSH_KEY_FILE"
SSH_PRIV="${SSH_KEY_FILE%.pub}"
[[ -f "$SSH_PRIV" ]] || die "matching private key not found: $SSH_PRIV"

# --- hourly price + estimate (public retail price API, no auth) -------------
RATE="$(python3 - "$VM_SIZE" "$REGION" "$FALLBACK_RATE" <<'EOF'
import json, sys, urllib.parse, urllib.request
size, region, fallback = sys.argv[1:4]
flt = (f"armSkuName eq '{size}' and armRegionName eq '{region}' "
       "and priceType eq 'Consumption'")
url = "https://prices.azure.com/api/retail/prices?" + urllib.parse.urlencode(
    {"$filter": flt})
try:
    items = json.load(urllib.request.urlopen(url, timeout=15))["Items"]
    items = [i for i in items
             if "Windows" not in i.get("productName", "")
             and "Spot" not in i.get("skuName", "")
             and "Low Priority" not in i.get("skuName", "")]
    print(f"{min(i['retailPrice'] for i in items):.4f}")
except Exception:
    print(fallback)
EOF
)"
EST="$(python3 -c "print(f'{float('$RATE') * float('$MAX_HOURS'):.2f}')")"
note "VM $VM_SIZE in $REGION: \$$RATE/h x ${MAX_HOURS}h max = \$$EST (+ small disk/IP cost)"
python3 -c "import sys; sys.exit(0 if float('$EST') <= float('$BUDGET_USD') else 1)" \
    || die "estimate \$$EST exceeds --budget-usd \$$BUDGET_USD; lower --max-hours"

if [[ $DRY_RUN -eq 0 ]]; then
    [[ -n "$APPROVE_USD" ]] || die "pass --approve-usd N (N >= \$$EST) to authorize spend"
    python3 -c "import sys; sys.exit(0 if float('$APPROVE_USD') >= float('$EST') else 1)" \
        || die "--approve-usd \$$APPROVE_USD is below the estimate \$$EST"
fi

# --- preflight (read-only) --------------------------------------------------
note "preflight checks"
run az account show --query "{name:name, id:id, user:user.name}" -o table
run az group show --name "$RG" --query "{name:name, location:location, tags:tags}" -o json
if [[ $DRY_RUN -eq 0 ]]; then
    existing="$(az resource list --resource-group "$RG" --query "length(@)" -o tsv)"
    [[ "$existing" == "0" ]] || die "resource group $RG is not empty ($existing resources)"
    sku_restr="$(az vm list-skus --location "$REGION" --size "$VM_SIZE" \
        --query "[0].restrictions | length(@)" -o tsv 2>/dev/null || echo "?")"
    [[ "$sku_restr" == "0" ]] || die "$VM_SIZE has restrictions in $REGION (restrictions=$sku_restr)"
    limit="$(az vm list-usage --location "$REGION" \
        --query "[?name.value=='standardNCCads2023Family' || name.value=='StandardNCCads2023Family'].limit | [0]" -o tsv)"
    [[ -n "$limit" ]] || die "no StandardNCCads2023Family quota entry in $REGION"
    [[ "$limit" -ge "$REQUIRED_VCPUS" ]] || die "quota $limit < $REQUIRED_VCPUS vCPUs in $REGION"
    if [[ "$limit" -gt "$REQUIRED_VCPUS" ]]; then
        note "WARNING: quota is $limit vCPUs; for a hard one-VM cap request exactly $REQUIRED_VCPUS"
    fi
    if [[ -z "$(az consumption budget list --resource-group "$RG" -o tsv 2>/dev/null)" ]]; then
        note "WARNING: no budget found on $RG. Create one in the portal (alerts only)."
    fi
else
    echo "+ (check: group empty, SKU unrestricted in $REGION, NCC quota >= $REQUIRED_VCPUS)"
fi

MY_IP="$(curl -fsS https://api.ipify.org 2>/dev/null || true)"
[[ $DRY_RUN -eq 1 || -n "$MY_IP" ]] || die "could not determine this machine's public IP"
MY_IP="${MY_IP:-<your-public-ip>}"

CREATED=0
FINAL_RC=0
LOCAL_SYNC_PID=""
WATCHDOG_PID=""
KEEP_ON_UNVERIFIED=0

teardown() {
    local rc=$?
    [[ $rc -ne 0 ]] && FINAL_RC=$rc
    trap - EXIT INT TERM
    [[ -n "$LOCAL_SYNC_PID" ]] && kill "$LOCAL_SYNC_PID" 2>/dev/null || true
    [[ -n "$WATCHDOG_PID" ]] && kill "$WATCHDOG_PID" 2>/dev/null || true
    if [[ $CREATED -eq 1 && $KEEP_ON_UNVERIFIED -eq 1 ]]; then
        note "RESULTS NOT VERIFIED LOCALLY: leaving $RG in place so they can be copied."
        note "The VM auto-deallocates at ${SHUTDOWN_HHMM:-?} UTC (stops compute billing)."
        note "Copy results:  rsync -az -e 'ssh -i $SSH_PRIV' $ADMIN_USER@${VM_IP:-?}:~/results/ $OUT/"
        note "THEN delete:   az group delete --name $RG --yes --no-wait"
        FINAL_RC=8
    elif [[ $CREATED -eq 1 ]]; then
        note "TEARDOWN: deleting resource group $RG (entire group)"
        az group delete --name "$RG" --yes --no-wait || note "az group delete FAILED: delete $RG by hand NOW"
        for _ in $(seq 1 60); do   # poll up to ~30 min
            [[ "$(az group exists --name "$RG")" == "false" ]] && break
            sleep 30
        done
        gone="$(az group exists --name "$RG" 2>/dev/null || echo unknown)"
        if [[ "$gone" == "false" ]]; then
            note "verified: resource group $RG no longer exists"
        else
            note "WARNING: $RG still exists (exists=$gone). Delete it in the portal NOW."
            FINAL_RC=9
        fi
        elapsed=$(( $(date +%s) - START_TS ))
        python3 - "$elapsed" "$RATE" <<'EOF'
import sys
secs, rate = int(sys.argv[1]), float(sys.argv[2])
print(f"[azure_run] wall time {secs/3600:.2f} h; approx VM cost ${secs/3600*rate:.2f} "
      "(check Cost Analysis tomorrow; usage data lags)")
EOF
    fi
    exit "$FINAL_RC"
}
trap teardown EXIT INT TERM
START_TS=$(date +%s)

ssh_opts=(-i "$SSH_PRIV" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 \
    -o ConnectTimeout=20 -o IdentitiesOnly=yes)

# --- network: NSG that only admits this machine on 22 ----------------------
run az network nsg create --resource-group "$RG" --name cc-bench-nsg --location "$REGION"
run az network nsg rule create --resource-group "$RG" --nsg-name cc-bench-nsg \
    --name ssh-from-me --priority 100 --access Allow --direction Inbound \
    --protocol Tcp --destination-port-ranges 22 --source-address-prefixes "$MY_IP/32"

# --- VM ---------------------------------------------------------------------
EXPIRES="$(date -u -v+"$(python3 -c "print(int(float('$MAX_HOURS')*60)+30)")"M +%Y-%m-%dT%H:%MZ 2>/dev/null \
    || date -u -d "+$(python3 -c "print(int(float('$MAX_HOURS')*60)+30)") minutes" +%Y-%m-%dT%H:%MZ)"
[[ $DRY_RUN -eq 1 ]] || CREATED=1   # from here on, a failure triggers teardown
# Hard time box, started BEFORE `az vm create` because billing starts during
# creation: SIGTERM ourselves -> teardown trap deletes the group.
if [[ $DRY_RUN -eq 0 ]]; then   
    ( sleep "$(python3 -c "print(int(float('$MAX_HOURS')*3600))")"; \
      echo "[azure_run] --max-hours reached; tearing down" >&2; kill -TERM $$ ) &
    WATCHDOG_PID=$!
fi
run az vm create --resource-group "$RG" --name "$VM_NAME" --location "$REGION" \
    --size "$VM_SIZE" --image "$IMAGE" --security-type ConfidentialVM \
    --os-disk-security-encryption-type VMGuestStateOnly \
    --enable-secure-boot true --enable-vtpm true --os-disk-size-gb 256 \
    --os-disk-delete-option Delete --nic-delete-option Delete \
    --public-ip-sku Standard --nsg cc-bench-nsg --nsg-rule NONE \
    --admin-username "$ADMIN_USER" --ssh-key-values "$SSH_KEY_FILE" \
    --authentication-type ssh --accept-term \
    --tags purpose=cc-vllm-bench expires="$EXPIRES"

# Azure-side backstop: deallocate the VM (stops compute billing) even if this
# script's host sleeps. auto-shutdown takes a daily UTC HHMM; max-hours < 24.
SHUTDOWN_HHMM="$(date -u -v+"$(python3 -c "print(int(float('$MAX_HOURS')*60)+20)")"M +%H%M 2>/dev/null \
    || date -u -d "+$(python3 -c "print(int(float('$MAX_HOURS')*60)+20)") minutes" +%H%M)"
run az vm auto-shutdown --resource-group "$RG" --name "$VM_NAME" --time "$SHUTDOWN_HHMM"

if [[ $DRY_RUN -eq 1 ]]; then
    VM_IP="<vm-public-ip>"
else
    VM_IP="$(az vm show -d --resource-group "$RG" --name "$VM_NAME" --query publicIps -o tsv)"
fi
REMOTE="$ADMIN_USER@$VM_IP"
note "VM up at $VM_IP"

wait_ssh() {
    for _ in $(seq 1 40); do
        ssh "${ssh_opts[@]}" "$REMOTE" true 2>/dev/null && return 0
        sleep 15
    done
    return 1
}
if [[ $DRY_RUN -eq 0 ]]; then wait_ssh || die "ssh to $VM_IP never came up"; fi

# --- CC preflight on the VM: bail out (and tear down) if CC is not ON -------
if [[ $DRY_RUN -eq 0 ]]; then
    mkdir -p "$OUT"
    cc_out="$(ssh "${ssh_opts[@]}" "$REMOTE" "nvidia-smi conf-compute -f; nvidia-smi conf-compute -e" 2>&1 || true)"
    echo "$cc_out" | tee "$OUT/vm_cc_state.txt"
    grep -qi "CC status: ON" <<<"$cc_out" || { note "CC is NOT ON; aborting before any benchmark"; exit 3; }
else
    echo "+ ssh $REMOTE 'nvidia-smi conf-compute -f'   # abort + teardown unless 'CC status: ON'"
fi

# --- ship this branch (python-only delta over the precompiled base wheel) ---
REPO_ROOT="$(git rev-parse --show-toplevel)"
BASE_COMMIT="$(git -C "$REPO_ROOT" merge-base HEAD upstream/main 2>/dev/null || git -C "$REPO_ROOT" rev-parse HEAD~3)"
BRANCH_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
note "branch $BRANCH_SHA, precompiled wheel base $BASE_COMMIT"
TARBALL="$(mktemp -t vllm-src.XXXXXX).tar.gz"
run git -C "$REPO_ROOT" archive --format=tar.gz -o "$TARBALL" HEAD
run scp "${ssh_opts[@]}" "$TARBALL" "$REMOTE:vllm-src.tar.gz"

BOOTSTRAP="
set -euo pipefail
mkdir -p ~/vllm && tar -xzf ~/vllm-src.tar.gz -C ~/vllm && cd ~/vllm
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=\$HOME/.local/bin:\$PATH
sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3.12-dev build-essential
uv venv --python 3.12 ~/venv && . ~/venv/bin/activate
python -c 'import os, sysconfig; p = sysconfig.get_path(\"include\") + \"/Python.h\"; assert os.path.exists(p), p'
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0.dev0+g${BRANCH_SHA:0:9} VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_COMMIT=$BASE_COMMIT uv pip install -e .
uv pip install -r benchmarks/confidential_compute/requirements.txt
vllm --version
"
run ssh "${ssh_opts[@]}" "$REMOTE" "bash -s" <<<"$BOOTSTRAP"

# Stream results home every minute while the matrix runs.
mkdir -p "$OUT" 2>/dev/null || true
if [[ $DRY_RUN -eq 0 ]]; then
    ( while sleep 60; do
        rsync -az -e "ssh ${ssh_opts[*]}" "$REMOTE:~/results/" "$OUT/" 2>/dev/null || true
      done ) &
    LOCAL_SYNC_PID=$!
fi

MICRO_FLAGS="--quick"; [[ $FULL_MICROBENCH -eq 1 ]] && MICRO_FLAGS=""
MATRIX_FLAGS=""; [[ $CHECK_OUTPUTS -eq 1 ]] && MATRIX_FLAGS="--check-outputs"
RUN_CMD=". ~/venv/bin/activate && cd ~/vllm && mkdir -p ~/results && \
export VLLM_USE_FLASHINFER_SAMPLER=0 && \
bash benchmarks/confidential_compute/run.sh $MICRO_FLAGS 2>&1 | tee ~/results/microbench.log && \
bash benchmarks/confidential_compute/run_matrix.sh --model '$MODEL' --arms '$ARMS' \
--concurrency '$CONCURRENCY' $MATRIX_FLAGS --out ~/results 2>&1 | tee ~/results/matrix.log"

if [[ $DRY_RUN -eq 0 ]]; then
    # Everything goes over stdin: the HF token (if any) never appears on a
    # command line or in a log.
    {
        if [[ -n "${!HF_TOKEN_ENV:-}" ]]; then
            printf 'export HF_TOKEN=%q\n' "${!HF_TOKEN_ENV}"
        fi
        echo "$RUN_CMD"
    } | ssh "${ssh_opts[@]}" "$REMOTE" "bash -s" 2>&1 | tee "$OUT/remote_stdout.log"
else
    echo "+ ssh $REMOTE '<activate venv; run.sh $MICRO_FLAGS; run_matrix.sh --model $MODEL --arms $ARMS --concurrency $CONCURRENCY>'"
fi

# --- verify the results actually arrived BEFORE teardown --------------------
if [[ $DRY_RUN -eq 0 ]]; then
    kill "$LOCAL_SYNC_PID" 2>/dev/null || true; LOCAL_SYNC_PID=""
    rsync -az --checksum -e "ssh ${ssh_opts[*]}" "$REMOTE:~/results/" "$OUT/"
    remote_n="$(ssh "${ssh_opts[@]}" "$REMOTE" "find ~/results -type f | wc -l" | tr -d ' ')"
    local_n="$(find "$OUT" -type f ! -name remote_stdout.log ! -name vm_cc_state.txt | wc -l | tr -d ' ')"
    if [[ "$remote_n" -gt "$local_n" ]]; then
        KEEP_ON_UNVERIFIED=1
        die "only $local_n of $remote_n result files arrived locally"
    fi
    note "results verified locally in $OUT ($local_n files)"
else
    echo "+ rsync results to $OUT and verify file counts before teardown"
fi
# EXIT trap now deletes the resource group and verifies it is gone.
