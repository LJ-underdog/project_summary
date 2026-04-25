#!/usr/bin/env bash
# ============================================================================
# Phase 0 PreFlight Check Script
# ----------------------------------------------------------------------------
# Purpose : Verify all prerequisites before starting verification pipeline.
# Usage   : bash /home/hanchang/project_fp8_tp4/verification_pipeline/phase0_preflight.sh
# Refs    : MASTER_PIPELINE.md §6 Phase 0 Checklist
#           PIPELINE_REVIEW_FINAL.md §5 Phase 0 amendments (0.8-0.11)
# ============================================================================

# Don't `set -e`: we want to keep running through all checks even on FAIL.
set -u
set -o pipefail

# ---------- color helpers ----------
if [[ -t 1 ]]; then
    C_RED=$'\033[0;31m'
    C_GREEN=$'\033[0;32m'
    C_YELLOW=$'\033[0;33m'
    C_BLUE=$'\033[0;34m'
    C_BOLD=$'\033[1m'
    C_RESET=$'\033[0m'
else
    C_RED=""; C_GREEN=""; C_YELLOW=""; C_BLUE=""; C_BOLD=""; C_RESET=""
fi

PASS_COUNT=0
WARN_COUNT=0
FAIL_COUNT=0
FAIL_ITEMS=()
WARN_ITEMS=()

pass() {
    # $1 = id, $2 = msg
    printf "%s[PASS]%s %-6s %s\n" "$C_GREEN" "$C_RESET" "$1" "$2"
    PASS_COUNT=$((PASS_COUNT + 1))
}
warn() {
    printf "%s[WARN]%s %-6s %s\n" "$C_YELLOW" "$C_RESET" "$1" "$2"
    WARN_COUNT=$((WARN_COUNT + 1))
    WARN_ITEMS+=("$1: $2")
}
fail() {
    printf "%s[FAIL]%s %-6s %s\n" "$C_RED" "$C_RESET" "$1" "$2"
    FAIL_COUNT=$((FAIL_COUNT + 1))
    FAIL_ITEMS+=("$1: $2")
}
section() {
    printf "\n%s%s=== %s ===%s\n" "$C_BOLD" "$C_BLUE" "$1" "$C_RESET"
}

# ----------------------------------------------------------------------------
ATOM_DIR="/home/hanchang/ATOM"
AITER_DIR="/home/hanchang/aiter"
PYTHON_BIN="/opt/venv/bin/python"
WORK_DIR="/home/hanchang/project_fp8_tp4/verification_pipeline"
HF_HUB_DIR="${HOME}/.cache/huggingface/hub"

# Required (or newer) commits
REQUIRED_ATOM_COMMIT="ccb64621"
REQUIRED_AITER_COMMIT="c38d0c9e6"

printf "%s%sPhase 0 PreFlight Check%s  (date: %s)\n" "$C_BOLD" "$C_BLUE" "$C_RESET" "$(date '+%Y-%m-%d %H:%M:%S')"
printf "Working dir: %s\n" "$WORK_DIR"

# ============================================================================
section "0.1  ATOM commit"
# ============================================================================
if [[ -d "$ATOM_DIR/.git" ]]; then
    ATOM_HEAD=$(git -C "$ATOM_DIR" log -1 --format='%H' 2>/dev/null || echo "")
    ATOM_HEAD_SHORT="${ATOM_HEAD:0:8}"
    if [[ -z "$ATOM_HEAD" ]]; then
        fail "0.1" "ATOM git log failed at $ATOM_DIR"
    else
        # Check if HEAD == required, or required is ancestor of HEAD
        if git -C "$ATOM_DIR" merge-base --is-ancestor "$REQUIRED_ATOM_COMMIT" "$ATOM_HEAD" 2>/dev/null; then
            pass "0.1" "ATOM HEAD=$ATOM_HEAD_SHORT contains required $REQUIRED_ATOM_COMMIT"
        else
            fail "0.1" "ATOM HEAD=$ATOM_HEAD_SHORT does NOT contain required commit $REQUIRED_ATOM_COMMIT (FP8 tp=4 fix). Run: git -C $ATOM_DIR log --oneline | head"
        fi
    fi
else
    fail "0.1" "ATOM repo not found at $ATOM_DIR"
fi

# ============================================================================
section "0.2  aiter commit"
# ============================================================================
if [[ -d "$AITER_DIR/.git" ]]; then
    AITER_HEAD=$(git -C "$AITER_DIR" log -1 --format='%H' 2>/dev/null || echo "")
    AITER_HEAD_SHORT="${AITER_HEAD:0:9}"
    if [[ -z "$AITER_HEAD" ]]; then
        fail "0.2" "aiter git log failed at $AITER_DIR"
    else
        if git -C "$AITER_DIR" merge-base --is-ancestor "$REQUIRED_AITER_COMMIT" "$AITER_HEAD" 2>/dev/null; then
            pass "0.2" "aiter HEAD=$AITER_HEAD_SHORT contains required $REQUIRED_AITER_COMMIT"
        else
            fail "0.2" "aiter HEAD=$AITER_HEAD_SHORT does NOT contain required commit $REQUIRED_AITER_COMMIT (FP8 blockscale). Run: git -C $AITER_DIR log --oneline | head"
        fi
    fi
else
    fail "0.2" "aiter repo not found at $AITER_DIR"
fi

# ============================================================================
section "0.3  aiter Python install path"
# ============================================================================
if [[ -x "$PYTHON_BIN" ]]; then
    AITER_PY_FILE=$(cd /tmp && "$PYTHON_BIN" -c "import aiter; print(aiter.__file__)" 2>/dev/null || echo "")
    if [[ -z "$AITER_PY_FILE" ]]; then
        fail "0.3" "Failed to import aiter (cd /tmp && $PYTHON_BIN -c 'import aiter')"
    elif [[ "$AITER_PY_FILE" == /home/hanchang/aiter/* ]]; then
        pass "0.3" "aiter -> $AITER_PY_FILE"
    else
        fail "0.3" "aiter resolved to $AITER_PY_FILE (expected /home/hanchang/aiter/...). Editable install missing."
    fi
else
    fail "0.3" "Python binary not found: $PYTHON_BIN"
fi

# ============================================================================
section "0.4a  aiter revert 3771835ac scope check"
# ============================================================================
if [[ -d "$AITER_DIR/.git" ]]; then
    REVERT_INFO=$(git -C "$AITER_DIR" show 3771835ac --stat 2>/dev/null | head -40 || echo "")
    if [[ -z "$REVERT_INFO" ]]; then
        warn "0.4a" "Commit 3771835ac not found in aiter repo. Check whether the revert was rebased away or never merged."
    else
        # Did the revert touch ATOM moe.py?
        if echo "$REVERT_INFO" | grep -qiE "(atom.*moe\.py|moe\.py.*atom)"; then
            warn "0.4a" "3771835ac touches ATOM moe.py -> V04 may need to be blocked / re-evaluated"
        else
            # Show files touched (filter to .py / .cu / .cpp lines for brevity)
            TOUCHED=$(echo "$REVERT_INFO" | grep -E "\| +[0-9]+ " | head -10 | sed 's/^/      /')
            pass "0.4a" "3771835ac scope OK (does not touch ATOM moe.py)"
            if [[ -n "$TOUCHED" ]]; then
                printf "%s\n" "$TOUCHED"
            fi
        fi
    fi
else
    fail "0.4a" "aiter repo missing; cannot inspect 3771835ac"
fi

# ============================================================================
section "0.5a  BF16 model cache"
# ============================================================================
BF16_HIT=$(ls "$HF_HUB_DIR" 2>/dev/null | grep -iE "Step-3\.5-Flash$|models--stepfun-ai--Step-3\.5-Flash$" || true)
if [[ -n "$BF16_HIT" ]]; then
    pass "0.5a" "BF16 cache found: $BF16_HIT"
else
    fail "0.5a" "BF16 model cache 'Step-3.5-Flash' not found under $HF_HUB_DIR"
fi

# ============================================================================
section "0.5b  FP8 model cache"
# ============================================================================
FP8_HIT=$(ls "$HF_HUB_DIR" 2>/dev/null | grep -iE "Step-3\.5-Flash-FP8" || true)
if [[ -n "$FP8_HIT" ]]; then
    pass "0.5b" "FP8 cache found: $FP8_HIT"
else
    warn "0.5b" "FP8 model cache 'Step-3.5-Flash-FP8' not found under $HF_HUB_DIR (V05/V06 will need download)"
fi

# ============================================================================
section "0.6  sliding_window value (config.json)"
# ============================================================================
SW_CONFIG=""
for d in "$HF_HUB_DIR"/models--stepfun-ai--Step-3.5-Flash/snapshots/*/; do
    if [[ -f "${d}config.json" ]]; then
        SW_CONFIG="${d}config.json"
        break
    fi
done
if [[ -z "$SW_CONFIG" ]]; then
    warn "0.6" "config.json not located under BF16 snapshot; sliding_window unverified"
else
    SW_VAL=$(cd /tmp && "$PYTHON_BIN" -c "import json; c=json.load(open('$SW_CONFIG')); print(c.get('sliding_window', 'MISSING'))" 2>/dev/null || echo "")
    if [[ -z "$SW_VAL" || "$SW_VAL" == "MISSING" ]]; then
        warn "0.6" "sliding_window field missing in $SW_CONFIG (val=$SW_VAL)"
    else
        pass "0.6" "sliding_window=$SW_VAL  (source: $SW_CONFIG)"
    fi
fi

# ============================================================================
section "0.7  GPU availability (8x MI350X)"
# ============================================================================
GPU_NAMES=$(cd /tmp && "$PYTHON_BIN" -c "
import torch
n = torch.cuda.device_count()
for i in range(n):
    try:
        p = torch.cuda.get_device_properties(i)
        print(f'GPU{i}: {p.name}')
    except Exception as e:
        print(f'GPU{i}: ERROR {e}')
" 2>/dev/null || echo "")
if [[ -z "$GPU_NAMES" ]]; then
    fail "0.7" "Failed to query GPU info via torch"
else
    GPU_COUNT=$(echo "$GPU_NAMES" | wc -l)
    if [[ "$GPU_COUNT" -ge 8 ]]; then
        pass "0.7" "Detected $GPU_COUNT GPUs"
    else
        warn "0.7" "Only $GPU_COUNT GPU(s) visible (expected 8)"
    fi
    # Print each GPU and flag GPU5
    while IFS= read -r line; do
        if [[ "$line" == GPU5:* ]]; then
            printf "      %s%s  <-- AVOID (hardware ~700ms/tensor anomaly per memory)%s\n" "$C_YELLOW" "$line" "$C_RESET"
        else
            printf "      %s\n" "$line"
        fi
    done <<< "$GPU_NAMES"
    warn "0.7-GPU5" "GPU5 has known hardware anomaly; exclude via HIP_VISIBLE_DEVICES=0,1,2,3,4,6,7 when possible"
fi

# ============================================================================
section "0.8  JIT cache state"
# ============================================================================
AITER_CACHE_DIR="/root/.cache/aiter"
if [[ -d "$AITER_CACHE_DIR" ]]; then
    AITER_SO_COUNT=$(ls "$AITER_CACHE_DIR" 2>/dev/null | wc -l)
    pass "0.8" "/root/.cache/aiter has $AITER_SO_COUNT entries"
else
    pass "0.8" "/root/.cache/aiter does not exist (clean)"
fi
ATOM_CACHE_COUNT=0
if [[ -d /root/.cache/atom ]]; then
    ATOM_CACHE_COUNT=$(ls /root/.cache/atom 2>/dev/null | wc -l)
fi
printf "      atom cache entries: %s\n" "$ATOM_CACHE_COUNT"
printf "      %sTo clean if encountering stale-kernel issues:%s\n" "$C_BOLD" "$C_RESET"
printf "        rm -rf /root/.cache/atom/* /root/.cache/aiter/* ~/.aiter_cache/*\n"

# ============================================================================
section "0.9  aiter logger INFO availability"
# ============================================================================
LOGGER_INFO=$(cd /tmp && "$PYTHON_BIN" -c "
import aiter, logging
if hasattr(aiter, 'logger'):
    lg = aiter.logger
    print(f'level={lg.level} ({logging.getLevelName(lg.level)}); name={lg.name}')
else:
    print('N/A')
" 2>/dev/null || echo "")
if [[ -z "$LOGGER_INFO" ]]; then
    warn "0.9" "Failed to query aiter.logger"
elif [[ "$LOGGER_INFO" == "N/A" ]]; then
    warn "0.9" "aiter has no .logger attribute (cannot toggle INFO via aiter.logger.setLevel)"
else
    pass "0.9" "aiter.logger -> $LOGGER_INFO"
fi

# ============================================================================
section "0.10  worktree / source-edit protocol reminder"
# ============================================================================
pass "0.10" "Reminder printed (no automated check)"
cat <<EOF
      ${C_BOLD}Source-edit protocol${C_RESET} (apply before any kernel/source modification):
        1. cd <repo> && git status --porcelain   # must be empty before touching files
        2. Use a worktree (git worktree add) for experimental changes when possible
        3. After edit: capture diff (git diff > /tmp/<tag>.patch) before re-running
        4. Clear caches if you changed kernels: rm -rf /root/.cache/atom/* /root/.cache/aiter/*
        5. Commit/push only from /home/hanchang/junlin12_repos/ (per global rules)
EOF

# ============================================================================
section "0.11  glm5 CSV row count"
# ============================================================================
GLM5_CSV="$AITER_DIR/aiter/configs/model_configs/glm5_bf16_tuned_gemm.csv"
if [[ -f "$GLM5_CSV" ]]; then
    GLM5_LINES=$(wc -l < "$GLM5_CSV")
    if [[ "$GLM5_LINES" -eq 72 ]]; then
        pass "0.11" "glm5_bf16_tuned_gemm.csv = $GLM5_LINES lines (fix applied)"
    elif [[ "$GLM5_LINES" -eq 73 ]]; then
        fail "0.11" "glm5 CSV = 73 lines -> fix NOT applied (expected 72 after deletion)"
    else
        warn "0.11" "glm5 CSV = $GLM5_LINES lines (expected 72; verify manually)"
    fi
else
    fail "0.11" "glm5 CSV not found at $GLM5_CSV"
fi

# ============================================================================
section "0.12  WORK_DIR and results/logs"
# ============================================================================
RESULTS_LOGS="$WORK_DIR/results/logs"
if mkdir -p "$RESULTS_LOGS" 2>/dev/null; then
    pass "0.12" "Ensured $RESULTS_LOGS exists"
else
    fail "0.12" "Could not create $RESULTS_LOGS"
fi

# ============================================================================
# Summary
# ============================================================================
printf "\n%s%s================ SUMMARY ================%s\n" "$C_BOLD" "$C_BLUE" "$C_RESET"
printf "  %s%d PASS%s, %s%d WARN%s, %s%d FAIL%s\n" \
    "$C_GREEN" "$PASS_COUNT" "$C_RESET" \
    "$C_YELLOW" "$WARN_COUNT" "$C_RESET" \
    "$C_RED" "$FAIL_COUNT" "$C_RESET"

if (( WARN_COUNT > 0 )); then
    printf "\n%sWarnings:%s\n" "$C_YELLOW" "$C_RESET"
    for w in "${WARN_ITEMS[@]}"; do
        printf "  - %s\n" "$w"
    done
fi

if (( FAIL_COUNT > 0 )); then
    printf "\n%sFailures (BLOCKING):%s\n" "$C_RED" "$C_RESET"
    for f in "${FAIL_ITEMS[@]}"; do
        printf "  - %s\n" "$f"
    done
    printf "\n%sPhase 0 BLOCKED. Fix FAIL items before proceeding.%s\n" "$C_RED" "$C_RESET"
    EXIT_CODE=1
else
    EXIT_CODE=0
fi

# ============================================================================
# Phase 1 launch hint (always print)
# ============================================================================
cat <<EOF

${C_BOLD}=== Phase 0 Complete ===${C_RESET}
下一步：
  1. 确认 GPU 分配方案（见 MASTER_PIPELINE.md §2.1）
  2. 以 Lead 身份启动 V01-MoE 验证（第一个阻断点）
  3. 参考 TEAM_CONFIG_verification.md 配置 agent team

关键文档：
  总计划：/home/hanchang/project_fp8_tp4/verification_pipeline/MASTER_PIPELINE.md
  分工配置：/home/hanchang/project_fp8_tp4/verification_pipeline/TEAM_CONFIG_verification.md
  V01 验证计划：/home/hanchang/project_fp8_tp4/verification_pipeline/V01_moe.md
EOF

exit $EXIT_CODE
