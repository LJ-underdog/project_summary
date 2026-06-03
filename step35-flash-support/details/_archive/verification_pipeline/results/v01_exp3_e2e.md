# V01 Exp3: End-to-End BF16 Inference (tp=2 + tp=4)

Date: 2026-04-25
Model: stepfun-ai/Step-3.5-Flash (BF16, kv_cache=bf16)
Config: --level 0 --temperature 0 --max-tokens 128 --max-num-batched-tokens 4096

## Results

### tp=2 (GPU 0,1)

| Metric  | Measured | Baseline | Delta | Status |
|---------|----------|----------|-------|--------|
| TTFT    | 85 ms    | 92 ms    | -7.6% | within +-20% |
| TPOT    | 18 ms    | 17 ms    | +5.9% | within +-20% |

Per-request log:
- Request 0: input=16, output=128, latency=2.32s, TTFT=0.085s, TPOT=0.018s
- Request 1: input=20, output=128, latency=2.32s, TTFT=0.085s, TPOT=0.018s
- Request 2: input=19, output=60 (eos), latency=1.12s, TTFT=0.085s, TPOT=0.018s
- Request 3: input=21, output=128, latency=2.32s, TTFT=0.085s, TPOT=0.018s

BOS-spam check: `grep -c "<s>"` = 0 (PASS, <= 1)
Exit status: clean (no crash)

### tp=4 (GPU 0,1,2,3)

| Metric  | Measured | Baseline | Delta | Status |
|---------|----------|----------|-------|--------|
| TTFT    | 84 ms    | 88 ms    | -4.5% | within +-20% |
| TPOT    | 18 ms    | 15.75 ms | +14.3% | within +-20% |

Per-request log:
- Request 0: input=16, output=128, latency=2.33s, TTFT=0.084s, TPOT=0.018s
- Request 1: input=20, output=128, latency=2.33s, TTFT=0.084s, TPOT=0.018s
- Request 2: input=19, output=60 (eos), latency=1.10s, TTFT=0.084s, TPOT=0.017s
- Request 3: input=21, output=128, latency=2.33s, TTFT=0.084s, TPOT=0.018s

BOS-spam check: `grep -c "<s>"` = 0 (PASS, <= 1)
Exit status: clean (no crash)

## Sample outputs (consistent across tp=2 and tp=4)

Prompt: `1+2+3=?`
Output: `We are asked: "1+2+3=?" This is a simple arithmetic sum. 1+2=3, then 3+3=6. So the answer is 6.</think>The sum of 1, 2, and 3 is 6.`

Prompt: `list all prime numbers within 100`
Output: `We are asked to list all prime numbers within 100. Prime numbers are natural numbers greater than 1 ... 2, 3, 5, 7, 11, 13, 17, 19, 23` (truncated at max_tokens=128)

Prompt: `introduce yourself`
Output (tp=2): `Hmm, the user simply asked me to introduce myself. This is a straightforward request with no complex context or hidden needs. I should provide a clear, friendly introduction covering my core capabilities and limitations...`

## Conclusion

- tp=2 BF16: **PASS** (TTFT=85ms, TPOT=18ms; both within +-20% of baseline; no BOS-spam; clean exit)
- tp=4 BF16: **PASS** (TTFT=84ms, TPOT=18ms; both within +-20% of baseline; no BOS-spam; clean exit)

Both BF16 configurations function correctly end-to-end with no regressions vs historical baselines.

Logs:
- /home/hanchang/project_fp8_tp4/verification_pipeline/results/logs/v01_exp3_tp2.log
- /home/hanchang/project_fp8_tp4/verification_pipeline/results/logs/v01_exp3_tp4.log
