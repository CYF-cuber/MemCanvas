#!/bin/bash
# Auto-evaluate after GRPO training completes.
#
# 1. Wait for training process to finish
# 2. Merge best RL LoRA checkpoint into model
# 3. Run full pipeline evaluation (RL vs SFT vs embedding vs no-memory)
#
# Usage:
#   nohup bash auto_eval_after_train.sh > /home/cyf/codex/auto_eval.log 2>&1 &

set -euo pipefail

TRAIN_PID=4066532
LOG_PREFIX="[auto_eval]"

echo "${LOG_PREFIX} $(date) — Waiting for training process (PID ${TRAIN_PID}) to finish..."

# Wait for training to complete
while kill -0 ${TRAIN_PID} 2>/dev/null; do
    sleep 60
done

echo "${LOG_PREFIX} $(date) — Training process finished!"
echo ""

# Activate conda env
source /home/cyf/miniconda3/etc/profile.d/conda.sh
conda activate verl

# Step 1: Merge best RL LoRA
echo "${LOG_PREFIX} $(date) — Step 1: Merging best RL LoRA checkpoint..."
cd /home/cyf/codex

python merge_rl_lora.py --step best --output /home/cyf/codex/agent_rl_merged

echo ""
echo "${LOG_PREFIX} $(date) — Step 2: Running full evaluation (compare all methods)..."

# Step 2: Run evaluation — quick test first (100 samples)
python eval_read_rl.py \
    --rl-model /home/cyf/codex/agent_rl_merged \
    --compare-all \
    --max-test 200 \
    --output-dir /home/cyf/codex/agent_rl_eval_output_quick

echo ""
echo "${LOG_PREFIX} $(date) — Quick eval (200 samples) done!"
echo ""

# Print quick results
if [ -f /home/cyf/codex/agent_rl_eval_output_quick/eval_summary.json ]; then
    echo "${LOG_PREFIX} Quick eval results:"
    cat /home/cyf/codex/agent_rl_eval_output_quick/eval_summary.json
fi

echo ""
echo "${LOG_PREFIX} $(date) — Step 3: Full evaluation (all test samples)..."

# Step 3: Full evaluation (all test samples)
python eval_read_rl.py \
    --rl-model /home/cyf/codex/agent_rl_merged \
    --compare-all \
    --output-dir /home/cyf/codex/agent_rl_eval_output

echo ""
echo "${LOG_PREFIX} $(date) — Full evaluation done!"

# Print final results
if [ -f /home/cyf/codex/agent_rl_eval_output/eval_summary.json ]; then
    echo ""
    echo "============================================================"
    echo "FINAL EVALUATION RESULTS"
    echo "============================================================"
    cat /home/cyf/codex/agent_rl_eval_output/eval_summary.json
fi

echo ""
echo "${LOG_PREFIX} $(date) — All done!"
