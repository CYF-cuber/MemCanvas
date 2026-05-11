#!/bin/bash
# End-to-end pipeline: wait for compression → build canvases → embed → evaluate
# Usage: bash run_pipeline.sh <level> [gpu_id]
# Example: bash run_pipeline.sh heavy 1

set -e
LEVEL=${1:?Usage: run_pipeline.sh <light|heavy> [gpu_id]}
GPU=${2:-0}
DIR="/home/cyf/memcanvas0402"
cd "$DIR"

echo "=== Pipeline for $LEVEL compression (GPU $GPU) ==="

# Step 1: Wait for compression to finish
COMPRESSED="$DIR/hotpotqa_${LEVEL}/compressed_texts.pkl"
echo "[1/4] Waiting for compression to complete..."
while true; do
    if [ -f "$COMPRESSED" ]; then
        N=$(python3 -c "import pickle; print(len(pickle.load(open('$COMPRESSED','rb'))))")
        echo "  $LEVEL: $N samples compressed"
        if [ "$N" -ge 3000 ]; then
            echo "  Compression complete!"
            break
        fi
    fi
    sleep 60
done

# Step 2: Build canvases
echo "[2/4] Building canvases..."
CUDA_VISIBLE_DEVICES=$GPU python -u build_compressed_canvases.py --level $LEVEL --phase render 2>&1 | tail -5

# Step 3: CLIP embeddings
echo "[3/4] Computing CLIP embeddings..."
CUDA_VISIBLE_DEVICES=$GPU python -u build_compressed_canvases.py --level $LEVEL --phase embed 2>&1 | tail -10

# Step 4: Evaluate with Qwen2.5-VL-7B
echo "[4/4] Evaluating with Qwen2.5-VL-7B..."
CUDA_VISIBLE_DEVICES=$GPU python -u eval_compressed.py --level $LEVEL --max-dev 500 2>&1 | tee "results_${LEVEL}.log"

echo "=== Pipeline complete for $LEVEL ==="
