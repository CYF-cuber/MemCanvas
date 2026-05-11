#!/bin/bash
# Run full pipeline for both compression levels sequentially
# Heavy finishes first, so run it first
set -e
DIR="/home/cyf/memcanvas0402"
cd "$DIR"

echo "============================================"
echo "Starting full compression experiment pipeline"
echo "============================================"

# Wait for BOTH compressions to finish
echo "Waiting for compression jobs to complete..."
for LEVEL in heavy light; do
    COMPRESSED="$DIR/hotpotqa_${LEVEL}/compressed_texts.pkl"
    while true; do
        if [ -f "$COMPRESSED" ]; then
            N=$(python3 -c "import pickle; print(len(pickle.load(open('$COMPRESSED','rb'))))")
            if [ "$N" -ge 3000 ]; then
                echo "  $LEVEL: $N samples - DONE"
                break
            else
                echo "  $LEVEL: $N/3000 samples..."
            fi
        else
            echo "  $LEVEL: no checkpoint yet..."
        fi
        sleep 120
    done
done

echo ""
echo "=== All compressions done. Building canvases ==="

# Build canvases (CPU only, can do both)
for LEVEL in heavy light; do
    echo "--- Building $LEVEL canvases ---"
    python -u build_compressed_canvases.py --level $LEVEL --phase render
done

echo ""
echo "=== Computing CLIP embeddings ==="

# CLIP embeddings (needs GPU, but lightweight)
for LEVEL in heavy light; do
    echo "--- CLIP for $LEVEL ---"
    CUDA_VISIBLE_DEVICES=0 python -u build_compressed_canvases.py --level $LEVEL --phase embed
done

echo ""
echo "=== Evaluating with Qwen2.5-VL-7B ==="

# Eval (needs GPU, 7B model is large)
# Run heavy first (skip baseline on light since it's the same)
echo "--- Eval heavy ---"
CUDA_VISIBLE_DEVICES=0,1 python -u eval_compressed.py --level heavy --max-dev 500

echo "--- Eval light (skip baseline, same as heavy) ---"
CUDA_VISIBLE_DEVICES=0,1 python -u eval_compressed.py --level light --max-dev 500 --skip-baseline

echo ""
echo "============================================"
echo "All done! Check results/"
echo "============================================"
