#!/bin/bash
echo "🔍 QwenDrive-QLoRA System Status"
echo "================================"

# Project info
echo
echo "📂 Project Status:"
echo "   Location: $(pwd)"
echo "   Python: $(python --version 2>&1)"
if [ -f "requirements.txt" ]; then
    echo "   Dependencies: $(wc -l < requirements.txt) packages"
fi

# Storage
echo
./scripts/check_sizes.sh

# GPU
echo
./scripts/check_gpu.sh

# Training status
echo
echo "🎯 Training Status:"
if [ -d "output/qwen3b-automotive" ]; then
    echo "   ✅ Model trained ($(ls output/qwen3b-automotive/*.safetensors 2>/dev/null | wc -l) adapter files)"
    echo "   📅 Last modified: $(stat -c %y output/qwen3b-automotive 2>/dev/null | cut -d' ' -f1)"
else
    echo "   ❌ No trained model found"
fi

if [ -f "output/training_sample.jsonl" ]; then
    SAMPLES=$(wc -l < output/training_sample.jsonl)
    echo "   📊 Training samples: $SAMPLES"
fi

echo
echo "🚀 Quick Commands:"
echo "   Train:        python train.py"
echo "   Check sizes:  ./scripts/check_sizes.sh"
echo "   Clean models: ./scripts/clean_models.sh"
echo "   Monitor GPU:  ./scripts/check_gpu.sh"