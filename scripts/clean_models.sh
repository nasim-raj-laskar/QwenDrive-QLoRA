#!/bin/bash
echo "🧹 Cleaning Model Cache..."

if [ -d "models/hf_cache" ]; then
    SIZE_BEFORE=$(du -sh models/hf_cache | cut -f1)
    echo "   Before: $SIZE_BEFORE"
    
    read -p "❓ Delete all cached models? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf models/hf_cache
        echo "   ✅ Deleted models/hf_cache"
        echo "   💾 Freed: $SIZE_BEFORE"
    else
        echo "   ❌ Cancelled"
    fi
else
    echo "   ℹ️  No models cache found"
fi