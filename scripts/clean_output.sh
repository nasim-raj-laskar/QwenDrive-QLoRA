#!/bin/bash
echo "🧹 Cleaning Training Output..."

if [ -d "output" ]; then
    SIZE_BEFORE=$(du -sh output | cut -f1)
    echo "   Before: $SIZE_BEFORE"
    
    read -p "❓ Delete all training outputs? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf output
        echo "   ✅ Deleted output/"
        echo "   💾 Freed: $SIZE_BEFORE"
    else
        echo "   ❌ Cancelled"
    fi
else
    echo "   ℹ️  No output directory found"
fi