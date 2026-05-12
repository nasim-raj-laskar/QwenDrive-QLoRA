#!/bin/bash
echo "🧹 Complete Project Cleanup"
echo "=========================="

TOTAL_BEFORE=$(du -sh . | cut -f1)
echo "Project size before cleanup: $TOTAL_BEFORE"
echo

# Show what will be deleted
echo "📋 Items to clean:"
[ -d "models/hf_cache" ] && echo "   🗂️  Model cache: $(du -sh models/hf_cache | cut -f1)"
[ -d "output" ] && echo "   📁 Training output: $(du -sh output | cut -f1)"
[ -d "__pycache__" ] && echo "   🐍 Python cache: $(du -sh __pycache__ | cut -f1)"
find . -name "*.pyc" -o -name "*.pyo" | head -5 | while read f; do echo "   🗑️  $f"; done

echo
read -p "❓ Proceed with cleanup? (y/N): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Clean model cache
    [ -d "models/hf_cache" ] && rm -rf models/hf_cache && echo "   ✅ Deleted model cache"
    
    # Clean output
    [ -d "output" ] && rm -rf output && echo "   ✅ Deleted training output"
    
    # Clean Python cache
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
    find . -name "*.pyc" -delete 2>/dev/null
    find . -name "*.pyo" -delete 2>/dev/null
    echo "   ✅ Deleted Python cache files"
    
    TOTAL_AFTER=$(du -sh . | cut -f1)
    echo
    echo "🎉 Cleanup complete!"
    echo "   Before: $TOTAL_BEFORE"
    echo "   After:  $TOTAL_AFTER"
else
    echo "   ❌ Cleanup cancelled"
fi