#!/bin/bash
echo "=== Project Storage Usage ==="
echo
echo "📁 Models Cache:"
if [ -d "models/hf_cache" ]; then
    du -sh models/hf_cache
    echo "   └─ $(find models/hf_cache -name "*.safetensors" | wc -l) model files"
else
    echo "   No models cached"
fi

echo
echo "📁 Training Output:"
if [ -d "output" ]; then
    du -sh output
    echo "   └─ $(find output -name "*.safetensors" | wc -l) adapter files"
else
    echo "   No output files"
fi

echo
echo "📁 Total Project Size:"
du -sh .
echo
echo "💾 Available Disk Space:"
df -h . | tail -1 | awk '{print "   Free: " $4 " / Total: " $2}'