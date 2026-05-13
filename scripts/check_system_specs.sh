#!/bin/bash

echo "=== System Specifications ==="
echo

# GPU Information
echo "GPU:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits | while read line; do
        echo "  $line"
    done
else
    echo "  No NVIDIA GPU detected"
fi
echo

# CPU Information
echo "CPU:"
echo "  $(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)"
echo "  Cores: $(nproc)"
echo

# Memory Information
echo "Memory:"
echo "  $(free -h | grep '^Mem:' | awk '{print $2}') total"
echo

# Storage Information
echo "Storage:"
df -h / | tail -1 | awk '{print "  " $2 " total, " $3 " used, " $4 " available"}'
echo

# Architecture
echo "Architecture:"
echo "  $(uname -m)"