#!/bin/bash
echo "🖥️  GPU Status Check"
echo

if command -v nvidia-smi &> /dev/null; then
    echo "📊 GPU Memory Usage:"
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
    while IFS=',' read -r name used total util; do
        echo "   GPU: $name"
        echo "   Memory: ${used}MB / ${total}MB ($(( used * 100 / total ))%)"
        echo "   Utilization: ${util}%"
    done
    
    echo
    echo "🔥 GPU Temperature:"
    nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | \
    while read temp; do
        echo "   ${temp}°C"
    done
else
    echo "   ❌ nvidia-smi not found"
    echo "   ℹ️  Install NVIDIA drivers to monitor GPU"
fi

echo
echo "💻 System Memory:"
free -h | grep "Mem:" | awk '{print "   Used: " $3 " / Total: " $2}'