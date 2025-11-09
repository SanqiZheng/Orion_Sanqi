#!/usr/bin/env bash

# ====================================
# 系统准备脚本 - 在运行训练前优化系统
# ====================================

echo "🔧 准备系统环境..."
echo "================================"

# 检查是否有root权限
if [ "$EUID" -ne 0 ]; then 
    echo "⚠️  建议以root权限运行此脚本以获得最佳效果"
    echo "   sudo bash prepare_system.sh"
    echo ""
fi

# 1. 显示当前内存状态
echo "📊 当前内存状态："
free -h
echo ""

# 2. 清理系统缓存（需要root权限）
if [ "$EUID" -eq 0 ]; then
    echo "🧹 清理系统缓存..."
    sync
    echo 3 > /proc/sys/vm/drop_caches
    echo "✅ 系统缓存已清理"
else
    echo "⏩ 跳过系统缓存清理（需要root权限）"
fi
echo ""

# 3. 清理GPU内存
echo "🎮 清理GPU进程..."
if command -v nvidia-smi &> /dev/null; then
    # 显示当前GPU使用情况
    echo "当前GPU状态："
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
    echo ""
    
    # 列出GPU上的进程
    gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
    if [ -n "$gpu_pids" ]; then
        echo "发现以下GPU进程："
        echo "$gpu_pids" | while read pid; do
            if [ -n "$pid" ]; then
                ps -p "$pid" -o pid,user,%cpu,%mem,cmd 2>/dev/null || echo "PID: $pid"
            fi
        done
        echo ""
        
        read -p "是否终止这些GPU进程? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "$gpu_pids" | while read pid; do
                [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null && echo "已终止进程: $pid"
            done
            sleep 2
            nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
        fi
    else
        echo "✅ 没有发现GPU进程"
    fi
else
    echo "⚠️  未找到nvidia-smi命令"
fi
echo ""

# 4. 检查并建议关闭占用内存的进程
echo "🔍 检查占用内存的进程（前10个）："
ps aux --sort=-%mem | head -n 11 | awk '{printf "%-10s %-8s %-8s %s\n", $1, $2, $4"%", $11}'
echo ""

echo "💡 建议："
echo "  - 关闭浏览器、IDE等大内存程序"
echo "  - 关闭不必要的后台服务"
echo ""

# 5. 检查swap空间
echo "💾 Swap空间状态："
free -h | grep Swap
swap_total=$(free -m | grep Swap | awk '{print $2}')
if [ "$swap_total" -lt 8192 ]; then
    echo "⚠️  Swap空间较小 (< 8GB)"
    echo "💡 如果内存不足，可以临时增加swap："
    echo "   sudo fallocate -l 8G /swapfile"
    echo "   sudo chmod 600 /swapfile"
    echo "   sudo mkswap /swapfile"
    echo "   sudo swapon /swapfile"
fi
echo ""

# 6. 设置vm参数（需要root权限）
if [ "$EUID" -eq 0 ]; then
    echo "⚙️  优化虚拟内存参数..."
    
    # 降低swappiness，减少使用swap
    echo 10 > /proc/sys/vm/swappiness
    echo "✅ swappiness = 10 (减少swap使用)"
    
    # 增加vfs_cache_pressure，更积极地回收缓存
    echo 100 > /proc/sys/vm/vfs_cache_pressure
    echo "✅ vfs_cache_pressure = 100"
    
    # 禁用内存过量分配（更严格的内存管理）
    # echo 2 > /proc/sys/vm/overcommit_memory
    # echo "✅ overcommit_memory = 2 (严格模式)"
else
    echo "⏩ 跳过虚拟内存参数优化（需要root权限）"
fi
echo ""

# 7. 最终内存状态
echo "✅ 准备完成！当前内存状态："
free -h
echo ""

if command -v nvidia-smi &> /dev/null; then
    echo "GPU状态："
    nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv
fi

echo ""
echo "================================"
echo "💡 现在可以运行训练脚本了："
echo "   bash adzoo/orion/orion_debug.sh"
echo "================================"

