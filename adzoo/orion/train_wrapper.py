#!/usr/bin/env python3
"""
训练包装器 - 添加内存监控和保护
防止训练过程中内存溢出导致系统死机
"""
import os
import sys
import signal
import psutil
import threading
import time
from datetime import datetime

# 内存监控配置
MEMORY_CHECK_INTERVAL = 5  # 每5秒检查一次
MEMORY_THRESHOLD_PERCENT = 90  # 内存使用超过90%时警告
MEMORY_CRITICAL_PERCENT = 95  # 内存使用超过95%时强制退出
MEMORY_AVAILABLE_MIN_MB = 1000  # 可用内存低于1GB时警告

class MemoryMonitor:
    """内存监控器"""
    
    def __init__(self, work_dir):
        self.work_dir = work_dir
        self.should_stop = False
        self.critical_hit = False
        self.monitor_thread = None
        
    def check_memory(self):
        """检查内存状态"""
        mem = psutil.virtual_memory()
        mem_percent = mem.percent
        mem_available_mb = mem.available / (1024 * 1024)
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 记录到文件
        log_file = os.path.join(self.work_dir, 'python_memory_monitor.log')
        with open(log_file, 'a') as f:
            f.write(f"[{timestamp}] 内存使用: {mem_percent:.1f}%, "
                   f"可用: {mem_available_mb:.0f}MB\n")
        
        # 检查阈值
        if mem_percent > MEMORY_CRITICAL_PERCENT or mem_available_mb < 500:
            print(f"\n🚨 [{timestamp}] 紧急：内存使用率 {mem_percent:.1f}% 或可用内存过低！")
            print(f"为防止系统死机，立即终止训练！")
            self.critical_hit = True
            self.should_stop = True
            # 发送信号终止主进程
            os.kill(os.getpid(), signal.SIGTERM)
            return True
            
        elif mem_percent > MEMORY_THRESHOLD_PERCENT or mem_available_mb < MEMORY_AVAILABLE_MIN_MB:
            print(f"\n⚠️  [{timestamp}] 警告：内存使用率 {mem_percent:.1f}%, "
                  f"可用 {mem_available_mb:.0f}MB")
            return False
            
        return False
    
    def monitor_loop(self):
        """监控循环"""
        while not self.should_stop:
            self.check_memory()
            time.sleep(MEMORY_CHECK_INTERVAL)
    
    def start(self):
        """启动监控"""
        self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.monitor_thread.start()
        print(f"✅ Python内存监控已启动 (阈值: {MEMORY_THRESHOLD_PERCENT}%)")
    
    def stop(self):
        """停止监控"""
        self.should_stop = True
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)

def signal_handler(signum, frame):
    """信号处理器"""
    print(f"\n\n🛑 收到信号 {signum}，正在清理...")
    sys.exit(1)

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python train_wrapper.py <work_dir> <train_script> [args...]")
        sys.exit(1)
    
    work_dir = sys.argv[1]
    train_script = sys.argv[2]
    train_args = sys.argv[3:]
    
    # 确保工作目录存在
    os.makedirs(work_dir, exist_ok=True)
    
    # 注册信号处理
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # 启动内存监控
    monitor = MemoryMonitor(work_dir)
    monitor.start()
    
    # 记录启动信息
    print(f"🚀 启动训练脚本: {train_script}")
    print(f"📂 工作目录: {work_dir}")
    print(f"⚙️  参数: {' '.join(train_args)}")
    print("=" * 60)
    
    try:
        # 导入并执行训练脚本
        sys.path.insert(0, os.path.dirname(train_script))
        
        # 将参数传递给训练脚本
        sys.argv = [train_script] + train_args
        
        # 执行训练脚本
        with open(train_script, 'r') as f:
            code = compile(f.read(), train_script, 'exec')
            exec(code, {'__name__': '__main__', '__file__': train_script})
            
    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")
        monitor.stop()
        sys.exit(130)
    except SystemExit as e:
        if monitor.critical_hit:
            print("\n🚨 由于内存压力过大，训练被强制终止")
            sys.exit(137)
        raise
    except Exception as e:
        print(f"\n❌ 训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        monitor.stop()
        sys.exit(1)
    finally:
        monitor.stop()

if __name__ == '__main__':
    main()

