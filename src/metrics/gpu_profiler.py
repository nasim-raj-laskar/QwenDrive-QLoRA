import torch
import time
import threading
import mlflow
from typing import Dict, List
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class GPUProfiler:
    def __init__(self):
        self.vram_peak = 0
        self.gpu_utilization_samples = []
        self.tokens_per_sec_samples = []
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start GPU monitoring in background thread."""
        if torch.cuda.is_available():
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_gpu)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            logger.info("GPU monitoring started")
    
    def stop_monitoring(self):
        """Stop GPU monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("GPU monitoring stopped")
    
    def _monitor_gpu(self):
        """Monitor GPU metrics in background."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            while self.monitoring:
                # VRAM usage
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                vram_used_gb = mem_info.used / 1024**3
                self.vram_peak = max(self.vram_peak, vram_used_gb)
                
                # GPU utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                self.gpu_utilization_samples.append(util.gpu)
                
                time.sleep(1)
        except (ImportError, Exception) as e:
            logger.warning(f"pynvml not available ({e}), using nvidia-smi fallback")
            while self.monitoring:
                try:
                    import subprocess
                    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'], 
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        lines = result.stdout.strip().split('\n')
                        if lines and lines[0]:
                            parts = lines[0].split(', ')
                            if len(parts) >= 2:
                                gpu_util = float(parts[0])
                                vram_mb = float(parts[1])
                                self.gpu_utilization_samples.append(gpu_util)
                                vram_gb = vram_mb / 1024
                                self.vram_peak = max(self.vram_peak, vram_gb)
                except Exception:
                    # Fallback to torch for VRAM only
                    if torch.cuda.is_available():
                        vram_used_gb = torch.cuda.memory_allocated() / 1024**3
                        self.vram_peak = max(self.vram_peak, vram_used_gb)
                time.sleep(1)
    
    def record_tokens_per_sec(self, tokens: int, duration: float):
        """Record tokens per second measurement."""
        if duration > 0:
            tps = tokens / duration
            self.tokens_per_sec_samples.append(tps)
    
    def get_metrics(self) -> Dict[str, float]:
        """Get aggregated GPU metrics."""
        metrics = {
            "gpu_vram_peak_gb": self.vram_peak,
            "gpu_utilization_avg": sum(self.gpu_utilization_samples) / len(self.gpu_utilization_samples) if self.gpu_utilization_samples else 0,
            "gpu_utilization_max": max(self.gpu_utilization_samples) if self.gpu_utilization_samples else 0,
            "tokens_per_sec_avg": sum(self.tokens_per_sec_samples) / len(self.tokens_per_sec_samples) if self.tokens_per_sec_samples else 0,
            "tokens_per_sec_max": max(self.tokens_per_sec_samples) if self.tokens_per_sec_samples else 0
        }
        return metrics
    
    def log_to_mlflow(self):
        """Log GPU metrics to MLflow."""
        metrics = self.get_metrics()
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        logger.info(f"GPU metrics logged to MLflow: {metrics}")
        return metrics