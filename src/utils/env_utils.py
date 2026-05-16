import torch
import platform
import sys
from importlib.metadata import version

def get_environment_info():
    """Get minimal environment info for experiment tracking."""
    try:
        env_info = {
            "env_python": platform.python_version(),
            "env_platform": platform.system(),
            "env_torch": version("torch"),
            "env_transformers": version("transformers"),
        }
        
        # CUDA info
        if torch.cuda.is_available():
            env_info["env_cuda"] = torch.version.cuda
            env_info["env_gpu_name"] = torch.cuda.get_device_name(0)
            env_info["env_gpu_memory_gb"] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}"
        else:
            env_info["env_cuda"] = "N/A"
            env_info["env_gpu_name"] = "N/A"
        
        # Optional libraries
        for lib in ["peft", "trl", "bitsandbytes", "unsloth"]:
            try:
                env_info[f"env_{lib}"] = version(lib)
            except:
                env_info[f"env_{lib}"] = "N/A"
                
        return env_info
    except Exception as e:
        return {"env_error": str(e)}