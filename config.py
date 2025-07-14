"""
Configuration file for LongAttn project
Supports both LLaMA-3 and DeepSeek V3 models
"""

class ModelConfig:
    """Model configuration class"""
    
    # LLaMA-3 Configuration
    LLAMA3_CONFIG = {
        "model_path": "meta-llama/Meta-Llama-3.1-70B",
        "tokenizer_path": "meta-llama/Meta-Llama-3-8B",
        "window_size": 32768,
        "rope_theta": 2500000.0,
        "max_length": 32768
    }
    
    # DeepSeek V3 Configuration
    DEEPSEEK_V3_CONFIG = {
        "model_path": "deepseek-ai/DeepSeek-V3-0324",
        "tokenizer_path": "deepseek-ai/DeepSeek-V3-0324", 
        "window_size": 131072,
        "rope_theta": 1000000.0,
        "max_length": 131072
    }
    
    # Default configuration (DeepSeek V3)
    DEFAULT_CONFIG = DEEPSEEK_V3_CONFIG
    
    @classmethod
    def get_config(cls, model_type="deepseek_v3"):
        """Get configuration for specified model type"""
        if model_type.lower() == "llama3":
            return cls.LLAMA3_CONFIG
        elif model_type.lower() in ["deepseek_v3", "deepseek", "v3"]:
            return cls.DEEPSEEK_V3_CONFIG
        else:
            print(f"Unknown model type: {model_type}. Using DeepSeek V3 as default.")
            return cls.DEFAULT_CONFIG
    
    @classmethod
    def list_models(cls):
        """List available model configurations"""
        return {
            "llama3": "LLaMA-3 with 32K context window",
            "deepseek_v3": "DeepSeek V3 with 128K context window"
        } 