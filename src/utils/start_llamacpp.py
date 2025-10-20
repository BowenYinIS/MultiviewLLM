#!/usr/bin/env python3
"""
Script to start llama.cpp server with various model configurations.

The script automatically configures server parameters (np, ctx_size) based on 
hostname and provides convenient access to multiple Qwen model variants.

Usage:
    # Generate shell script with all model commands
    python start_llamacpp.py

    # Run a specific model directly
    python start_llamacpp.py qwen3-instruct-30b
    python start_llamacpp.py qwen3-instruct-4b
    python start_llamacpp.py qwen3-instruct-4b-finetuned
    python start_llamacpp.py qwen3-thinking-30b
    python start_llamacpp.py qwen2.5-instruct-7b
    python start_llamacpp.py qwen2-instruct-7b

    # List available models with configuration details
    python start_llamacpp.py --list

    # Show help
    python start_llamacpp.py --help

Available Models:
    - qwen3-instruct-30b: Qwen3 30B instruction model (Q4_K_XL, -hf)
    - qwen3-instruct-4b: Qwen3 4B instruction model (Q4_K_XL, -hf)
    - qwen3-instruct-4b-finetuned: Qwen3 4B finetuned model (Q8_0, -m local)
    - qwen3-thinking-30b: Qwen3 30B thinking model (Q4_K_XL, -hf)
    - qwen2.5-instruct-7b: Qwen2.5 7B instruction model (Q4_K_M, -hf)
    - qwen2-instruct-7b: Qwen2 7B instruction model (Q4_K_M, -hf)

Host Configuration:
    The script requires hostname to be configured in HOST_CONFIGS with:
    - np: Number of parallel slots
    - ctx_size: Total context size
    
    Configured hosts:
    - yu-lerner: np=8, ctx_size=262144
    - hopper: np=16, ctx_size=524288
"""
import os
import socket
import subprocess
import sys
from dataclasses import dataclass
from typing import NamedTuple

from src.config.paths import paths

class HostConfig(NamedTuple):
    """Host-specific configuration for llama.cpp server."""
    np: int
    ctx_size: int


# Host-specific configurations
HOST_CONFIGS = {
    "yu-lerner": HostConfig(np=8, ctx_size=262144),
    "hopper": HostConfig(np=16, ctx_size=524288),
}


def get_host_config() -> HostConfig:
    """
    Get np and ctx_size based on hostname.
    
    Returns:
        HostConfig: Configuration with np and ctx_size
    
    Raises:
        KeyError: If hostname is not configured
    """
    hostname = socket.gethostname()
    
    if hostname not in HOST_CONFIGS:
        raise KeyError(
            f"Unknown hostname '{hostname}'. "
            f"Please add configuration to HOST_CONFIGS. "
            f"Available hosts: {list(HOST_CONFIGS.keys())}"
        )
    
    return HOST_CONFIGS[hostname]


@dataclass
class ModelConfig:
    """Configuration for a llama.cpp model server.
    
    Note: Either hf_model or model_path should be provided, not both.
    If model_path is provided, it takes precedence and hf_model is ignored.
    """
    ctx_size: int
    np: int
    hf_model: str | None = None
    model_path: str | None = None
    temp: float | None = None
    top_p: float | None = None
    min_p: float | None = None
    top_k: int | None = None
    port: int = 8080
    gpu_id: int = 0
    n: int = 8192
    b: int = 32768
    ub: int = 16384
    presence_penalty: float = 1.0
    llama_cpp_path: str = "~/App/llama.cpp/build/bin/llama-server"
    flash_attn: bool = True
    ngl: int = -1
    threads: int = -1
    ctk: str = "q8_0"
    ctv: str = "q8_0"
    host: str = "0.0.0.0"
    jinja: bool = True


# Explicit model settings - modify these to change model behavior
# Note: Use either 'hf_model' (-hf flag) or 'model_path' (-m flag) for model specification
# If both are provided, 'model_path' takes precedence
# temp, top_p, min_p, and top_k are optional - omit them to not include in command
MODEL_SETTINGS = {
    "qwen3-instruct-30b": {
        "hf_model": "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_XL",
        "temp": 0.7,
        "top_p": 0.80,
        "min_p": 0.0,
        "top_k": 20,
        "presence_penalty": 1,
    },
    "qwen3-instruct-4b": {
        "hf_model": "unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_XL",
        "temp": 0.7,
        "top_p": 0.80,
        "min_p": 0.0,
        "top_k": 20,
        "presence_penalty": 1,
    },
    "qwen3-instruct-4b-finetuned": {
        "model_path": str(paths.checkpoint_dir / "benchmark/qwen3-4b-finetuned/gguf/qwen3-4b-instruct-2507-finetuned.Q8_0.gguf"),
        "temp": 0.7,
        "top_p": 0.80,
        "min_p": 0.0,
        "top_k": 20,
        "presence_penalty": 1,
    },
    "qwen3-thinking-30b": {
        "hf_model": "unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF:Q4_K_XL",
        "temp": 0.6,
        "top_p": 0.95,
        "min_p": 0.0,
        "top_k": 20,
        "presence_penalty": 1,
    },
    "qwen2.5-instruct-7b": {
        "hf_model": "Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M",
        "temp": 0.7,
        "top_p": 0.8,
        "repetition_penalty": 1.05,
    },
    "qwen2-instruct-7b": {
        "hf_model": "Qwen/Qwen2-7B-Instruct-GGUF:Q4_K_M",
        "temp": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "repetition_penalty": 1.05,
    },
}


def get_models() -> dict[str, ModelConfig]:
    """
    Get model configurations with host-specific np and ctx_size.
    
    Returns:
        dict[str, ModelConfig]: Dictionary of model configurations
    """
    host_config = get_host_config()
    
    models = {}
    for name, settings in MODEL_SETTINGS.items():
        models[name] = ModelConfig(
            ctx_size=host_config.ctx_size,
            np=host_config.np,
            hf_model=settings.get("hf_model"),
            model_path=settings.get("model_path"),
            temp=settings.get("temp"),
            top_p=settings.get("top_p"),
            min_p=settings.get("min_p"),
            top_k=settings.get("top_k"),
            port=settings.get("port", 8080),
        )
    
    return models


def build_command(config: ModelConfig) -> list[str]:
    """Build the llama-server command from configuration."""
    cmd = [os.path.expanduser(config.llama_cpp_path)]
    
    # Add model specification (-m takes precedence over -hf)
    if config.model_path:
        cmd.extend(["-m", config.model_path])
    elif config.hf_model:
        cmd.extend(["-hf", config.hf_model])
    
    if config.jinja:
        cmd.append("--jinja")
    
    cmd.extend([
        "-ngl", str(config.ngl),
        "--threads", str(config.threads),
        "--ctx-size", str(config.ctx_size),
    ])
    
    if config.flash_attn:
        cmd.extend(["--flash-attn", "on"])
    
    cmd.extend([
        "-b", str(config.b),
        "-ub", str(config.ub),
        "-n", str(config.n),
        "-ctk", config.ctk,
        "-ctv", config.ctv,
        "-np", str(config.np),
        "--host", config.host,
        "--port", str(config.port),
    ])
    
    # Only add sampling parameters if specified
    if config.temp is not None:
        cmd.extend(["--temp", str(config.temp)])
    
    if config.min_p is not None:
        cmd.extend(["--min-p", str(config.min_p)])
    
    if config.top_p is not None:
        cmd.extend(["--top-p", str(config.top_p)])
    
    if config.top_k is not None:
        cmd.extend(["--top-k", str(config.top_k)])
    
    cmd.extend(["--presence-penalty", str(config.presence_penalty)])
    
    return cmd


def generate_shell_script():
    """Generate shell script with all model commands."""
    models = get_models()
    hostname = socket.gethostname()
    host_config = get_host_config()
    
    print("#!/bin/bash")
    print()
    print("# Generated by start_llamacpp.py")
    print(f"# Host: {hostname}")
    print(f"# Configuration: np={host_config.np}, ctx_size={host_config.ctx_size:,}")
    print("# This script contains all available llama.cpp server configurations")
    print()
    print("# Choose model (-hf or -m) from:")
    for name, config in models.items():
        model_spec = config.model_path if config.model_path else config.hf_model
        model_type = "-m" if config.model_path else "-hf"
        print(f"# - {name}: {model_type} {model_spec}")
    print()
    print("# Key settings:")
    print("# - --ctx-size: Total context divided by -np slots")
    print("# - -np: Number of parallel slots")
    print("# - Each slot: ctx-size / np tokens")
    print("# - -n: Max output tokens per request")
    print("# - -b/-ub: High values for continuous batching efficiency")
    print()
    
    for name, config in models.items():
        cmd = build_command(config)
        print(f"# {name}")
        print(f"CUDA_VISIBLE_DEVICES={config.gpu_id} {' '.join(cmd)}")
        print()


def run_model(model_name: str):
    """Execute a specific model server."""
    models = get_models()
    
    if model_name not in models:
        print(f"Error: Unknown model '{model_name}'", file=sys.stderr)
        print(f"Available models: {', '.join(models.keys())}", file=sys.stderr)
        sys.exit(1)
    
    config = models[model_name]
    cmd = build_command(config)
    
    # Set environment variables
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
    
    print(f"Starting {model_name}...")
    if config.model_path:
        print(f"Model path (-m): {config.model_path}")
    elif config.hf_model:
        print(f"HF model (-hf): {config.hf_model}")
    print(f"Port: {config.port}")
    print(f"Context size: {config.ctx_size}")
    print(f"Parallel slots: {config.np}")
    print(f"GPU: {config.gpu_id}")
    print()
    print(f"Command: CUDA_VISIBLE_DEVICES={config.gpu_id} {' '.join(cmd)}")
    print()
    
    try:
        subprocess.run(cmd, env=env)
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        print(f"Error running server: {e}", file=sys.stderr)
        sys.exit(1)


def list_models():
    """List all available model configurations."""
    models = get_models()
    hostname = socket.gethostname()
    host_config = get_host_config()
    
    print(f"Host: {hostname}")
    print(f"Configuration: np={host_config.np}, ctx_size={host_config.ctx_size:,}")
    print()
    print("Available models:")
    print()
    for name, config in models.items():
        print(f"  {name}")
        if config.model_path:
            print(f"    Model path (-m): {config.model_path}")
        elif config.hf_model:
            print(f"    HF model (-hf): {config.hf_model}")
        print(f"    Context: {config.ctx_size:,} tokens")
        print(f"    Slots: {config.np} (≈{config.ctx_size // config.np:,} tokens/slot)")
        print(f"    Port: {config.port}")
        print(f"    GPU: {config.gpu_id}")
        print(f"    Temperature: {config.temp}")
        print()


def main():
    """Main entry point."""
    if len(sys.argv) == 1:
        # No arguments: generate shell script
        generate_shell_script()
    elif len(sys.argv) == 2:
        arg = sys.argv[1]
        if arg in ["--list", "-l"]:
            list_models()
        elif arg in ["--help", "-h"]:
            print(__doc__)
        else:
            # Run specific model
            run_model(arg)
    else:
        print("Error: Too many arguments", file=sys.stderr)
        print(__doc__, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

