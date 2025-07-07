#!/usr/bin/env python3
"""
Comprehensive diagnostic script for dev container issues.
Run this inside the container to diagnose Python environment and remote extension problems.
"""

import sys
import os
import subprocess
import json
from pathlib import Path


def run_command(cmd, description):
    """Run a command and return its output."""
    print(f"\n🔍 {description}")
    print("=" * 60)
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {result.stdout.strip()}")
        else:
            print(f"❌ Error (code {result.returncode}): {result.stderr.strip()}")
        return result
    except Exception as e:
        print(f"❌ Exception: {e}")
        return None


def check_paths_and_environment():
    """Check Python paths and environment variables."""
    print("\n🐍 PYTHON ENVIRONMENT DIAGNOSTICS")
    print("=" * 60)
    
    # Python executable and version
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Python path: {sys.path[:3]}...")  # First few paths
    
    # Environment variables
    print(f"\nVIRTUAL_ENV: {os.environ.get('VIRTUAL_ENV', 'Not set')}")
    print(f"PATH (first 3): {':'.join(os.environ.get('PATH', '').split(':')[:3])}")
    
    # Virtual environment validation
    venv_path = Path('/app/.venv')
    if venv_path.exists():
        print(f"✅ Virtual environment exists at {venv_path}")
        print(f"   - bin directory: {list(venv_path.glob('bin/python*'))}")
        print(f"   - site-packages: {(venv_path / 'lib/python3.10/site-packages').exists()}")
    else:
        print(f"❌ Virtual environment NOT found at {venv_path}")


def check_key_packages():
    """Check if key packages are importable."""
    print("\n📦 PACKAGE IMPORT TESTS")
    print("=" * 60)
    
    packages = [
        'jax', 'torch', 'numpy', 'pandas', 'matplotlib', 
        'jupyterlab', 'streamlit', 'sklearn'
    ]
    
    for package in packages:
        try:
            if package == 'sklearn':
                import sklearn
                version = sklearn.__version__
            else:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
            print(f"✅ {package}: {version}")
        except ImportError as e:
            print(f"❌ {package}: Import failed - {e}")
        except Exception as e:
            print(f"⚠️  {package}: {e}")


def check_gpu_environment():
    """Check GPU-related environment variables."""
    print("\n🎮 GPU ENVIRONMENT VARIABLES")
    print("=" * 60)
    
    gpu_env_vars = [
        'XLA_PYTHON_CLIENT_PREALLOCATE',
        'XLA_PYTHON_CLIENT_ALLOCATOR', 
        'XLA_PYTHON_CLIENT_MEM_FRACTION',
        'JAX_PLATFORM_NAME',
        'XLA_FLAGS',
        'JAX_DISABLE_JIT',
        'JAX_ENABLE_X64',
        'JAX_PREALLOCATION_SIZE_LIMIT_BYTES',
        'TF_FORCE_GPU_ALLOW_GROWTH',
        'NVIDIA_VISIBLE_DEVICES',
        'NVIDIA_DRIVER_CAPABILITIES'
    ]
    
    for var in gpu_env_vars:
        value = os.environ.get(var, 'Not set')
        print(f"   {var}: {value}")


def check_gpu_support():
    """Check GPU support for JAX and PyTorch with enhanced diagnostics."""
    print("\n🎮 ENHANCED GPU SUPPORT CHECK")
    print("=" * 60)
    
    # JAX GPU check with detailed info
    try:
        import jax
        print(f"JAX version: {jax.__version__}")
        
        devices = jax.devices()
        print(f"JAX devices: {devices}")
        
        if devices:
            for i, device in enumerate(devices):
                print(f"   Device {i}: {device}")
                
        if any('gpu' in str(device).lower() or 'cuda' in str(device).lower() for device in devices):
            print("✅ JAX GPU/CUDA support detected!")
            
            # Test a simple computation
            try:
                import jax.numpy as jnp
                x = jnp.ones((1000, 1000))
                result = jnp.sum(x)
                print(f"   ✅ JAX GPU computation test passed: sum = {result}")
            except Exception as e:
                print(f"   ⚠️  JAX GPU computation test failed: {e}")
        else:
            print("⚠️  JAX GPU support not detected")
            print("   This might be due to GPU architecture compatibility")
            
    except Exception as e:
        print(f"❌ JAX GPU check failed: {e}")
    
    # PyTorch GPU check with enhanced info
    try:
        import torch
        print(f"\nPyTorch version: {torch.__version__}")
        print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"✅ PyTorch CUDA device count: {device_count}")
            
            for i in range(device_count):
                try:
                    device_name = torch.cuda.get_device_name(i)
                    memory_total = torch.cuda.get_device_properties(i).total_memory
                    print(f"   Device {i}: {device_name}")
                    print(f"     Total memory: {memory_total / (1024**3):.1f} GB")
                except Exception as e:
                    print(f"   Device {i}: Error getting info - {e}")
            
            # Test a simple computation
            try:
                device = torch.device('cuda:0')
                x = torch.ones(1000, 1000, device=device)
                result = torch.sum(x)
                print(f"   ✅ PyTorch GPU computation test passed: sum = {result}")
            except Exception as e:
                print(f"   ⚠️  PyTorch GPU computation test failed: {e}")
        else:
            print("⚠️  PyTorch CUDA not available")
            print("   Check CUDA installation and GPU compatibility")
            
    except Exception as e:
        print(f"❌ PyTorch GPU check failed: {e}")


def check_workspace_mount():
    """Check if workspace is properly mounted."""
    print("\n📁 WORKSPACE MOUNT CHECK")
    print("=" * 60)
    
    workspace_path = Path('/workspace')
    if workspace_path.exists():
        print(f"✅ /workspace directory exists")
        try:
            contents = list(workspace_path.iterdir())[:10]  # First 10 items
            print(f"   Contents (first 10): {[p.name for p in contents]}")
            
            # Check for specific expected files
            expected_files = ['.devcontainer', 'pyproject.toml', 'docker-compose.yml']
            for file in expected_files:
                if (workspace_path / file).exists():
                    print(f"   ✅ Found: {file}")
                else:
                    print(f"   ❌ Missing: {file}")
        except Exception as e:
            print(f"   ❌ Error reading workspace: {e}")
    else:
        print(f"❌ /workspace directory does not exist")


def check_dev_container_config():
    """Check dev container configuration."""
    print("\n⚙️  DEV CONTAINER CONFIG CHECK")
    print("=" * 60)
    
    config_path = Path('/workspace/.devcontainer/devcontainer.json')
    if config_path.exists():
        print("✅ devcontainer.json found")
        try:
            with open(config_path) as f:
                config = json.load(f)
            print(f"   Name: {config.get('name', 'Not specified')}")
            print(f"   Python path: {config.get('customizations', {}).get('vscode', {}).get('settings', {}).get('python.defaultInterpreterPath', 'Not specified')}")
            print(f"   Workspace folder: {config.get('workspaceFolder', 'Not specified')}")
        except Exception as e:
            print(f"   ❌ Error reading config: {e}")
    else:
        print("❌ devcontainer.json not found")


def main():
    """Run all diagnostic checks."""
    print("🔍 DEV CONTAINER COMPREHENSIVE DIAGNOSTICS")
    print("=" * 80)
    print(f"Running from: {os.getcwd()}")
    print(f"User: {os.getenv('USER', 'unknown')}")
    print(f"Container hostname: {os.getenv('HOSTNAME', 'unknown')}")
    
    # System commands
    run_command("uv --version", "UV Version")
    run_command("which python", "Python Location")
    run_command("ls -la /app/.venv/", "Virtual Environment Contents")
    run_command("mount | grep workspace", "Workspace Mount Status")
    run_command("nvidia-smi", "NVIDIA GPU Status")
    
    # Python-based checks
    check_paths_and_environment()
    check_gpu_environment()
    check_key_packages()
    check_gpu_support()
    check_workspace_mount()
    check_dev_container_config()
    
    print("\n" + "=" * 80)
    print("🎯 SUMMARY & RECOMMENDATIONS")
    print("=" * 80)
    print("If you see issues:")
    print("1. ❌ Virtual env missing → Check Dockerfile uv sync step")
    print("2. ❌ Workspace not mounted → Check devcontainer.json mounts config")
    print("3. ❌ Packages missing → Check uv.lock and pip install steps")
    print("4. ⚠️  GPU not detected → Check docker-compose.yml gpu settings")
    print("5. 🔧 For VS Code issues → Check python.defaultInterpreterPath setting")
    print("6. 🎮 For GPU issues → Check NVIDIA drivers and CUDA compatibility")
    print("\n✅ All checks passed = ready for development!")


if __name__ == "__main__":
    main() 
