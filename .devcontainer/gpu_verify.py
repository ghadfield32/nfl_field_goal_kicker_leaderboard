#!/usr/bin/env python3
"""
Verify that the GPU is accessible and JAX is correctly configured.
This script is used during container startup.
"""

import sys

def check_gpu():
    print("Checking GPU availability...")
    try:
        import jax
        jax.config.update('jax_platform_name', 'gpu')
        
        # Get device count and details
        devices = jax.devices()
        device_count = len(devices)
        print(f"JAX version: {jax.__version__}")
        print(f"Available devices: {device_count}")
        
        for i, device in enumerate(devices):
            print(f"Device {i}: {device}")
        
        if device_count == 0 or 'gpu' not in str(devices[0]).lower():
            print("WARNING: No GPU devices found by JAX!")
            return False
        
        # Check CUDA configuration
        import jax.tools.jax_jit
        jit_info = jax.tools.jax_jit.get_jax_jit_flags()
        print(f"JIT configuration: {jit_info}")
        
        # Run a simple GPU computation
        print("Running a test computation on GPU...")
        import numpy as np
        x = np.ones((1000, 1000))
        result = jax.numpy.sum(x, axis=0)
        print(f"Test computation result shape: {result.shape}")
        
        print("JAX GPU verification completed successfully!")
        return True
    
    except ImportError:
        print("JAX not found! Make sure JAX is installed with GPU support.")
        return False
    except Exception as e:
        print(f"Error during GPU verification: {e}")
        return False

if __name__ == "__main__":
    success = check_gpu()
    if not success:
        print("WARNING: GPU verification failed!")
        # Not exiting with error to allow container to start anyway
        # sys.exit(1)
    else:
        sys.exit(0) 
