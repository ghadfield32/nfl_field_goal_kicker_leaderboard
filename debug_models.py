#!/usr/bin/env python3
"""Debug script to check model availability and directory structure."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.nfl_kicker_analysis.config import config
    print("✅ Config loaded successfully")
    print(f"PROJECT_ROOT: {config.PROJECT_ROOT}")
    print(f"MODELS_DIR: {config.MODELS_DIR}")
except Exception as e:
    print(f"❌ Error loading config: {e}")
    sys.exit(1)

# Check directory structure
print("\n=== Directory Structure ===")
main_mlruns = config.PROJECT_ROOT / "mlruns"
point_estimate_dir = config.MODELS_DIR / "mlruns" / "models"

print(f"Main mlruns dir: {main_mlruns}")
print(f"Main mlruns exists: {main_mlruns.exists()}")

if main_mlruns.exists():
    models_dir = main_mlruns / "models"
    print(f"Main models dir: {models_dir}")
    print(f"Main models dir exists: {models_dir.exists()}")
    if models_dir.exists():
        print(f"Main models contents: {list(models_dir.iterdir())}")

print(f"\nPoint estimate dir: {point_estimate_dir}")
print(f"Point estimate dir exists: {point_estimate_dir.exists()}")
if point_estimate_dir.exists():
    print(f"Point estimate contents: {list(point_estimate_dir.iterdir())}")

# Check model utils
try:
    from src.nfl_kicker_analysis.utils.model_utils import list_registered_models, list_saved_models
    print("\n=== MLflow Registered Models ===")
    mlflow_models = list_registered_models()
    print(f"Registered models: {mlflow_models}")
    
    print("\n=== Filesystem Models ===")
    fs_models = list_saved_models(point_estimate_dir)
    print(f"Filesystem models (point_estimate_dir): {fs_models}")
    
    # Also check main mlruns/models
    if main_mlruns.exists():
        main_models_dir = main_mlruns / "models"
        if main_models_dir.exists():
            fs_models_main = list_saved_models(main_models_dir)
            print(f"Filesystem models (main mlruns): {fs_models_main}")
    
except Exception as e:
    print(f"❌ Error checking model utils: {e}")
    import traceback
    traceback.print_exc() 