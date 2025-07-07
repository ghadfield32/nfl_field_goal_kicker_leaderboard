#!/usr/bin/env python3
"""Setup script to ensure all files are properly configured for Streamlit Cloud deployment."""

import shutil
import os
from pathlib import Path

def setup_for_streamlit_cloud():
    """Set up the project for Streamlit Cloud deployment."""
    
    print("Setting up project for Streamlit Cloud deployment...")
    
    # 1. Copy leaderboard files to accessible locations
    print("\nCopying leaderboard files...")
    copy_leaderboards()
    
    # 2. Ensure data directories exist
    print("\nCreating necessary directories...")
    directories = [
        "data/processed",
        "data/raw", 
        "output",
        "models/bayesian",
        "mlruns/models"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"Created/verified directory: {dir_path}")
    
    # 3. Create a streamlit config file
    print("\nCreating Streamlit configuration...")
    streamlit_config_dir = Path(".streamlit")
    streamlit_config_dir.mkdir(exist_ok=True)
    
    config_content = """[server]
port = 8501
enableCORS = false
enableXsrfProtection = false
maxUploadSize = 200

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#FF6B35"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
"""
    
    with open(streamlit_config_dir / "config.toml", "w") as f:
        f.write(config_content)
    print("Created Streamlit config file")
    
    # 4. Create a simple startup script for cloud deployment
    print("\nCreating startup script...")
    startup_script = """#!/bin/bash
# Streamlit Cloud startup script

echo "Starting NFL Kicker Leaderboard app..."

# Copy leaderboard files if they exist
if [ -d "output" ]; then
    echo "Copying leaderboard files..."
    python copy_leaderboards.py
fi

# Start the Streamlit app
echo "Starting Streamlit..."
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
"""
    
    with open("start.sh", "w") as f:
        f.write(startup_script)
    
    # Make it executable (on Unix systems)
    try:
        os.chmod("start.sh", 0o755)
    except:
        pass  # Windows doesn't need this
    
    print("Created startup script")
    
    # 5. Verify critical files exist
    print("\nVerifying critical files...")
    critical_files = [
        "app.py",
        "requirements.txt", 
        "copy_leaderboards.py"
    ]
    
    for file_path in critical_files:
        if Path(file_path).exists():
            print(f"OK: {file_path} exists")
        else:
            print(f"ERROR: {file_path} missing!")
    
    # 6. Check for leaderboard files
    print("\nChecking leaderboard files...")
    leaderboard_files = list(Path(".").glob("*leaderboard.csv"))
    if leaderboard_files:
        print(f"Found {len(leaderboard_files)} leaderboard files in project root")
        for file in leaderboard_files:
            print(f"   {file}")
    else:
        print("WARNING: No leaderboard files found in project root")
    
    # 7. Check model files
    print("\nChecking model files...")
    mlruns_models = Path("mlruns/models")
    if mlruns_models.exists():
        model_dirs = [d for d in mlruns_models.iterdir() if d.is_dir()]
        print(f"Found {len(model_dirs)} model directories in MLflow")
        for model_dir in model_dirs:
            print(f"   {model_dir.name}")
    else:
        print("WARNING: No MLflow models directory found")
    
    print("\nSetup complete! Your project is ready for Streamlit Cloud deployment.")
    print("\nNext steps:")
    print("   1. Commit and push all changes to your GitHub repository")
    print("   2. Deploy to Streamlit Cloud using your GitHub repository")
    print("   3. The app should automatically start with all leaderboards available")

def copy_leaderboards():
    """Copy leaderboard files to multiple locations for compatibility."""
    
    # Define source and target directories
    source_dir = Path("output")
    target_dirs = [
        Path("."),  # Project root
        Path("data/processed"),
    ]
    
    # Ensure target directories exist
    for target_dir in target_dirs:
        target_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy all leaderboard CSV files
    if source_dir.exists():
        for csv_file in source_dir.glob("*leaderboard.csv"):
            print(f"   Found leaderboard file: {csv_file}")
            
            for target_dir in target_dirs:
                target_file = target_dir / csv_file.name
                try:
                    shutil.copy2(csv_file, target_file)
                    print(f"   Copied {csv_file.name} -> {target_dir}/")
                except Exception as e:
                    print(f"   ERROR: Failed to copy {csv_file} -> {target_file}: {e}")
    else:
        print(f"   ERROR: Source directory {source_dir} does not exist")
    
    # Also copy the main leaderboard.csv if it exists
    main_leaderboard = Path("output/leaderboard.csv")
    if main_leaderboard.exists():
        for target_dir in target_dirs:
            target_file = target_dir / "leaderboard.csv"
            try:
                shutil.copy2(main_leaderboard, target_file)
                print(f"   Copied main leaderboard -> {target_dir}/")
            except Exception as e:
                print(f"   ERROR: Failed to copy main leaderboard -> {target_file}: {e}")

if __name__ == "__main__":
    setup_for_streamlit_cloud() 