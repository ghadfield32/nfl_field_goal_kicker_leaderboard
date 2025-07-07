#!/usr/bin/env python3
"""
Comprehensive setup script for Streamlit Cloud deployment.
Ensures all required files are in the correct locations for both 
point estimate and Bayesian models to work properly.
"""

import shutil
from pathlib import Path
import sys

def setup_streamlit_cloud():
    """Setup all required files for Streamlit Cloud deployment."""
    print("🚀 Setting up files for Streamlit Cloud deployment...")
    
    project_root = Path(__file__).parent.absolute()
    print(f"📁 Project root: {project_root}")
    
    # === Point Estimate Model Setup ===
    print("\n📊 Setting up Point Estimate Models...")
    
    # Copy leaderboard files from output/ to project root
    output_dir = project_root / "output"
    if output_dir.exists():
        leaderboard_files = list(output_dir.glob("*leaderboard.csv"))
        print(f"   Found {len(leaderboard_files)} leaderboard files in output/")
        
        for csv_file in leaderboard_files:
            target_file = project_root / csv_file.name
            if not target_file.exists():
                shutil.copy2(csv_file, target_file)
                print(f"   ✅ Copied {csv_file.name} to project root")
            else:
                print(f"   ⏭️  {csv_file.name} already exists in project root")
    else:
        print("   ⚠️  output/ directory not found")
    
    # Copy leaderboard files to data/processed/ as well
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    if output_dir.exists():
        for csv_file in output_dir.glob("*leaderboard.csv"):
            target_file = processed_dir / csv_file.name
            if not target_file.exists():
                shutil.copy2(csv_file, target_file)
                print(f"   ✅ Copied {csv_file.name} to data/processed/")
    
    # === Bayesian Model Setup ===
    print("\n🔬 Setting up Bayesian Models...")
    
    # Copy bayesian_features.csv to multiple locations
    bayesian_features_source = project_root / "output" / "bayesian_features.csv"
    if bayesian_features_source.exists():
        print(f"   ✅ Found bayesian_features.csv in output/")
        
        # Copy to project root
        target_root = project_root / "bayesian_features.csv"
        if not target_root.exists():
            shutil.copy2(bayesian_features_source, target_root)
            print(f"   ✅ Copied bayesian_features.csv to project root")
        else:
            print(f"   ⏭️  bayesian_features.csv already exists in project root")
        
        # Copy to data/processed/
        target_processed = processed_dir / "bayesian_features.csv"
        if not target_processed.exists():
            shutil.copy2(bayesian_features_source, target_processed)
            print(f"   ✅ Copied bayesian_features.csv to data/processed/")
        else:
            print(f"   ⏭️  bayesian_features.csv already exists in data/processed/")
    else:
        print("   ⚠️  bayesian_features.csv not found in output/")
    
    # Check Bayesian model directories
    bayesian_models_dir = project_root / "models" / "bayesian"
    if bayesian_models_dir.exists():
        suite_dirs = [d for d in bayesian_models_dir.iterdir() 
                     if d.is_dir() and (d/"meta.json").exists() and (d/"trace.nc").exists()]
        print(f"   ✅ Found {len(suite_dirs)} valid Bayesian suite(s)")
        for suite_dir in suite_dirs:
            print(f"      - {suite_dir.name}")
    else:
        print("   ⚠️  models/bayesian/ directory not found")
    
    # === MLflow Model Setup ===
    print("\n🤖 Checking MLflow Models...")
    
    mlflow_models_dir = project_root / "mlruns" / "models"
    if mlflow_models_dir.exists():
        model_names = [d.name for d in mlflow_models_dir.iterdir() if d.is_dir()]
        print(f"   ✅ Found {len(model_names)} MLflow model(s): {model_names}")
    else:
        print("   ⚠️  mlruns/models/ directory not found")
    
    # === Verify Key Files ===
    print("\n🔍 Verifying key files...")
    
    key_files = [
        "app.py",
        "requirements.txt",
        "data/raw/kickers.csv",
        "data/raw/field_goal_attempts.csv",
    ]
    
    for file_path in key_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - MISSING")
    
    # === Summary ===
    print("\n📋 Setup Summary:")
    
    # Count leaderboard files
    leaderboard_count = len(list(project_root.glob("*leaderboard.csv")))
    print(f"   📊 Point estimate leaderboards: {leaderboard_count}")
    
    # Check Bayesian data
    bayesian_data_exists = (project_root / "bayesian_features.csv").exists()
    print(f"   🔬 Bayesian features data: {'✅' if bayesian_data_exists else '❌'}")
    
    # Check Bayesian models
    if bayesian_models_dir.exists():
        suite_count = len([d for d in bayesian_models_dir.iterdir() 
                          if d.is_dir() and (d/"meta.json").exists() and (d/"trace.nc").exists()])
        print(f"   🔬 Bayesian model suites: {suite_count}")
    else:
        print(f"   🔬 Bayesian model suites: 0")
    
    # Check MLflow models
    if mlflow_models_dir.exists():
        mlflow_count = len([d for d in mlflow_models_dir.iterdir() if d.is_dir()])
        print(f"   🤖 MLflow models: {mlflow_count}")
    else:
        print(f"   🤖 MLflow models: 0")
    
    print("\n🎉 Setup complete! Your app should now work in Streamlit Cloud.")
    print("\n💡 Next steps:")
    print("   1. Commit and push these changes to your repository")
    print("   2. Deploy to Streamlit Cloud")
    print("   3. Check the app logs if any issues occur")

if __name__ == "__main__":
    setup_streamlit_cloud() 