#!/usr/bin/env python3
"""Script to ensure leaderboard files are in the correct locations for Streamlit Cloud."""

import shutil
from pathlib import Path

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
            print(f"Found leaderboard file: {csv_file}")
            
            for target_dir in target_dirs:
                target_file = target_dir / csv_file.name
                try:
                    shutil.copy2(csv_file, target_file)
                    print(f"✅ Copied {csv_file} -> {target_file}")
                except Exception as e:
                    print(f"❌ Failed to copy {csv_file} -> {target_file}: {e}")
    else:
        print(f"❌ Source directory {source_dir} does not exist")
    
    # Also copy the main leaderboard.csv if it exists
    main_leaderboard = Path("output/leaderboard.csv")
    if main_leaderboard.exists():
        print(f"Found main Bayesian leaderboard: {main_leaderboard}")
        
        # Copy to target directories
        for target_dir in target_dirs:
            target_file = target_dir / "leaderboard.csv"
            if not target_file.exists():
                shutil.copy2(main_leaderboard, target_file)
                print(f"✅ Copied {main_leaderboard} to {target_file}")
            else:
                print(f"⏭️  {target_file} already exists")

        # Also copy to project root for Streamlit Cloud
        project_root_target = Path("leaderboard.csv")
        if not project_root_target.exists():
            shutil.copy2(main_leaderboard, project_root_target)
            print(f"✅ Copied {main_leaderboard} to project root")
        else:
            print(f"⏭️  leaderboard.csv already exists in project root")

if __name__ == "__main__":
    copy_leaderboards() 