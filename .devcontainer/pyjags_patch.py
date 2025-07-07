#!/usr/bin/env python3
import os
import sys

def patch_pyjags_sources():
    print("Downloading and patching PyJAGS source...")
    os.system("pip download --no-binary :all: pyjags==1.3.8")
    os.system("tar -xzf pyjags-1.3.8.tar.gz")
    os.chdir("pyjags-1.3.8")
    
    # Add cstdint include to all cpp files
    for root, dirs, files in os.walk("src"):
        for file in files:
            if file.endswith(".cpp") or file.endswith(".h"):
                filepath = os.path.join(root, file)
                with open(filepath, 'r') as f:
                    content = f.read()
                if "#include <cstdint>" not in content:
                    with open(filepath, 'w') as f:
                        f.write("#include <cstdint>\n" + content)
                    print(f"Patched {filepath}")
    
    # Build and install
    os.system("pip install --no-build-isolation .")
    print("PyJAGS installation complete!")
    return 0

if __name__ == "__main__":
    sys.exit(patch_pyjags_sources()) 
