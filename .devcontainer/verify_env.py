# In your host terminal:
cat << 'EOF' > .devcontainer/verify_env.py
#!/usr/bin/env python3
import encodings, jupyterlab, torch, jax, sys, os

print("## Python & library diagnostics ##")
print("Python:", sys.executable, sys.version.split()[0])
print("🟢 encodings OK")
print("🟢 jupyterlab OK")
print("🟢 torch", torch.__version__, "CUDA:", torch.cuda.is_available())
print("🟢 jax", jax.__version__, "devices:", jax.devices())
EOF
chmod +x .devcontainer/verify_env.py
