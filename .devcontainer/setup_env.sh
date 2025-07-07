#!/usr/bin/env sh
# Copy the template only on first run so local secrets are not overwritten
set -eu
if [ ! -f /workspace/.env ]; then
  echo "📝  Generating default .env from template"
  cp /workspace/.env.template /workspace/.env
fi 
