#!/bin/bash
# Build and push the MONARCHS Docker image to Docker Hub.
# Runnable from any directory - paths resolve relative to the repo root.
# Replace "jelsey92/monarchs" with "<username>/monarchs" for your own Docker Hub.
set -euo pipefail

# repo root is the parent of this script's directory (Docker/), so the build
# context is the whole repo (needed for the source install and .git/hatch-vcs)
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

docker build -t jelsey92/monarchs -f "$repo_root/Docker/Dockerfile" "$repo_root"
docker push jelsey92/monarchs