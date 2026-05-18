#!/bin/sh
set -e

# Ensure the HF cache directory exists and is writable.
# This matters when /var/cache/nanovllm is a host-mounted volume
# (Docker creates the host dir as root, overriding build-time chown).
mkdir -p "${HF_HOME:-/var/cache/nanovllm/hf}"

exec "$@"
