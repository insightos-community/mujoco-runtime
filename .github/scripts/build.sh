#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
mkdir -p .output/payload
uv sync --frozen --python 3.10.19 --extra dev
make test
uv build
uv build --project packages/mujoco-visuals --out-dir dist
uv run --frozen --extra dev python ../automation/.github/scripts/runtime.py
