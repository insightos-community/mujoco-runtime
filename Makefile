# Copyright 2026 InsightOS
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

.PHONY: install lint test test-contracts test-native test-profiles test-robosuite-real test-libero-real test-libero-pro-real build ci runtime-pack-native runtime-pack-robosuite runtime-pack-libero runtime-packs run run-robosuite run-libero demo

PYTEST_ENV = PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

LIBERO_REVISION = 8f1084e3132a39270c3a13ebe37270a43ece2a01
LIBERO_PRO_REVISION = 0bcf73621c789ffd6ed8858467a89df9ca94fd6b

install:
	uv sync --frozen --extra dev

lint:
	uv run ruff check src tests

test:
	$(PYTEST_ENV) uv run pytest -p pytest_cov -m 'not native' --cov=plugin_mujoco --cov-report=term-missing --cov-report=xml

test-contracts:
	$(PYTEST_ENV) uv run pytest -q \
		tests/test_v040_runtime_architecture.py::test_public_contract_examples_match_models

test-native:
	test -n "$$MUJOCO_ASSET_ROOT"
	$(PYTEST_ENV) uv run pytest -p pytest_cov -m native -vv --cov=plugin_mujoco.native --cov-config=.coveragerc-native --cov-report=term-missing

test-profiles:
	$(PYTEST_ENV) uv run --project profiles/common pytest -p no:cacheprovider profiles/common/tests
	uv run --project profiles/common ruff check profiles/common/src profiles/common/tests

# 真实 robosuite 门禁直接构造 Lift 环境并检查命令实际收敛，不能用 Fake Adapter
# 或“依赖不存在就 skip”替代。该目标在隔离 profile 锁文件中运行。
test-robosuite-real:
	$(PYTEST_ENV) MUJOCO_GL=$${MUJOCO_GL:-egl} uv run --project profiles/robosuite \
		--frozen python -m pytest -q profiles/robosuite/tests

# 固定 task/init-state/seed 的真实 LIBERO 证据。源码版本变量、实际 Git
# 提交和输出目录缺一不可，不能回退到下载、临时源码或 pytest skip。
test-libero-real:
	test -n "$$SEMANTIC_LIBERO_ROOT"
	test -n "$$SEMANTIC_LIBERO_REVISION"
	test "$$SEMANTIC_LIBERO_REVISION" = "$(LIBERO_REVISION)"
	test "$$(git -C "$$SEMANTIC_LIBERO_ROOT" rev-parse HEAD)" = "$$SEMANTIC_LIBERO_REVISION"
	test -n "$$SIMULATION_EVIDENCE_DIR"
	mkdir -p "$$SIMULATION_EVIDENCE_DIR/libero"
	$(PYTEST_ENV) MUJOCO_GL=$${MUJOCO_GL:-egl} uv run --project profiles/libero --frozen \
		semantic-sim-profile libero --libero-root "$$SEMANTIC_LIBERO_ROOT" \
		--suite libero_spatial --task-id 0 --init-state-id 0 --seed 7 --steps 10 \
		--output-dir "$$SIMULATION_EVIDENCE_DIR/libero"
	test -s "$$SIMULATION_EVIDENCE_DIR/libero/report.json"
	test -n "$$(find "$$SIMULATION_EVIDENCE_DIR/libero" -name '*.png' -print -quit)"

# LIBERO-Pro 复用同一 LIBERO 环境，额外校验固定扰动源码并同时保存
# base/perturbed 报告和 comparison.json。
test-libero-pro-real:
	test -n "$$SEMANTIC_LIBERO_ROOT"
	test -n "$$SEMANTIC_LIBERO_REVISION"
	test "$$SEMANTIC_LIBERO_REVISION" = "$(LIBERO_REVISION)"
	test "$$(git -C "$$SEMANTIC_LIBERO_ROOT" rev-parse HEAD)" = "$$SEMANTIC_LIBERO_REVISION"
	test -n "$$SEMANTIC_LIBERO_PRO_ROOT"
	test -n "$$SEMANTIC_LIBERO_PRO_REVISION"
	test "$$SEMANTIC_LIBERO_PRO_REVISION" = "$(LIBERO_PRO_REVISION)"
	test "$$(git -C "$$SEMANTIC_LIBERO_PRO_ROOT" rev-parse HEAD)" = "$$SEMANTIC_LIBERO_PRO_REVISION"
	test -n "$$SIMULATION_EVIDENCE_DIR"
	mkdir -p "$$SIMULATION_EVIDENCE_DIR/libero-pro"
	$(PYTEST_ENV) MUJOCO_GL=$${MUJOCO_GL:-egl} uv run --project profiles/libero --frozen \
		semantic-sim-profile libero-pro \
		--libero-root "$$SEMANTIC_LIBERO_ROOT" \
		--libero-pro-root "$$SEMANTIC_LIBERO_PRO_ROOT" \
		--evaluation-config profiles/libero/libero-pro.example.yaml \
		--perturbation environment \
		--suite libero_spatial --task-id 0 --init-state-id 0 --seed 7 --steps 10 \
		--output-dir "$$SIMULATION_EVIDENCE_DIR/libero-pro"
	test -s "$$SIMULATION_EVIDENCE_DIR/libero-pro/base/report.json"
	test -s "$$SIMULATION_EVIDENCE_DIR/libero-pro/perturbed/report.json"
	test -s "$$SIMULATION_EVIDENCE_DIR/libero-pro/comparison.json"
	test -n "$$(find "$$SIMULATION_EVIDENCE_DIR/libero-pro" -name '*.png' -print -quit)"

build:
	uv build

# 与功能分支流水线常驻 Job 对齐：lint / 单测 / 合同 / 构建 / gitleaks。
# native、robosuite、LIBERO、runtime-pack 仍只在专项 Runner / 标签上跑。
ci: lint test test-contracts build
	@if command -v gitleaks >/dev/null 2>&1; then \
		gitleaks detect --source . --config .gitleaks.toml --verbose; \
	elif [[ -x .ci-bin/gitleaks ]]; then \
		.ci-bin/gitleaks detect --source . --config .gitleaks.toml --verbose; \
	else \
		bash ci/install-gitleaks.sh && .ci-bin/gitleaks detect --source . --config .gitleaks.toml --verbose; \
	fi

PACK_VERSION ?= 0.4.0-dev.0

runtime-pack-native:
	uv run --with PyYAML python tools/build_runtime_pack.py --profile native-mujoco --version $(PACK_VERSION)

runtime-pack-robosuite:
	uv run --with PyYAML python tools/build_runtime_pack.py --profile robosuite-1.5 --version $(PACK_VERSION)

runtime-pack-libero:
	uv run --with PyYAML python tools/build_runtime_pack.py --profile libero-robosuite-1.4 --version $(PACK_VERSION)

runtime-packs: runtime-pack-native runtime-pack-robosuite runtime-pack-libero

run:
	uv run python -m plugin_mujoco.main

run-robosuite:
	MUJOCO_GL=${MUJOCO_GL:-egl} SEMANTIC_SIM_PROFILE=robosuite-1.5 \
		PLUGIN_MUJOCO_PORT=${PLUGIN_MUJOCO_PORT:-8091} \
		uv run --project profiles/robosuite --frozen semantic-sim-runtime

run-libero:
	test -n "$SEMANTIC_LIBERO_ROOT"
	MUJOCO_GL=${MUJOCO_GL:-egl} SEMANTIC_SIM_PROFILE=libero-robosuite-1.4 \
		PLUGIN_MUJOCO_PORT=${PLUGIN_MUJOCO_PORT:-8092} \
		uv run --project profiles/libero --frozen semantic-sim-runtime

demo:
	uv run python -m plugin_mujoco.testing.demo
