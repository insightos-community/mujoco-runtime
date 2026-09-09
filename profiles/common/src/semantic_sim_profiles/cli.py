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

"""隔离 profile 的命令行入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from semantic_sim_profiles.libero import LiberoAdapter, generate_perturbed_bddl
from semantic_sim_profiles.models import ComparisonReport, ProfileRunRequest
from semantic_sim_profiles.robosuite import RobosuiteAdapter
from semantic_sim_profiles.runner import run_episode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Semantic MuJoCo benchmark profile")
    subparsers = parser.add_subparsers(dest="profile", required=True)

    robosuite = subparsers.add_parser("robosuite")
    robosuite.add_argument("--environment", choices=["Lift", "Stack"], required=True)
    _common_arguments(robosuite)

    libero = subparsers.add_parser("libero")
    _libero_arguments(libero)

    libero_pro = subparsers.add_parser("libero-pro")
    _libero_arguments(libero_pro)
    libero_pro.add_argument("--libero-pro-root", type=Path, required=True)
    libero_pro.add_argument("--evaluation-config", type=Path, required=True)
    libero_pro.add_argument(
        "--perturbation",
        choices=["environment", "spatial", "object", "language", "task"],
        required=True,
    )
    return parser


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument(
        "--camera",
        action="append",
        dest="camera_names",
        default=None,
        help="可重复指定；默认 agentview 和 robot0_eye_in_hand",
    )


def _libero_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--libero-root", type=Path, required=True)
    _common_arguments(parser)


def main() -> None:
    args = build_parser().parse_args()
    cameras = tuple(args.camera_names or ["agentview", "robot0_eye_in_hand"])
    if args.profile == "robosuite":
        request = ProfileRunRequest(
            profile="robosuite",
            environment=args.environment,
            seed=args.seed,
            steps=args.steps,
            output_dir=args.output_dir,
            camera_names=cameras,
            width=args.width,
            height=args.height,
        )
        adapter = RobosuiteAdapter(
            args.environment,
            camera_names=cameras,
            width=args.width,
            height=args.height,
            horizon=max(args.steps, 1),
        )
        report = _run_and_close(adapter, request)
        if report.failure_reason:
            _print_report(report.to_dict())
            raise SystemExit(1)
        _print_report(report.to_dict())
        return

    base_request = ProfileRunRequest(
        profile="libero",
        environment=args.suite + ":" + str(args.task_id),
        seed=args.seed,
        steps=args.steps,
        output_dir=args.output_dir / "base" if args.profile == "libero-pro" else args.output_dir,
        camera_names=cameras,
        width=args.width,
        height=args.height,
        suite=args.suite,
        task_id=args.task_id,
        init_state_id=args.init_state_id,
    )
    base_adapter = LiberoAdapter(
        source_root=args.libero_root,
        config_root=base_request.output_dir / ".libero-config",
        suite_name=args.suite,
        task_id=args.task_id,
        init_state_id=args.init_state_id,
        seed=args.seed,
        camera_names=cameras,
        width=args.width,
        height=args.height,
        horizon=max(args.steps, 1),
    )
    base_report = _run_and_close(base_adapter, base_request)
    if base_report.failure_reason:
        _print_report(base_report.to_dict())
        raise SystemExit(1)
    if args.profile == "libero":
        _print_report(base_report.to_dict())
        return

    perturbed_bddl = generate_perturbed_bddl(
        libero_pro_root=args.libero_pro_root,
        original_bddl=base_adapter.bddl_path,
        suite_name=args.suite,
        perturbation_name=args.perturbation,
        evaluation_config=args.evaluation_config,
        output_dir=args.output_dir / "generated",
        seed=args.seed,
    )
    perturbed_request = ProfileRunRequest(
        profile="libero-pro",
        environment=args.suite + ":" + str(args.task_id),
        seed=args.seed,
        steps=args.steps,
        output_dir=args.output_dir / "perturbed",
        camera_names=cameras,
        width=args.width,
        height=args.height,
        suite=args.suite,
        task_id=args.task_id,
        init_state_id=args.init_state_id,
        perturbation=args.perturbation,
    )
    perturbed_adapter = LiberoAdapter(
        source_root=args.libero_root,
        config_root=perturbed_request.output_dir / ".libero-config",
        suite_name=args.suite,
        task_id=args.task_id,
        init_state_id=args.init_state_id,
        seed=args.seed,
        camera_names=cameras,
        width=args.width,
        height=args.height,
        horizon=max(args.steps, 1),
        bddl_file=perturbed_bddl,
        use_init_state=False,
    )
    perturbed_report = _run_and_close(perturbed_adapter, perturbed_request)
    if perturbed_report.failure_reason:
        _print_report(perturbed_report.to_dict())
        raise SystemExit(1)
    comparison = ComparisonReport(
        suite=args.suite,
        task_id=args.task_id,
        init_state_id=args.init_state_id,
        perturbation=args.perturbation,
        base=base_report,
        perturbed=perturbed_report,
        differences={
            "reward": perturbed_report.reward - base_report.reward,
            "success_changed": perturbed_report.success != base_report.success,
            "base_success": base_report.success,
            "perturbed_success": perturbed_report.success,
        },
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = comparison.to_dict()
    (args.output_dir / "comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _print_report(payload)


def _run_and_close(adapter: Any, request: ProfileRunRequest) -> Any:
    try:
        return run_episode(adapter, request)
    finally:
        adapter.close()


def _print_report(payload: Dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
