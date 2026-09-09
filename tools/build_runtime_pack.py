#!/usr/bin/env python3
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

"""从固定源码 Tag 构建不依赖源码目录的 Runtime Pack。

脚本只调用参数数组形式的 uv/pip/tar，不执行 shell。每个 Pack 包含项目 Wheel、
离线依赖 Wheelhouse、导出的固定 requirements.lock、场景索引、smoke 请求、
许可提示和版本验证文件。外部资产与 benchmark 数据只写需求，不复制进 Pack。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
VERSION_RE = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(-[0-9A-Za-z.-]+)?$")


@dataclass(frozen=True)
class PackProfile:
    pack_id: str
    python: str
    project: Path
    runner: str
    endpoint: str
    smoke_scene: str
    profile: dict[str, object]
    hardware_requirements: dict[str, str]
    content_requirements: dict[str, dict[str, object]]
    wheel_projects: tuple[Path, ...]
    omit_packages: tuple[str, ...] = ()


def profile_table() -> dict[str, PackProfile]:
    common_capabilities = {
        "viewer": True,
        "viewer_camera_modes": ["fixed"],
        "scene_step": True,
        "scene_reset": True,
        "sensor_kinds": ["rgb", "depth", "contact", "holding"],
    }
    return {
        "native-mujoco": PackProfile(
            pack_id="native-mujoco",
            python="3.10.16",
            project=ROOT,
            runner="native-mujoco",
            endpoint="http://127.0.0.1:8090",
            smoke_scene="palletizing_depalletizing_001",
            profile={
                "runtime_profile_id": "native-mujoco",
                "name": "Native MuJoCo",
                "engine": "mujoco",
                "loader": "native",
                "api_version": "v1",
                "scene_kinds": ["scene_document", "asset_scene"],
                "environment": "runtime-pack",
                "environment_ready": True,
                "available": True,
                "availability_known": True,
                "capabilities": {
                    **common_capabilities,
                    "editable_scene": True,
                    "native_evaluator": False,
                    "viewer_camera_modes": ["free", "fixed"],
                    "robot_models": ["r1_pro_chassis"],
                },
            },
            hardware_requirements={
                "architecture": "amd64",
                "renderer": "egl",
                "gpu": "optional",
            },
            content_requirements={"mujoco_assets": {"required": True}},
            wheel_projects=(ROOT,),
        ),
        "robosuite-1.5": PackProfile(
            pack_id="robosuite-1.5",
            python="3.10.16",
            project=ROOT / "profiles" / "robosuite",
            runner="robosuite-1.5",
            endpoint="http://127.0.0.1:8091",
            smoke_scene="Lift",
            profile={
                "runtime_profile_id": "robosuite-1.5",
                "name": "robosuite 1.5",
                "engine": "mujoco",
                "loader": "robosuite",
                "api_version": "v1",
                "scene_kinds": ["robosuite-task"],
                "environment": "runtime-pack",
                "environment_ready": True,
                "available": True,
                "availability_known": True,
                "capabilities": {
                    **common_capabilities,
                    "editable_scene": False,
                    "native_evaluator": True,
                    "robot_models": ["franka_panda"],
                },
            },
            hardware_requirements={
                "architecture": "amd64",
                "renderer": "egl",
                "gpu": "optional",
            },
            content_requirements={"franka_model": {"required": True}},
            wheel_projects=(ROOT / "profiles" / "common",),
            omit_packages=("semantic-sim-profiles",),
        ),
        "libero-robosuite-1.4": PackProfile(
            pack_id="libero-robosuite-1.4",
            python="3.8.20",
            project=ROOT / "profiles" / "libero",
            runner="libero-robosuite-1.4",
            endpoint="http://127.0.0.1:8092",
            smoke_scene="libero_spatial:0",
            profile={
                "runtime_profile_id": "libero-robosuite-1.4",
                "name": "LIBERO / LIBERO-Pro",
                "engine": "mujoco",
                "loader": "libero",
                "api_version": "v1",
                "scene_kinds": ["libero-task", "libero-pro-evaluation"],
                "environment": "runtime-pack",
                "environment_ready": True,
                "available": True,
                "availability_known": True,
                "capabilities": {
                    **common_capabilities,
                    "editable_scene": False,
                    "native_evaluator": True,
                    "robot_models": ["franka_panda"],
                },
            },
            hardware_requirements={
                "architecture": "amd64",
                "renderer": "egl",
                "gpu": "optional",
            },
            content_requirements={
                "franka_model": {"required": True},
                "libero_source": {
                    "required": True,
                    "revision": "8f1084e3132a39270c3a13ebe37270a43ece2a01",
                    "license_confirmation": "LIBERO",
                },
                "libero_pro_source": {
                    "required": True,
                    "revision": "0bcf73621c789ffd6ed8858467a89df9ca94fd6b",
                    "license_confirmation": "LIBERO-Pro",
                },
            },
            wheel_projects=(ROOT / "profiles" / "common",),
            omit_packages=("semantic-sim-profiles",),
        ),
    }


def run(*arguments: str, cwd: Path = ROOT, stdout: Path | None = None) -> None:
    target = stdout.open("wb") if stdout else None
    try:
        subprocess.run(arguments, cwd=cwd, check=True, stdout=target)
    finally:
        if target:
            target.close()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def file_record(root: Path, path: Path) -> dict[str, str]:
    return {"path": path.relative_to(root).as_posix(), "sha256": digest(path)}


def python_wheel_version(pack_version: str) -> str:
    """把产品 SemVer 转成 Wheel METADATA 使用的 PEP 440 版本。"""
    base, separator, prerelease = pack_version.partition("-")
    if not separator:
        return base
    kind, dot, number = prerelease.partition(".")
    if dot and kind in {"dev", "rc"} and number.isdigit():
        return f"{base}.{kind}{number}" if kind == "dev" else f"{base}{kind}{number}"
    raise ValueError("Runtime Pack 预发布版本只支持 -dev.N 或 -rc.N")


def wheel_metadata_version(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        metadata_files = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_files) != 1:
            raise RuntimeError(f"Wheel {path.name} 缺少唯一 METADATA")
        for line in archive.read(metadata_files[0]).decode("utf-8").splitlines():
            if line.startswith("Version: "):
                return line.removeprefix("Version: ").strip()
    raise RuntimeError(f"Wheel {path.name} 缺少 Version")


def build_wheels(profile: PackProfile, stage: Path, pack_version: str) -> list[Path]:
    wheel_dir = stage / "wheels"
    wheel_dir.mkdir()
    for project in profile.wheel_projects:
        run("uv", "build", "--wheel", "--project", str(project), "--out-dir", str(wheel_dir))
    wheels = sorted(wheel_dir.glob("*.whl"))
    if not wheels:
        raise RuntimeError("没有生成 Runtime Wheel")
    expected = python_wheel_version(pack_version)
    for wheel in wheels:
        actual = wheel_metadata_version(wheel)
        if actual != expected:
            raise RuntimeError(
                f"Pack {pack_version} 不能包含 {wheel.name}（Wheel 版本 {actual}，期望 {expected}）"
            )
    return wheels


def export_and_build_dependencies(profile: PackProfile, stage: Path) -> tuple[Path, list[Path]]:
    lock = stage / "locks" / "requirements.lock"
    lock.parent.mkdir()
    command = [
        "uv",
        "export",
        "--project",
        str(profile.project),
        "--frozen",
        "--no-dev",
        "--no-hashes",
        "--no-emit-project",
        "--format",
        "requirements-txt",
        "--output-file",
        str(lock),
    ]
    for package in profile.omit_packages:
        command.extend(("--no-emit-package", package))
    run(*command)
    wheelhouse = stage / "wheelhouse"
    wheelhouse.mkdir()
    run(
        "uv",
        "run",
        "--isolated",
        "--no-project",
        "--python",
        profile.python,
        "--with",
        "pip",
        "python",
        "-m",
        "pip",
        "wheel",
        "--wheel-dir",
        str(wheelhouse),
        "--requirement",
        str(lock),
    )
    dependencies = sorted(wheelhouse.iterdir())
    if not dependencies or any(not path.is_file() for path in dependencies):
        raise RuntimeError("CI 生成的离线 Wheelhouse 为空或包含非文件")
    return lock, dependencies


def copy_metadata(
    profile: PackProfile, version: str, stage: Path
) -> tuple[Path, list[Path], Path, Path]:
    source = ROOT / "runtime-packs" / profile.pack_id
    catalog_target = stage / "catalog"
    shutil.copytree(source / "catalog", catalog_target)
    catalog = catalog_target / "catalog.yaml"
    stamp_catalog_version(catalog, version)
    smoke = stage / "smoke" / "request.json"
    smoke.parent.mkdir()
    shutil.copy2(source / "smoke-request.json", smoke)
    licenses = stage / "licenses"
    shutil.copytree(ROOT / "runtime-packs" / "LICENSES", licenses)
    verification = stage / "verification" / "version.json"
    verification.parent.mkdir()
    commit = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip()
    verification.write_text(
        json.dumps(
            {
                "pack_id": profile.pack_id,
                "pack_version": version,
                "source_commit": commit,
                "python_version": profile.python,
                "builder": "tools/build_runtime_pack.py/v1",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    resources = sorted(
        path for path in catalog_target.rglob("*") if path.is_file() and path != catalog
    )
    return catalog, resources, smoke, verification


def stamp_catalog_version(catalog: Path, version: str) -> None:
    """让 Pack 场景目录与它要求的资产目录使用同一个发布版本。"""
    document = yaml.safe_load(catalog.read_text(encoding="utf-8"))
    document["catalog_version"] = version
    for entry in document.get("entries", []):
        for scene_version in entry.get("versions", []):
            authoring = scene_version.get("authoring") or {}
            if "asset_catalog_version" in authoring:
                authoring["asset_catalog_version"] = version
    catalog.write_text(
        yaml.safe_dump(document, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def build(profile: PackProfile, version: str, output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="semantic-runtime-pack-") as temporary:
        stage = Path(temporary)
        wheels = build_wheels(profile, stage, version)
        lock, dependencies = export_and_build_dependencies(profile, stage)
        catalog, resources, smoke, verification = copy_metadata(profile, version, stage)
        license_files = sorted((stage / "licenses").iterdir())
        manifest = {
            "schema_version": 1,
            "pack_id": profile.pack_id,
            "pack_version": version,
            "profile": profile.profile,
            "runner": profile.runner,
            "python_version": profile.python,
            "endpoint": profile.endpoint,
            "hardware_requirements": profile.hardware_requirements,
            "requirements_lock": file_record(stage, lock),
            "wheels": [file_record(stage, path) for path in wheels],
            "wheelhouse": [file_record(stage, path) for path in dependencies],
            "scene_catalog": file_record(stage, catalog),
            "scene_resources": [file_record(stage, path) for path in resources],
            "licenses": [file_record(stage, path) for path in license_files],
            "verification_files": [file_record(stage, verification)],
            "smoke_scene_key": profile.smoke_scene,
            "smoke_request": file_record(stage, smoke),
            "content_requirements": profile.content_requirements,
        }
        (stage / "runtime-pack.yaml").write_text(
            yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        archive = output / f"semantic-{profile.pack_id}-{version}.runtime.tar.zst"
        top_level = sorted(path.name for path in stage.iterdir())
        run("tar", "--zstd", "-cf", str(archive), "-C", str(stage), *top_level)
        (archive.with_suffix(archive.suffix + ".sha256")).write_text(
            f"{digest(archive)}  {archive.name}\n", encoding="utf-8"
        )
        return archive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(profile_table()), required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "dist-runtime-packs")
    arguments = parser.parse_args()
    if not VERSION_RE.fullmatch(arguments.version):
        parser.error("--version 必须是 SemVer（例如 0.4.0 或 0.4.0-rc.1）")
    archive = build(
        profile_table()[arguments.profile], arguments.version, arguments.output.resolve()
    )
    print(archive)


if __name__ == "__main__":
    main()
