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
import os
import re
import shutil
import subprocess
import sys
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
            wheel_projects=(ROOT / "profiles" / "common", ROOT / "packages" / "mujoco-visuals"),
            omit_packages=("semantic-sim-profiles", "semantic-mujoco-visuals"),
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
                    "scene_previews": True,
                    "robot_models": ["franka_panda"],
                },
            },
            hardware_requirements={
                "architecture": "amd64",
                "renderer": "egl",
                "gpu": "optional",
            },
            content_requirements={},
            wheel_projects=(ROOT / "profiles" / "common", ROOT / "packages" / "mujoco-visuals"),
            omit_packages=("semantic-sim-profiles", "semantic-mujoco-visuals"),
        ),
    }


def _curl_wheel_dir() -> Path:
    """定位共享的 curl Wheel 下载器（semantic-framework/scripts）。

    各仓库在不同工作区布局下位置不一，按 $SEMANTIC、ROOT 各级父目录依次查找。
    """
    semantic = os.environ.get("SEMANTIC", "").strip()
    candidates = []
    if semantic:
        candidates.append(Path(semantic) / "semantic-framework/scripts")
    for parent in [ROOT, *ROOT.parents]:
        candidates.append(parent / "semantic-framework/scripts")
    for candidate in candidates:
        if (candidate / "curl_wheel.py").is_file():
            return candidate
    raise FileNotFoundError(
        "找不到 semantic-framework/scripts/curl_wheel.py；"
        "请把各仓库放在同一工作区下，或 export SEMANTIC=<工作区根目录>"
    )


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
        "requirements.txt",
        "--output-file",
        str(lock),
    ]
    for package in profile.omit_packages:
        command.extend(("--no-emit-package", package))
    run(*command)
    wheelhouse = stage / "wheelhouse"
    wheelhouse.mkdir()
    command = [
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
    ]
    # 修改目录或 smoke 配置时复用既有锁定 Wheel，构建不重复下载仿真依赖。
    # 种子来源：显式 SEMANTIC_RUNTIME_WHEELHOUSE，或约定共享路径下的 wheelhouse。
    seeds = []
    if seed := os.environ.get("SEMANTIC_RUNTIME_WHEELHOUSE"):
        seeds.append(seed)
    env_wh = os.environ.get("SEMANTIC_WHEELHOUSE")
    for base in ([env_wh] if env_wh is not None else ["/data/wheelhouse"]):
        if not base or not os.path.isdir(base):
            continue
        if any(name.endswith(".whl") for name in os.listdir(base)):
            seeds.append(base)
        for entry in sorted(os.listdir(base)):
            sub = os.path.join(base, entry)
            if os.path.isdir(sub) and any(name.endswith(".whl") for name in os.listdir(sub)):
                seeds.append(sub)
    if seeds:
        command.append("--no-index")
        for seed in seeds:
            command.extend(("--find-links", seed))
        run(*command)
    else:
        # 无缓存时先用 curl 下载器取 PyPI 上的 Wheel：实测 pip 在 GB 级文件上会连接
        # 僵死（317 MB 的 cublas 卡 25 分钟），curl 同文件 28 秒完成；剩余源码包由
        # 下面的 pip 兜底。
        curl_collected = False
        try:
            sys.path.insert(0, str(_curl_wheel_dir()))
            from curl_wheel import collect as curl_collect
            index = os.environ.get("UV_DEFAULT_INDEX") or os.environ.get("PIP_INDEX_URL") \
                or "https://mirrors.aliyun.com/pypi/simple"
            # 进度实时可见（日志重定向时 Python 默认块缓冲会吞掉输出）。
            count = curl_collect(Path(lock), wheelhouse, index, [], python=profile.python,
                                 progress=lambda message: print(message, flush=True))
            curl_collected = count > 0
        except Exception as error:
            print(f"[warn] curl 下载器不可用（{error}），改用 pip 收集", flush=True)
        command.extend(("--find-links", str(wheelhouse)))
        if curl_collected:
            # curl 已取到 PyPI 上的全部 Wheel，剩余（源码包/本地产品 Wheel）也都在
            # find-links 里；加 --no-index 避免 pip 再去访问索引并重新下载大文件。
            command.append("--no-index")
        else:
            # 抗僵死兜底：pip 需联网，仍加超时避免无限等待。
            command.extend(("--timeout", "60", "--retries", "5"))
        run(*command)
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


def build(profile: PackProfile, version: str, output: Path, upstream_source: Path | None = None) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="semantic-runtime-pack-") as temporary:
        stage = Path(temporary)
        wheels = build_wheels(profile, stage, version)
        if profile.runner == "libero-robosuite-1.4":
            from libero_packages import build_loader
            if upstream_source is None:
                raise ValueError("构建 LIBERO Runtime 需要 --upstream-source；安装时无需此目录")
            wheels.append(build_loader(upstream_source.resolve(), stage / "wheels"))
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
        # 新的 LIBERO 引擎包只交付代码与锁定环境；原生任务单独由场景包登记。
        # 其他 Profile 暂时保留原已发布包结构，避免影响现有拆码垛安装。
        if profile.runner == "libero-robosuite-1.4":
            for key in ("scene_catalog", "scene_resources", "smoke_scene_key", "smoke_request"):
                manifest.pop(key)
            shutil.rmtree(stage / "catalog")
            shutil.rmtree(stage / "smoke")
        settings = ROOT / "runtime-packs" / profile.profile["runtime_profile_id"] / "runtime-settings.json"
        if settings.is_file():
            target = stage / "runtime-settings.json"
            shutil.copy2(settings, target)
            manifest["settings"] = file_record(stage, target)
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
    parser.add_argument("--upstream-source", type=Path, help="构建期的固定 LIBERO 源码；不会作为安装依赖")
    parser.add_argument("--output", type=Path, default=ROOT / "dist-runtime-packs")
    arguments = parser.parse_args()
    if not VERSION_RE.fullmatch(arguments.version):
        parser.error("--version 必须是 SemVer（例如 0.4.0 或 0.4.0-rc.1）")
    archive = build(
        profile_table()[arguments.profile], arguments.version, arguments.output.resolve(), arguments.upstream_source
    )
    print(archive)


if __name__ == "__main__":
    main()
