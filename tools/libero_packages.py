"""LIBERO 固定源码分别构建加载器 Wheel 与原生场景数据包。"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def verify_source(source: Path) -> str:
    # 发布构建在唯一的来源锁文件中选择版本，Runtime 运行代码不承担版本锁定。
    # 返回实际校验过的提交，归档和场景元数据必须使用同一值。
    lock = yaml.safe_load((ROOT / "profiles/sources.lock.yaml").read_text(encoding="utf-8"))
    expected = lock["sources"]["libero"]["commit"]
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != expected:
        raise ValueError("LIBERO 源码与 sources.lock.yaml 不匹配，期望 " + expected + "，实际 " + revision)
    subprocess.run(["git", "-C", str(source), "diff", "--exit-code", "HEAD", "--", "libero", "setup.py"], check=True)
    return revision


def build_loader(source: Path, destination: Path) -> Path:
    """源码依赖在构建期固化；部署环境无需 Git 目录或 --libero-root。

    git archive 只读取固定提交中的文件，排除本机训练缓存及未跟踪实验代码。
    数据目录由场景包单独交付，Wheel 保留原始上游 Python 实现和许可证。
    """
    revision = verify_source(source)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="libero-loader-") as directory:
        root = Path(directory)
        archive = root / "upstream.tar"
        subprocess.run(["git", "-C", str(source), "archive", "--format=tar", "--output", str(archive), revision], check=True)
        checkout = root / "source"
        checkout.mkdir()
        subprocess.run(["tar", "-xf", str(archive), "-C", str(checkout)], check=True)
        for name in ("assets", "bddl_files", "init_files"):
            shutil.rmtree(checkout / "libero" / "libero" / name)
        # 上游外层 libero 是 namespace package，原 setup.py 的 find_packages
        # 会生成不含代码的空 Wheel。仅替换构建元数据，保留加载器源码原样。
        (checkout / "setup.py").rename(checkout / "upstream-setup.txt")
        (checkout / "pyproject.toml").write_text(
            '[build-system]\nrequires = ["setuptools==75.3.4", "wheel==0.45.1"]\nbuild-backend = "setuptools.build_meta"\n'
            '[project]\nname = "libero"\nversion = "0.1.0"\nrequires-python = ">=3.8"\nlicense = {file = "LICENSE"}\n'
            '[tool.setuptools.packages.find]\ninclude = ["libero.libero*"]\nnamespaces = true\n', encoding="utf-8"
        )
        subprocess.run(["uv", "build", "--wheel", "--out-dir", str(destination), str(checkout)], check=True)
    wheel = next(destination.glob("libero-0.1.0-*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        if "libero/libero/envs/env_wrapper.py" not in archive.namelist():
            raise ValueError("LIBERO Wheel 缺少原生加载器代码")
        if any("/assets/" in name or "/init_files/" in name or "/bddl_files/" in name for name in archive.namelist()):
            raise ValueError("LIBERO Runtime Wheel 包含了场景数据")
    return wheel


def build_scenes(source: Path, destination: Path, version: str, preview_directory: Path | None = None) -> Path:
    """一个场景数据包可登记多个任务，同一 Runtime 逐次加载任一已选任务。

    此构建命令在 LIBERO 开发环境执行，以原生 benchmark 的任务顺序和初态
    数量生成目录；发布数据仍直接来自 Git 提交，不重新生成或调整初态。
    """
    revision = verify_source(source)
    sys.path.insert(0, str(ROOT / "profiles/common/src"))
    from semantic_sim_profiles.libero import installed_task_catalog

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="libero-scenes-") as directory:
        root = Path(directory)
        tasks = installed_task_catalog(source, root / "config")
        entries = []
        for task in tasks:
            if not task["available"]:
                raise ValueError(task["name"] + ": " + task["unavailable_reason"])
            entries.append({
                "scene_id": task["scene_key"].replace("_", "-").replace(":", "-"),
                "name": task["name"], "description": task["task"]["language"], "tags": ["libero", "franka"],
                "engine": "mujoco", "loader": "libero", "source": "libero",
                "compatible_runtime_profile": "libero-robosuite-1.4",
                "content_root": "content",
                "versions": [{
                    "version": version, "runtime_scene_key": task["scene_key"],
                    "published": True, "robot_models": ["franka_panda"],
                    "variants": [{"variant_id": layout, "name": layout, "kind": "init_state"} for layout in task["layouts"]],
                    "capabilities": ["viewer", "rgb", "depth", "contact", "evaluation"],
                    "authoring": {"mode": "none"},
                    "evaluation": {"provider": "libero", "evaluation_kind": "task_success", "metrics": ["success", "reward"], "supports_comparison": False},
                }],
            })
        manifest = {"schema_version": 1, "kind": "scene_catalog", "name": "libero-scenes", "version": version, "source_revision": revision, "scene_catalog": "catalog/catalog.yaml"}
        previews = {}
        if preview_directory:
            # 只携带与当前上游资产版本一致的预览；未生成的任务仍由安装器补齐。
            by_key = {entry["versions"][0]["runtime_scene_key"]: entry for entry in entries}
            for request_file in preview_directory.glob("*/request.json"):
                request = json.loads(request_file.read_text())
                # 旧版恢复后立即截图，可能仍有悬空物体，不随新包继续传播。
                if request.get("preview_version") != 2:
                    continue
                entry = by_key.get(request.get("scene_key"))
                result_file = request_file.parent / "result.json"
                metadata_file = Path(request.get("content_root", "")) / "scene-content.json"
                if not entry or not result_file.is_file() or not metadata_file.is_file():
                    continue
                if json.loads(metadata_file.read_text()).get("source_revision") != revision:
                    continue
                result = json.loads(result_file.read_text())
                entry["description"] = result.get("description", entry["description"])
                for variant in entry["versions"][0]["variants"]:
                    cached = result.get("variants", {}).get(variant["variant_id"])
                    if not cached:
                        continue
                    image = request_file.parent / cached["preview"]
                    if not image.is_file() or image.parent != request_file.parent:
                        continue
                    relative = "previews/%s/%s.jpg" % (entry["scene_id"], variant["variant_id"])
                    variant.update(preview=relative, description=cached.get("description", ""))
                    entry.setdefault("preview", relative)
                    previews["catalog/content/" + relative] = image
        # 压缩包只含场景数据和清单，不包含 Python 代码、虚拟环境或其他组件包。
        files = subprocess.check_output(["git", "-C", str(source), "ls-tree", "-r", "--name-only", "HEAD", "libero/libero/assets", "libero/libero/bddl_files", "libero/libero/init_files"], text=True).splitlines()
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("semantic-component.yaml", yaml.safe_dump(manifest, sort_keys=False))
            archive.writestr("catalog/catalog.yaml", yaml.safe_dump({"schema_version": 1, "catalog_version": version, "entries": entries}, allow_unicode=True, sort_keys=False))
            archive.writestr("catalog/content/scene-content.json", json.dumps({"source_revision": revision}))
            archive.write(source / "LICENSE", "catalog/content/LICENSE")
            for relative, image in previews.items():
                archive.write(image, relative)
            for relative in files:
                path = source / relative
                if path.is_file():
                    archive.write(path, "catalog/content/" + str(Path(relative).relative_to("libero/libero")))
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="独立构建 LIBERO 原生场景包")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--preview-directory", type=Path, help="可选：携带安装期已生成的预览缓存")
    args = parser.parse_args()
    print(build_scenes(args.source.resolve(), args.output.resolve(), args.version, args.preview_directory))
