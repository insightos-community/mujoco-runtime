from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from semantic_sim_profiles import libero


def test_activate_libero_source_writes_isolated_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "LIBERO"
    package = root / "libero" / "libero"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(libero, "_git_commit", lambda _: "installed-revision")

    config_root = tmp_path / "config"
    commit = libero._activate_libero_source(root, config_root)

    assert commit == "installed-revision"
    config = yaml.safe_load((config_root / "config.yaml").read_text(encoding="utf-8"))
    assert config["bddl_files"] == str(package / "bddl_files")
    assert config["init_states"] == str(package / "init_files")
    assert str(root.resolve()) in sys.path
    sys.path.remove(str(root.resolve()))


@pytest.mark.parametrize("revision", ["another-revision", None])
def test_activate_libero_source_accepts_other_revision_or_source_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, revision
) -> None:
    root = tmp_path / "LIBERO"
    package = root / "libero" / "libero"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(libero, "_git_commit", lambda _: revision)

    assert libero._activate_libero_source(root, tmp_path / "config") == revision
    sys.path.remove(str(root.resolve()))


def test_libero_pro_loader_delays_python310_annotations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "LIBERO-PRO"
    source.mkdir()
    (source / "perturbation.py").write_text(
        "class Example:\n    value: str | None = None\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(libero, "_git_commit", lambda _: "another-pro-revision")

    loaded = libero._load_perturbation_module(source)

    assert loaded.Example.__annotations__["value"] == "str | None"
    sys.path.remove(str(source.resolve()))


@pytest.mark.parametrize("revision", ["scene-package-revision", None])
def test_scene_package_reports_its_own_revision(tmp_path, monkeypatch, revision):
    root = tmp_path / "scene"
    for name in ("assets", "bddl_files", "init_files"):
        (root / name).mkdir(parents=True)
    (root / "scene-content.json").write_text(json.dumps({"source_revision": revision}))
    objects = SimpleNamespace(
        **{
            name: SimpleNamespace()
            for name in (
                "articulated_objects",
                "google_scanned_objects",
                "hope_objects",
                "turbosquid_objects",
            )
        }
    )
    domain = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "libero.libero.envs", SimpleNamespace(bddl_base_domain=domain))
    monkeypatch.setitem(sys.modules, "libero.libero.envs.objects", objects)
    assert libero._activate_libero_source(root, tmp_path / "config") == revision
    assert domain.DIR_PATH == str(root / "envs")
    assert all(module.absolute_path == root for module in vars(objects).values())
    assert str(root) not in sys.path


def test_scene_package_still_requires_assets(tmp_path):
    (tmp_path / "scene-content.json").write_text('{"source_revision":"new"}')
    with pytest.raises(FileNotFoundError, match="assets"):
        libero._activate_libero_source(tmp_path, tmp_path / "config")


def test_source_archive_does_not_report_parent_repository_commit(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("没有独立 .git 的源码不能读取父仓库的版本")

    monkeypatch.setattr(libero.subprocess, "check_output", unexpected)
    assert libero._git_commit(tmp_path) is None


def test_git_worktree_revision_is_recorded(tmp_path, monkeypatch):
    (tmp_path / ".git").write_text("gitdir: /unused/worktree")
    monkeypatch.setattr(
        libero.subprocess, "check_output", lambda *args, **kwargs: "worktree-head\n"
    )
    assert libero._git_commit(tmp_path) == "worktree-head"
