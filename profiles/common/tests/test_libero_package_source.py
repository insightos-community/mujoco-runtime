"""发布来源锁与包内来源记录使用同一版本，运行时代码无需硬编码 commit。"""

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest
import yaml

from semantic_sim_profiles import libero


@pytest.fixture
def builder(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[3] / "tools/libero_packages.py"
    spec = importlib.util.spec_from_file_location("libero_packages_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    (tmp_path / "profiles").mkdir()
    return module


def write_lock(root, revision):
    (root / "profiles/sources.lock.yaml").write_text(
        yaml.safe_dump({"sources": {"libero": {"commit": revision}}})
    )


@pytest.mark.parametrize("revision", ["first", "updated"])
def test_build_reads_updated_source_lock(builder, tmp_path, monkeypatch, revision):
    write_lock(tmp_path, revision)
    monkeypatch.setattr(builder.subprocess, "check_output", lambda *a, **kw: revision + "\n")
    monkeypatch.setattr(builder.subprocess, "run", lambda *a, **kw: None)
    assert builder.verify_source(tmp_path / "source") == revision


def test_build_rejects_source_that_differs_from_lock(builder, tmp_path, monkeypatch):
    write_lock(tmp_path, "expected")
    monkeypatch.setattr(builder.subprocess, "check_output", lambda *a, **kw: "actual\n")
    with pytest.raises(ValueError, match="期望 expected，实际 actual"):
        builder.verify_source(tmp_path / "source")


def test_scene_manifest_records_verified_source(builder, tmp_path, monkeypatch):
    write_lock(tmp_path, "updated")
    source = tmp_path / "source"
    source.mkdir()
    (source / "LICENSE").write_text("test license")
    monkeypatch.setattr(
        builder.subprocess,
        "check_output",
        lambda args, **kw: "updated\n" if "rev-parse" in args else "",
    )
    monkeypatch.setattr(builder.subprocess, "run", lambda *a, **kw: None)
    monkeypatch.setattr(libero, "installed_task_catalog", lambda *a: [])
    destination = builder.build_scenes(source, tmp_path / "scenes.zip", "1.0.0")
    with zipfile.ZipFile(destination) as archive:
        assert (
            json.loads(archive.read("catalog/content/scene-content.json"))["source_revision"]
            == "updated"
        )
        assert (
            yaml.safe_load(archive.read("semantic-component.yaml"))["source_revision"] == "updated"
        )
