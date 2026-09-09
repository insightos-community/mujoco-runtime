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

from __future__ import annotations

import sys
from pathlib import Path

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
    monkeypatch.setattr(libero, "_git_commit", lambda _: libero.LIBERO_COMMIT)

    config_root = tmp_path / "config"
    commit = libero._activate_libero_source(root, config_root)

    assert commit == libero.LIBERO_COMMIT
    config = yaml.safe_load((config_root / "config.yaml").read_text(encoding="utf-8"))
    assert config["bddl_files"] == str(package / "bddl_files")
    assert config["init_states"] == str(package / "init_files")
    assert str(root.resolve()) in sys.path
    sys.path.remove(str(root.resolve()))


def test_activate_libero_source_rejects_wrong_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "LIBERO"
    package = root / "libero" / "libero"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(libero, "_git_commit", lambda _: "wrong")

    with pytest.raises(RuntimeError, match="源码提交不匹配"):
        libero._activate_libero_source(root, tmp_path / "config")


def test_libero_pro_loader_delays_python310_annotations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "LIBERO-PRO"
    source.mkdir()
    (source / "perturbation.py").write_text(
        "class Example:\n    value: str | None = None\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(libero, "_git_commit", lambda _: libero.LIBERO_PRO_COMMIT)

    loaded = libero._load_perturbation_module(source)

    assert loaded.Example.__annotations__["value"] == "str | None"
    sys.path.remove(str(source.resolve()))
