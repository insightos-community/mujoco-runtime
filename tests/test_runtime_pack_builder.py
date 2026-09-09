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

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _builder():
    spec = importlib.util.spec_from_file_location(
        "semantic_runtime_pack_builder", ROOT / "tools" / "build_runtime_pack.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pack_profiles_are_isolated_and_have_offline_catalog_inputs():
    profiles = _builder().profile_table()
    assert set(profiles) == {"native-mujoco", "robosuite-1.5", "libero-robosuite-1.4"}
    assert {item.python for item in profiles.values()} == {"3.10.16", "3.8.20"}
    for profile in profiles.values():
        assert profile.hardware_requirements == {
            "architecture": "amd64",
            "renderer": "egl",
            "gpu": "optional",
        }
        source = ROOT / "runtime-packs" / profile.pack_id
        assert (source / "smoke-request.json").is_file()
        request = json.loads((source / "smoke-request.json").read_text())
        assert request["request_id"] and request["runtime_profile_id"] == profile.pack_id
        catalog = yaml.safe_load((source / "catalog" / "catalog.yaml").read_text())
        assert catalog["schema_version"] == 1 and catalog["entries"]
        for entry in catalog["entries"]:
            assert entry["compatible_runtime_profile"] == profile.pack_id
            if profile.pack_id == "native-mujoco":
                assert entry["scene_id"] == "depalletizing-r1pro"
                assert entry["versions"][0]["runtime_scene_key"] == (
                    "palletizing_depalletizing_tote_v1"
                )
                assert {
                    item["variant_id"] for item in entry["versions"][0]["variants"]
                } == {"layout001", "layout002", "layout003", "layout_smoke"}
            for version in entry["versions"]:
                authoring = version["authoring"]
                if profile.pack_id == "native-mujoco":
                    assert authoring["mode"] == "layout_only"
                    assert authoring["template_ref"]
                    assert authoring["locked_nodes"]
                    for variant in version["variants"]:
                        assert variant["authoring_ref"]
                else:
                    assert authoring == {"mode": "none"}


def test_formal_profile_projects_no_longer_use_editable_common_package():
    for profile in ("robosuite", "libero"):
        pyproject = (ROOT / "profiles" / profile / "pyproject.toml").read_text()
        lock = (ROOT / "profiles" / profile / "uv.lock").read_text()
        assert "editable = true" not in pyproject
        assert 'source = { editable = "../common" }' not in lock
        assert 'source = { directory = "../common" }' in lock


def test_pack_semver_must_match_python_wheel_version():
    builder = _builder()
    assert builder.python_wheel_version("0.4.0") == "0.4.0"
    assert builder.python_wheel_version("0.4.0-dev.0") == "0.4.0.dev0"
    assert builder.python_wheel_version("0.4.0-rc.2") == "0.4.0rc2"
    with pytest.raises(ValueError, match="只支持"):
        builder.python_wheel_version("0.4.0-preview")


def test_pack_catalog_is_stamped_with_exact_pack_version(tmp_path):
    builder = _builder()
    source = ROOT / "runtime-packs" / "native-mujoco" / "catalog" / "catalog.yaml"
    target = tmp_path / "catalog.yaml"
    target.write_text(source.read_text())

    builder.stamp_catalog_version(target, "0.4.0-dev.0")

    catalog = yaml.safe_load(target.read_text())
    assert catalog["catalog_version"] == "0.4.0-dev.0"
    authoring = catalog["entries"][0]["versions"][0]["authoring"]
    assert authoring["asset_catalog_version"] == "0.4.0-dev.0"
