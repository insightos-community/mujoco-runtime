from __future__ import annotations

import pytest

from plugin_mujoco.compiler.rules import is_relative_asset_key


@pytest.mark.parametrize("value", ["/etc/passwd", r"C:\Windows\file", "C:relative", r"\\server\share\file", r"\rooted", "../escape", r"asset\..\escape", ""])
def test_asset_identifier_rejects_host_paths_on_every_platform(value):
    assert not is_relative_asset_key(value)


@pytest.mark.parametrize("value", ["robots/r1/model.xml", "objects/周转箱/model.xml"])
def test_asset_identifier_accepts_repository_paths(value):
    assert is_relative_asset_key(value)
