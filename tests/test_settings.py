from pathlib import Path

from plugin_mujoco.settings import Settings


def test_process_settings_default_to_realtime(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MUJOCO_ASSET_ROOT", str(tmp_path))
    monkeypatch.delenv("PLUGIN_MUJOCO_REALTIME", raising=False)

    assert Settings.from_env().realtime is True

    monkeypatch.setenv("PLUGIN_MUJOCO_REALTIME", "0")
    assert Settings.from_env().realtime is False
