"""独立引擎安装无需场景，场景内容通过启动请求选择。"""

from pathlib import Path
from unittest.mock import patch

from semantic_sim_profiles.runtime_service import ProfileRuntimeService, _adapter_factory


def test_empty_runtime_is_ready_without_source_or_scenes(monkeypatch):
    monkeypatch.delenv("SEMANTIC_LIBERO_ROOT", raising=False)
    service = ProfileRuntimeService("libero-robosuite-1.4")
    assert service.runtime_info()["state"] == "ready"
    assert service.scenes() == []


def test_scene_request_selects_content_without_source_environment(monkeypatch, tmp_path):
    monkeypatch.delenv("SEMANTIC_LIBERO_ROOT", raising=False)
    monkeypatch.delenv("SEMANTIC_RUNTIME_CONFIG", raising=False)
    with patch("semantic_sim_profiles.runtime_service.LiberoAdapter") as adapter:
        for task in (0, 1):
            factory = _adapter_factory(
                "libero-robosuite-1.4", f"libero_spatial:{task}", "init-0",
                {"scene_content_root": str(tmp_path)},
            )
            factory()
            assert adapter.call_args.kwargs["source_root"] == Path(tmp_path)
            assert adapter.call_args.kwargs["task_id"] == task
