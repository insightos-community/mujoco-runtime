import json

import numpy as np
import pytest

from semantic_sim_profiles.scene_preview import describe_difference, prepare


def test_difference_uses_native_object_names_and_actual_state():
    a = {"objects": {"arbitrary_cup": {"position": [0, 0, 0], "rotation": [1, 0, 0, 0]}},
         "joints": {"drawer": {"value": 0, "unit": "m"}}}
    b = {"objects": {"arbitrary_cup": {"position": [.03, 0, 0], "rotation": [0, 0, 0, 1]}},
         "joints": {"drawer": {"value": .1, "unit": "m"}}}
    text = describe_difference(a, b)
    assert "arbitrary_cup" in text and "3.0 cm" in text and "180.0°" in text
    assert "drawer" in text and "0.100 m" in text
    assert "未见" in describe_difference(a, a)


def test_complete_cache_needs_no_environment(tmp_path, monkeypatch):
    (tmp_path / "0.jpg").write_bytes(b"cached-image")
    (tmp_path / "result.json").write_text(json.dumps({"description": "official language",
        "variants": {"init-0": {"preview": "0.jpg", "description": "initial"}}}))
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"output_dir": str(tmp_path), "variants": ["init-0"]}))
    monkeypatch.setattr(
        "semantic_sim_profiles.scene_preview.LiberoAdapter",
        lambda **kw: pytest.fail("cache must not load engine"),
    )
    prepare(request)
    assert (tmp_path / "0.jpg").read_bytes() == b"cached-image"


def test_partial_cache_and_arbitrary_suite(tmp_path, monkeypatch):
    calls = []
    class Fake:
        def __init__(self, **kw):
            assert kw["suite_name"] == "libero_goal" and kw["task_id"] == 7
            self._task = type("Task", (), {"language": "open any drawer"})()
            self._env = type("Wrapper", (), {"env": type("Env", (), {
                "parsed_problem": {
                    "obj_of_interest": ["drawer"],
                    "goal_state": [["Open", "drawer"]],
                }})()})()
        def reset(self, seed): assert seed == 0
        def reset_initial_state(self, index):
            calls.append(index)
            return {"agentview_image": np.zeros((3, 3, 3), dtype=np.uint8)}
        def step(self, action):
            assert np.array_equal(action, np.zeros(7))
            calls.append("settle")
            return {"agentview_image": np.ones((3, 3, 3), dtype=np.uint8)}, 0, False, {}
        def close(self): calls.append("close")
    monkeypatch.setattr("semantic_sim_profiles.scene_preview.LiberoAdapter", Fake)
    monkeypatch.setattr(
        "semantic_sim_profiles.scene_preview.state_summary",
        lambda env: {"objects": {}, "joints": {}},
    )
    (tmp_path / "0.jpg").write_bytes(b"existing")
    (tmp_path / "result.json").write_text(
        json.dumps({"variants": {"init-0": {"preview": "0.jpg"}}})
    )
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"output_dir": str(tmp_path), "content_root": str(tmp_path),
        "scene_key": "libero_goal:7", "variants": ["init-0", "init-9"]}))
    prepare(request)
    result = json.loads((tmp_path / "result.json").read_text())
    assert calls == [0] + ["settle"] * 5 + [9] + ["settle"] * 5 + ["close"]
    assert "open any drawer" in result["description"]
    assert set(result["variants"]) == {"init-0", "init-9"}
    assert (tmp_path / "0.jpg").read_bytes() == b"existing"
