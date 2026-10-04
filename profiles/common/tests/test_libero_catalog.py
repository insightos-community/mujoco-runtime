import sys
from pathlib import Path
from types import SimpleNamespace

import semantic_sim_profiles.runtime_service as runtime_module
from semantic_sim_profiles import libero
from semantic_sim_profiles.libero import _benchmark_catalog
from semantic_sim_profiles.runtime_service import ProfileRuntimeService


def test_catalog_preserves_native_task_order_and_available_initial_states(tmp_path):
    first = tmp_path / "z_native_first.bddl"
    second = tmp_path / "a_native_second.bddl"
    first.touch()
    second.touch()

    class Suite:
        def get_num_tasks(self):
            return 2

        def get_task(self, index):
            return SimpleNamespace(name=[first.stem, second.stem][index], language="pick cup")

        def get_task_bddl_file_path(self, index):
            return [first, second][index]

        def get_task_init_states(self, index):
            return [[index]] * (3 + index)

    scenes = _benchmark_catalog({"libero_spatial": Suite, "libero_goal": Suite}, "fixed")
    assert [item["scene_key"] for item in scenes] == [
        "libero_spatial:0", "libero_spatial:1", "libero_goal:0", "libero_goal:1",
    ]
    assert scenes[0]["task"]["name"] == first.stem
    assert scenes[0]["layouts"] == ["init-0", "init-1", "init-2"]
    assert scenes[1]["task"]["initial_state_count"] == 4
    assert all(item["available"] for item in scenes)
    assert all(item["task"]["source_revision"] == "fixed" for item in scenes)


def test_catalog_does_not_offer_missing_native_files(tmp_path):
    class Suite:
        def get_num_tasks(self):
            return 1

        def get_task(self, index):
            return SimpleNamespace(name="missing", language="pick cup")

        def get_task_bddl_file_path(self, index):
            return tmp_path / "missing.bddl"

        def get_task_init_states(self, index):
            raise FileNotFoundError("missing initial states")

    scene = _benchmark_catalog({"libero_spatial": Suite}, "fixed")[0]
    assert not scene["available"]
    assert scene["layouts"] == []
    assert "BDDL" in scene["unavailable_reason"]
    assert "初态" in scene["unavailable_reason"]


def test_runtime_catalog_is_cached_without_starting_physics(tmp_path, monkeypatch):
    monkeypatch.setenv("SEMANTIC_LIBERO_ROOT", str(tmp_path))
    monkeypatch.setenv("SEMANTIC_LIBERO_CONFIG_ROOT", str(tmp_path / "config"))
    calls = []

    def catalog(source: Path, config: Path):
        calls.append((source, config))
        return [{"scene_key": "libero_spatial:0", "layouts": ["init-0", "init-1"]}]

    monkeypatch.setattr(runtime_module, "installed_task_catalog", catalog)
    runtime = ProfileRuntimeService("libero-robosuite-1.4")
    first = runtime.scenes()
    first[0]["layouts"].clear()
    assert runtime.scenes()[0]["layouts"] == ["init-0", "init-1"]
    assert calls == [(tmp_path, tmp_path / "config")]
    assert runtime.runtime_info()["active_instance_id"] is None


def test_runtime_without_scene_data_exposes_an_empty_catalog(monkeypatch):
    monkeypatch.delenv("SEMANTIC_LIBERO_ROOT", raising=False)
    # Runtime 与场景独立安装：仅安装引擎时目录为空，仍可就绪并等待场景包。
    runtime = ProfileRuntimeService("libero-robosuite-1.4")
    assert runtime.scenes() == []
    assert runtime.runtime_info()["active_instance_id"] is None


def test_catalog_uses_declared_upstream_suites_not_auxiliary_registered_classes(monkeypatch):
    class EmptySuite:
        def get_num_tasks(self):
            return 0

    def auxiliary():
        raise AssertionError("上游辅助注册项不应作为可导入套件")

    benchmark = SimpleNamespace(
        libero_suites=["libero_spatial"],
        get_benchmark_dict=lambda: {"libero_spatial": EmptySuite, "libero_100": auxiliary},
    )
    monkeypatch.setitem(sys.modules, "libero.libero", SimpleNamespace(benchmark=benchmark))
    monkeypatch.setattr(libero, "_activate_libero_source", lambda *args: "fixed")
    assert libero.installed_task_catalog(Path("source"), Path("config")) == []


def test_libero_camera_and_controller_are_deployment_configuration(tmp_path, monkeypatch):
    monkeypatch.setenv("SEMANTIC_LIBERO_ROOT", str(tmp_path))
    monkeypatch.setattr(runtime_module, "LiberoAdapter", lambda **kwargs: kwargs)
    for key in ("SEMANTIC_LIBERO_CONTROLLER", "SEMANTIC_LIBERO_CAMERA_WIDTH",
                "SEMANTIC_LIBERO_CAMERA_HEIGHT"):
        monkeypatch.delenv(key, raising=False)
    factory = runtime_module._adapter_factory(
        "libero-robosuite-1.4", "libero_spatial:0", "init-0", {},
    )
    old = factory()
    assert (old["width"], old["height"], old["controller"]) == (640, 480, "JOINT_POSITION")
    monkeypatch.setenv("SEMANTIC_LIBERO_CONTROLLER", "OSC_POSE")
    monkeypatch.setenv("SEMANTIC_LIBERO_CAMERA_WIDTH", "256")
    monkeypatch.setenv("SEMANTIC_LIBERO_CAMERA_HEIGHT", "256")
    vla = factory()
    assert (vla["width"], vla["height"], vla["controller"]) == (256, 256, "OSC_POSE")


def test_libero_uses_installed_runtime_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("SEMANTIC_LIBERO_ROOT", str(tmp_path))
    monkeypatch.setattr(runtime_module, "LiberoAdapter", lambda **kwargs: kwargs)
    for key in ("SEMANTIC_LIBERO_CONTROLLER", "SEMANTIC_LIBERO_CAMERA_WIDTH",
                "SEMANTIC_LIBERO_CAMERA_HEIGHT"):
        monkeypatch.delenv(key, raising=False)
    settings = tmp_path / "runtime-settings.json"
    settings.write_text('{"camera_width":256,"camera_height":256,"controller":"OSC_POSE"}')
    monkeypatch.setenv("SEMANTIC_RUNTIME_CONFIG", str(settings))
    adapter = runtime_module._adapter_factory(
        "libero-robosuite-1.4", "libero_spatial:0", "init-0", {}
    )()
    assert (adapter["width"], adapter["height"], adapter["controller"]) == (256, 256, "OSC_POSE")
