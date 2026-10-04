from types import SimpleNamespace

import numpy as np
import pytest

from semantic_sim_profiles.libero import LiberoAdapter
from semantic_sim_profiles.libero_scene import body_sources, scene_metadata


def native_scene():
    model = SimpleNamespace(
        nbody=6, body_parentid=[0, 0, 1, 0, 3, 0],
        body_name2id=lambda name: {"panda": 3}[name],
        ngeom=2, geom_bodyid=[2, 5], geom_type=[6, 7],
        geom_size=np.array([[0.1, 0.2, 0.3], [0, 0, 0]]),
        geom_dataid=[-1, 0], mesh_vertadr=[0], mesh_vertnum=[2],
        mesh_vert=np.array([[-0.1, -0.1, -0.2], [0.1, 0.1, 0.2]]),
        site_name2id=lambda name: 0,
    )
    data = SimpleNamespace(
        body_xpos=np.zeros((6, 3)), body_xmat=np.tile(np.eye(3), (6, 1, 1)),
        body_xquat=np.tile([1, 0, 0, 0], (6, 1)),
        geom_xpos=np.array([[0.2, 0, 0], [0, 0, 0]]),
        geom_xmat=np.tile(np.eye(3), (2, 1, 1)),
        site_xpos=np.array([[1, 2, 3]]), site_xmat=np.array([np.eye(3)]),
    )
    return SimpleNamespace(
        sim=SimpleNamespace(model=model, data=data),
        robots=[SimpleNamespace(robot_model=SimpleNamespace(root_body="panda"))],
        obj_body_id={"bowl": 1, "bowl_large": 5}, fixtures_dict={},
        objects_dict={
            name: SimpleNamespace(category_name="bowl") for name in ("bowl", "bowl_large")
        },
        object_sites_dict={"bowl_inside": SimpleNamespace(
            parent_name="bowl", size=[0.1, 0.2, 0.01], site_type="box",
        )},
    )


def test_visual_sources_use_exact_native_roots_not_name_substrings():
    assert body_sources(native_scene()) == {
        1: "bowl", 2: "bowl", 3: "franka-0", 4: "franka-0", 5: "bowl_large",
    }


def test_metadata_reads_native_pose_geometry_and_site_region():
    env = native_scene()
    result = scene_metadata(env)
    first, second = result["objects"]
    assert first["source_id"] == "bowl"
    assert first["category"] == "bowl"
    assert first["extent"] == pytest.approx([0.6, 0.4, 0.6])
    assert second["extent"] == pytest.approx([0.2, 0.2, 0.4])
    assert first["pose"]["position"] == [0, 0, 0]  # 不为显示居中而改变物体原点。
    region = result["regions"][0]
    assert region["pose"]["position"] == [1, 2, 3]
    assert region["extent"] == [0.2, 0.4, 0.02]
    assert region["properties"]["parent_source_id"] == "bowl"
    env.sim.data.body_xpos[1] = [1, 0, 0]
    env.sim.data.geom_xpos[0] += [1, 0, 0]
    updated = scene_metadata(env)["objects"][0]
    assert updated["pose"]["position"] == [1, 0, 0]
    assert updated["extent"] == pytest.approx(first["extent"])


def test_contacts_are_object_specific_not_global_holding():
    env = SimpleNamespace(
        sim=SimpleNamespace(data=SimpleNamespace(ncon=10)),
        robots=[SimpleNamespace(gripper=SimpleNamespace(important_geoms={
            "left_fingerpad": ["left_pad"], "right_fingerpad": ["right_pad"],
        }))],
        objects_dict={name: SimpleNamespace(contact_geoms=[name]) for name in ("cup", "plate")},
        check_contact=lambda pads, geoms: geoms == ["cup"],
    )
    adapter = object.__new__(LiberoAdapter)
    adapter._env = SimpleNamespace(env=env)
    report = adapter.contact_state()
    assert report["count"] == 10
    assert not report["holding"]
    assert report["contacts"] == [{
        "source_id": "cup", "left_contact": True, "right_contact": True,
        "bilateral_contact": True,
    }]
