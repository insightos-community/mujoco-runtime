"""从 LIBERO 原生对象注册表读取地图，不从观测字段或名称子串猜对象。"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from semantic_sim_profiles.pose import mat2quat_xyzw


def body_sources(env: Any) -> Dict[int, str]:
    """子 body 继承最近的已登记根对象，保证 GLB 节点和地图共享标识。"""
    model = env.sim.model
    roots = {int(body): name for name, body in env.obj_body_id.items()}
    for robot in env.robots:
        roots[int(model.body_name2id(robot.robot_model.root_body))] = "franka-0"
    result = {}
    for body in range(1, model.nbody):
        parent = body
        while parent:
            if parent in roots:
                result[body] = roots[parent]
                break
            parent = int(model.body_parentid[parent])
    return result


def _extent(env: Any, source: str, root: int, sources: Dict[int, str]) -> Optional[list]:
    """按编译后几何计算根坐标系包围尺寸，包含关节子部件。

    物体公共 pose 必须仍是原生 body 原点，不能为了居中显示而改变抓取目标。
    因此尺寸采用关于该原点对称的保守包围盒，并在状态中标明不是碰撞形状。
    mesh 顶点已经由 MuJoCo 编译缩放；不再次乘 XML scale。
    """
    model, data = env.sim.model, env.sim.data
    rotation = np.asarray(data.body_xmat[root]).reshape(3, 3)
    origin = np.asarray(data.body_xpos[root])
    limit = np.zeros(3)
    found = False
    for geom in range(model.ngeom):
        if sources.get(int(model.geom_bodyid[geom])) != source:
            continue
        kind = int(model.geom_type[geom])
        size = np.asarray(model.geom_size[geom])
        world_rotation = np.asarray(data.geom_xmat[geom]).reshape(3, 3)
        local_rotation = rotation.T @ world_rotation
        center = rotation.T @ (data.geom_xpos[geom] - origin)
        if kind == 7:  # mesh：编译后的顶点位于 geom 局部坐标系。
            mesh = int(model.geom_dataid[geom])
            start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
            vertices = np.asarray(model.mesh_vert[start:start + count])
            bounds = np.max(np.abs(vertices @ local_rotation.T + center), axis=0)
        else:
            if kind == 2:  # sphere
                half = np.repeat(size[0], 3)
            elif kind == 3:  # capsule 的 size[1] 不包含两端半球。
                half = np.array([size[0], size[0], size[1] + size[0]])
            elif kind == 5:  # cylinder
                half = np.array([size[0], size[0], size[1]])
            elif kind in (4, 6):  # ellipsoid / box
                half = size
            else:
                continue  # 无限平面及高度场不伪造有限的物品尺寸。
            bounds = np.abs(center) + np.abs(local_rotation) @ half
        limit = np.maximum(limit, bounds)
        found = True
    return (2 * limit).tolist() if found else None


def scene_metadata(env: Any) -> Dict[str, list]:
    """仅由 Runtime worker 调用；区域位姿直接读原生 site，不解释 BDDL。"""
    model, data = env.sim.model, env.sim.data
    sources = body_sources(env)
    objects = []
    for name, obj in {**env.fixtures_dict, **env.objects_dict}.items():
        root = int(env.obj_body_id[name])
        quat = np.asarray(data.body_xquat[root])
        objects.append({
            "source_id": name, "name": name, "category": obj.category_name,
            "pose": {"frame_id": "world", "position": data.body_xpos[root].tolist(),
                     "quaternion_xyzw": quat[[1, 2, 3, 0]].tolist()},
            "extent": _extent(env, name, root, sources),
            "state": {"fixture": name in env.fixtures_dict,
                      "extent_kind": "root_centered_geometry_bounds"},
        })
    regions = []
    for name, site in env.object_sites_dict.items():
        site_id = int(model.site_name2id(name))
        regions.append({
            "source_id": name, "name": name,
            "pose": {"frame_id": "world", "position": data.site_xpos[site_id].tolist(),
                     "quaternion_xyzw": mat2quat_xyzw(data.site_xmat[site_id]).tolist()},
            "extent": (2 * np.asarray(site.size)).tolist(),
            "properties": {"parent_source_id": site.parent_name, "shape": site.site_type},
        })
    return {"objects": objects, "regions": regions}
