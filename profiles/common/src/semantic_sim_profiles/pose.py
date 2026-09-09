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

"""Profile Runtime 共用的 Robot 基座位姿读取与四元数转换。"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def read_root_body_pose(robot: Any) -> Dict[str, Any]:
    """从 robosuite Robot 的根 body 读取当前世界位姿。

    robosuite 1.5 的 ``robot.base_ori`` 实际保存 3x3 旋转矩阵，而
    LIBERO 使用的 robosuite 1.4 保存 xyzw 四元数。为了避免两个 Profile
    对同一属性作出不同解释，这里只使用两版共同支持的 root body 查询接口。

    本函数会访问 MuJoCo ``sim.data``，因此只能由 Profile worker/物理线程
    调用。HTTP、WebSocket 等 API 线程必须读取 Runtime 已保存的位姿副本。
    公共接口统一使用米、世界坐标系和 ``xyzw`` 四元数顺序。
    """

    root_body = robot.robot_model.root_body
    position = np.asarray(robot.sim.data.get_body_xpos(root_body), dtype=np.float64)
    rotation = np.asarray(robot.sim.data.get_body_xmat(root_body), dtype=np.float64).reshape(3, 3)
    if position.shape != (3,) or not bool(np.isfinite(position).all()):
        raise RuntimeError("Franka 根 body 的世界位置无效")
    if not bool(np.isfinite(rotation).all()):
        raise RuntimeError("Franka 根 body 的世界旋转无效")

    quaternion = mat2quat_xyzw(rotation)
    return {
        "position": [float(value) for value in position],
        "quaternion_xyzw": [float(value) for value in quaternion],
        "frame_id": "world",
    }


def mat2quat_xyzw(rotation: np.ndarray) -> np.ndarray:
    """把 3x3 旋转矩阵转换成归一化的 ``xyzw`` 四元数。

    算法与 robosuite 1.4/1.5 的 ``transform_utils.mat2quat`` 一致，放在
    common 包中是为了避免 common 测试环境反向依赖任一隔离 Profile。
    四元数 ``q`` 与 ``-q`` 表示相同旋转；这里固定 ``w >= 0``，使连续状态和
    测试证据具有确定表示。
    """

    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    m00, m01, m02 = matrix[0]
    m10, m11, m12 = matrix[1]
    m20, m21, m22 = matrix[2]
    symmetric = np.asarray(
        [
            [m00 - m11 - m22, m01 + m10, m02 + m20, m21 - m12],
            [m01 + m10, m11 - m00 - m22, m12 + m21, m02 - m20],
            [m02 + m20, m12 + m21, m22 - m00 - m11, m10 - m01],
            [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
        ],
        dtype=np.float64,
    )
    symmetric /= 3.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    # 上述矩阵的特征向量顺序是 xyzw，因此不再做 MuJoCo wxyz 转换。
    quaternion = eigenvectors[:, int(np.argmax(eigenvalues))]
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12 or not np.isfinite(norm):
        raise RuntimeError("Franka 根 body 的世界四元数无效")
    quaternion = quaternion / norm
    if quaternion[3] < 0:
        quaternion = -quaternion
    return quaternion
