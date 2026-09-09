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

"""隔离 Profile 的 MuJoCo body 名称到公共 source_id 映射。"""

from __future__ import annotations

from typing import Iterable, Optional


def public_source_for_body(
    model: object,
    body_id: int,
    *,
    robot_source_id: str,
    object_source_ids: Iterable[str],
) -> Optional[str]:
    """沿 body 父链匹配公开 Robot/Object，绝不返回引擎 body 名。

    robosuite 与 LIBERO 会给 body 增加实例前缀或后缀。匹配前只保留字母数字，
    并优先选择最长公开 ID，避免 cube 与 cube_a 同时存在时误选。
    """
    candidates = sorted(
        ((source_id, _normalized(source_id)) for source_id in object_source_ids),
        key=lambda item: len(item[1]),
        reverse=True,
    )
    current = int(body_id)
    while current > 0:
        name = _body_name(model, current)
        normalized = _normalized(name)
        if normalized:
            if "robot0" in normalized or "panda" in normalized:
                return robot_source_id
            for source_id, public_name in candidates:
                if public_name and public_name in normalized:
                    return source_id
        parents = getattr(model, "body_parentid", ())
        if current >= len(parents):
            break
        parent = int(parents[current])
        if parent == current:
            break
        current = parent
    return None


def _body_name(model: object, body_id: int) -> str:
    resolver = getattr(model, "body_id2name", None)
    if callable(resolver):
        return str(resolver(body_id) or "")
    names = getattr(model, "body_names", ())
    return str(names[body_id] if body_id < len(names) else "")


def _normalized(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())
