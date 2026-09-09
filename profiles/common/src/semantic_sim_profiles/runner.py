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

"""所有 benchmark profile 共用的 episode 执行和证据输出。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Protocol, Tuple

from PIL import Image

from semantic_sim_profiles.models import EpisodeReport, ProfileRunRequest


class ProfileAdapter(Protocol):
    @property
    def language(self) -> str: ...

    def reset(self, seed: int) -> Dict[str, Any]: ...

    def neutral_action(self) -> Any: ...

    def base_pose(self) -> Dict[str, Any]: ...

    def gripper_opening(self) -> float: ...

    def visual_model_data(self) -> Tuple[Any, Any]: ...

    def visual_source_for_body(
        self, body_id: int, object_source_ids: Iterable[str]
    ) -> str | None: ...

    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]: ...

    def success(self) -> bool: ...

    def native_metrics(self) -> Dict[str, Any]: ...

    def close(self) -> None: ...


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_episode(adapter: ProfileAdapter, request: ProfileRunRequest) -> EpisodeReport:
    """执行一个确定性 smoke episode；低层 action 只存在于 profile 内部。"""
    request.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = utc_iso()
    completed_steps = 0
    reward = 0.0
    terminated = False
    success = False
    observation: Dict[str, Any] = {}
    failure_reason = None
    try:
        observation = adapter.reset(request.seed)
        _save_observation_evidence(observation, request.output_dir, "initial")
        for _ in range(max(request.steps, 0)):
            observation, reward, terminated, _ = adapter.step(adapter.neutral_action())
            completed_steps += 1
            success = adapter.success()
            if terminated or success:
                break
        evidence = _save_observation_evidence(observation, request.output_dir, "final")
    except Exception as exc:
        evidence = {}
        failure_reason = str(exc)
    ended_at = utc_iso()
    report = EpisodeReport(
        profile=request.profile,
        environment=request.environment,
        seed=request.seed,
        requested_steps=request.steps,
        completed_steps=completed_steps,
        language=adapter.language or None,
        reward=float(reward),
        success=bool(success),
        terminated=bool(terminated),
        observation_fields=summarize_observation(observation),
        evidence_files=evidence,
        native_metrics=adapter.native_metrics(),
        started_at=started_at,
        ended_at=ended_at,
        failure_reason=failure_reason,
    )
    report_path = request.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def summarize_observation(observation: Dict[str, Any]) -> list:
    fields = []
    for key in sorted(observation):
        value = observation[key]
        shape = list(getattr(value, "shape", ()))
        dtype = str(getattr(value, "dtype", type(value).__name__))
        fields.append({"key": str(key), "shape": shape, "dtype": dtype})
    return fields


def _save_observation_evidence(
    observation: Dict[str, Any], output_dir: Path, prefix: str
) -> Dict[str, str]:
    files: Dict[str, str] = {}
    for key, value in _array_observations(observation):
        lowered = key.lower()
        if lowered.endswith("_image") or lowered.endswith("_rgb"):
            path = output_dir / (prefix + "-" + _safe_name(key) + ".png")
            image = _to_rgb_image(value)
            image.save(path, format="PNG")
            files[prefix + ":" + key] = str(path)
        elif lowered.endswith("_depth"):
            path = output_dir / (prefix + "-" + _safe_name(key) + "-depth16.png")
            image = _to_depth_image(value)
            image.save(path, format="PNG")
            files[prefix + ":" + key] = str(path)
    return files


def _array_observations(observation: Dict[str, Any]) -> Iterable[Tuple[str, Any]]:
    for key, value in observation.items():
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            yield str(key), value


def _to_rgb_image(value: Any) -> Image.Image:
    import numpy as np

    array = np.asarray(value)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.shape[-1] > 3:
        array = array[..., :3]
    if array.dtype != np.uint8:
        maximum = float(np.nanmax(array)) if array.size else 0.0
        if maximum <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _to_depth_image(value: Any) -> Image.Image:
    import numpy as np

    array = np.asarray(value, dtype=np.float32).squeeze()
    finite = np.isfinite(array)
    if not finite.any():
        encoded = np.zeros(array.shape, dtype=np.uint16)
    else:
        minimum = float(array[finite].min())
        maximum = float(array[finite].max())
        scale = 65535.0 / max(maximum - minimum, 1e-9)
        encoded = np.zeros(array.shape, dtype=np.uint16)
        encoded[finite] = np.clip((array[finite] - minimum) * scale, 0, 65535).astype(np.uint16)
    return Image.fromarray(encoded, mode="I;16")


def _safe_name(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "_" for character in value
    )
