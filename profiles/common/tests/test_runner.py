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

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from semantic_sim_profiles.models import ProfileRunRequest
from semantic_sim_profiles.runner import run_episode, summarize_observation


class FakeAdapter:
    profile_name = "fake"
    environment_name = "Lift"

    @property
    def language(self) -> str:
        return "fake task"

    def __init__(self) -> None:
        self.steps = 0
        self.closed = False

    def reset(self, seed: int):
        self.steps = 0
        return self._observation()

    def neutral_action(self):
        return np.zeros(7, dtype=np.float32)

    def step(self, action):
        self.steps += 1
        return self._observation(), float(self.steps), self.steps >= 2, {}

    def success(self) -> bool:
        return self.steps >= 2

    def render_evidence(self, observation):
        return {
            "rgb": observation["agentview_image"],
            "depth": observation["agentview_depth"],
        }

    def native_metrics(self):
        return {"steps": self.steps}

    def close(self) -> None:
        self.closed = True

    def _observation(self):
        return {
            "agentview_image": np.full((4, 5, 3), 127, dtype=np.uint8),
            "agentview_depth": np.full((4, 5), 0.5, dtype=np.float32),
            "robot0_proprio-state": np.zeros(9, dtype=np.float32),
        }


def test_profile_runner_writes_report_and_image_evidence(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    request = ProfileRunRequest(
        profile="fake",
        environment="Lift",
        seed=7,
        steps=4,
        output_dir=tmp_path,
    )

    report = run_episode(adapter, request)
    adapter.close()

    assert report.success is True
    assert report.completed_steps == 2
    assert report.seed == 7
    assert adapter.closed is True
    assert (tmp_path / "report.json").is_file()
    assert (tmp_path / "final-agentview_image.png").is_file()
    assert (tmp_path / "final-agentview_depth-depth16.png").is_file()
    saved = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert saved["native_metrics"]["steps"] == 2
    fields = {item["key"]: item for item in saved["observation_fields"]}
    assert fields["agentview_image"]["shape"] == [4, 5, 3]


def test_profile_runner_records_failure_and_still_closes(tmp_path: Path) -> None:
    class BrokenAdapter(FakeAdapter):
        def step(self, action):
            raise RuntimeError("step failed")

    adapter = BrokenAdapter()
    request = ProfileRunRequest(
        profile="fake",
        environment="Lift",
        seed=1,
        steps=1,
        output_dir=tmp_path,
    )

    report = run_episode(adapter, request)
    adapter.close()

    assert report.success is False
    assert "step failed" in (report.failure_reason or "")
    assert adapter.closed is True


def test_summarize_observation_keeps_shape_and_scalar_values() -> None:
    summary = summarize_observation(
        {
            "image": np.zeros((2, 3, 3), dtype=np.uint8),
            "reward": 1.5,
            "done": True,
            "ignored": object(),
        }
    )
    fields = {item["key"]: item for item in summary}
    assert fields["image"] == {
        "key": "image",
        "shape": [2, 3, 3],
        "dtype": "uint8",
    }
    assert fields["reward"]["dtype"] == "float"
    assert fields["done"]["dtype"] == "bool"
