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

"""Profile runner 使用的稳定输入和验收报告。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class ProfileRunRequest:
    profile: str
    environment: str
    seed: int = 0
    steps: int = 10
    output_dir: Path = Path("profile-output")
    camera_names: Sequence[str] = ("agentview", "robot0_eye_in_hand")
    width: int = 256
    height: int = 256
    suite: Optional[str] = None
    task_id: Optional[int] = None
    init_state_id: Optional[int] = None
    perturbation: Optional[str] = None


@dataclass
class EpisodeReport:
    profile: str
    environment: str
    seed: int
    requested_steps: int
    completed_steps: int
    language: Optional[str]
    reward: float
    success: bool
    terminated: bool
    observation_fields: List[Dict[str, Any]]
    evidence_files: Dict[str, str]
    native_metrics: Dict[str, Any]
    started_at: str
    ended_at: str
    failure_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ComparisonReport:
    suite: str
    task_id: int
    init_state_id: int
    perturbation: str
    base: EpisodeReport
    perturbed: EpisodeReport
    differences: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
