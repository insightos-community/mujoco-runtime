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

import os
import socket
import subprocess
import sys
import time

import httpx
import pytest


def _free_port() -> int:
    with socket.socket() as value:
        value.bind(("127.0.0.1", 0))
        return int(value.getsockname()[1])


def _wait_health(port: int, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=0.2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.05)
    raise AssertionError("Runtime 进程未在限定时间内启动")


@pytest.mark.process
def test_process_sigterm_and_port_conflict(asset_root, tmp_path):
    port = _free_port()
    environment = {
        **os.environ,
        "MUJOCO_ASSET_ROOT": str(asset_root),
        "PLUGIN_MUJOCO_BACKEND": "fake",
        "PLUGIN_MUJOCO_PORT": str(port),
        "PYTHONPATH": "src",
    }
    # Windows venv python.exe is a redirector; its PID is not the interpreter PID.
    # The harness records the actual child identity before invoking the real entry.
    pid_file = tmp_path / "runtime.pid"
    environment["TEST_RUNTIME_PID_FILE"] = str(pid_file)
    entry = (
        "import os, pathlib, runpy; "
        "pathlib.Path(os.environ['TEST_RUNTIME_PID_FILE']).write_text(str(os.getpid())); "
        "runpy.run_module('plugin_mujoco.main', run_name='__main__')"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", entry],
        cwd=os.getcwd(),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    second = None
    try:
        _wait_health(port)
        runtime_pid = int(pid_file.read_text())
        if sys.platform == "win32":
            from plugin_mujoco.windows_stop import request_stop

            with pytest.raises(ValueError, match="identity changed"):
                request_stop(runtime_pid, "0000000000000000")
            _wait_health(port)
        second = subprocess.Popen(
            [sys.executable, "-m", "plugin_mujoco.main"],
            cwd=os.getcwd(),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert second.wait(timeout=5) != 0
        output = (second.stdout.read() or "").lower()
        assert any(message in output for message in ("address already in use", "address in use", "10048"))
    finally:
        if sys.platform == "win32":
            from plugin_mujoco.windows_stop import identity, request_stop

            request_stop(int(pid_file.read_text()), identity(int(pid_file.read_text())))
        else:
            process.terminate()
        assert process.wait(timeout=5) in {0, -15}
        output = process.stdout.read() if process.stdout else ""
        assert "Application shutdown complete" in output
        if second is not None and second.poll() is None:
            second.terminate()
            second.wait(timeout=5)
