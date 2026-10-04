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
def test_process_sigterm_and_port_conflict(asset_root):
    port = _free_port()
    environment = {
        **os.environ,
        "MUJOCO_ASSET_ROOT": str(asset_root),
        "PLUGIN_MUJOCO_BACKEND": "fake",
        "PLUGIN_MUJOCO_PORT": str(port),
        "PYTHONPATH": "src",
    }
    process = subprocess.Popen(
        [sys.executable, "-m", "plugin_mujoco.main"],
        cwd=os.getcwd(),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    second = None
    try:
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
        assert "address already in use" in (second.stdout.read() or "").lower()
    finally:
        process.terminate()
        assert process.wait(timeout=5) in {0, -15}
        output = process.stdout.read() if process.stdout else ""
        assert "Application shutdown complete" in output
        if second is not None and second.poll() is None:
            second.terminate()
            second.wait(timeout=5)
