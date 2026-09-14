# Copyright 2026 InsightOS
# SPDX-License-Identifier: Apache-2.0
"""Windows stop event shared with the native Semantic supervisor.

A creation-time identity prevents a reused process ID from receiving a stale
stop request. The default process-token DACL restricts endpoint access.
"""
from __future__ import annotations

import ctypes
import os
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from ctypes import wintypes

kernel = ctypes.WinDLL("kernel32", use_last_error=True)
kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel.OpenProcess.restype = wintypes.HANDLE
kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
kernel.GetProcessTimes.restype = wintypes.BOOL
kernel.CloseHandle.argtypes = [wintypes.HANDLE]
kernel.CloseHandle.restype = wintypes.BOOL
kernel.CreateEventW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
kernel.CreateEventW.restype = wintypes.HANDLE
kernel.OpenEventW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
kernel.OpenEventW.restype = wintypes.HANDLE
kernel.SetEvent.argtypes = [wintypes.HANDLE]
kernel.SetEvent.restype = wintypes.BOOL
kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
kernel.WaitForSingleObject.restype = wintypes.DWORD


def identity(pid: int) -> str:
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        times = [wintypes.FILETIME() for _ in range(4)]
        if not kernel.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
            raise ctypes.WinError(ctypes.get_last_error())
        return f"{times[0].dwHighDateTime:08x}{times[0].dwLowDateTime:08x}"
    finally:
        kernel.CloseHandle(handle)


def _name(pid: int, created: str) -> str:
    return f"Local\\InsightOS.Semantic.Stop.{pid}.{created}"


def request_stop(pid: int, created: str) -> None:
    if not created or identity(pid) != created:
        raise ValueError("Process identity changed; refusing stop")
    event = kernel.OpenEventW(0x0002, False, _name(pid, created))
    if not event:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if not kernel.SetEvent(event):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.CloseHandle(event)


@contextmanager
def stop_endpoint(callback: Callable[[], None]) -> Iterator[None]:
    event = kernel.CreateEventW(None, True, False, _name(os.getpid(), identity(os.getpid())))
    error = ctypes.get_last_error()
    if not event:
        raise ctypes.WinError(error)
    if error == 183:  # ERROR_ALREADY_EXISTS: never reuse an unknown endpoint.
        kernel.CloseHandle(event)
        raise FileExistsError("Windows stop endpoint already exists")
    done = threading.Event()

    def wait() -> None:
        while not done.is_set():
            result = kernel.WaitForSingleObject(event, 50)
            if result == 0:
                callback()
                return
            if result != 258:  # WAIT_TIMEOUT
                callback()
                return

    worker = threading.Thread(target=wait, name="semantic-stop", daemon=True)
    worker.start()
    try:
        yield
    finally:
        done.set()
        worker.join()
        kernel.CloseHandle(event)
