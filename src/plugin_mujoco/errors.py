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

"""运行时可预期错误。

API 层只需要把这里的错误转换成稳定的 HTTP 响应，不需要理解 MuJoCo 内部异常。
"""


class RuntimeErrorBase(Exception):
    """所有可返回给 SDK 客户端的运行时错误。"""

    code = "runtime_error"
    status_code = 409

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class NotFoundError(RuntimeErrorBase):
    code = "not_found"
    status_code = 404


class ConflictError(RuntimeErrorBase):
    code = "conflict"
    status_code = 409


class ValidationRuntimeError(RuntimeErrorBase):
    code = "invalid_request"
    status_code = 422


class BackendUnavailableError(RuntimeErrorBase):
    code = "backend_unavailable"
    status_code = 503


class BackendFailureError(RuntimeErrorBase):
    code = "backend_failure"
    status_code = 500
