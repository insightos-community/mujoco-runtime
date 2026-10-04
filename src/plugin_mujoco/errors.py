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
