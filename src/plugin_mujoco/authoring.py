"""兼容导入；正式场景编译实现位于 plugin_mujoco.compiler。"""

from plugin_mujoco.compiler import (
    AuthoringIssue,
    RuntimeBundle,
    RuntimeBundleResult,
    SceneDocument,
    compile_scene_document,
    validate_scene_document,
)

__all__ = [
    "AuthoringIssue",
    "RuntimeBundle",
    "RuntimeBundleResult",
    "SceneDocument",
    "compile_scene_document",
    "validate_scene_document",
]
