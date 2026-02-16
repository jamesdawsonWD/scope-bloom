"""Mock scope modules if scope is not installed (standalone test environment)."""

import sys
from unittest.mock import MagicMock


def _install_scope_mocks():
    try:
        from scope.core.pipelines.interface import Pipeline  # noqa: F401

        return
    except (ImportError, ModuleNotFoundError):
        pass

    scope = MagicMock()

    # --- scope.core.pipelines.interface ---

    class Pipeline:
        @classmethod
        def get_config_class(cls):
            raise NotImplementedError

    class Requirements:
        def __init__(self, input_size=1):
            self.input_size = input_size

    interface = MagicMock()
    interface.Pipeline = Pipeline
    interface.Requirements = Requirements

    # --- scope.core.pipelines.base_schema ---

    from pydantic import BaseModel

    class BasePipelineConfig(BaseModel):
        model_config = {"arbitrary_types_allowed": True}

    class ModeDefaults:
        def __init__(self, default=False):
            self.default = default

    from enum import Enum

    class UsageType(str, Enum):
        PREPROCESSOR = "preprocessor"
        POSTPROCESSOR = "postprocessor"

    def ui_field_config(**kwargs):
        return kwargs

    base_schema = MagicMock()
    base_schema.BasePipelineConfig = BasePipelineConfig
    base_schema.ModeDefaults = ModeDefaults
    base_schema.UsageType = UsageType
    base_schema.ui_field_config = ui_field_config

    # --- scope.core.plugins.hookspecs ---

    hookspecs = MagicMock()
    hookspecs.hookimpl = lambda fn: fn

    # Register every module path scope imports use
    sys.modules.setdefault("scope", scope)
    sys.modules.setdefault("scope.core", scope.core)
    sys.modules.setdefault("scope.core.plugins", scope.core.plugins)
    sys.modules.setdefault("scope.core.plugins.hookspecs", hookspecs)
    sys.modules.setdefault("scope.core.pipelines", scope.core.pipelines)
    sys.modules.setdefault("scope.core.pipelines.interface", interface)
    sys.modules.setdefault("scope.core.pipelines.base_schema", base_schema)


_install_scope_mocks()
