import importlib
import pathlib
import json
import os
from types import ModuleType
from typing import Dict, Any, List, Optional

BASE_DIR = pathlib.Path(__file__).parent


class ToolLoader:
    def __init__(self):
        self.tools: Dict[str, Any] = {}
        self.loaded_tools: Dict[str, Any] = {}  # Expected by tests
        self._tool_registry = self._load_registry()

    def _load_registry(self) -> List[Dict[str, Any]]:
        """Load tool registry from JSON file"""
        registry_path = BASE_DIR / "toolregistry.json"
        if registry_path.exists():
            with open(registry_path, "r") as f:
                return json.load(f)
        return []

    def load_tools(self, tools_dir: str = None) -> None:
        """Load tools from directory by scanning modules"""
        if tools_dir and not os.path.exists(tools_dir):
            print(f"Tools directory not found: {tools_dir}")
            return

        # Load tools from registry
        for tool_def in self._tool_registry:
            if tool_def.get("type") == "function":
                fn = tool_def.get("function", {})
                module_name = fn.get("module")
                class_name = fn.get("class")
                tool_name = fn.get("name")

                if module_name and class_name and tool_name:
                    module = self._load_tool_module(module_name)
                    if module:
                        tool_class = self._find_tool_class_in_module(module, class_name)
                        if tool_class:
                            try:
                                instance = tool_class()
                                self.tools[tool_name] = instance
                                self.loaded_tools[tool_name] = instance
                                print(f"✅ Loaded tool: {tool_name} ({module_name}.{class_name})")
                            except Exception as e:
                                print(f"❌ Failed to instantiate tool {tool_name}: {e}")

    def _load_tool_module(self, name: str) -> Optional[ModuleType]:
        """Load a tool module by name"""
        try:
            return importlib.import_module(f"termnet.tools.{name}")
        except ImportError as e:
            print(f"❌ Failed to import module termnet.tools.{name}: {e}")
            return None

    def _find_tool_class_in_module(self, module: ModuleType, fallback_name: str) -> Optional[type]:
        """Find a tool class in the module"""
        # First try the exact fallback name
        if hasattr(module, fallback_name):
            cls = getattr(module, fallback_name)
            if isinstance(cls, type):
                return cls

        # Then try to find any class ending with "Tool"
        for attr_name in dir(module):
            if attr_name.endswith('Tool'):
                cls = getattr(module, attr_name)
                if isinstance(cls, type):
                    return cls

        return None

    def get_tool_definitions(self, registry_path: str = 'toolregistry.json') -> List[Dict[str, Any]]:
        """Get tool definitions, filtering by enabled=true and type=function"""
        if registry_path != 'toolregistry.json':
            # Load from custom path if specified
            try:
                with open(registry_path, "r") as f:
                    registry = json.load(f)
            except FileNotFoundError:
                return []
        else:
            registry = self._tool_registry

        # Filter for enabled function types only
        filtered = []
        for tool_def in registry:
            # Check if it's a function type
            if tool_def.get("type") == "function":
                # Check if enabled (default to True if not specified)
                enabled = tool_def.get("enabled", True)
                if enabled:
                    filtered.append(tool_def)

        return filtered

    def get_tool_instance(self, tool_name: str) -> Optional[Any]:
        """Get a tool instance by name"""
        return self.loaded_tools.get(tool_name)
