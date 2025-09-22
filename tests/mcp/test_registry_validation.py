import unittest

from mcp.registry import MCPRegistry


class TestMCPRegistryValidation(unittest.TestCase):
    def test_registry_entries_have_implementations(self) -> None:
        registry = MCPRegistry()
        issues = registry.validate()
        self.assertFalse(
            issues,
            msg="\n".join(f"{issue.kind}:{issue.tool_id}:{issue.detail}" for issue in issues),
        )


if __name__ == "__main__":
    unittest.main()
