from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

import mcp.servers.ally_shell_server as ally_server


class AllyShellServerTests(unittest.TestCase):
    def test_dry_run_returns_metadata(self) -> None:
        result = ally_server.run_command({
            "command": ["echo", "hello"],
            "dry_run": True,
        })
        self.assertEqual(result["executor"], "dry_run")
        self.assertIsNone(result["exit_code"])
        self.assertFalse(result["timed_out"])

    def test_subprocess_execution_invoked(self) -> None:
        completed = subprocess.CompletedProcess(args=["ls"], returncode=0, stdout="ok", stderr="")
        with patch.object(subprocess, "run", return_value=completed) as mock_run:
            result = ally_server.run_command({
                "command": ["ls"],
                "timeout_seconds": 5,
                "workdir": ".",
                "use_ally": False,
            })

        mock_run.assert_called_once()
        self.assertEqual(result["stdout"], "ok")
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["executor"], "subprocess")

    def test_requires_command_tokens(self) -> None:
        with self.assertRaises(ValueError):
            ally_server.run_command({"command": []})

        with self.assertRaises(ValueError):
            ally_server.run_command({"command": [123]})  # type: ignore[arg-type]

    def test_invalid_workdir_rejected(self) -> None:
        outside = Path("/")
        with self.assertRaises(ValueError):
            ally_server.run_command({
                "command": ["pwd"],
                "workdir": str(outside),
            })


if __name__ == "__main__":
    unittest.main()
