from __future__ import annotations

import subprocess
import unittest
from unittest.mock import patch

import mcp.servers.semtools_server as semtools


class SemtoolsServerTests(unittest.TestCase):
    def test_semtools_parse_invokes_cli(self) -> None:
        with patch.object(subprocess, "run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="parsed", stderr=""
            )
            result = semtools.semtools_parse({
                "paths": [
                    __file__,
                ]
            })
        mock_run.assert_called_once()
        self.assertEqual(result["output"], "parsed")

    def test_semtools_parse_structured(self) -> None:
        markdown = "# Title\ncontent line\n## Section\nmore"
        with patch.object(subprocess, "run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout=markdown, stderr=""
            )
            result = semtools.semtools_parse({
                "paths": [__file__],
                "structured_output": True,
            })
        self.assertEqual(len(result["structured"]), 2)
        self.assertEqual(result["structured"][0]["heading"], "Title")

    def test_semtools_search_invokes_cli(self) -> None:
        with patch.object(subprocess, "run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="results", stderr=""
            )
            result = semtools.semtools_search({
                "paths": [__file__],
                "query": "keyword",
                "max_distance": 0.2,
            })
        mock_run.assert_called_once()
        self.assertEqual(result["results"], "results")

    def test_invalid_path_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            semtools.semtools_parse({"paths": ["/tmp/does-not-exist"]})


if __name__ == "__main__":
    unittest.main()
