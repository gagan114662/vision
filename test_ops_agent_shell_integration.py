"""
Test script to validate Ops agent integration with shell MCP.

This demonstrates how an Ops agent would use the ops.shell.run_command tool
for terminal access with proper provenance logging.
"""
import asyncio
import json
from datetime import datetime
from mcp.servers import ally_shell_server


class MockOpsAgent:
    """Mock Ops agent that uses shell MCP integration."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.command_history = []

    async def execute_shell_command(
        self,
        command: list,
        workdir: str = ".",
        dry_run: bool = False,
        timeout_seconds: int = 30
    ):
        """Execute shell command through MCP with provenance logging."""

        # Log the command request
        request_log = {
            "agent_id": self.agent_id,
            "timestamp": datetime.now().isoformat(),
            "action": "shell_command_request",
            "command": command,
            "workdir": workdir,
            "dry_run": dry_run,
            "timeout_seconds": timeout_seconds
        }

        print(f"🤖 Agent {self.agent_id}: Requesting shell command execution")
        print(f"   Command: {' '.join(command)}")
        print(f"   Workdir: {workdir}")
        print(f"   Dry run: {dry_run}")

        # Execute through shell MCP
        try:
            result = ally_shell_server.run_command({
                "command": command,
                "workdir": workdir,
                "dry_run": dry_run,
                "timeout_seconds": timeout_seconds,
                "use_ally": False  # Not using Ally routing for this test
            })

            # Log the result with provenance
            provenance_log = {
                "agent_id": self.agent_id,
                "timestamp": datetime.now().isoformat(),
                "action": "shell_command_completed",
                "request": request_log,
                "result": {
                    "executor": result["executor"],
                    "exit_code": result["exit_code"],
                    "duration_seconds": result["duration_seconds"],
                    "timed_out": result["timed_out"],
                    "start_time": result["start_time"],
                    "end_time": result["end_time"],
                    "workdir": result["workdir"],
                    "stdout_length": len(result["stdout"]) if result["stdout"] else 0,
                    "stderr_length": len(result["stderr"]) if result["stderr"] else 0
                }
            }

            self.command_history.append(provenance_log)

            print(f"✅ Command executed successfully:")
            print(f"   Executor: {result['executor']}")
            print(f"   Exit code: {result['exit_code']}")
            print(f"   Duration: {result['duration_seconds']:.3f}s")
            print(f"   Timed out: {result['timed_out']}")

            if result["stdout"]:
                print(f"   Stdout: {len(result['stdout'])} characters")
            if result["stderr"]:
                print(f"   Stderr: {len(result['stderr'])} characters")

            return result

        except Exception as e:
            error_log = {
                "agent_id": self.agent_id,
                "timestamp": datetime.now().isoformat(),
                "action": "shell_command_error",
                "request": request_log,
                "error": str(e)
            }

            self.command_history.append(error_log)

            print(f"❌ Command failed: {e}")
            raise

    def get_provenance_report(self):
        """Get complete provenance report for all commands."""
        return {
            "agent_id": self.agent_id,
            "total_commands": len(self.command_history),
            "command_history": self.command_history,
            "report_generated": datetime.now().isoformat()
        }


async def test_ops_agent_workflow():
    """Test complete Ops agent workflow with shell MCP integration."""

    print("🚀 Testing Ops Agent Shell MCP Integration")
    print("=" * 50)

    # Create mock Ops agent
    agent = MockOpsAgent("ops-agent-001")

    # Test 1: Simple directory listing
    print("\n📁 Test 1: Directory Exploration")
    await agent.execute_shell_command(["pwd"])
    await agent.execute_shell_command(["ls", "-la", "mcp/servers/"], timeout_seconds=10)

    # Test 2: Git status check
    print("\n🔍 Test 2: Git Repository Status")
    await agent.execute_shell_command(["git", "status", "--porcelain"])
    await agent.execute_shell_command(["git", "log", "--oneline", "-5"])

    # Test 3: Python environment check
    print("\n🐍 Test 3: Python Environment Validation")
    await agent.execute_shell_command(["python3", "--version"])
    await agent.execute_shell_command(["pip3", "list"], timeout_seconds=15)

    # Test 4: Lean CLI integration
    print("\n⚡ Test 4: Lean CLI Integration")
    await agent.execute_shell_command(["lean", "--version"])

    # Test 5: Dry run mode
    print("\n🧪 Test 5: Dry Run Mode")
    await agent.execute_shell_command(
        ["lean", "backtest", "VisionMonthlyAlpha/"],
        dry_run=True
    )

    # Test 6: File system boundaries (should work within repo)
    print("\n🛡️ Test 6: Workspace Boundaries")
    await agent.execute_shell_command(["find", ".", "-name", "*.py", "-type", "f"], timeout_seconds=10)

    # Generate provenance report
    print("\n📊 Provenance Report")
    print("=" * 30)

    report = agent.get_provenance_report()
    print(f"Agent ID: {report['agent_id']}")
    print(f"Total commands executed: {report['total_commands']}")
    print(f"Report generated: {report['report_generated']}")

    # Show command summary
    successful_commands = [
        cmd for cmd in report['command_history']
        if cmd['action'] == 'shell_command_completed' and cmd['result']['exit_code'] == 0
    ]

    failed_commands = [
        cmd for cmd in report['command_history']
        if cmd['action'] == 'shell_command_completed' and cmd['result']['exit_code'] != 0
    ]

    error_commands = [
        cmd for cmd in report['command_history']
        if cmd['action'] == 'shell_command_error'
    ]

    print(f"\n✅ Successful commands: {len(successful_commands)}")
    print(f"❌ Failed commands: {len(failed_commands)}")
    print(f"💥 Error commands: {len(error_commands)}")

    # Show timing analysis
    if successful_commands:
        durations = [cmd['result']['duration_seconds'] for cmd in successful_commands]
        avg_duration = sum(durations) / len(durations)
        max_duration = max(durations)
        min_duration = min(durations)

        print(f"\n⏱️ Timing Analysis:")
        print(f"   Average duration: {avg_duration:.3f}s")
        print(f"   Max duration: {max_duration:.3f}s")
        print(f"   Min duration: {min_duration:.3f}s")

    # Show executor analysis
    executors = [cmd['result']['executor'] for cmd in successful_commands]
    executor_counts = {}
    for executor in executors:
        executor_counts[executor] = executor_counts.get(executor, 0) + 1

    print(f"\n🔧 Executor Usage:")
    for executor, count in executor_counts.items():
        print(f"   {executor}: {count} commands")

    print("\n" + "=" * 50)
    print("✅ Ops Agent Shell MCP Integration Test Complete!")

    return report


if __name__ == "__main__":
    # Run the test
    report = asyncio.run(test_ops_agent_workflow())

    # Save provenance report
    with open("ops_agent_shell_provenance.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n📋 Full provenance report saved to: ops_agent_shell_provenance.json")