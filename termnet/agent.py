import sys
import json
import time
import hashlib
import aiohttp
import asyncio
from typing import Dict, List, Tuple

from termnet.config import CONFIG
from termnet.toolloader import ToolLoader


class TermNetAgent:
    def __init__(self, terminal):
        self.terminal = terminal
        self.session_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        self.cache: Dict[str, Tuple[float, str, int, bool]] = {}
        self.current_goal = ""

        # 🔌 Load tools dynamically
        self.tool_loader = ToolLoader()
        self.tool_loader.load_tools()

        # True conversation history (persist across turns)
        self.conversation_history: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": """

You are TermNet with tool access.

RULES:
- Use any too, any time you need to if it will speed up the task.
- Use the tool output to decide your next step.
- Summarize naturally when you have enough info.
-Call tools first, then respond. \\
""",
            }
        ]

    # -----------------------------
    # TOOLS
    # -----------------------------
    def _get_tool_definitions(self):
        tools = self.tool_loader.get_tool_definitions()
        #print("🔧 Registered tools:", [t["function"]["name"] for t in tools])
        return tools

    async def _execute_tool(self, tool_name: str, args: dict, reasoning: str) -> str:
        print(f"\n🛠 Executing tool: {tool_name}")
        #print(f"Reasoning: {reasoning}")
        print(f"Args: {args}")

        tool_instance = self.tool_loader.get_tool_instance(tool_name)
        if not tool_instance:
            obs = f"❌ Tool {tool_name} not found"
            self.conversation_history.append(
                {"role": "tool", "name": tool_name, "content": obs}
            )
            return obs

        try:
            if tool_name == "terminal_execute":
                method = getattr(tool_instance, "execute_command", None)
            else:
                method_name = tool_name.split("_", 1)[-1]
                method = getattr(tool_instance, method_name, None)

            if not method:
                obs = f"❌ Tool {tool_name} has no valid method"
            elif asyncio.iscoroutinefunction(method):
                obs = await method(**args)
            else:
                obs = method(**args)
        except Exception as e:
            obs = f"❌ Tool execution error: {e}"
            print(f"❌ Tool execution error: {e}")

        # ✅ Feed result back into conversation
        self.conversation_history.append(
            {"role": "tool", "name": tool_name, "content": str(obs)}
        )

        return str(obs)

    # -----------------------------
    # LLM
    # -----------------------------
    async def _llm_chat_stream(self, tools: List[Dict]):
        """Stream responses from the LLM (Ollama chat API)."""
        payload = {
            "model": CONFIG["MODEL_NAME"],
            "messages": self.conversation_history,
            "tools": tools,
            "stream": True,
            "options": {"temperature": CONFIG["LLM_TEMPERATURE"]},
        }

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
            async with session.post(f"{CONFIG['OLLAMA_URL']}/api/chat", json=payload) as r:
                async for line in r.content:
                    if not line:
                        continue
                    try:
                        data = json.loads(line.decode())
                        msg = data.get("message", {})
                        if "content" in msg and msg["content"]:
                            yield ("CONTENT", msg["content"])
                        if "tool_calls" in msg and msg["tool_calls"]:
                            yield ("TOOL", msg["tool_calls"])
                    except Exception:
                        continue

    # -----------------------------
    # MAIN LOOP
    # -----------------------------
    async def chat(self, goal: str):
        self.current_goal = goal

        # Append user input (don’t reset history!)
        self.conversation_history.append({"role": "user", "content": goal})

        tools = self._get_tool_definitions()

        for step in range(CONFIG["MAX_AI_STEPS"]):
            collected_text = ""

            async for tag, chunk in self._llm_chat_stream(tools):
                if tag == "TOOL":
                    calls = chunk
                    if calls:
                        fn = calls[0].get("function", {})
                        name = fn.get("name", "")
                        args = fn.get("arguments", {})

                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except Exception:
                                args = {}

                        # 🔑 Execute tool → result goes into conversation
                        await self._execute_tool(name, args, fn.get("reasoning", ""))

                        # Stop inner stream → outer loop restarts with updated history
                        break

                elif tag == "CONTENT":
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
                    collected_text += chunk
                    await asyncio.sleep(CONFIG["STREAM_CHUNK_DELAY"])

            if collected_text.strip():
                # Store assistant answer
                self.conversation_history.append(
                    {"role": "assistant", "content": collected_text.strip()}
                )
                print()
                return

        print("\nReached step limit.")
