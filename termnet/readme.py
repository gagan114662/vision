#!/usr/bin/env python3
import pathlib

README_CONTENT = """# TermNet

TermNet is an **AI-powered terminal assistant** that connects a Large Language Model (LLM) with shell command execution, browser search, and dynamically loaded tools.  
It streams responses in real-time, executes tools one at a time, and maintains conversational memory across steps.

⚠️ **Disclaimer:** This project is experimental. **Use at your own risk.**

---

## ✨ Features

- 🖥️ **Terminal integration**  
  Safely execute shell commands with sandboxed handling, timeout control, and a built-in safety filter that blocks destructive commands (`rm -rf /`, `shutdown`, etc.).

- 🔧 **Dynamic tool loading**  
  Easily extend functionality by editing `toolregistry.json`. No need to modify core files — tools are auto-discovered.

- 🌐 **Browser search**  
  Use Playwright-powered search and page scraping to fetch information dynamically from the web.

- 🧠 **Memory system**  
  Tracks the agent’s planning, actions, observations, reflections, and errors across multiple steps.

- ⚡ **Streaming LLM output**  
  Integrates with [Ollama](https://ollama.ai) for real-time streaming chat responses.

- 🛡️ **Safety layer**  
  Risky commands (e.g., `rm`, `chmod`, `mv`) are not blocked, but the agent provides clear warnings before running them.

---

## 📂 Project Structure

~~~
termnet/
├── agent.py          # Core TermNetAgent: manages chat loop, tool calls, and LLM streaming
├── main.py           # CLI entrypoint for running the agent
├── config.py         # Loads configuration from config.json
├── config.json       # Model and runtime configuration
├── memory.py         # Memory system for reasoning steps
├── safety.py         # Safety checker for shell commands
├── toolloader.py     # Dynamic tool importer based on toolregistry.json
├── toolregistry.json # Registered tools & their schemas
├── browsersearch.py  # Browser search & scraping tool (Playwright + BeautifulSoup)
├── scratchpad.py     # Note-taking / planning scratchpad
├── terminal.py       # Terminal session wrapper with safety & timeout handling
~~~

---

## ⚙️ Installation

### Requirements

- Python **3.9+**
- [Ollama](https://ollama.ai) running locally or accessible via API
- Chromium (installed automatically by Playwright)

### Setup

1. Clone the repository:

   ~~~bash
   git clone https://github.com/RawdogReverend/termnet.git
   cd termnet
   ~~~

2. Install dependencies:

   ~~~bash
   pip install -r requirements.txt
   ~~~

3. Install Playwright browser binaries:

   ~~~bash
   playwright install chromium
   ~~~

---

## 🚀 Usage

You can start TermNet in two ways:

### Directly with Python

~~~bash
python -m termnet.main
~~~

or (depending on your system):

~~~bash
python3 -m termnet.main
~~~

### With the provided script

On Linux/macOS:

~~~bash
./run.sh
~~~

On Windows:

~~~bat
run.bat
~~~

### Chatting

- Type natural language queries.
- TermNet may:
  - Suggest and run safe terminal commands
  - Search the web via `browser_search`
  - Use custom tools defined in `toolregistry.json`
- Exit at any time with:

~~~bash
exit
~~~

---

## ⚙️ Configuration

Configuration is stored in `config.json`.

| Key                | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| `OLLAMA_URL`       | Base URL for the Ollama server (default: `http://127.0.0.1:11434`)          |
| `MODEL_NAME`       | The model name/tag to use (e.g., `gpt-oss:20b`)                             |
| `LLM_TEMPERATURE`  | Randomness in responses (0 = deterministic, 1 = creative)                   |
| `MAX_AI_STEPS`     | Maximum reasoning/tool-execution steps per user query                       |
| `COMMAND_TIMEOUT`  | Max seconds allowed for a terminal command before being killed              |
| `MAX_OUTPUT_LENGTH`| Maximum number of characters per LLM response                               |
| `MEMORY_WINDOW`    | Number of past interactions kept in context                                 |
| `MEMORY_SUMMARY_LIMIT` | Max characters when summarizing memory                                  |
| `CACHE_TTL_SEC`    | Time-to-live for cached tool results                                        |
| `STREAM_CHUNK_DELAY` | Delay between streamed chunks of LLM output                               |

To change models or API endpoints, edit this file and restart TermNet.

---

## 🛠️ Adding Tools

Tools are defined in `toolregistry.json` and implemented in `termnet/tools/`.

### 1. Register the tool
Add a new JSON entry in `toolregistry.json`:

~~~json
{
  "type": "function",
  "function": {
    "name": "my_custom_tool",
    "description": "Describe what this tool does",
    "module": "mytool",
    "class": "MyTool",
    "method": "run",
    "parameters": {
      "type": "object",
      "properties": {
        "arg1": { "type": "string" }
      },
      "required": ["arg1"]
    }
  }
}
~~~

### 2. Implement the tool
Create `termnet/tools/mytool.py`:

~~~python
class MyTool:
    async def run(self, arg1: str):
        return f"Tool executed with arg1={arg1}"
~~~

### 3. Run TermNet
The tool will auto-load at startup:

~~~bash
python -m termnet.main
~~~

✅ No need to edit the agent itself — tools are discovered dynamically.

---

## ⚠️ Safety Notes

- Dangerous commands (like `rm -rf /`) are **blocked**.
- Risky commands (like `rm`, `mv`, `chmod`) are **allowed with warnings**.
- Always review what the agent suggests before execution.

---

## 📜 License

This project is licensed under the MIT License.  
See LICENSE for details.
"""

def main():
    readme_path = pathlib.Path(__file__).parent / "README.md"
    readme_path.write_text(README_CONTENT, encoding="utf-8")
    print(f"✅ README.md has been written to {readme_path.resolve()}")

if __name__ == "__main__":
    main()
