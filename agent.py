"""
Tiny local AI agent — runs 100% on your PC with Ollama. No tokens, no internet.

Setup (once):
  curl -fsSL https://ollama.com/install.sh | sh
  ollama pull qwen3:4b
  python3 -m venv ~/agent-env
  source ~/agent-env/bin/activate
  pip install ollama

Run:
  source ~/agent-env/bin/activate
  python agent.py

Try asking: "what time is it?", "what files are in my home folder?",
"read ~/notes.txt and summarize it". Type "quit" to exit.
"""
import datetime
import os

import ollama

MODEL = "llama3.2"  # small enough for a 4GB GPU like the GTX 970


# ---- Tools: plain Python functions the agent can choose to call ----
# The docstrings matter: the model reads them to decide when to use each tool.

def get_time() -> str:
    """Get the current date and time."""
    return datetime.datetime.now().strftime("%A %B %d, %Y %I:%M %p")


def list_files(path: str = ".") -> str:
    """List the files in a folder.

    Args:
        path: the folder to list, e.g. "~" for the home folder
    """
    try:
        return "\n".join(sorted(os.listdir(os.path.expanduser(path)))) or "(empty)"
    except Exception as e:
        return f"Error: {e}"


def read_file(path: str) -> str:
    """Read a text file (first 2000 characters).

    Args:
        path: the file to read, e.g. "~/notes.txt"
    """
    try:
        with open(os.path.expanduser(path), errors="ignore") as f:
            return f.read(2000)
    except Exception as e:
        return f"Error: {e}"


TOOLS = {fn.__name__: fn for fn in (get_time, list_files, read_file)}

# ---- The agent loop: think -> use a tool -> look at result -> repeat ----

messages = [{
    "role": "system",
    "content": "You are a helpful local assistant. Use your tools when they "
               "help answer the question. Keep answers short.",
}]

print(f"Local agent running on {MODEL}. Type 'quit' to exit.")

while True:
    user = input("\nYou: ").strip()
    if user.lower() in ("quit", "exit"):
        break
    if not user:
        continue
    messages.append({"role": "user", "content": user})

    for _ in range(5):  # max tool steps per question, so it can't loop forever
        response = ollama.chat(model=MODEL, messages=messages, tools=list(TOOLS.values()))
        msg = response.message
        messages.append(msg)

        if not msg.tool_calls:  # no tool needed -> final answer
            print(f"\nAgent: {msg.content}")
            break

        for call in msg.tool_calls:  # run each tool the model asked for
            name, args = call.function.name, call.function.arguments
            print(f"  [using tool] {name}({args})")
            fn = TOOLS.get(name)
            result = fn(**args) if fn else f"Unknown tool: {name}"
            messages.append({"role": "tool", "content": str(result), "tool_name": name})
