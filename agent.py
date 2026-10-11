"""
Tiny local AI agent — runs 100% on your PC with Ollama. No tokens, no internet.

Setup (once):
  curl -fsSL https://ollama.com/install.sh | sh
  ollama pull llama3.2
  python3 -m venv ~/agent-env
  source ~/agent-env/bin/activate
  pip install ollama

Run:
  source ~/agent-env/bin/activate
  python agent.py

Try asking: "what time is it?", "what files are in my home folder?",
"is 8.8.8.8 reachable?", "is port 22 open on localhost?". Type "quit" to exit.
"""
import datetime
import os
import socket
import subprocess

import ollama

MODEL = "llama3.2"  # ~2GB, fits fully on a 4GB GPU like the GTX 970


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


def ping_host(host: str) -> str:
    """Check whether a host is reachable on the network by pinging it once.

    Args:
        host: a hostname or IP address, e.g. "8.8.8.8" or "localhost"
    """
    # Only scan machines you own or have permission to test.
    try:
        result = subprocess.run(
            ["ping", "-c", "1", "-W", "2", host],  # 1 packet, 2-second timeout
            capture_output=True, text=True, timeout=5,
        )
        return f"{host} is UP" if result.returncode == 0 else f"{host} is DOWN or not responding"
    except Exception as e:
        return f"Error: {e}"


def check_port(host: str, port: int) -> str:
    """Check whether a single TCP port is open on a host.

    Args:
        host: a hostname or IP address, e.g. "localhost"
        port: the port number, e.g. 22 for SSH or 80 for HTTP
    """
    # Only scan machines you own or have permission to test.
    try:
        with socket.create_connection((host, int(port)), timeout=2):
            return f"Port {port} on {host} is OPEN"
    except (ConnectionRefusedError, socket.timeout, OSError):
        return f"Port {port} on {host} is CLOSED or filtered"
    except Exception as e:
        return f"Error: {e}"


TOOLS = {fn.__name__: fn for fn in (get_time, list_files, read_file, ping_host, check_port)}

# ---- The agent loop: think -> use a tool -> look at result -> repeat ----

messages = [{
    "role": "system",
    "content": "You are a helpful local assistant. When a tool can answer the "
               "question, call it instead of describing it. Keep answers short.",
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
