# Local AI Agent (Ollama + Python)

A small AI agent that runs 100% on my own hardware: no API keys, no tokens, no internet after setup. Built to understand how agents actually work before building bigger ones.

**Stack:** Parrot OS · Python 3 · [Ollama](https://ollama.com) · `llama3.2` (3B) · GTX 970 (4GB VRAM)

## What it does

The agent takes a question, decides on its own whether it needs a tool, runs the tool, reads the result, and answers. That plan → act → check → repeat loop is what makes it an agent instead of a chatbot.

Tools it can call (plain Python functions in `agent.py`):

| Tool | What it does |
|------|--------------|
| `get_time()` | Current date and time |
| `list_files(path)` | List a folder |
| `read_file(path)` | Read the first 2000 chars of a text file |

Example:

```
You: what files are in my home folder?
  [using tool] list_files({'path': '~'})
Agent: Your home folder contains Desktop, Documents, Pictures, Videos, and hidden config files like .bash_history and .profile.
```

## How it works

1. The model is sent the conversation plus the tools' docstrings (that's how it knows what each tool is for).
2. If the model replies with a tool call, Python runs that function and sends the result back.
3. Loop until the model answers in plain text, capped at 5 tool steps per question so it can't run forever.

## Setup

```bash
curl -fsSL https://ollama.com/install.sh | sh   # install Ollama
ollama pull llama3.2                              # ~2GB model
python3 -m venv ~/agent-env
source ~/agent-env/bin/activate                   # needed in every new terminal
pip install ollama
python agent.py
```

## Troubleshooting log

**Problem:** First version used `qwen3:4b`. Answers took minutes and CPU sat at 100%.

**Diagnosis:**
- `nvidia-smi` showed Ollama *was* on the GPU, but VRAM was at 3.2 / 4 GB with the desktop using ~500MB.
- `ollama ps` showed `10%/90% CPU/GPU`: the model didn't fully fit, so 10% spilled to the CPU, which bottlenecked everything.
- `qwen3` also "thinks" silently before answering, adding hundreds of hidden tokens per reply.

**Fix:** Switched to `llama3.2` (2GB, no thinking mode). It fits entirely in VRAM and answers in seconds.

**Lesson:** For local AI, VRAM is the limit that matters. A model that 90% fits runs at CPU speed.

**Second problem:** `llama3.2` sometimes *says* it will call a tool instead of calling it. Small models are unreliable about *when* to act; a follow-up nudge usually works, and `qwen2.5:3b` is a stronger tool-caller at the same size.

## Security notes

This agent can read any file the user can. Asked nicely, it will happily read `/etc/passwd`. That's fine for a learning box but is exactly why real agents need sandboxing, allow-lists, and least privilege. Next step is to red-team it and then lock it down.

## Next steps

- [ ] Add a network tool (ping / port check) and a note-saving tool
- [ ] Restrict `read_file` and `list_files` to an allow-listed folder
- [ ] Red-team it (prompt injection via file contents) and document what breaks
- [ ] Try a 12GB GPU and see what 8–14B models change
