# Agent 34: Power Management Agent

Part of the Agentic Shell project. Turns requests such as
"Put my laptop into maximum battery saving mode" into controlled changes to the CPU power
policy and the screen backlight.

It runs on three system backends, chosen automatically (`POWER_AGENT_BACKEND=auto|linux|windows|demo`):

| Backend | Used on | How it changes settings |
|---|---|---|
| `linux` | Linux | writes `/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor` and `/sys/class/backlight/*/brightness` |
| `windows` | Windows 10/11 | `powercfg` processor min/max state of the active power plan, WMI for brightness; no admin rights needed |
| `demo` | any OS (`--demo`) | a simulated Linux laptop in a temp folder; the real sysfs code runs against it, so it is safe for presentations |

On Windows the Linux governors map to processor-state limits:
`powersave` = 5-50 %, `conservative` = 5-70 %, `schedutil` = 5-100 % (Windows default behaviour),
`performance` = 100-100 %. Per-CPU governors are a Linux-only feature.

It runs entirely on **free, open-weight models**. No paid provider is needed.

## Which model, and why

| Order | Backend | Model | Free-tier limits | Notes |
|---|---|---|---|---|
| 1 | Groq | `openai/gpt-oss-120b` (Apache 2.0) | 30 RPM, 1,000 RPD, 8K TPM, 200K TPD | tool use supported, no parallel calls |
| 2 | Groq | `qwen/qwen3.8-27b` | same numbers, **separate bucket** | tool use and parallel tool calls |
| 3 | Groq | `openai/gpt-oss-20b` | same numbers, separate bucket | smaller and faster |
| 4 | OpenRouter | `openai/gpt-oss-120b:free` | 20 RPM, 50 RPD (1,000 RPD after a one-time $10 top-up) | optional backup |
| 5 | Ollama (local) | `qwen3:8b` by default | none | optional, offline, needs your own RAM |
| last | built-in offline parser | none | none | always available |

Groq limits are per model and per organisation, so the three Groq entries give three
separate token budgets. **Tokens per minute (8K) is the limit that binds**, not requests.
`python -m power_agent.budget` prints the planning estimate: about 860 tokens per
request, so roughly 8 requests a minute and 200 a day per Groq model. Common
requests need only one model call (economy mode, see below).
Exact limits for your account are on https://console.groq.com/settings/limits.

## Quick start on Windows

```bat
setup_windows.bat     :: installs dependencies (once)
run_demo.bat          :: interactive agent on a SIMULATED laptop (safe to present)
run_agent.bat         :: interactive agent on THIS laptop (really changes CPU limits / brightness)
run_tests.bat         :: 65 tests, no hardware or network needed
```

Try: `Put my laptop into maximum battery saving mode`, `Dim the screen a bit`,
`What is my current status?`, `Restore my settings`.
Optional: copy `.env.example` to `.env` and add a free Groq key to use the LLM instead of the offline parser.

## Setup (free, any OS)

```bash
pip install -r requirements.txt
export GROQ_API_KEY=gsk_...        # (Windows cmd: set GROQ_API_KEY=gsk_...) free key from https://console.groq.com/keys, no card needed
python live_check.py               # one live check per configured backend
sudo -E python -m power_agent      # interactive; root needed for sysfs writes
```

Optional extras: `export OPENROUTER_API_KEY=...` and/or `export OLLAMA_MODEL=qwen3:8b`
(with `ollama serve` running). `POWER_AGENT_CHAIN=groq-gpt-oss-20b,ollama-local` sets
an explicit order.

## How the free tiers are handled

* **Economy mode (default):** after state-changing tool calls the agent writes the reply
  itself from the tool results, so a typical command costs one model request, not two.
  Set `POWER_AGENT_SUMMARY=llm` to let the model word the reply.
* **Budget tracker:** a sliding window per backend keeps requests under the published
  RPM/TPM/RPD/TPD, waiting up to 8 s before moving on.
* **429 handling:** HTTP 429 moves the request to the next model and pauses the rate-limited one
  for the `retry-after` period.
* **Provider quirks:** `tool_use_failed` (the provider could not parse the model's tool call)
  is retried once; an unsupported optional parameter is dropped and retried; bad keys disable
  that backend for the session.
* **Static prompt kept small and first** (about 700 tokens) so provider-side prompt caching
  can apply where it is offered.
* If everything is exhausted, the offline parser answers the request.

## Layout

| File | Role |
|---|---|
| `power_agent/agent.py` | `PowerManagementAgent`: tool loop, confirmation handling, economy mode |
| `power_agent/llm.py` | Free-model client (urllib), budget tracker, fallback chain, Ollama adapter |
| `power_agent/tools.py` | The seven tools and their JSON schemas |
| `power_agent/guardrails.py` | Validation rules (governor allow-list, brightness floor, confirmations) |
| `power_agent/sysfs.py` | sysfs reads/writes, `sudo -n tee` escalation, configurable root |
| `power_agent/win_tools.py` | Windows backend (`powercfg` + WMI), same tool interface |
| `power_agent/backend.py` | picks linux / windows / demo backend |
| `power_agent/demo.py` | builds the simulated laptop used by `--demo` |
| `power_agent/local_router.py` | Offline intent parser |
| `power_agent/mcp_server.py` | The same tools exposed as an MCP server |
| `power_agent/budget.py` | Free-tier planning estimate |
| `live_check.py` | Live check of your keys and the chosen models |
| `tests/` | 65 tests: simulated sysfs, plus a mock server for the LLM layer |
| `evaluate.py` | Evaluation harness (`--llm` uses the free-model chain) |

## Plugging into the shell

```python
from power_agent import PowerManagementAgent
agent = PowerManagementAgent()          # builds the chain from the environment
print(agent.handle("Dim the screen a bit"))
```

The router only needs `power_agent.agent.AGENT_ID` (34), `AGENT_NAME` and `ROUTING_HINT`.

## Other commands

```bash
python -m power_agent "Switch to powersave"   # single command
python -m power_agent.mcp_server              # MCP server over stdio
pytest                                        # no hardware, no network, no key needed
python evaluate.py                            # offline router
python evaluate.py --llm --pace 10            # free-model chain, 10 s between prompts
```

Without root the agent tries `sudo -n tee` for governors and the logind
`SetBrightness` call for the backlight. `POWER_AGENT_SYSROOT` points the agent at a
directory shaped like `/` for testing.

## Environment variables

`GROQ_API_KEY`, `GROQ_MODEL`, `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, `OLLAMA_HOST`,
`OLLAMA_MODEL`, `POWER_AGENT_CHAIN`, `POWER_AGENT_SUMMARY`, `POWER_AGENT_SYSROOT`,
`POWER_AGENT_STATE`, `POWER_AGENT_BACKEND`.
