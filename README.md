```
<div align="center">

# 🤖 Free Claude Code

### Use Claude Code CLI & VSCode for free. No Anthropic API key required.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Python 3.14](https://img.shields.io/badge/python-3.14-3776ab.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json&style=for-the-badge)](https://github.com/astral-sh/uv)
[![Tested with Pytest](https://img.shields.io/badge/testing-Pytest-00c0ff.svg?style=for-the-badge)](https://github.com/Alishahryar1/free-claude-code/actions/workflows/tests.yml)
[![Type checking: Ty](https://img.shields.io/badge/type%20checking-ty-ffcc00.svg?style=for-the-badge)](https://pypi.org/project/ty/)
[![Code style: Ruff](https://img.shields.io/badge/code%20formatting-ruff-f5a623.svg?style=for-the-badge)](https://github.com/astral-sh/ruff)
[![Logging: Loguru](https://img.shields.io/badge/logging-loguru-4ecdc4.svg?style=for-the-badge)](https://github.com/Delgan/loguru)

A lightweight proxy that routes Claude Code's Anthropic API calls to **NVIDIA NIM** (40 req/min free), **OpenRouter** (hundreds of models), **DeepSeek** (direct API), **LM Studio** (fully local), or **llama.cpp** (local with Anthropic endpoints).

[Quick Start](#quick-start) · [Providers](#providers) · [Discord Bot](#discord-bot) · [Configuration](#configuration) · [Development](#development) · [Contributing](#contributing)

---

</div>

> **Personal fork note:** I primarily use this with OpenRouter pointed at `google/gemini-2.5-flash` for day-to-day tasks and NVIDIA NIM for heavier Sonnet-class work. Switched from `gemini-flash-1.5` — 2.5-flash handles tool calls noticeably better in my testing. YMMV with other models.
>
> **Tip:** If you hit OpenRouter rate limits during long sessions, set `OPENROUTER_FALLBACK_MODEL=google/gemini-2.5-flash-lite` as a cheaper fallback — it's fast enough for most file edits and bash commands.

<div align="center">
  <img src="pic.png" alt="Free Claude Code in action" width="700">
  <p><em>Claude Code running via NVIDIA NIM, completely free</em></p>
</div>

## Features

| Feature                    | Description                                                                                     |
| -------------------------- | ----------------------------------------------------------------------------------------------- |
| **Zero Cost**              | 40 req/min free on NVIDIA NIM. Free models on OpenRouter. Fully local with LM Studio            |
| **Drop-in Replacement**    | Set 2 env vars. No modifications to Claude Code CLI or VSCode extension needed                  |
| **5 Providers**            | NVIDIA NIM, OpenRouter, DeepSeek, LM Studio (local), llama.cpp (`llama-server`)                  |
| **Per-Model Mapping**      | Route Opus / Sonnet / Haiku to different models and providers. Mix providers freely             |
| **Thinking Token Support** | Parses `<think>` tags and `reasoning_content` into native Claude thinking blocks                |
| **Heuristic Tool Parser**
```