# Opencode CLI Reference

This document serves as a verified reference for using `opencode` within the Ralph Universal environment, based on [official documentation](https://opencode.ai/docs/) and empirical testing in the Edge Lab.

## Critical Findings (Jan 22, 2026)

### 1. Model Refresh
If a paid model (e.g., `zai-coding-plan/glm-4.7`) is missing from the CLI but visible in the UI or valid in the API, you **MUST** refresh the local cache:

```bash
opencode models --refresh
```

### 2. Provider Naming Conventions
The `opencode` CLI uses specific provider prefixes that may differ from marketing names.
- **Wrong:** `zai/glm-4.7`, `opencode/glm-4-plus`
- **Correct:** `zai-coding-plan/glm-4.7`

Always use `opencode models` to list valid IDs.

### 3. Large Prompt Handling (Stability Fix)
When passing large context (like PRD + History) to `opencode run`, using command-line arguments can cause `EBADF` or `Argument list too long` errors.
**Solution:** Pass the prompt via **STDIN**.

```python
# Python Example (Robust)
subprocess.run(
    ["opencode", "run", "--model", "zai-coding-plan/glm-4.7"],
    input=large_prompt_string,  # <-- Pass here
    text=True,
    capture_output=True
)
```

## Essential Commands

### Models
- `opencode models`: List all available models (cached).
- `opencode models --refresh`: Force update of model list from network.
- `opencode models <provider>`: Filter by provider (e.g., `opencode models anthropic`).

### MCP (Model Context Protocol)
- `opencode mcp ls`: List active MCP servers.
- `opencode mcp add`: Wizard to add a new MCP server.
- `opencode mcp auth`: Authenticate with OAuth-enabled MCP servers.

### Run (Headless Mode)
- `opencode run "prompt"`: Simple execution.
- `opencode run --file context.txt`: Attach a file to the prompt.
- `opencode run -`: Read prompt from STDIN (implied if piped).
- `opencode run --attach http://localhost:4096`: Attach to a running `opencode serve` instance (faster, no cold boot).

## Configuration
Config is stored in `~/.config/opencode/config.json` or `.opencode/config.json`.
Reference: [Config Schema](https://opencode.ai/docs/config/).

### Key Options
```json
{
  "disabled_providers": ["openai"],
  "enabled_providers": ["anthropic", "opencode"],
  "instructions": [".cursor/rules/*.md"]
}
```

### Advanced: Rules & Context Management
Opencode supports a powerful "Rules" system to teach agents project specific patterns.

1.  **`AGENTS.md`**: Run `/init` to create this file. It acts as the "Constitution" for the project.
2.  **External References**: You can reference other files in `AGENTS.md` using `@filename`.
3.  **Lazy Loading**: Instruct the agent to read referenced files only on a "need-to-know" basis.

**Best Practice:**
Use `opencode.json` (or `~/.config/opencode/config.json`) to globally define instructions:

```json
{
  "instructions": [
    "docs/standards.md",
    ".cursor/rules/*.md"
  ]
}
```

## Links
- [Official CLI Docs](https://opencode.ai/docs/cli/)
- [Config Documentation](https://opencode.ai/docs/config/)
- [MCP Documentation](https://opencode.ai/docs/mcp/)
- [Rules & Context](https://opencode.ai/docs/rules)
- [Zen (Curated Models)](https://opencode.ai/docs/zen)
