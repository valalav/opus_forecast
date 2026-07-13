# Antigravity Skills Guide

**Skills** are modular capabilities that extend Ralph's abilities beyond basic coding. They follow the [official Google Antigravity specification](https://antigravity.google/docs/skills).

## 1. What is a Skill?

A Skill is a directory containing a definition file (`SKILL.md`) and optional assets (scripts, templates, reference docs). Ideally, it encapsulates a specific workflow (e.g., "Web Search", "Database Migration", "Git Formatting").

## 2. Directory Structure

All skills must be located in `edge_lab/skills/`.

```
edge_lab/skills/
├── my-skill/
│   ├── SKILL.md       # The Brain: Instructions & Metadata
│   ├── scripts/       # The Hands: Python/Bash scripts
│   └── references/    # The Knowledge: Templates/Docs
```

## 3. Creating a Skill

### Step 1: Create the Directory
```bash
mkdir -p edge_lab/skills/web_search
```

### Step 2: Create SKILL.md
The `SKILL.md` file MUST have YAML frontmatter and a Markdown body.

```markdown
---
name: web_search
description: Use this skill when the user asks to search the web, find external documentation, or look up recent data.
---

# Web Search Skill

**Goal**: Retrieve information from the internet using `googlesearch-python` or `curl`.

## Capabilities
1. **Google Search**: Find URLs for a query.
2. **Page Fetch**: Download content from a specific URL.

## Instructions
1. Use `scripts/search.py` to perform the search.
2. If that fails, try `curl`.

## Constraints
- Do not download binary files.
- Summarize long pages.
```

### Step 3: Add Scripts (Optional)
If your skill needs to run code, put it in `scripts/`.
*Example: `edge_lab/skills/web_search/scripts/search.py`*

## 4. Using Skills

The Worker Agent has been instructed to check `edge_lab/skills/` for capabilities.

**To use a skill in a Task:**
Simply mention it in the Task Description.
> "Use the Web Search skill to find the latest inflation data."

The Worker will:
1. Read `edge_lab/skills/web_search/SKILL.md`.
2. Follow the instructions defined there.
3. Execute the scripts as needed.

## 5. Verification
The Critic will verify that the skill was used correctly by checking:
- Did the Worker follow the "Verification" steps in the skill file?
- Did the scripts run successfully?
