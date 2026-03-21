# Skill Authoring Reference

Complete reference for all SKILL.md frontmatter fields, tool names, and variables.

---

## All Frontmatter Fields

### Core Fields

| Field           | Type   | Default         | Description                                                                                      |
| --------------- | ------ | --------------- | ------------------------------------------------------------------------------------------------ |
| `name`          | string | directory name  | Slash-command name. Lowercase, hyphens, max 64 chars. Must match directory name.                 |
| `description`   | string | first paragraph | What the skill does + trigger phrases. **Primary signal for auto-triggering.**                   |
| `allowed-tools` | array  | none            | Tools Claude can use without permission during skill execution. Array or comma-separated string. |

### Invocation Control

| Field                      | Type    | Default | Description                                                                                   |
| -------------------------- | ------- | ------- | --------------------------------------------------------------------------------------------- |
| `disable-model-invocation` | boolean | `false` | `true` = Claude cannot auto-invoke. Description not loaded into context. Manual `/name` only. |
| `user-invocable`           | boolean | `true`  | `false` = hidden from `/` menu. Claude can still auto-invoke. Use for background knowledge.   |

### Execution Context

| Field     | Type   | Default           | Description                                                                                        |
| --------- | ------ | ----------------- | -------------------------------------------------------------------------------------------------- |
| `model`   | string | session default   | Override model for this skill. Values: `opus`, `sonnet`, `haiku`, or full model ID.                |
| `context` | string | none              | `fork` = run in isolated subagent. Skill content becomes subagent prompt. No conversation history. |
| `agent`   | string | `general-purpose` | Subagent type when `context: fork`. Options: `Explore`, `Plan`, `general-purpose`.                 |

### User Experience

| Field           | Type   | Default | Description                                                        |
| --------------- | ------ | ------- | ------------------------------------------------------------------ |
| `argument-hint` | string | none    | Autocomplete hint for expected arguments. E.g., `"[issue-number]"` |

### Skill-Scoped Hooks

```yaml
hooks:
  PreToolUse:
    - matcher: "Bash"
      hooks:
        - type: command
          command: "./scripts/validate.sh"
  PostToolUse:
    - matcher: "Edit|Write"
      hooks:
        - type: command
          command: "./scripts/lint.sh"
```

Only `PreToolUse`, `PostToolUse`, and `Stop` events are supported in skill frontmatter.

---

## Complete Tool Names for allowed-tools

### Standard Tools

| Tool              | Permission needed | Purpose                        |
| ----------------- | ----------------- | ------------------------------ |
| `Read`            | No                | Read file contents             |
| `Write`           | Yes               | Create or overwrite files      |
| `Edit`            | Yes               | Replace text in existing files |
| `Glob`            | No                | Find files by pattern          |
| `Grep`            | No                | Search file contents           |
| `Bash`            | Yes               | Execute shell commands         |
| `WebFetch`        | No                | Fetch and process web content  |
| `WebSearch`       | No                | Search the web                 |
| `Task`            | Yes               | Spawn subagents                |
| `NotebookEdit`    | Yes               | Edit Jupyter notebooks         |
| `AskUserQuestion` | No                | Request input from user        |

### Bash with Command Filtering

Restrict Bash to specific commands using patterns:

```yaml
allowed-tools:
  - "Bash(npm run *)"
  - "Bash(git commit *)"
  - "Bash(npx shadcn@latest add *)"
```

`*` is a wildcard. `Bash` without a pattern allows all commands.

### MCP Tools

Format: `mcp__<server-name>__<tool-name>`

```yaml
allowed-tools:
  - "mcp__memory__create_entities"
  - "mcp__filesystem__read_file"
```

---

## Variables Available in Skill Content

| Variable               | Description                                          |
| ---------------------- | ---------------------------------------------------- |
| `$ARGUMENTS`           | All arguments passed to the skill                    |
| `$ARGUMENTS[N]`        | Nth argument (0-indexed)                             |
| `$0`, `$1`, `$2`       | Shorthand for `$ARGUMENTS[0]`, `$ARGUMENTS[1]`, etc. |
| `${CLAUDE_SESSION_ID}` | Current session ID                                   |

### Shell Preprocessing

Use `` !`command` `` to run shell commands before Claude sees the content:

```markdown
Current branch: !`git branch --show-current`
Recent changes: !`git log --oneline -5`
```

The output replaces the placeholder before Claude processes the skill.

---

## Frontmatter Template

```yaml
---
name: my-skill
description: What this does. Use when {trigger1}, {trigger2}, {trigger3}, or asking about {topic1}, {topic2}.
allowed-tools:
  - Read
  - Write
  - Edit
  - Grep
  - Glob
# argument-hint: "[optional-args]"
# disable-model-invocation: false
# model: sonnet
# context: fork
# agent: Explore
---
```

---

## Auto-Triggering Tips

Skill descriptions compete for a ~16KB context budget. With many skills:

- Keep descriptions under 2 sentences
- Front-load the most important trigger phrases
- Combine related skills if their descriptions overlap significantly
- Use `disable-model-invocation: true` for skills that should only run manually (deploys, destructive operations)

If a skill isn't auto-triggering reliably, make the description more specific rather than longer.
