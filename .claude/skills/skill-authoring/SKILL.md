---
name: skill-authoring
description: Best practices for writing and auditing Claude Code SKILL.md files. Use when creating a new skill, editing an existing skill, auditing skills, improving skill frontmatter, optimizing skill descriptions for auto-triggering, or asking about skill file structure, frontmatter fields, or allowed-tools configuration.
allowed-tools:
  - Read
  - Write
  - Edit
  - Grep
  - Glob
---

# Skill Authoring Guide

Best practices for writing effective Claude Code skills. A skill is a `SKILL.md` file in `.claude/skills/<name>/` that teaches Claude how to handle specific tasks.

## Quick Reference

| Topic           | File                         | Description                              |
| --------------- | ---------------------------- | ---------------------------------------- |
| **Frontmatter** | [reference.md](reference.md) | All YAML frontmatter fields with details |
| **Tool names**  | [reference.md](reference.md) | Complete allowed-tools list              |

---

## Core Principles

1. **Description is the trigger** — Claude decides whether to load a skill based solely on its `description`. Make it specific with concrete trigger phrases.
2. **Lean SKILL.md, heavy reference files** — Only the description is always in context. Full skill content loads on invocation. Keep SKILL.md focused; move detailed docs to separate files.
3. **Minimize permission friction** — List every tool the skill needs in `allowed-tools` so Claude doesn't pause to ask.
4. **No fluff** — Every line should teach Claude something actionable. Cut marketing language, redundant explanations, and obvious statements.

---

## Writing the Description

The `description` field is the single most important part of a skill. It controls auto-triggering.

**Pattern:** `{What it does}. Use when {trigger phrase 1}, {trigger phrase 2}, ... or asking about {topic 1}, {topic 2}.`

**Good:**

```yaml
description: Component patterns and UI library usage. Use when creating components, using shadcn/ui, implementing theming, working with forms, or asking about component organization.
```

**Bad:**

```yaml
description: Helps with frontend components and UI stuff.
```

Rules:

- Include 3-8 specific trigger phrases that a user would naturally say
- Use action verbs: "creating", "implementing", "debugging", "adding", "configuring"
- Include tool/library names: "shadcn/ui", "Legend-State", "zod"
- Never use second person ("you") or marketing language

---

## Choosing allowed-tools

Only include tools the skill actually needs. Read-only tools (`Read`, `Grep`, `Glob`, `WebSearch`, `WebFetch`) don't require permission anyway, but listing them documents intent.

**The tools that matter most** (these are the ones that prompt for permission):

| Tool    | Include when the skill...                             |
| ------- | ----------------------------------------------------- |
| `Write` | Creates new files                                     |
| `Edit`  | Modifies existing files                               |
| `Bash`  | Runs shell commands (use patterns: `Bash(npm run *)`) |
| `Task`  | Spawns subagents for parallel work                    |

**Common combos:**

- **Read-only skill:** `Read, Grep, Glob`
- **Code-writing skill:** `Read, Write, Edit, Grep, Glob`
- **Full autonomy skill:** `Read, Write, Edit, Grep, Glob, Bash, WebSearch, WebFetch`

---

## SKILL.md Structure

Follow this structure for consistency:

```markdown
---
name: my-skill
description: {What it does}. Use when {triggers}.
allowed-tools:
  - {tools}
---

# {Skill Name} Guide

One-line summary of what this skill covers.

## Related Skills (optional)

Links to related skills for cross-referencing.

## Quick Reference (optional)

Table linking to supporting files.

## {Core Sections}

The actual instructions, patterns, and examples.
Keep to what Claude needs to act. No preambles.

## Checklist (optional)

Steps for common tasks within this skill's domain.

## Detailed Documentation (optional)

Links to reference files for deep content.
```

---

## Content Guidelines

**Do:**

- Use tables for structured reference (locations, conventions, mappings)
- Show code examples from the actual codebase, not hypotheticals
- Use checklists for multi-step procedures
- Link to reference files for anything over ~20 lines of detail

**Don't:**

- Explain why skills exist or how Claude works
- Repeat information available in referenced files
- Add "Notes:", "Important:", or "Remember:" prefixes — just state the instruction
- Include code examples for things Claude already knows (basic JS, HTML, etc.)

---

## Supporting Files

Move heavy content out of SKILL.md into the skill's directory:

```plaintext
my-skill/
├── SKILL.md           # Core instructions (~100-200 lines)
├── patterns.md        # Code patterns and examples
├── reference.md       # Detailed API/config reference
└── conventions.md     # Naming and style rules
```

Claude loads these lazily — only when it follows a link or decides it needs more detail. This keeps the context window lean.

---

## Auditing Checklist

When reviewing an existing skill:

- [ ] **Description** — Contains 3-8 specific trigger phrases? Uses action verbs?
- [ ] **allowed-tools** — Lists every tool the skill needs? No missing write/edit/bash?
- [ ] **Content length** — SKILL.md under ~200 lines? Heavy content in reference files?
- [ ] **No fluff** — Every line is actionable? No marketing, no obvious statements?
- [ ] **Code examples** — From the actual codebase, not generic?
- [ ] **Cross-references** — Links to related skills where relevant?
- [ ] **Frontmatter fields** — Using `argument-hint` if the skill takes args? `disable-model-invocation` for dangerous workflows?

---

## Detailed Documentation

- [reference.md](reference.md) — Complete frontmatter field reference and tool name list
