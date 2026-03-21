---
name: feature-spec
description: Write feature specification documents (SPEC.md). Use when creating a new feature spec, documenting feature requirements, defining feature scope and boundaries, or when the user asks to write or create a spec for a feature.
allowed-tools:
  - Read
  - Write
  - Grep
  - Glob
  - AskUserQuestion
---

# Feature Specification Guide

Write SPEC.md documents that define feature intent, scope, behaviors, and dependencies.

---

## When to Use

- Starting a new module (spec-first development)
- Documenting an existing module
- Clarifying module boundaries before implementation
- Creating a source of truth for testing

---

## SPEC.md Template

Create `SPEC.md` in the module's directory (e.g., `src/memory_smolvla/memory/SPEC.md`):

```markdown
# {Module Name}

## Purpose

2-3 sentences explaining WHY this module exists and what problem it solves.
Focus on what it enables for the overall memory-augmented VLA system.

## Scope

**Does:**

- Bullet list of what this module handles
- Be specific and concrete
- Each item should be verifiable

**Does NOT:**

- Explicit boundaries (what this module is NOT responsible for)
- Clarifies ownership between modules
- Prevents scope creep

## Key Behaviors

Plain language descriptions of expected behaviors. Each becomes a test case:

1. When [trigger], [expected result]
2. If [condition], then [outcome]
3. Given [context], when [action], then [result]

Focus on observable behaviors (tensor shapes, gradient flow, memory state).

## Dependencies

List modules and external packages this module relies on:

- `module-name` - what we use from it
- External: `torch.nn.MultiheadAttention` - cross-attention retrieval
- External: `einops` - tensor reshaping

## Known Gaps

Honest accounting of current limitations:

- Things that don't work yet
- Intentional simplifications
- Deferred functionality
- Edge cases not handled

## Files

Brief description of key files:

| File | Purpose |
| ---- | ------- |
| `bank.py` | Episodic memory storage and consolidation |
| `retrieval.py` | Cross-attention memory retrieval |
```

---

## Writing Guidelines

### Purpose Section

- Answer "why does this exist?" not "what does it do?"
- Focus on what it enables for the memory-augmented policy
- Keep to 2-3 sentences maximum

**Good:** "Provides episodic memory storage so the action expert can condition on earlier observations in the episode, enabling multi-step task reasoning."

**Bad:** "Stores tensors in a list and consolidates by cosine similarity."

### Key Behaviors

- Write as testable assertions
- One behavior per line
- Include tensor shape expectations where relevant

**Good:**

1. When write() is called, the tensor is detached before storage (no gradient leakage)
2. If bank exceeds max_size, the two most similar entries are merged
3. Given an empty bank, read_all() returns zero-sized tensors on the correct device

### Dependencies

- Only list direct dependencies
- Note what specifically is used from each dependency
- Include both internal modules and external packages

---

## File Location

```plaintext
src/memory_smolvla/{module}/
├── SPEC.md              # Module specification
├── __init__.py
├── bank.py
├── retrieval.py
└── ...
```

---

## Checklist

Before finalizing a SPEC.md:

- [ ] Purpose explains WHY, not just WHAT
- [ ] Scope "Does" items are concrete and verifiable
- [ ] Scope "Does NOT" clarifies boundaries with other modules
- [ ] Key Behaviors can each become a test case
- [ ] Dependencies are accurate and specific
- [ ] Known Gaps are honest about limitations
- [ ] No implementation details (those belong in code)
- [ ] Another developer could understand module scope from this alone
