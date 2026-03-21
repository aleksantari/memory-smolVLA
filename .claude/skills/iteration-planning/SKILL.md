---
name: iteration-planning
description: Create and manage iteration plans. Use when starting a new iteration, writing an iteration plan, breaking down requirements into tasks, planning sprint work, or asking about iteration plan format and structure.
argument-hint: "iteration number"
allowed-tools:
  - Read
  - Write
  - Edit
  - Grep
  - Glob
  - "Bash(gh issue *)"
  - "Bash(gh api *)"
---

# Iteration Planning Skill

## Related Skills

- `github-issues` — create the issues from the task breakdown
- `retrospective` — close out the iteration after work is done

## Process

1. Read the template at `docs/iteration-plan-template.md`
2. Ask the user for the iteration number and requirements (or read from a PRD if available)
3. **Draft** `docs/iteration-<number>-plan.md` with the 3 required sections:
   - **Requirements & Acceptance Criteria** — each requirement gets a title, description, and testable acceptance criteria checklist
   - **Coordination & Design Decisions** — architecture choices, API contracts, shared interfaces, responsibilities, dependencies
   - **Task Breakdown** — table of tasks with type, assignee(s), and issue number
4. **Ask the user to review the draft** — the team owns the final plan, not the AI. In particular:
   - Are the acceptance criteria complete? Any edge cases missing?
   - Do the coordination decisions (API contracts, schemas, shared interfaces) match what the team agreed on?
   - Is the task breakdown realistic? Are assignments correct?
5. Incorporate feedback and finalize the plan

## Timeline

- **Iteration plan due:** Tuesday 6 PM of Week 1
- **Iteration work period:** remainder of Week 1 + Week 2
- **Retrospective due:** Monday end-of-day after the iteration ends

## Guidelines

- Each requirement must have clear, testable acceptance criteria
- Break requirements into issues that can be completed by 1-2 people
- Follow issue ownership rules from the `github-issues` skill (not a fixed number of assignees)
- Only create issues for the current iteration — never for future iterations
- Apply the correct label to each issue: `feature`, `task`, or `bug`
- Set the milestone to the current iteration (e.g., "Iteration 1")

## After Planning

Offer to create the GitHub issues from the task breakdown using the `github-issues` skill conventions:

- Each issue gets a descriptive title, description, and acceptance criteria
- Apply the correct label and milestone
- Assign the correct team members
