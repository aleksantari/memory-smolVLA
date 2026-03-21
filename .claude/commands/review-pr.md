---
agent: code-review
description: Review a pull request for code quality, conventions, and correctness
---

Review pull request $ARGUMENTS.

**Before running your normal review process**, do these PR-specific checks first:

1. Fetch the PR metadata: `gh pr view $ARGUMENTS --json title,body,baseRefName,headRefName,additions,deletions`
2. Fetch the diff: `gh pr diff $ARGUMENTS`
3. Check PR conventions (see `.claude/rules/`):
   - Branch name follows `<author>/<type>/issue-<number>-<short-description>`
   - Commits reference issue numbers
   - PR body includes `Closes #<number>` or `Fixes #<number>`
   - PR is under ~400 changed lines
4. Check for correctness:
   - Tensor shapes are consistent through the memory pipeline
   - Memory bank detaches stored tensors
   - Episode boundaries are respected in data loading
   - Only memory module parameters are trainable
5. Report any convention violations before proceeding.

Then run your full code review process on the PR diff.
