#!/bin/bash
# Runs validation before git commit commands.
# Blocks the commit if validation fails.

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command')

# Only intercept git commit commands
if [[ ! "$COMMAND" =~ git[[:space:]]+commit ]]; then
  exit 0
fi

# Run Python checks if src/ or tests/ have staged changes
if git diff --cached --name-only | grep -q "^src/\|^tests/\|^scripts/"; then
  echo "Running Python checks..."
  ruff check src/ tests/ scripts/ 2>&1 || { echo "ruff check failed" >&2; exit 2; }
fi

exit 0
