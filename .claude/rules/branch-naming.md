# Branch Naming Convention

When creating branches, ALWAYS follow this pattern:

`<author>/<type>/issue-<number>-<short-description>`

Where:

- `author` is the GitHub username of the branch creator
- `type` matches the issue label: `feature`, `bug`, or `task`
- `number` is the GitHub issue number
- `short-description` is 2-4 lowercase words separated by hyphens

Examples:

- `asmith/feature/issue-5-user-authentication`
- `jdoe/bug/issue-12-fix-login-redirect`
- `mchen/task/issue-3-setup-ci-pipeline`

Never push directly to master. Always create a branch and open a PR.
