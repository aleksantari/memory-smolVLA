# PR Workflow Reference

## Common Commands

### Create a PR

Use the PR template from `.github/PULL_REQUEST_TEMPLATE.md` for the body structure:

```bash
gh pr create --title "Title" --body "body matching the template" --reviewer "username"
```

### List open PRs

```bash
gh pr list
```

### View a PR

```bash
gh pr view <number>
```

### Check PR diff size

```bash
gh pr diff <number> --stat
```

### Merge a PR (merge commit only)

```bash
gh pr merge <number> --merge --delete-branch
```

## Collaborative Feature Branch Flow

```plaintext
main
  └── feature-branch (created by feature owner)
        ├── contributor-1-branch → PR into feature-branch
        └── contributor-2-branch → PR into feature-branch
```

### Create the feature branch

```bash
git checkout main
git pull
git checkout -b <owner>/<type>/issue-<number>-<description>
git push -u origin <owner>/<type>/issue-<number>-<description>
```

### Create a sub-branch

```bash
git checkout <feature-branch>
git pull
git checkout -b <author>/<type>/issue-<number>-<description>
```

### PR for sub-branch (targets feature branch, not main)

```bash
gh pr create --base <feature-branch> --title "Title" --body "Closes #N (partial)"
```

### Final PR to main

```bash
gh pr create --base main --title "Feature title" --body "Closes #N"
```
