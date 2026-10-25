# Antigravity Workspace Configuration & Rules

## GitHub Auto-Sync Policy
- **Automatic GitHub Sync**: Whenever Antigravity completes a task that creates, modifies, or deletes files in this repository, it MUST automatically stage, commit, and push the changes to GitHub.
- **Git Workflow**:
  1. `git add -A`
  2. `git commit -m "<concise descriptive summary of changes>"`
  3. `git push origin main`
- If a push encounters an authentication or network error, report the error clearly to the user.
