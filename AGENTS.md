# Project workflow

## Finish features before changing scope

- Treat a feature as finished only when its behavior works end to end, its focused verification passes, and its user-visible changes are recorded in `CHANGELOG.md`.
- If the user changes or expands scope while a feature is in progress, remind them that the current feature should be finished, verified, added to the changelog, and committed before starting unrelated work.
- Do not mix unrelated scope into the same feature commit. If the user explicitly abandons the current feature, do not describe or commit it as complete.

## Commits

- Commit after finishing each feature. Include the implementation, generated assets, tests or smoke-check changes, documentation, and changelog entry that belong to that feature.
- Run the focused verification before committing. Never commit a known-broken feature.
- Use a concise imperative commit subject that names the completed feature.
