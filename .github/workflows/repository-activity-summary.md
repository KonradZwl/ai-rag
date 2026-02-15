---
description: Generate a summary of repository activity and create an issue with the report
engine: copilot
on:
  workflow_dispatch:
permissions:
  contents: read
  issues: read
  pull-requests: read
tools:
  github:
    toolsets: [default]
safe-outputs:
  create-issue:
    max: 1
    close-older-issues: true
---

# Repository Activity Summary

You are an AI agent tasked with generating a comprehensive activity report for this repository.

## Your Task

Analyze recent repository activity and create a summary issue with the following information:

### 1. Recent Issues
- List recent issues created
- Include issue number, title, and author
- Categorize by type if labels are present (bug, enhancement, question, etc.)
- Note any recently closed issues

### 2. Pull Requests
- List recent pull requests
- Include PR number, title, author, and current status
- Note any PRs that were recently merged
- Highlight PRs awaiting review

### 3. Commits
- Summarize recent commit activity
- Highlight the most active contributors
- Note any significant changes or patterns

### 4. Repository Statistics
- Current number of open issues
- Current number of open pull requests
- Recent stars or forks (if changed)

## Output Format

Create an issue using the `create-issue` safe output with the following structure:

**Title:** `Repository Activity Summary - [Date]`

**Body:**
```markdown
# 📊 Repository Activity Summary - [Date]

## 🆕 Recent Issues
[List recent issues or state "No recent issues"]

## 🔀 Pull Requests
### Recently Opened
[List recent PRs or state "No recent pull requests"]

### Recently Merged
[List recently merged PRs or state "No PRs recently merged"]

### Awaiting Review
[List PRs awaiting review or state "No PRs awaiting review"]

## 💻 Recent Commits
- **Total commits:** [number]
- **Top contributors:** [list top contributors with commit counts]
- **Key changes:** [brief summary of significant commits if any]

## 📈 Repository Stats
- **Open Issues:** [count]
- **Open PRs:** [count]
- **Recent Activity:** [any notable trends]

---
*Generated manually by Repository Activity Summary workflow*
```

## Instructions

1. Use the GitHub toolset to query repository data:
   - Search for recent issues (e.g., last 7-30 days)
   - Search for recent pull requests
   - Get recent commits from the default branch
   - Get current repository statistics

2. Analyze the data and format it according to the template above

3. Use the `create-issue` safe output to post the report as a new issue

4. Use emoji icons to make the report more readable and engaging

5. If there's no activity in a category, clearly state that instead of leaving it empty

6. Keep the report concise but informative - aim for clarity over verbosity

## Error Handling

- If you encounter API errors, mention them in the report
- If no data is available for the time period, create a brief report stating there was minimal activity
- Always create the issue even if there's minimal activity
