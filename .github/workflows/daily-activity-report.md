---
description: Generate a daily summary of repository activity and create an issue with the report
on:
  schedule: daily
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

# Daily Repository Activity Report

You are an AI agent tasked with generating a comprehensive daily activity report for this repository.

## Your Task

Analyze recent repository activity from the past 24 hours and create a daily summary issue with the following information:

### 1. New Issues
- List any new issues created in the last 24 hours
- Include issue number, title, and author
- Categorize by type if labels are present (bug, enhancement, question, etc.)

### 2. Pull Requests
- List new pull requests opened in the last 24 hours
- Include PR number, title, author, and current status
- Note any PRs that were merged in the last 24 hours

### 3. Commits
- Summarize commit activity (total number of commits)
- Highlight the most active contributors
- Note any significant changes or patterns

### 4. Repository Statistics
- Current number of open issues
- Current number of open pull requests
- Recent stars or forks (if changed)

## Output Format

Create an issue using the `create-issue` safe output with the following structure:

**Title:** `Daily Activity Report - [Date]`

**Body:**
```markdown
# 📊 Daily Activity Report - [Date]

## 🆕 New Issues
[List new issues or state "No new issues in the last 24 hours"]

## 🔀 Pull Requests
### Opened
[List new PRs or state "No new pull requests"]

### Merged
[List merged PRs or state "No PRs merged"]

## 💻 Commits
- **Total commits:** [number]
- **Top contributors:** [list top 3 contributors with commit counts]
- **Key changes:** [brief summary of significant commits if any]

## 📈 Repository Stats
- **Open Issues:** [count]
- **Open PRs:** [count]
- **Recent Activity:** [any notable trends]

---
*Generated automatically by Daily Activity Report workflow*
```

## Instructions

1. Use the GitHub toolset to query repository data:
   - Search for issues created in the last 24 hours
   - Search for pull requests created or updated in the last 24 hours
   - Get recent commits from the default branch
   - Get current repository statistics

2. Analyze the data and format it according to the template above

3. Use the `create-issue` safe output to post the report as a new issue

4. Use emoji icons to make the report more readable and engaging

5. If there's no activity in a category, clearly state that instead of leaving it empty

6. Keep the report concise but informative - aim for clarity over verbosity

## Error Handling

- If you encounter API errors, mention them in the report
- If no data is available for the time period, create a brief report stating there was no activity
- Always create the issue even if there's minimal activity to maintain the daily cadence
