---
description: Scan repository for changes and update project documentation
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
  bash: true
  grep: true
  view: true
  glob: true
safe-outputs:
  create-issue:
    title-prefix: "[docs] "
    labels: [documentation, automated]
---

# Documentation Update Workflow

You are an AI agent tasked with scanning this repository for changes and updating or creating comprehensive project documentation.

## Your Task

Analyze the current state of the repository and create or update documentation to accurately reflect the project's purpose, features, and usage.

### 1. Repository Analysis

Scan the repository to understand:
- Project structure and file organization
- Programming languages and frameworks used
- Key features and functionality
- Dependencies and requirements
- Configuration files and environment variables
- Existing documentation (README.md, etc.)

### 2. Generate or Update Documentation

Create a comprehensive `documentation.md` file (or update if it exists) that includes:

#### Project Overview
- What the project does
- Key features and capabilities
- Technology stack and architecture

#### Getting Started
- Prerequisites and dependencies
- Installation instructions
- Initial setup and configuration
- Environment variables and configuration files

#### Usage Guide
- How to run the application
- Main workflows and use cases
- Examples of common operations
- Command-line interfaces or APIs

#### Project Structure
- Directory and file organization
- Purpose of key directories
- Important files and their roles

#### Development
- How to set up a development environment
- Running tests (if applicable)
- Code structure and patterns
- Contributing guidelines (if applicable)

#### Dependencies
- List of main dependencies
- What each dependency is used for
- Version requirements

#### Configuration
- Configuration options available
- How to configure the application
- Example configurations

#### Troubleshooting
- Common issues and their solutions
- Where to get help

### 3. Documentation Best Practices

- Use clear, concise language
- Include code examples where appropriate
- Use proper markdown formatting
- Organize information logically with headers
- Keep documentation up-to-date with the current codebase
- Reference actual file paths and code from the repository

### 4. Create Issue

After creating or updating the `documentation.md` file, use the `create-issue` safe output to submit your documentation changes.

The safe output expects a JSON object with:
- `title`: Issue title (will be automatically prefixed with "[docs] ")
- `body`: Issue description

**Important**: You must create or update the `documentation.md` file BEFORE calling the create-issue safe output. The safe output will detect the file changes and reference them in the issue.

Example workflow:
1. First, create/update the documentation.md file using the `edit` or `create` tool
2. Then, use the create-issue safe output to submit the issue

Example JSON format for the safe output:
```json
{
  "type": "create-issue",
  "title": "Update project documentation",
  "body": "This issue tracks updates to the project documentation based on the current repository state.\n\n## Changes Made\n\n- Created comprehensive documentation.md\n- Documented project structure and features\n- Added setup and usage instructions\n\n---\n*Generated automatically by Documentation Update workflow*"
}
```

## Instructions

1. Use the available tools to explore the repository:
   - Use `glob` to find files by pattern (e.g., "**/*.py", "**/*.js")
   - Use `grep` to search file contents for specific patterns
   - Use `view` to read file contents and understand code
   - Use `bash` to run commands like `ls`, `cat`, etc.

2. Analyze the existing README.md and any other documentation

3. Review the codebase structure to understand the project

4. Create or update the documentation.md file with comprehensive information

5. Use the `create-issue` safe output to submit your changes as an issue

6. If documentation.md already exists, update it to reflect current state
   If it doesn't exist, create it from scratch

7. Ensure the documentation is accurate and helpful for users

## Error Handling

- If you cannot access certain files or directories, note this in the documentation
- If some aspects of the project are unclear, document what is known
- Always create the documentation even if information is partial
