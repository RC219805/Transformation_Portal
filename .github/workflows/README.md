# GitHub Actions Workflows

This directory contains GitHub Actions workflow definitions for the 800 Picacho Lane LUTs repository.

## Workflows

### python-app.yml
Main CI workflow for Python testing and linting.
- Runs on push to main and pull requests
- Tests with Python 3.10
- Linting with flake8
- Testing with pytest

### pylint.yml
Pylint code quality checks.

### codeql.yml
Security analysis with CodeQL.

### summary.yml
**Status**: Experimental - Debugging Required

This workflow attempts to use AI to automatically summarize new GitHub issues when they are opened.

#### Current Implementation
- Triggers on issue open events
- Attempts to use `actions/ai-inference@v2` (non-existent action)
- Posts AI-generated summary as a comment on the issue

#### Known Issues
⚠️ **The `actions/ai-inference@v2` action does not exist in the GitHub Actions marketplace.**

This workflow will fail at the AI inference step. The debug logging has been added to help diagnose issues and provide visibility into the failure.

#### Debug Features
The workflow now includes comprehensive debugging:
- Prints issue details (number, title, author, body)
- Reports inference step outcome and status
- Checks for response output
- Provides fallback notification when AI summarization fails

#### Possible Solutions
To make this workflow functional, consider one of these alternatives:

1. **Use GitHub Copilot API** (if available to your organization)
   - Replace with actual GitHub AI/Copilot API endpoints
   
2. **Use OpenAI API**
   ```yaml
   - name: Generate summary with OpenAI
     env:
       OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
     run: |
       # Call OpenAI API to generate summary
   ```

3. **Use GitHub Actions marketplace alternatives**
   - Search for community-maintained AI summary actions
   
4. **Disable the workflow**
   - Comment out or remove if AI summarization is not critical

#### Testing
Since the AI inference action doesn't exist, this workflow will:
- Print debug information about the issue
- Fail gracefully at the inference step (with `continue-on-error: true`)
- Post a notification comment that summarization failed
- Not block other workflows or issue creation

#### Maintenance Notes
- Added debug logging in commit 6ca3996
- Error handling ensures workflow doesn't block issue creation
- Requires action implementation or replacement before production use
