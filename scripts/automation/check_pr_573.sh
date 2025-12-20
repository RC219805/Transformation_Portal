#!/bin/bash
# PR #573 Status Checker
# Quick script to monitor CI progress

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PR #573 - Status Check"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo

# Get PR status
gh pr view 573 --json title,state,mergeable,statusCheckRollup | \
  jq -r '
    "Title: \(.title)",
    "State: \(.state)",
    "Mergeable: \(.mergeable)",
    "",
    "Status Checks:",
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
    (.statusCheckRollup[] | 
      if .status == "COMPLETED" then
        if .conclusion == "SUCCESS" then
          "✅ \(.name | if length > 40 then .[0:37] + "..." else . end)"
        elif .conclusion == "NEUTRAL" then
          "⚪ \(.name | if length > 40 then .[0:37] + "..." else . end)"
        else
          "❌ \(.name | if length > 40 then .[0:37] + "..." else . end) - \(.conclusion)"
        end
      else
        "🔄 \(.name | if length > 40 then .[0:37] + "..." else . end) - \(.status)"
      end
    ),
    "",
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  '

echo
echo "View full details: https://github.com/RC219805/Transformation_Portal/pull/573/checks"
echo

# Count statuses
total=$(gh pr view 573 --json statusCheckRollup | jq '.statusCheckRollup | length')
completed=$(gh pr view 573 --json statusCheckRollup | jq '[.statusCheckRollup[] | select(.status == "COMPLETED")] | length')
success=$(gh pr view 573 --json statusCheckRollup | jq '[.statusCheckRollup[] | select(.conclusion == "SUCCESS")] | length')

echo "Summary: $success successful, $completed/$total completed"
echo
