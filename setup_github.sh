#!/bin/bash
# Run this script from the dts-fall-detection-github folder to initialize git and push to GitHub.
# Usage: bash setup_github.sh

set -e

REPO_NAME="dts-fall-detection"
GITHUB_USER=$(gh api user --jq .login 2>/dev/null || echo "")

echo "=== Setting up git repo ==="

# Clean up any broken .git from a previous attempt
if [ -d ".git" ]; then
  echo "Removing existing .git directory..."
  rm -rf .git
fi

# Initialize fresh repo
git init
git branch -m main
git config user.email "rudrachopra023@gmail.com"
git config user.name "Rudra Chopra"

# Stage everything
git add .
git commit -m "Initial commit: DTS fall detection research code and paper"

echo ""
echo "=== Creating GitHub repo and pushing ==="

if command -v gh &>/dev/null && [ -n "$GITHUB_USER" ]; then
  # Use GitHub CLI if available and authenticated
  gh repo create "$REPO_NAME" --public --source=. --remote=origin --push
  echo ""
  echo "Done! Repo is live at: https://github.com/$GITHUB_USER/$REPO_NAME"
else
  echo "GitHub CLI (gh) not found or not authenticated."
  echo ""
  echo "Please:"
  echo "  1. Go to https://github.com/new and create a public repo named '$REPO_NAME' (leave it empty, no README/license)"
  echo "  2. Then run these two commands:"
  echo ""
  echo "     git remote add origin https://github.com/YOUR_USERNAME/$REPO_NAME.git"
  echo "     git push -u origin main"
fi
