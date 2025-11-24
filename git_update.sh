#!/bin/bash

# 1. Navigate to the project folder
cd /home/andrew/.ssh/Trading/Knowledge_Graph/

# 2. Check if the CSV file has changed
if [[ -n $(git status --porcelain live_risk_scores.csv) ]]; then
  echo "🚀 Risk scores changed. Pushing to GitHub..."
  
  # 3. Git commands to push the file
  git add live_risk_scores.csv
  git commit -m "🤖 Auto-Update: Fresh GNN Risk Scores"
  git push origin main
  
  echo "✅ Successfully pushed to GitHub."
else
  echo "💤 No changes in risk scores. Skipping push."
fi