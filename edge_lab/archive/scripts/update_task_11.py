#!/usr/bin/env python3
import json
from pathlib import Path

prd_path = Path("/home/valalav/_projects/sirena-kbr/edge_lab/tasks/prd.json")

with open(prd_path, 'r') as f:
    prd = json.load(f)

# Update task 11
for task in prd['user_stories']:
    if task['id'] == 11:
        task['status'] = 'PENDING_REVIEW'
        print(f"Updated Task {task['id']}: {task['status']}")
        break

with open(prd_path, 'w') as f:
    json.dump(prd, f, indent=2)

print("PRD updated successfully")
