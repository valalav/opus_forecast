"""
Task Refiner Agent

Third agent in the Ralph Universal system that:
1. Watches for BLOCKED tasks
2. Analyzes why they failed (from feedback)
3. Researches relevant files (ls, head, file structure)
4. Creates refined subtasks with proper MVAC criteria
"""

import json
import os
import re
import subprocess
import time
from datetime import datetime
from core.state import StateManager
from core.agent_wrapper import AgentWrapper
from config import MAX_ITERATIONS, SLEEP_INTERVAL, BASE_DIR, PROJECT_ROOT

# Refiner-specific config
REFINER_SLEEP = 30  # Check every 30 seconds
MAX_SUBTASKS = 4
MIN_SUBTASKS = 2


def find_file_paths(text: str) -> list[str]:
    """Extract potential file paths from task description."""
    patterns = [
        r'data/[^\s\'"]+\.\w+',          # data/file.csv
        r'sirena/[^\s\'"]+\.\w+',         # sirena/models/x.py
        r'scripts/[^\s\'"]+\.\w+',        # scripts/x.py
        r'archive/[^\s\'"]+',             # archive/...
        r'/home/[^\s\'"]+',               # absolute paths
        r'\w+\.(?:csv|xlsx|json|py|md)'   # any file with common extensions
    ]
    
    paths = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        paths.extend(matches)
    
    return list(set(paths))


def research_file(file_path: str, base_dir: str) -> dict:
    """Examine a file and return its structure info."""
    info = {'path': file_path, 'exists': False}
    
    # Try relative to project root first, then absolute
    candidates = [
        os.path.join(base_dir, file_path),
        os.path.join(PROJECT_ROOT.parent, file_path),  # sirena-kbr root
        file_path
    ]
    
    actual_path = None
    for candidate in candidates:
        if os.path.exists(candidate):
            actual_path = candidate
            break
    
    if not actual_path:
        return info
    
    info['exists'] = True
    info['actual_path'] = actual_path
    
    try:
        # File stats
        stat = os.stat(actual_path)
        info['size_mb'] = round(stat.st_size / (1024 * 1024), 2)
        info['is_dir'] = os.path.isdir(actual_path)
        
        if info['is_dir']:
            # List directory contents
            contents = os.listdir(actual_path)[:20]  # First 20 items
            info['contents'] = contents
            info['total_files'] = len(os.listdir(actual_path))
        else:
            # Get file type
            result = subprocess.run(['file', actual_path], capture_output=True, text=True)
            info['file_type'] = result.stdout.split(':')[-1].strip()
            
            # Get first lines if text file
            if 'text' in info['file_type'].lower() or actual_path.endswith(('.csv', '.json', '.py', '.md')):
                with open(actual_path, 'r', encoding='utf-8', errors='ignore') as f:
                    info['head'] = f.read(2000)  # First 2KB
                    
            # For Excel files, try to get sheet names
            if actual_path.endswith(('.xlsx', '.xls')):
                try:
                    import openpyxl
                    wb = openpyxl.load_workbook(actual_path, read_only=True)
                    info['sheets'] = wb.sheetnames
                    wb.close()
                except:
                    pass
                    
    except Exception as e:
        info['error'] = str(e)
    
    return info


def build_refiner_prompt(task: dict, file_research: list[dict]) -> str:
    """Build prompt for LLM to analyze and decompose task."""
    
    # Format file research
    files_info = ""
    for f in file_research:
        if f['exists']:
            files_info += f"\n### {f['path']}\n"
            files_info += f"- Size: {f.get('size_mb', 'N/A')} MB\n"
            if f.get('is_dir'):
                files_info += f"- Type: Directory ({f.get('total_files', 0)} items)\n"
                files_info += f"- Contents: {', '.join(f.get('contents', []))}\n"
            else:
                files_info += f"- Type: {f.get('file_type', 'unknown')}\n"
                if f.get('sheets'):
                    files_info += f"- Excel Sheets: {', '.join(f['sheets'])}\n"
                if f.get('head'):
                    files_info += f"- Preview:\n```\n{f['head'][:500]}...\n```\n"
        else:
            files_info += f"\n### {f['path']}\n- NOT FOUND\n"
    
    prompt = f"""# Task Refinement Request

## Original Task (BLOCKED after 3 attempts)

**ID**: {task.get('id')}
**Title**: {task.get('title')}
**Priority**: {task.get('priority', 'medium')}

**Description**:
{task.get('description', 'N/A')}

**Original Acceptance Criteria**:
{chr(10).join('- ' + c for c in task.get('acceptance_criteria', []))}

**Failure Feedback**:
{task.get('feedback', 'No feedback provided')}

## File Research
{files_info if files_info else 'No files found to research.'}

---

## Your Task

Analyze why this task failed and create {MIN_SUBTASKS}-{MAX_SUBTASKS} smaller, more achievable subtasks.

For each subtask provide:
1. **title** - Clear, specific title
2. **description** - Detailed description with hints based on file research
3. **acceptance_criteria** - 2-3 MVAC criteria that are specific and testable
4. **priority** - same as parent or adjusted based on difficulty

Output as JSON array:
```json
[
  {{
    "title": "Subtask 1 Title",
    "description": "Detailed description with specific hints...",
    "acceptance_criteria": [
      "@file: path/to/expected/output.csv exists",
      "@functional: script exits with code 0",
      "@metric: output contains > N rows"
    ],
    "priority": "high"
  }},
  ...
]
```

Be specific! Include file paths, column names, sheet names, row counts from the research.
"""
    return prompt


def parse_subtasks_from_output(output: str) -> list[dict]:
    """Parse LLM output to extract subtasks JSON."""
    # Try to find JSON array in output
    json_match = re.search(r'\[\s*\{.*?\}\s*\]', output, re.DOTALL)
    
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    
    # Fallback: try to parse entire output as JSON
    try:
        return json.loads(output)
    except:
        pass
    
    return []


def create_subtasks(state: StateManager, parent_task: dict, subtasks_data: list[dict]) -> list[int]:
    """Create subtask entries in prd.json."""
    prd = state.read_prd()
    
    # Get next ID
    existing_ids = [t.get('id', 0) for t in prd.get('user_stories', [])]
    next_id = max(existing_ids, default=0) + 1
    
    created_ids = []
    parent_id = parent_task['id']
    
    for i, sub in enumerate(subtasks_data[:MAX_SUBTASKS]):
        new_task = {
            "id": next_id + i,
            "title": sub.get('title', f"Subtask {i+1} of Task {parent_id}"),
            "description": sub.get('description', ''),
            "acceptance_criteria": sub.get('acceptance_criteria', []),
            "priority": sub.get('priority', parent_task.get('priority', 'medium')),
            "status": "TODO",
            "parent_id": parent_id,
            "created_at": datetime.now().isoformat(),
            "created_by": "refiner"
        }
        prd['user_stories'].append(new_task)
        created_ids.append(new_task['id'])
    
    # Mark parent as DECOMPOSED
    for t in prd['user_stories']:
        if t['id'] == parent_id:
            t['status'] = 'DECOMPOSED'
            t['subtask_ids'] = created_ids
            break
    
    state.save_prd(prd)
    return created_ids


def main():
    state = StateManager()
    agent = AgentWrapper("refiner")
    
    print(f"🔬 Refiner started in {BASE_DIR}")
    print(f"   Watching for BLOCKED tasks every {REFINER_SLEEP}s")
    
    processed_ids = set()  # Don't reprocess same task
    
    for i in range(MAX_ITERATIONS):
        time.sleep(REFINER_SLEEP)
        
        # Find BLOCKED tasks
        tasks = state.read_prd().get("user_stories", [])
        blocked = [t for t in tasks if t.get("status") == "BLOCKED" and t['id'] not in processed_ids]
        
        if not blocked:
            print(f"🔬 Refiner: No BLOCKED tasks. Waiting...")
            continue
        
        # Process first blocked task
        task = blocked[0]
        task_id = task['id']

        # Circuit Breaker: Check attempts
        attempts = task.get('refinement_attempts', 0)
        if attempts >= 3:
            print(f"🔬 Refiner: Skipping Task {task_id} (Max attempts reached: {attempts})")
            processed_ids.add(task_id)
            continue
            
        print(f"🔬 Refiner: Analyzing BLOCKED Task {task_id}: {task['title']} (Attempt {attempts + 1}/3)")
        
        # Increment attempt counter immediately to prevent infinite loops on crash
        task['refinement_attempts'] = attempts + 1
        state.save_prd(state.read_prd())
        
        state.append_progress(f"Refiner starting analysis of BLOCKED Task {task_id} (Attempt {attempts + 1})", "REFINER")
        
        # 1. Research files mentioned in task
        all_text = f"{task.get('title', '')} {task.get('description', '')} {task.get('feedback', '')}"
        file_paths = find_file_paths(all_text)
        
        print(f"   📁 Found {len(file_paths)} potential file paths to research")
        
        file_research = []
        for fp in file_paths[:10]:  # Max 10 files
            info = research_file(fp, str(PROJECT_ROOT.parent))
            file_research.append(info)
            if info['exists']:
                print(f"   ✓ {fp}: {info.get('size_mb', 0)} MB")
            else:
                print(f"   ✗ {fp}: not found")
        
        # 2. Build prompt and call LLM
        prompt = build_refiner_prompt(task, file_research)
        
        print(f"   🧠 Calling LLM to analyze and decompose...")
        state.append_progress(f"Refiner calling LLM to analyze Task {task_id}", "REFINER")
        
        try:
            output = agent.run(prompt)
        except Exception as e:
            print(f"   ❌ LLM call failed: {e}")
            state.append_progress(f"Refiner LLM failed for Task {task_id}: {e}", "REFINER")
            processed_ids.add(task_id)
            continue
        
        # 3. Parse subtasks
        subtasks = parse_subtasks_from_output(output)
        
        if not subtasks or len(subtasks) < MIN_SUBTASKS:
            print(f"   ⚠️ Could not parse valid subtasks from LLM output")
            state.append_progress(f"Refiner could not generate valid subtasks for Task {task_id}", "REFINER")
            processed_ids.add(task_id)
            continue
        
        # 4. Create subtasks in prd.json
        created_ids = create_subtasks(state, task, subtasks)
        
        print(f"   ✅ Created {len(created_ids)} subtasks: {created_ids}")
        state.append_progress(
            f"Refiner decomposed Task {task_id} into {len(created_ids)} subtasks: {created_ids}", 
            "REFINER"
        )
        
        processed_ids.add(task_id)
        
        # Log summary of created subtasks
        for i, sub in enumerate(subtasks[:len(created_ids)]):
            print(f"      Subtask {created_ids[i]}: {sub.get('title', 'N/A')}")


if __name__ == "__main__":
    main()
