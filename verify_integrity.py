
import sys
from pathlib import Path

def check_file_content(path, must_contain):
    try:
        content = Path(path).read_text()
        missing = [phrase for phrase in must_contain if phrase not in content]
        if missing:
            print(f"❌ {path}: Missing phrases: {missing}")
            return False
        print(f"✅ {path}: passed integrity check.")
        return True
    except Exception as e:
        print(f"❌ {path}: Error - {e}")
        return False

def verify_integrity():
    files_to_check = {
        "edge_lab/system/worker.py": [
            "PHASE 3: SELF-VERIFICATION (CRITICAL)",
            "You MUST run the verification command YOURSELF",
            "Fake Work"
        ],
        "edge_lab/system/critic.py": [
            "CHECK EVIDENCE IN WORKER OUTPUT",
            "Verification commands run:",
            "Ghost Work"
        ]
    }
    
    all_passed = True
    for path, phrases in files_to_check.items():
        full_path = f"/home/valalav/_projects/sirena-kbr/{path}"
        if not check_file_content(full_path, phrases):
            all_passed = False
            
    if all_passed:
        print("\n🎉 INTEGRITY UPGRADE VERIFIED: Ralph is now stricter.")
        sys.exit(0)
    else:
        print("\n🔥 INTEGRITY CHECK FAILED")
        sys.exit(1)

if __name__ == "__main__":
    verify_integrity()
