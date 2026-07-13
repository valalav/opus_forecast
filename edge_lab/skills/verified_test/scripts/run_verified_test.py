
import sys
import argparse
import subprocess
import json
import time
import hashlib
from pathlib import Path

def generate_signature(data: str) -> str:
    """Generate a simple hash signature."""
    return hashlib.sha256(data.encode()).hexdigest()[:16]

def run_test(target: str, verbose: bool = False):
    print(f"🔒 RUNNING VERIFIED TEST: {target}")
    start_time = time.time()
    
    cmd = ["pytest", target]
    if verbose:
        cmd.append("-v")
        
    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.time() - start_time
    
    passed = result.returncode == 0
    output_tail = result.stdout[-500:] if result.stdout else ""
    error_tail = result.stderr[-500:] if result.stderr else ""
    
    # Construct receipt
    receipt = {
        "timestamp": time.time(),
        "target": target,
        "passed": passed,
        "duration_seconds": round(duration, 2),
        "output_tail": output_tail,
        "error_tail": error_tail,
        "verified": True
    }
    
    receipt_str = json.dumps(receipt, sort_keys=True)
    receipt["signature"] = generate_signature(receipt_str)
    
    # OUTPUT MUST BE IN JSON BLOCK
    print("\n=== VERIFIED RECEIPT (COPY BELOW) ===")
    print("```json")
    print(json.dumps(receipt, indent=2))
    print("```")
    print("=====================================\n")
    
    if not passed:
        print("❌ TESTS FAILED")
        print(result.stdout)
        print(result.stderr)
        sys.exit(1)
    else:
        print("✅ TESTS PASSED")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="Test file or directory to run")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    
    run_test(args.target, args.verbose)

if __name__ == "__main__":
    main()
