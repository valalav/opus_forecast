
import argparse
import subprocess
import sys
import shutil
from pathlib import Path

def run_opencode_analysis(file_path: str, query: str):
    # check if opencode exists
    if not shutil.which("opencode"):
        print("❌ Error: 'opencode' CLI not found in PATH.")
        sys.exit(1)
        
    # Read data file (head only to save context)
    path = Path(file_path)
    if not path.exists():
        print(f"❌ Error: File {file_path} not found.")
        sys.exit(1)
        
    # Read first 100 lines to fit in context
    with open(path, "r") as f:
        head_data = "".join(f.readlines()[:100])
        
    system_prompt = """
    You are a Senior Econometrician for the Opus Forecast project.
    Your expertise includes Time Series Analysis, Regressions, and Macroeconomics.
    
    Analyze the provided data snippet based on the user's query.
    - Be rigorous. Mention statistical tests (ADF, KPSS) if relevant.
    - Point out anomalies or structural breaks.
    - Provide a concise Markdown report.
    """
    
    full_prompt = f"{system_prompt}\n\nDATA SNIPPET:\n{head_data}\n\nQUERY:\n{query}"
    
    print(f"🤔 Asking Econometrician (via Opencode)...")
    
    try:
        # Using STDIN method as per opencode_reference.md (Stability Fix)
        result = subprocess.run(
            ["opencode", "run", "-"], 
            input=full_prompt,
            text=True,
            capture_output=True
        )
        
        if result.returncode != 0:
            print(f"❌ Opencode Error: {result.stderr}")
            sys.exit(1)
            
        print("\n=== Econometric Analysis Report_ ===\n")
        print(result.stdout)
        print("\n==================================\n")
        
    except Exception as e:
        print(f"❌ Execution Error: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to CSV data file")
    parser.add_argument("--query", required=True, help="Question for the econometrician")
    args = parser.parse_args()
    
    run_opencode_analysis(args.file, args.query)

if __name__ == "__main__":
    main()
