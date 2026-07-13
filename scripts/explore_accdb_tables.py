"""Quick exploration of all ACCDB tables: schema + first 5 rows + last 3 rows."""
import subprocess
import sys

DB_PATH = "data/db_cpi_store.accdb"

# Tables we already know well: data_indices, regions_names
# Tables we need to explore: data_price, data_seasonalised, data_weights, items_names,
#   items_set, items_structure, region_structure, weights, z_max_day

EXPLORE_TABLES = [
    "z_max_day",        # probably small
    "items_set",        # probably small  
    "items_structure",  # probably small
    "region_structure", # probably small
    "weights",          # could be large or small
]

def run_cmd(cmd, timeout=300):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout, r.stderr, r.returncode
    except subprocess.TimeoutExpired:
        return "", f"TIMEOUT ({timeout}s)", -1

for table in EXPLORE_TABLES:
    print(f"\n{'='*60}")
    print(f"TABLE: {table}")
    print(f"{'='*60}")
    
    # Schema
    stdout, stderr, rc = run_cmd(["mdb-schema", "--table", table, DB_PATH], timeout=60)
    if stdout.strip():
        print(stdout.strip()[:500])
    
    # First 5 + last 3 rows
    stdout, stderr, rc = run_cmd(["mdb-export", DB_PATH, table], timeout=120)
    if stdout.strip():
        lines = stdout.strip().split("\n")
        total = len(lines) - 1
        print(f"\nTotal rows: {total}")
        for line in lines[:6]:  # header + 5
            print(f"  {line}")
        if total > 8:
            print("  ...")
            for line in lines[-3:]:
                print(f"  {line}")
    else:
        print(f"  Error/Empty: {stderr}")
