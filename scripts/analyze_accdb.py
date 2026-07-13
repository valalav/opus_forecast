"""
Lightweight analyzer for large .accdb files.
Reads header + uses mdb-tables with short timeout, then samples each table.
"""
import subprocess
import struct
import os
import sys

DB_PATH = "data/db_cpi_store.accdb"

def read_header(path):
    """Read Access header to determine version."""
    with open(path, 'rb') as f:
        header = f.read(256)
    
    # Access version detection from magic bytes at offset 0x14
    version_map = {
        0x00: "JET3 (Access 97)",
        0x01: "JET4 (Access 2000/2002/2003)",
        0x02: "ACE12 (Access 2007)",
        0x03: "ACE14 (Access 2010)",
        0x04: "ACE15 (Access 2013)",
        0x05: "ACE16 (Access 2016+)",
        0x06: "ACE17 (Access 365)",
    }
    
    if len(header) >= 0x15:
        ver_byte = header[0x14]
        version = version_map.get(ver_byte, f"Unknown (0x{ver_byte:02x})")
    else:
        version = "Unknown"
    
    # Page size from offset 0x08 (2 bytes, little-endian)
    if len(header) >= 0x0A:
        page_size = struct.unpack_from('<H', header, 0x08)[0]
    else:
        page_size = 0
    
    return version, page_size

def run_cmd(cmd, timeout=600):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout, r.stderr, r.returncode
    except subprocess.TimeoutExpired:
        return "", f"TIMEOUT ({timeout}s)", -1

def main():
    size_bytes = os.path.getsize(DB_PATH)
    size_mb = size_bytes / (1024*1024)
    
    version, page_size = read_header(DB_PATH)
    
    print(f"File: {DB_PATH}")
    print(f"Size: {size_mb:.1f} MB ({size_bytes:,} bytes)")
    print(f"Version: {version}")
    print(f"Page size: {page_size} bytes")
    print()
    
    # Get tables (with generous timeout for large file)
    print("=== TABLES ===")
    print("(Waiting up to 10 minutes for mdb-tables on 1.2GB file...)")
    sys.stdout.flush()
    
    stdout, stderr, rc = run_cmd(["mdb-tables", "-1", DB_PATH], timeout=600)
    if rc != 0 or not stdout.strip():
        print(f"ERROR getting tables: {stderr}")
        sys.exit(1)
    
    tables = [t.strip() for t in stdout.strip().split("\n") if t.strip()]
    print(f"Found {len(tables)} tables:")
    for t in tables:
        print(f"  - {t}")
    
    # Schema & sample for each table
    for table in tables:
        print(f"\n{'='*60}")
        print(f"TABLE: {table}")
        print(f"{'='*60}")
        
        # Schema
        print("\n--- Schema (CREATE TABLE) ---")
        sys.stdout.flush()
        stdout, stderr, rc = run_cmd(["mdb-schema", "--table", table, DB_PATH], timeout=300)
        if stdout.strip():
            print(stdout.strip())
        else:
            print(f"  Error: {stderr}")
        
        # Export first 10 rows
        print("\n--- Sample data (first 10 rows) ---")
        sys.stdout.flush()
        stdout, stderr, rc = run_cmd(["mdb-export", table, DB_PATH], timeout=300)
        if stdout.strip():
            lines = stdout.strip().split("\n")
            total = len(lines) - 1  # minus header
            print(f"  Total rows: ~{total}")
            for line in lines[:11]:  # header + 10 rows
                print(f"  {line}")
            if total > 10:
                print(f"  ... ({total - 10} more rows)")
            
            # Also show last 3 rows to understand date range
            if total > 13:
                print(f"\n  --- Last 3 rows ---")
                for line in lines[-3:]:
                    print(f"  {line}")
        else:
            print(f"  Error: {stderr}")
    
    print("\n=== ANALYSIS COMPLETE ===")

if __name__ == "__main__":
    main()
