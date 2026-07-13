
import requests
import argparse
import sys
import time
import urllib3
from pathlib import Path
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# Suppress insecure request warnings (Rosstat often has bad certs)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def get_session():
    session = requests.Session()
    
    # Retry logic
    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # Headers to mimic browser (critical for Rosstat)
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    })
    
    return session

def fetch_file(url: str, output_path: str):
    print(f"🌍 Fetching: {url}")
    session = get_session()
    
    try:
        response = session.get(url, verify=False, timeout=30, stream=True)
        response.raise_for_status()
        
        # Save file
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        size_kb = path.stat().st_size / 1024
        print(f"✅ Downloaded to {path} ({size_kb:.1f} KB)")
        
    except Exception as e:
        print(f"❌ Error fetching {url}: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="URL to download")
    parser.add_argument("--output", required=True, help="Output file path")
    args = parser.parse_args()
    
    fetch_file(args.url, args.output)

if __name__ == "__main__":
    main()
