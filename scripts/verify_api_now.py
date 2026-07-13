import requests
import time
import sys

def verify_health():
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        print(f"Status Code: {response.status_code}")
        print(f"Response Body: {response.json()}")
        
        data = response.json()
        if "status" in data and "version" in data:
            print("SUCCESS: Health endpoint keys found.")
            return True
        else:
            print("FAILURE: Missing keys.")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    if verify_health():
        sys.exit(0)
    else:
        sys.exit(1)
