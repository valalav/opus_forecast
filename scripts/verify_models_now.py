import requests
import sys

def verify_models():
    try:
        response = requests.get("http://localhost:8000/models", timeout=5)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print(f"Response Body (truncated): {str(response.json())[:200]}...")
            data = response.json()
            if "models" in data and len(data["models"]) > 0:
                print("SUCCESS: /models endpoint works.")
                return True
        print("FAILURE: /models check failed.")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    if verify_models():
        sys.exit(0)
    else:
        sys.exit(1)
