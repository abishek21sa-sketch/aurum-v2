import requests
import os
from dotenv import load_dotenv
load_dotenv()

key = os.getenv("FRED_API_KEY")
r = requests.get(
    f"https://api.stlouisfed.org/fred/releases",
    params={"api_key": key, "file_type": "json", "limit": 20}
)
print(f"Status: {r.status_code}")
data = r.json()
for rel in data.get("releases", []):
    print(f"  {rel['id']}: {rel['name']}")