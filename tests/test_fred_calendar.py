import requests
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()

key = os.getenv("FRED_API_KEY")

# Key release IDs we care about
RELEASES_OF_INTEREST = {
    10: "Consumer Price Index",
    50: "Employment Situation (NFP)",
    21: "H.6 Money Stock",
    175: "GDP",
    180: "FOMC Press Release",
}

today = datetime.now().strftime("%Y-%m-%d")
ninety_days = (datetime.now() + timedelta(days=90)).strftime("%Y-%m-%d")

print(f"Upcoming releases ({today} to {ninety_days}):\n")

for release_id, name in RELEASES_OF_INTEREST.items():
    r = requests.get(
        "https://api.stlouisfed.org/fred/release/dates",
        params={
            "api_key": key,
            "release_id": release_id,
            "file_type": "json",
            "realtime_start": today,
            "realtime_end": ninety_days,
            "limit": 5,
            "sort_order": "asc",
            "include_release_dates_with_no_data": "true"
        }
    )
    data = r.json()
    dates = [d["date"] for d in data.get("release_dates", [])]
    print(f"  {name} (ID {release_id}): {dates[:3]}")