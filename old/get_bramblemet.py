import requests
from bs4 import BeautifulSoup
import csv
from datetime import datetime

def fetch_bramblemet(date_str):
    # Format must be dd/mm/yyyy
    url = "https://www.bramblemet.co.uk/search.aspx"
    payload = {
        "ctl00$MainContent$txtDate": date_str,
        "ctl00$MainContent$btnSubmit": "Submit"
    }

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Content-Type": "application/x-www-form-urlencoded"
    }

    response = requests.post(url, data=payload, headers=headers)
    if response.status_code != 200:
        print(f"Error: Received HTTP {response.status_code}")
        return

    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table")
    if not table:
        print("No data table found.")
        return

    # Extract rows
    rows = []
    for tr in table.find_all("tr"):
        cols = [td.get_text(strip=True) for td in tr.find_all("td")]
        if cols:
            rows.append(cols)

    if not rows:
        print("No data found for that date.")
        return

    # Save to CSV
    out_name = f"bramblemet_{date_str.replace('/', '-')}.csv"
    with open(out_name, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"✅ Data saved to {out_name}")

# Example usage
if __name__ == "__main__":
    date_input = input("Enter date (dd/mm/yyyy): ")
    try:
        datetime.strptime(date_input, "%d/%m/%Y")
        fetch_bramblemet(date_input)
    except ValueError:
        print("❌ Invalid date format. Use dd/mm/yyyy.")
