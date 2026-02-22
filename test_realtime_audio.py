import time
import sqlite3
import requests
import os

from sms_alert import send_sms
from location import get_live_location

APP_URL = "http://127.0.0.1:5000"

ALERT_COOLDOWN = 30
last_alert_time = 0

print("🚨 Emergency alert service started")

# ================= HELPERS =================
def get_latest_location():
    try:
        r = requests.get(APP_URL + "/get_location", timeout=2)
        return r.json()
    except:
        return {"lat": None, "lon": None}


def get_current_user():
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("""
        SELECT first_name, last_name, relative_phone1, relative_phone2
        FROM users ORDER BY id DESC LIMIT 1
    """)
    row = cur.fetchone()
    conn.close()

    if not row:
        return None, []

    first, last, r1, r2 = row
    return f"{first} {last}", [r1, r2]


# ================= MAIN LOOP =================
def watch_emergency():
    global last_alert_time

    while True:
        try:
            r = requests.get(APP_URL + "/listener_status", timeout=2)
            state = r.json()
        except:
            time.sleep(2)
            continue

        if state.get("emergency"):
            now = time.time()

            if now - last_alert_time >= ALERT_COOLDOWN:
                print("🚨 EMERGENCY CONFIRMED")

                name, relatives = get_current_user()
                loc = get_latest_location()
                link = get_live_location(loc["lat"], loc["lon"])

                for num in relatives:
                    if num:
                        send_sms(num, name, link)
                        print("📨 SMS sent to", num)

                last_alert_time = now

        time.sleep(2)


# ================= RUN =================
if __name__ == "__main__":
    watch_emergency()
