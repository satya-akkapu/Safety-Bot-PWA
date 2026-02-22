import os
from twilio.rest import Client

ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER")

client = Client(ACCOUNT_SID, AUTH_TOKEN)

def send_sms(to_number, name, location):
    message = f"""🚨 EMERGENCY ALERT 🚨

{name} is in danger and needs immediate help!

📍 Live Location:
{location}
"""

    client.messages.create(
        body=message,
        from_=TWILIO_NUMBER,
        to=to_number
    )