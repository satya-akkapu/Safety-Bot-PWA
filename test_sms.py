from sms_alert import send_sms

send_sms(
    "+917702312244",   # verified number
    "Test User",
    "https://www.google.com/maps?q=18.1067,83.3956"
)

print("Done")
