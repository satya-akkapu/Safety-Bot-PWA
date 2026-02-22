def get_live_location(lat, lon):
    if lat is None or lon is None:
        return "Location unavailable"

    return f"https://www.google.com/maps?q={lat},{lon}"


