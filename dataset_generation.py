# dataset_generation.py

import pandas as pd
import numpy as np
from faker import Faker
import random
import geoip2.database

# ================================
# 🔷 Initialize Faker and GeoIP2
# ================================
fake = Faker()

# 🔴 Replace with your actual extracted .mmdb path (use raw string to avoid unicode issues)
reader = geoip2.database.Reader(
    r"C:\Users\MY PC\Desktop\CS\web dev\GeoLite2-City_20250715\GeoLite2-City_20250715\GeoLite2-City.mmdb"
)

# ================================
# 🔷 Generate Users and Devices
# ================================
num_users = 1000
user_ids = [f"user_{str(i).zfill(3)}" for i in range(1, num_users + 1)]

# Assign 1-3 devices per user
user_devices = {}
for user in user_ids:
    num_devices = random.randint(1, 3)
    devices = [f"device_{fake.lexify(text='????')}" for _ in range(num_devices)]
    user_devices[user] = devices

# ================================
# 🔷 Generate Login Records
# ================================
num_records = 10000
records = []

for _ in range(num_records):
    user = random.choice(user_ids)
    
    # 85% chance of known device, else new device
    if random.random() < 0.85:
        device_id = random.choice(user_devices[user])
        device_novelty = "Known"
    else:
        device_id = f"device_{fake.lexify(text='????')}"
        device_novelty = "New"
    
    # Generate public IPv4 address
    ip_address = fake.ipv4_public()
    
    # 🔷 GeoIP lookup for city and country
    try:
        response = reader.city(ip_address)
        city = response.city.name if response.city.name else "Unknown City"
        country = response.country.name if response.country.name else "Unknown Country"
    except:
        city = "Unknown City"
        country = "Unknown Country"
    
    # Random timestamp within last 90 days
    timestamp = fake.date_time_between(start_date='-90d', end_date='now')
    login_hour = timestamp.hour
    login_dayofweek = timestamp.weekday()  # 0=Monday, 6=Sunday
    login_isweekend = 1 if login_dayofweek >= 5 else 0
    
    # Calculate deviation from user's usual login hour (8-10 AM)
    usual_login_hour = random.randint(8, 10)
    login_hour_deviation = abs(login_hour - usual_login_hour)
    
    # Failed attempts in past 24h (skewed towards 0-1)
    failed_attempts_24h = np.random.choice([0,1,2,3,4,5], p=[0.5,0.3,0.1,0.05,0.03,0.02])
    
    # 🔷 Add Risk Label logic
    if device_novelty == "New" and login_hour_deviation > 2 and failed_attempts_24h > 0:
        label = 1  # Risky
    else:
        label = 0  # Genuine
    
    # Append record with all new engineered features
    records.append({
        "UserID": user,
        "DeviceID": device_id,
        "IP Address": ip_address,
        "City": city,
        "Country": country,
        "Timestamp": timestamp,
        "Login hour deviation": login_hour_deviation,
        "Login_DayOfWeek": login_dayofweek,
        "Login_IsWeekend": login_isweekend,
        "Device novelty": device_novelty,
        "Failed attempts (24h)": failed_attempts_24h,
        "Label": label
    })

# ================================
# 🔷 Convert to DataFrame and Save
# ================================
df = pd.DataFrame(records)

# Sort by timestamp for realism
df = df.sort_values(by="Timestamp").reset_index(drop=True)

# Save as CSV
df.to_csv("login_dataset.csv", index=False)

# Close GeoIP reader
reader.close()

print("✅ Dataset generation complete with new engineered features. File saved as login_dataset.csv")
