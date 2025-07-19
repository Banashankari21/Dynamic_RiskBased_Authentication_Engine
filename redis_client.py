import redis

# Connect to Redis server
r = redis.Redis(host='localhost', port=6379, db=0)

# Test connection
try:
    r.ping()
    print("✅ Connected to Redis successfully")
except Exception as e:
    print("❌ Redis connection error:", e)
