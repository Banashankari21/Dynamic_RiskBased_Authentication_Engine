from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import os
from kafka import KafkaProducer
import json
import redis
from fastapi.middleware.cors import CORSMiddleware

# ============================
# 🔷 Initialize FastAPI app
# ============================
app = FastAPI(
    title="Dynamic Risk-Based Authentication Engine",
    description="API to predict login risk score dynamically using trained ML model",
    version="1.0.0"
)

# ============================
# 🔷 Enable CORS for frontend calls
# ============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace * with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================
# 🔷 Load ML model at startup
# ============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "risk_model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ============================
# 🔷 Setup Kafka Producer
# ============================
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# ============================
# 🔷 Setup Redis client
# ============================
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# ============================
# 🔷 Define request schema
# ============================
class LoginData(BaseModel):
    login_hour_deviation: float
    device_novelty: int
    failed_attempts_24h: int
    country_encoded: int
    city_encoded: int
    login_success_rate: float
    ip_risk_score: float

# ============================
# 🔷 Root endpoint
# ============================
@app.get("/")
def read_root():
    return {"message": "Dynamic Risk-Based Authentication API is running."}

# ============================
# 🔷 Predict risk endpoint
# ============================
@app.post("/predict")
def predict_risk(data: LoginData):
    # Convert input data to NumPy array in correct order
    input_data = np.array([[ 
        data.login_hour_deviation,
        data.device_novelty,
        data.failed_attempts_24h,
        data.country_encoded,
        data.city_encoded,
        data.login_success_rate,
        data.ip_risk_score
    ]])

    # Get prediction
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]  # probability of being risky

    # Prepare message
    message = {
        "user_id": "user_001",  # Modify if dynamic user ID is passed later
        "risk_label": int(pred),
        "risk_probability": float(prob)
    }

    # Send to Kafka
    producer.send("login_risk_predictions", message)
    producer.flush()

    # Store in Redis as list for historical tracking
    redis_client.lpush(f"history:{message['user_id']}", json.dumps(message))

    # Return response
    return message

# ============================
# 🔷 Get latest risk data endpoint
# ============================
@app.get("/risk/{user_id}")
def get_risk(user_id: str):
    data = redis_client.lindex(f"history:{user_id}", 0)  # get latest entry from list
    if data:
        return json.loads(data)
    else:
        return {"error": "User not found"}

# ============================
# 🔷 Get full risk history endpoint
# ============================
@app.get("/risk/history/{user_id}")
def get_risk_history(user_id: str):
    history = redis_client.lrange(f"history:{user_id}", 0, -1)
    data = [json.loads(item) for item in history]
    return data
