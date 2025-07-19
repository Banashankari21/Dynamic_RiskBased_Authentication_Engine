from kafka import KafkaProducer
import json

# Initialize the producer
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Example message
message = {
    "user_id": "user_001",
    "risk_label": 1,
    "risk_probability": 0.88
}

# Send it to the topic
producer.send('login_risk_predictions', message)
producer.flush()

print("✅ Message sent to Kafka topic.")
