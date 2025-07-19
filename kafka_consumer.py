from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    'login_risk_predictions',
    bootstrap_servers=['localhost:9092'],
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='my-group',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

print("🔍 Waiting for messages...")
for message in consumer:
    print(f"✅ Received: {message.value}")
