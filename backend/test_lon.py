from confluent_kafka import Consumer
import json
import uuid

c = Consumer({
    "bootstrap.servers": "localhost:9092",
    "group.id": str(uuid.uuid4()),
    "auto.offset.reset": "earliest"
})
c.subscribe(['raw-flows'])

counts = {"MUM-01": 0, "USE-01": 0, "LON-01": 0}
for i in range(2000):
    msg = c.poll(1.0)
    if msg is None: continue
    if msg.error(): continue
    try:
        val = json.loads(msg.value().decode('utf-8'))
        sid = val.get("server_id")
        if sid in counts:
            counts[sid] += 1
    except:
        pass

print("Counts from latest 2000 messages:", counts)
