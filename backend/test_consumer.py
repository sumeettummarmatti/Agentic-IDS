from confluent_kafka import Consumer
import uuid

c = Consumer({
    "bootstrap.servers": "localhost:9092",
    "group.id": f"test-group-{uuid.uuid4()}",
    "auto.offset.reset": "latest"
})
c.subscribe(["raw-flows"])

counts = {"MUM-01": 0, "USE-01": 0, "LON-01": 0}
for _ in range(500):
    msg = c.poll(1.0)
    if msg is None: continue
    if msg.error(): continue
    key = msg.key().decode('utf-8') if msg.key() else "none"
    if key in counts:
        counts[key] += 1
    else:
        counts[key] = 1

print("Results:", counts)
