"""
Kafka-shaped transaction producer.

Two backends, selected by the KAFKA_MODE env var (see src/streaming/__init__.py):

  - "local" (default): an in-process queue.Queue-based broker emulation.
    Topics are just named queues held in a process-wide registry so multiple
    LocalProducer/LocalConsumer instances in the same process can talk to
    each other, exactly like independent producer/consumer processes would
    talk through a real Kafka topic.

  - "real": a thin wrapper around kafka-python's KafkaProducer, using the
    same `send(topic, value)` interface. Requires a running broker (see
    docker-compose.yml's `kafka` + `zookeeper` services) — not exercised in
    this sandbox (no live broker here), but the code path is real and would
    work unmodified against a real cluster.

Messages are plain JSON dicts matching the transaction schema produced by
data/generate_synthetic_transactions.py, so the same message shape flows
through both backends.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from typing import Any

from src.streaming import KAFKA_MODE

# Process-wide registry of "topics" for the local backend, so independent
# LocalProducer/LocalConsumer instances share the same underlying queues.
_LOCAL_TOPICS: dict[str, "queue.Queue"] = {}
_LOCAL_TOPICS_LOCK = threading.Lock()


def _get_local_topic(topic: str) -> "queue.Queue":
    with _LOCAL_TOPICS_LOCK:
        if topic not in _LOCAL_TOPICS:
            _LOCAL_TOPICS[topic] = queue.Queue()
        return _LOCAL_TOPICS[topic]


class LocalProducer:
    """queue.Queue-backed producer emulating Kafka's `send(topic, value)`."""

    def __init__(self, client_id: str = "local-producer"):
        self.client_id = client_id
        self._sent_count = 0

    def send(self, topic: str, value: dict[str, Any]) -> None:
        q = _get_local_topic(topic)
        message = {
            "topic": topic,
            "value": value,
            "timestamp": time.time(),
            "offset": q.qsize(),
        }
        q.put(message)
        self._sent_count += 1

    def flush(self) -> None:
        # No-op for the in-memory backend; present for interface parity with kafka-python.
        pass

    def close(self) -> None:
        pass

    @property
    def sent_count(self) -> int:
        return self._sent_count


class RealKafkaProducer:
    """Thin wrapper around kafka-python's KafkaProducer.

    Not exercised against a live broker in this sandbox (no broker running
    here) — included for realism / production-mode parity. See
    docker-compose.yml for the `kafka` service this would connect to
    (KAFKA_BOOTSTRAP_SERVERS env var, default 'kafka:9092').
    """

    def __init__(self, bootstrap_servers: str = "localhost:9092", client_id: str = "real-producer"):
        from kafka import KafkaProducer  # kafka-python; imported lazily

        self._producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            client_id=client_id,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
            acks="all",
            retries=3,
        )

    def send(self, topic: str, value: dict[str, Any]) -> None:
        key = value.get("card_id") or value.get("transaction_id")
        self._producer.send(topic, key=key, value=value)

    def flush(self) -> None:
        self._producer.flush()

    def close(self) -> None:
        self._producer.close()


def get_producer(**kwargs):
    """Factory returning the producer implementation selected by KAFKA_MODE."""
    if KAFKA_MODE == "real":
        return RealKafkaProducer(**kwargs)
    return LocalProducer(**kwargs)


if __name__ == "__main__":
    import argparse
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    import pandas as pd

    from src.streaming import TRANSACTIONS_TOPIC

    parser = argparse.ArgumentParser(description="Stream synthetic transactions onto the transactions topic.")
    parser.add_argument("--data", type=str, default=os.path.join(os.path.dirname(__file__), "..", "..", "data", "transactions.csv"))
    parser.add_argument("--n", type=int, default=20, help="Number of transactions to stream")
    parser.add_argument("--delay", type=float, default=0.05, help="Seconds to sleep between messages")
    args = parser.parse_args()

    df = pd.read_csv(args.data).sample(n=args.n, random_state=int(time.time())).reset_index(drop=True)
    producer = get_producer()
    print(f"[producer] KAFKA_MODE={KAFKA_MODE}; streaming {len(df)} messages to topic '{TRANSACTIONS_TOPIC}'")

    for _, row in df.iterrows():
        message = row.to_dict()
        message["timestamp"] = str(message["timestamp"])
        producer.send(TRANSACTIONS_TOPIC, message)
        print(f"[producer] sent transaction_id={message['transaction_id']} amount={message['amount']:.2f}")
        time.sleep(args.delay)

    producer.flush()
    print(f"[producer] done. sent={producer.sent_count if hasattr(producer, 'sent_count') else len(df)}")
