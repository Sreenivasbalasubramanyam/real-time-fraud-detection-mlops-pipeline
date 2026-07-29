"""
End-to-end local streaming demo: runs the producer (in a background thread)
and the scoring consumer (in the main thread) against the in-process
queue.Queue backend (KAFKA_MODE=local, the default) — no Kafka broker
required. This demonstrates the full "streaming payment risk scoring" loop
described in the project README.

Usage:
    python run_streaming_demo.py --n 25
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd

from src.streaming import KAFKA_MODE, TRANSACTIONS_TOPIC
from src.streaming.consumer import run_scoring_consumer
from src.streaming.producer import get_producer

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "transactions.csv")


def _produce(n: int, delay: float):
    df = pd.read_csv(DATA_PATH).sample(n=n, random_state=7).reset_index(drop=True)
    producer = get_producer(client_id="demo-producer")
    print(f"[producer] streaming {len(df)} messages to topic '{TRANSACTIONS_TOPIC}' (KAFKA_MODE={KAFKA_MODE})")
    for _, row in df.iterrows():
        message = row.to_dict()
        message["timestamp"] = str(message["timestamp"])
        producer.send(TRANSACTIONS_TOPIC, message)
        time.sleep(delay)
    producer.flush()
    print(f"[producer] done, sent {len(df)} messages")


def main():
    parser = argparse.ArgumentParser(description="Run producer + consumer end-to-end demo.")
    parser.add_argument("--n", type=int, default=25)
    parser.add_argument("--delay", type=float, default=0.03)
    args = parser.parse_args()

    producer_thread = threading.Thread(target=_produce, args=(args.n, args.delay), daemon=True)
    producer_thread.start()

    time.sleep(0.2)  # let a few messages land before the consumer starts polling
    results = run_scoring_consumer(max_messages=args.n, idle_timeout=3.0)
    producer_thread.join(timeout=5)

    n_flagged = sum(1 for r in results if r["is_flagged"])
    print(f"\n=== Demo summary === messages_scored={len(results)} flagged_as_fraud={n_flagged}")


if __name__ == "__main__":
    main()
