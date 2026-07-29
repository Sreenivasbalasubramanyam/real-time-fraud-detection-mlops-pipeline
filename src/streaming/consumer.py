"""
Kafka-shaped transaction consumer + real-time scoring.

Mirrors src/streaming/producer.py: a "local" (queue.Queue) backend and a
"real" (kafka-python) backend behind the same polling interface, selected by
KAFKA_MODE. The consumer reads transaction messages, scores each one with
the trained fraud model (src/models/score.py), and publishes the scoring
result onto the `fraud-scores` topic — exactly the shape of a real-time
payment risk scoring service sitting downstream of a payments topic.
"""

from __future__ import annotations

import json
import queue
import time
from typing import Any, Iterator

from src.streaming import KAFKA_MODE
from src.streaming.producer import _get_local_topic


class LocalConsumer:
    """queue.Queue-backed consumer emulating Kafka's poll/iterate interface."""

    def __init__(self, topic: str, group_id: str = "local-consumer-group", timeout: float = 1.0):
        self.topic = topic
        self.group_id = group_id
        self.timeout = timeout
        self._queue: "queue.Queue" = _get_local_topic(topic)
        self._received_count = 0

    def poll(self, timeout: float | None = None) -> dict[str, Any] | None:
        """Return the next message's `value` dict, or None if nothing arrives
        within the timeout (mirrors kafka-python's poll-with-timeout pattern,
        simplified to one message at a time for readability in this demo).
        """
        t = timeout if timeout is not None else self.timeout
        try:
            message = self._queue.get(timeout=t)
            self._received_count += 1
            return message["value"]
        except queue.Empty:
            return None

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return self

    def __next__(self) -> dict[str, Any]:
        value = self.poll()
        if value is None:
            raise StopIteration
        return value

    def close(self) -> None:
        pass

    @property
    def received_count(self) -> int:
        return self._received_count


class RealKafkaConsumer:
    """Thin wrapper around kafka-python's KafkaConsumer.

    Not exercised against a live broker in this sandbox — included for
    realism / production-mode parity with docker-compose.yml's `kafka`
    service.
    """

    def __init__(
        self,
        topic: str,
        bootstrap_servers: str = "localhost:9092",
        group_id: str = "fraud-scoring-service",
        auto_offset_reset: str = "earliest",
    ):
        from kafka import KafkaConsumer  # kafka-python; imported lazily

        self._consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            auto_offset_reset=auto_offset_reset,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            enable_auto_commit=True,
        )

    def poll(self, timeout: float | None = 1.0) -> dict[str, Any] | None:
        records = self._consumer.poll(timeout_ms=int((timeout or 1.0) * 1000), max_records=1)
        for _, batch in records.items():
            for record in batch:
                return record.value
        return None

    def __iter__(self):
        for record in self._consumer:
            yield record.value

    def close(self) -> None:
        self._consumer.close()


def get_consumer(topic: str, **kwargs):
    """Factory returning the consumer implementation selected by KAFKA_MODE."""
    if KAFKA_MODE == "real":
        return RealKafkaConsumer(topic, **kwargs)
    return LocalConsumer(topic, **kwargs)


def run_scoring_consumer(max_messages: int | None = None, idle_timeout: float = 2.0) -> list[dict]:
    """Consume from the transactions topic, score each message, and publish
    the result to the scores topic. Returns the list of scoring results
    produced (for demo/test purposes) and also emits them to `SCORES_TOPIC`.
    """
    import sys
    import os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.models.score import FraudScorer
    from src.streaming import SCORES_TOPIC, TRANSACTIONS_TOPIC
    from src.streaming.producer import get_producer

    scorer = FraudScorer()
    consumer = get_consumer(TRANSACTIONS_TOPIC)
    score_producer = get_producer(client_id="scoring-service-producer")

    results = []
    n_flagged = 0
    while True:
        if max_messages is not None and len(results) >= max_messages:
            break
        txn = consumer.poll(timeout=idle_timeout)
        if txn is None:
            print(f"[consumer] no message within {idle_timeout}s, stopping.")
            break

        result = scorer.score_transaction(txn)
        score_producer.send(SCORES_TOPIC, json.loads(result.to_json()))
        results.append(json.loads(result.to_json()))
        if result.is_flagged:
            n_flagged += 1
        flag_str = "FLAGGED" if result.is_flagged else "ok"
        print(
            f"[consumer] txn={result.transaction_id} card={result.card_id} "
            f"p(fraud)={result.fraud_probability:.4f} -> {flag_str}"
        )

    print(f"[consumer] processed {len(results)} messages, flagged {n_flagged} as fraud "
          f"(threshold={scorer.threshold:.4f})")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the streaming fraud-scoring consumer.")
    parser.add_argument("--max-messages", type=int, default=None)
    parser.add_argument("--idle-timeout", type=float, default=2.0)
    args = parser.parse_args()

    print(f"[consumer] KAFKA_MODE={KAFKA_MODE}")
    run_scoring_consumer(max_messages=args.max_messages, idle_timeout=args.idle_timeout)
