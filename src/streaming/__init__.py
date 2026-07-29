"""
Streaming backend selection.

This package emulates a Kafka producer/consumer pair behind one interface,
switchable via the `KAFKA_MODE` environment variable:

    KAFKA_MODE=local   (default) -> queue.Queue + threading backend, no broker
                                      required. Good for local dev, tests,
                                      and this sandbox (no live Kafka).
    KAFKA_MODE=real               -> kafka-python backend, talking to a real
                                      Kafka broker (see docker-compose.yml for
                                      the confluentinc/cp-kafka +
                                      confluentinc/cp-zookeeper services that
                                      make up "production mode").

Both backends implement the same tiny interface:

    Producer.send(topic: str, value: dict) -> None
    Consumer.poll(timeout: float | None) -> dict | None   (blocking-ish poll)
    Consumer.__iter__ / __next__                          (streaming iteration)

so `src/streaming/producer.py` / `consumer.py` and any code that imports them
(e.g. a demo script or the Docker consumer entrypoint) does not need to know
which backend is active.
"""

import os

KAFKA_MODE = os.environ.get("KAFKA_MODE", "local").lower()

if KAFKA_MODE not in ("local", "real"):
    raise ValueError(f"Invalid KAFKA_MODE={KAFKA_MODE!r}; expected 'local' or 'real'")

TRANSACTIONS_TOPIC = "payment-transactions"
SCORES_TOPIC = "fraud-scores"

__all__ = ["KAFKA_MODE", "TRANSACTIONS_TOPIC", "SCORES_TOPIC"]
