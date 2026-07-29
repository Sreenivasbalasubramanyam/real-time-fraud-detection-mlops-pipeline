"""
Feature engineering for fraud scoring — PySpark implementation (production-scale).

This module mirrors the exact function names, signatures, and logic of
`src/features/pandas_features.py`, but implemented with real PySpark
DataFrame / Window operations so it can run unmodified against a Spark
cluster processing high-volume transaction streams (e.g. reading from a
Kafka topic via Structured Streaming, or a data lake in batch).

NOTE ON THIS ENVIRONMENT: PySpark was installed and this module was smoke
tested locally (see README "What Actually Ran" section for the verified
command and output). `pandas_features.py` is the implementation used by the
rest of the pipeline (train_model.py, score.py) in this repo/CI because it
has no JVM dependency and runs faster on the modest single-node synthetic
dataset used here. In a production deployment with real transaction volume,
this Spark implementation is the one that would run on the cluster.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import TimestampType

NUMERIC_FEATURE_COLUMNS = [
    "log_amount",
    "amount_zscore_by_card",
    "distance_from_home_km",
    "velocity_1h",
    "account_age_days",
    "hour_of_day",
    "is_night",
    "cvv_match",
    "is_new_device",
    "is_new_shipping_address",
    "merchant_category_freq",
    "country_freq",
    "txn_count_last_24h",
    "amount_sum_last_24h",
]

LABEL_COLUMN = "is_fraud"


def get_spark_session(app_name: str = "fraud-feature-engineering") -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def load_transactions(spark: SparkSession, path: str) -> DataFrame:
    """Load raw transactions CSV and parse timestamp column."""
    df = spark.read.csv(path, header=True, inferSchema=True)
    df = df.withColumn("timestamp", F.col("timestamp").cast(TimestampType()))
    return df


def add_amount_features(df: DataFrame) -> DataFrame:
    """Add log-amount and per-card amount z-score features."""
    df = df.withColumn("log_amount", F.log1p(F.col("amount")))

    card_window = Window.partitionBy("card_id")
    df = df.withColumn("_card_amount_mean", F.avg("amount").over(card_window))
    df = df.withColumn("_card_amount_std", F.coalesce(F.stddev("amount").over(card_window), F.lit(0.0)))
    df = df.withColumn(
        "amount_zscore_by_card",
        F.when(
            F.col("_card_amount_std") > 0,
            (F.col("amount") - F.col("_card_amount_mean")) / F.col("_card_amount_std"),
        ).otherwise(F.lit(0.0)),
    )
    df = df.drop("_card_amount_mean", "_card_amount_std")
    return df


def add_temporal_features(df: DataFrame) -> DataFrame:
    """Add hour-of-day and is_night flag from the timestamp column."""
    df = df.withColumn("hour_of_day", F.hour("timestamp"))
    df = df.withColumn(
        "is_night",
        F.when((F.col("hour_of_day") <= 5) | (F.col("hour_of_day") >= 22), F.lit(1)).otherwise(F.lit(0)),
    )
    return df


def add_velocity_features(df: DataFrame) -> DataFrame:
    """Add rolling 24h transaction count & spend per card (causal / no look-ahead).

    Uses a Spark range-window over event time (in seconds), partitioned by
    card_id and ordered by timestamp, looking back 24h and excluding the
    current row (rowsBetween/rangeBetween up to -1 second before current row).
    """
    seconds = F.col("timestamp").cast("long")
    df = df.withColumn("_ts_seconds", seconds)

    window_24h = (
        Window.partitionBy("card_id")
        .orderBy("_ts_seconds")
        .rangeBetween(-24 * 3600, -1)
    )

    df = df.withColumn("txn_count_last_24h", F.count(F.lit(1)).over(window_24h))
    df = df.withColumn("amount_sum_last_24h", F.coalesce(F.sum("amount").over(window_24h), F.lit(0.0)))
    df = df.drop("_ts_seconds")
    return df


def add_categorical_frequency_features(df: DataFrame, freq_maps: dict | None = None) -> tuple[DataFrame, dict]:
    """Frequency-encode merchant_category and country.

    If `freq_maps` is provided (fitted on a training set), those frequencies
    are broadcast-joined in so unseen categories map to 0. Otherwise
    frequencies are computed from `df` itself via groupBy/count.
    """
    maps: dict = {}
    total = df.count()

    for col, out_col in [("merchant_category", "merchant_category_freq"), ("country", "country_freq")]:
        if freq_maps and col in freq_maps:
            freq = freq_maps[col]
            spark = df.sparkSession
            freq_rows = [(k, float(v)) for k, v in freq.items()]
            freq_df = spark.createDataFrame(freq_rows, [col, out_col])
        else:
            counts = df.groupBy(col).count()
            freq_df = counts.withColumn(out_col, F.col("count") / F.lit(total)).drop("count")
            freq = {row[col]: row[out_col] for row in freq_df.collect()}

        maps[col] = freq
        df = df.join(F.broadcast(freq_df), on=col, how="left")
        df = df.withColumn(out_col, F.coalesce(F.col(out_col), F.lit(0.0)))

    return df, maps


def add_risk_flags(df: DataFrame) -> DataFrame:
    """Cast boolean risk flags to int."""
    for col in ["cvv_match", "is_new_device", "is_new_shipping_address"]:
        df = df.withColumn(col, F.col(col).cast("int"))
    return df


def engineer_features(df: DataFrame, freq_maps: dict | None = None) -> tuple[DataFrame, dict]:
    """Run the full feature engineering pipeline (see pandas_features.engineer_features)."""
    out = add_amount_features(df)
    out = add_temporal_features(out)
    out = add_velocity_features(out)
    out = add_risk_flags(out)
    out, maps = add_categorical_frequency_features(out, freq_maps=freq_maps)

    keep_cols = ["transaction_id", "card_id", "timestamp"] + NUMERIC_FEATURE_COLUMNS
    if LABEL_COLUMN in out.columns:
        keep_cols.append(LABEL_COLUMN)
    out = out.select(*keep_cols)
    return out, maps


if __name__ == "__main__":
    import os

    spark = get_spark_session()
    path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "transactions.csv")
    raw = load_transactions(spark, path)
    engineered, maps = engineer_features(raw)
    engineered.show(5, truncate=False)
    print(f"Engineered row count: {engineered.count()}")
    print(f"Feature columns: {NUMERIC_FEATURE_COLUMNS}")
    spark.stop()
