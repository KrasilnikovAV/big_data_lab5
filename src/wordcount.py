import argparse

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, lower, split


class WordCountJob:
    def __init__(self, input_path):
        self.input_path = input_path
        self.spark = None

    @classmethod
    def from_args(cls):
        parser = argparse.ArgumentParser(description="Run Spark WordCount example.")
        parser.add_argument(
            "--input",
            default="data/data_for_wordcount.txt",
            help="Path to text file for WordCount.",
        )
        args = parser.parse_args()
        return cls(input_path=args.input)

    def build_spark(self):
        spark = (
            SparkSession.builder
            .appName("WordCount")
            .master("local[*]")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("ERROR")
        return spark

    def run(self):
        self.spark = self.build_spark()
        try:
            counts = self.count_words()
            counts.show()
        finally:
            self.spark.stop()

    def count_words(self):
        df = self.spark.read.text(self.input_path)
        words = df.select(
            explode(split(lower(col("value")), r"\s+")).alias("word")
        )
        return (
            words
            .where(col("word") != "")
            .groupBy("word")
            .count()
            .orderBy(col("count").desc(), col("word"))
        )


if __name__ == "__main__":
    WordCountJob.from_args().run()
