import argparse
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col, count, regexp_replace


FEATURE_COLUMNS = [
    "energy-kcal_100g",
    "fat_100g",
    "carbohydrates_100g",
    "sugars_100g",
    "proteins_100g",
    "salt_100g",
]
META_COLUMNS = ["product_name", "brands"]


class KMeansJob:
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = args.run_id or self.build_run_id()
        self.spark = None

    @classmethod
    def from_args(cls):
        parser = argparse.ArgumentParser(
            description="Train a Spark ML KMeans model on OpenFoodFacts nutrition data."
        )
        parser.add_argument(
            "--input",
            default="data/sample_openfoodfacts.csv",
            help="Path to OpenFoodFacts TSV/CSV file.",
        )
        parser.add_argument(
            "--sep",
            default="\t",
            help="Input delimiter. OpenFoodFacts exports use tab by default.",
        )
        parser.add_argument(
            "--k",
            type=int,
            default=5,
            help="Number of clusters.",
        )
        parser.add_argument(
            "--sample-fraction",
            type=float,
            default=1.0,
            help="Fraction of cleaned rows to use for training.",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Random seed for sampling and KMeans.",
        )
        parser.add_argument(
            "--output-dir",
            default="outputs/openfoodfacts_kmeans",
            help="Directory for metrics, summaries and saved model.",
        )
        parser.add_argument(
            "--run-id",
            default=None,
            help="Optional run identifier for metrics output.",
        )
        parser.add_argument(
            "--spark-master",
            default="local[*]",
            help="Spark master URL.",
        )
        return cls(parser.parse_args())

    def run(self):
        self.spark = self.build_spark()
        try:
            source_df = self.load_source_df()
            selected_df, clean_df = self.clean_product_nutrition(source_df)
            rows_before = selected_df.count()
            rows_after_cleaning = clean_df.count()

            training_df = self.sample_training_rows(clean_df)
            rows_for_training = training_df.count()
            if rows_for_training < self.args.k:
                raise ValueError(
                    f"Need at least {self.args.k} cleaned rows for KMeans, got {rows_for_training}."
                )

            model, predictions = self.train(training_df)
            silhouette = self.evaluate(predictions)
            cluster_stats = self.summarize(predictions)

            model_path = self.save_local_outputs(model, cluster_stats)
            metrics = self.build_metrics(
                rows_before=rows_before,
                rows_after_cleaning=rows_after_cleaning,
                rows_for_training=rows_for_training,
                silhouette=silhouette,
                model_path=model_path,
            )
            self.write_metrics_file(metrics)
            self.print_results(metrics, cluster_stats)
        finally:
            if self.spark is not None:
                self.spark.stop()

    def build_spark(self):
        spark = (
            SparkSession.builder
            .appName("OpenFoodFactsKMeans")
            .master(self.args.spark_master)
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("ERROR")
        return spark

    def load_source_df(self):
        return (
            self.spark.read
            .option("header", True)
            .option("sep", self.args.sep)
            .csv(self.args.input)
        )

    def clean_product_nutrition(self, source_df):
        missing = sorted(set(META_COLUMNS + FEATURE_COLUMNS) - set(source_df.columns))
        if missing:
            raise ValueError(
                f"Input file '{self.args.input}' is missing columns: {', '.join(missing)}"
            )

        df = source_df.select(*(META_COLUMNS + FEATURE_COLUMNS))
        for feature in FEATURE_COLUMNS:
            df = df.withColumn(
                feature,
                regexp_replace(col(feature), ",", ".").cast("double"),
            )

        clean_df = df.dropna(subset=FEATURE_COLUMNS).where(
            (col("energy-kcal_100g") >= 0)
            & (col("energy-kcal_100g") <= 1000)
            & (col("fat_100g") >= 0)
            & (col("fat_100g") <= 100)
            & (col("carbohydrates_100g") >= 0)
            & (col("carbohydrates_100g") <= 100)
            & (col("sugars_100g") >= 0)
            & (col("sugars_100g") <= 100)
            & (col("proteins_100g") >= 0)
            & (col("proteins_100g") <= 100)
            & (col("salt_100g") >= 0)
            & (col("salt_100g") <= 20)
        )
        return df, clean_df

    def sample_training_rows(self, clean_df):
        if self.args.sample_fraction >= 1.0:
            return clean_df
        return clean_df.sample(
            withReplacement=False,
            fraction=self.args.sample_fraction,
            seed=self.args.seed,
        )

    def train(self, training_df):
        pipeline = Pipeline(
            stages=[
                VectorAssembler(inputCols=FEATURE_COLUMNS, outputCol="raw_features"),
                StandardScaler(
                    inputCol="raw_features",
                    outputCol="features",
                    withStd=True,
                    withMean=True,
                ),
                KMeans(
                    k=self.args.k,
                    seed=self.args.seed,
                    featuresCol="features",
                    predictionCol="cluster_id",
                ),
            ]
        )
        model = pipeline.fit(training_df)
        return model, model.transform(training_df)

    def evaluate(self, predictions):
        evaluator = ClusteringEvaluator(
            featuresCol="features",
            predictionCol="cluster_id",
            metricName="silhouette",
        )
        return evaluator.evaluate(predictions)

    def summarize(self, predictions):
        return (
            predictions.groupBy("cluster_id")
            .agg(
                count("*").alias("cluster_count"),
                *[avg(feature).alias(f"avg_{feature}") for feature in FEATURE_COLUMNS],
            )
            .orderBy("cluster_id")
        )

    def save_local_outputs(self, model, cluster_stats):
        model_path = self.output_dir / "model"
        self.remove_if_exists(model_path)
        model.write().overwrite().save(str(model_path))

        self.write_single_csv(cluster_stats, self.output_dir / "cluster_stats")
        return model_path

    def build_metrics(
        self,
        rows_before,
        rows_after_cleaning,
        rows_for_training,
        silhouette,
        model_path,
    ):
        return {
            "run_id": self.run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "input": self.args.input,
            "k": self.args.k,
            "seed": self.args.seed,
            "sample_fraction": self.args.sample_fraction,
            "features": FEATURE_COLUMNS,
            "rows_before_cleaning": rows_before,
            "rows_after_cleaning": rows_after_cleaning,
            "rows_for_training": rows_for_training,
            "silhouette": silhouette,
            "model_path": str(model_path),
        }

    def write_metrics_file(self, metrics):
        with (self.output_dir / "metrics.json").open("w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2)

    def print_results(self, metrics, cluster_stats):
        print("Run ID:", metrics["run_id"])
        print("Input file:", metrics["input"])
        print("Rows before cleaning:", metrics["rows_before_cleaning"])
        print("Rows after cleaning:", metrics["rows_after_cleaning"])
        print("Rows for training:", metrics["rows_for_training"])
        print("KMeans k:", metrics["k"])
        print("Silhouette:", round(metrics["silhouette"], 4))
        print("Model saved to:", metrics["model_path"])
        print("Cluster statistics:")
        cluster_stats.show(truncate=False)

    def write_single_csv(self, df, path):
        self.remove_if_exists(path)
        df.coalesce(1).write.mode("overwrite").option("header", True).csv(str(path))

    def remove_if_exists(self, path):
        target = Path(path)
        if target.exists():
            shutil.rmtree(target)

    @staticmethod
    def build_run_id():
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def run_kmeans(args):
    KMeansJob(args).run()


if __name__ == "__main__":
    KMeansJob.from_args().run()
