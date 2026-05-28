import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col, count, regexp_replace, row_number
from pyspark.sql.window import Window


@dataclass
class KMeansConfig:
    input_path: str = "data/sample_openfoodfacts.csv"
    sep: str = "\t"
    k: int = 5
    sample_fraction: float = 1.0
    seed: int = 42
    output_dir: str = "outputs/openfoodfacts_kmeans"


class OpenFoodFactsKMeansTrainer:
    FEATURES = [
        "energy-kcal_100g",
        "fat_100g",
        "carbohydrates_100g",
        "sugars_100g",
        "proteins_100g",
        "salt_100g",
    ]
    META_COLS = ["product_name", "brands"]

    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.spark = None

    @classmethod
    def from_args(cls):
        parser = argparse.ArgumentParser(
            description="Train a Spark ML KMeans model on OpenFoodFacts nutrition data."
        )
        parser.add_argument(
            "--input",
            default=KMeansConfig.input_path,
            help="Path to OpenFoodFacts TSV/CSV file.",
        )
        parser.add_argument(
            "--sep",
            default=KMeansConfig.sep,
            help="Input delimiter. OpenFoodFacts exports use tab by default.",
        )
        parser.add_argument(
            "--k",
            type=int,
            default=KMeansConfig.k,
            help="Number of clusters.",
        )
        parser.add_argument(
            "--sample-fraction",
            type=float,
            default=KMeansConfig.sample_fraction,
            help="Fraction of cleaned rows to use for training.",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=KMeansConfig.seed,
            help="Random seed for sampling and KMeans.",
        )
        parser.add_argument(
            "--output-dir",
            default=KMeansConfig.output_dir,
            help="Directory for metrics, summaries, examples and saved model.",
        )
        args = parser.parse_args()
        return cls(
            KMeansConfig(
                input_path=args.input,
                sep=args.sep,
                k=args.k,
                sample_fraction=args.sample_fraction,
                seed=args.seed,
                output_dir=args.output_dir,
            )
        )

    def run(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.spark = self.build_spark()
        try:
            source_df, clean_df = self.load_and_clean()
            rows_before = source_df.count()
            rows_after_cleaning = clean_df.count()

            training_df = self.build_training_df(clean_df)
            rows_for_training = training_df.count()

            model, predictions = self.train(training_df)
            silhouette = self.evaluate(predictions)
            cluster_sizes, cluster_means, examples = self.summarize(predictions)

            model_path = self.save_outputs(
                model=model,
                cluster_sizes=cluster_sizes,
                cluster_means=cluster_means,
                examples=examples,
            )
            self.write_metrics(
                rows_before=rows_before,
                rows_after_cleaning=rows_after_cleaning,
                rows_for_training=rows_for_training,
                silhouette=silhouette,
                model_path=model_path,
            )
            self.print_results(
                rows_before=rows_before,
                rows_after_cleaning=rows_after_cleaning,
                rows_for_training=rows_for_training,
                silhouette=silhouette,
                model_path=model_path,
                cluster_sizes=cluster_sizes,
                cluster_means=cluster_means,
                examples=examples,
            )
        finally:
            self.spark.stop()

    def build_spark(self):
        spark = (
            SparkSession.builder
            .appName("OpenFoodFactsKMeans")
            .master("local[*]")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("ERROR")
        return spark

    def load_and_clean(self):
        df = (
            self.spark.read
            .option("header", True)
            .option("sep", self.config.sep)
            .csv(self.config.input_path)
        )

        df = df.select(*(self.META_COLS + self.FEATURES))

        for feature in self.FEATURES:
            df = df.withColumn(
                feature,
                regexp_replace(col(feature), ",", ".").cast("double"),
            )

        clean_df = df.dropna(subset=self.FEATURES).where(
            (col("energy-kcal_100g") >= 0) &
            (col("energy-kcal_100g") <= 1000) &
            (col("fat_100g") >= 0) &
            (col("fat_100g") <= 100) &
            (col("carbohydrates_100g") >= 0) &
            (col("carbohydrates_100g") <= 100) &
            (col("sugars_100g") >= 0) &
            (col("sugars_100g") <= 100) &
            (col("proteins_100g") >= 0) &
            (col("proteins_100g") <= 100) &
            (col("salt_100g") >= 0) &
            (col("salt_100g") <= 20)
        )

        return df, clean_df

    def build_training_df(self, clean_df):
        if self.config.sample_fraction >= 1.0:
            return clean_df
        return clean_df.sample(
            withReplacement=False,
            fraction=self.config.sample_fraction,
            seed=self.config.seed,
        )

    def train(self, training_df):
        assembler = VectorAssembler(
            inputCols=self.FEATURES,
            outputCol="raw_features",
        )
        scaler = StandardScaler(
            inputCol="raw_features",
            outputCol="features",
            withStd=True,
            withMean=True,
        )
        kmeans = KMeans(
            k=self.config.k,
            seed=self.config.seed,
            featuresCol="features",
            predictionCol="cluster",
        )

        pipeline = Pipeline(stages=[assembler, scaler, kmeans])
        model = pipeline.fit(training_df)
        return model, model.transform(training_df)

    def evaluate(self, predictions):
        evaluator = ClusteringEvaluator(
            featuresCol="features",
            predictionCol="cluster",
            metricName="silhouette",
        )
        return evaluator.evaluate(predictions)

    def summarize(self, predictions):
        cluster_sizes = predictions.groupBy("cluster").count().orderBy("cluster")
        cluster_means = (
            predictions
            .groupBy("cluster")
            .agg(
                count("*").alias("count"),
                *[avg(feature).alias(f"avg_{feature}") for feature in self.FEATURES],
            )
            .orderBy("cluster")
        )

        window = Window.partitionBy("cluster").orderBy("product_name")
        examples = (
            predictions
            .select("cluster", "product_name", "brands", *self.FEATURES)
            .where(col("product_name").isNotNull())
            .withColumn("rn", row_number().over(window))
            .where(col("rn") <= 5)
            .drop("rn")
            .orderBy("cluster", "product_name")
        )

        return cluster_sizes, cluster_means, examples

    def save_outputs(self, model, cluster_sizes, cluster_means, examples):
        model_path = self.output_dir / "model"
        self.remove_if_exists(model_path)
        model.write().overwrite().save(str(model_path))

        self.write_single_csv(cluster_sizes, self.output_dir / "cluster_sizes")
        self.write_single_csv(cluster_means, self.output_dir / "cluster_means")
        self.write_single_csv(examples, self.output_dir / "cluster_examples")

        return model_path

    def write_metrics(
        self,
        rows_before,
        rows_after_cleaning,
        rows_for_training,
        silhouette,
        model_path,
    ):
        metrics = {
            "input": self.config.input_path,
            "k": self.config.k,
            "seed": self.config.seed,
            "sample_fraction": self.config.sample_fraction,
            "features": self.FEATURES,
            "rows_before_cleaning": rows_before,
            "rows_after_cleaning": rows_after_cleaning,
            "rows_for_training": rows_for_training,
            "silhouette": silhouette,
            "model_path": str(model_path),
        }

        with (self.output_dir / "metrics.json").open("w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2)

    def print_results(
        self,
        rows_before,
        rows_after_cleaning,
        rows_for_training,
        silhouette,
        model_path,
        cluster_sizes,
        cluster_means,
        examples,
    ):
        print("Rows before cleaning:", rows_before)
        print("Rows after cleaning:", rows_after_cleaning)
        print("Rows for training:", rows_for_training)
        print("KMeans k:", self.config.k)
        print("Silhouette:", round(silhouette, 4))
        print("Model saved to:", model_path)
        print("Cluster sizes:")
        cluster_sizes.show()
        print("Cluster means:")
        cluster_means.show(truncate=False)
        print("Examples:")
        examples.show(self.config.k * 5, truncate=False)

    def write_single_csv(self, df, path):
        self.remove_if_exists(path)
        (
            df.coalesce(1)
            .write
            .mode("overwrite")
            .option("header", True)
            .csv(str(path))
        )

    @staticmethod
    def remove_if_exists(path):
        target = Path(path)
        if target.exists():
            shutil.rmtree(target)


if __name__ == "__main__":
    OpenFoodFactsKMeansTrainer.from_args().run()
