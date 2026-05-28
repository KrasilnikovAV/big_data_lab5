import argparse
import json
import shutil
from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col, count, regexp_replace, row_number
from pyspark.sql.window import Window


FEATURES = [
    "energy-kcal_100g",
    "fat_100g",
    "carbohydrates_100g",
    "sugars_100g",
    "proteins_100g",
    "salt_100g",
]

META_COLS = ["product_name", "brands"]


def parse_args():
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
        help="Directory for metrics, summaries, examples and saved model.",
    )
    return parser.parse_args()


def build_spark():
    spark = (
        SparkSession.builder
        .appName("OpenFoodFactsKMeans")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def remove_if_exists(path):
    target = Path(path)
    if target.exists():
        shutil.rmtree(target)


def write_single_csv(df, path):
    remove_if_exists(path)
    (
        df.coalesce(1)
        .write
        .mode("overwrite")
        .option("header", True)
        .csv(str(path))
    )


def load_and_clean(spark, input_path, sep):
    df = (
        spark.read
        .option("header", True)
        .option("sep", sep)
        .csv(input_path)
    )

    df = df.select(*(META_COLS + FEATURES))

    for feature in FEATURES:
        df = df.withColumn(
            feature,
            regexp_replace(col(feature), ",", ".").cast("double"),
        )

    df_clean = df.dropna(subset=FEATURES).where(
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

    return df, df_clean


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    spark = build_spark()

    source_df, clean_df = load_and_clean(spark, args.input, args.sep)
    rows_before = source_df.count()
    rows_after_cleaning = clean_df.count()

    training_df = clean_df
    if args.sample_fraction < 1.0:
        training_df = clean_df.sample(
            withReplacement=False,
            fraction=args.sample_fraction,
            seed=args.seed,
        )

    rows_for_training = training_df.count()

    assembler = VectorAssembler(
        inputCols=FEATURES,
        outputCol="raw_features",
    )
    scaler = StandardScaler(
        inputCol="raw_features",
        outputCol="features",
        withStd=True,
        withMean=True,
    )
    kmeans = KMeans(
        k=args.k,
        seed=args.seed,
        featuresCol="features",
        predictionCol="cluster",
    )

    pipeline = Pipeline(stages=[assembler, scaler, kmeans])
    model = pipeline.fit(training_df)
    predictions = model.transform(training_df)

    evaluator = ClusteringEvaluator(
        featuresCol="features",
        predictionCol="cluster",
        metricName="silhouette",
    )
    silhouette = evaluator.evaluate(predictions)

    cluster_sizes = predictions.groupBy("cluster").count().orderBy("cluster")
    cluster_means = (
        predictions
        .groupBy("cluster")
        .agg(
            count("*").alias("count"),
            *[avg(feature).alias(f"avg_{feature}") for feature in FEATURES],
        )
        .orderBy("cluster")
    )

    window = Window.partitionBy("cluster").orderBy("product_name")
    examples = (
        predictions
        .select("cluster", "product_name", "brands", *FEATURES)
        .where(col("product_name").isNotNull())
        .withColumn("rn", row_number().over(window))
        .where(col("rn") <= 5)
        .drop("rn")
        .orderBy("cluster", "product_name")
    )

    model_path = output_dir / "model"
    remove_if_exists(model_path)
    model.write().overwrite().save(str(model_path))

    write_single_csv(cluster_sizes, output_dir / "cluster_sizes")
    write_single_csv(cluster_means, output_dir / "cluster_means")
    write_single_csv(examples, output_dir / "cluster_examples")

    metrics = {
        "input": args.input,
        "k": args.k,
        "seed": args.seed,
        "sample_fraction": args.sample_fraction,
        "features": FEATURES,
        "rows_before_cleaning": rows_before,
        "rows_after_cleaning": rows_after_cleaning,
        "rows_for_training": rows_for_training,
        "silhouette": silhouette,
        "model_path": str(model_path),
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("Rows before cleaning:", rows_before)
    print("Rows after cleaning:", rows_after_cleaning)
    print("Rows for training:", rows_for_training)
    print("KMeans k:", args.k)
    print("Silhouette:", round(silhouette, 4))
    print("Model saved to:", model_path)
    print("Cluster sizes:")
    cluster_sizes.show()
    print("Cluster means:")
    cluster_means.show(truncate=False)
    print("Examples:")
    examples.show(args.k * 5, truncate=False)

    spark.stop()


if __name__ == "__main__":
    main()
