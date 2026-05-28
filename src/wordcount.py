from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split, col, lower

spark = (
    SparkSession.builder
    .appName("WordCount")
    .master("local[*]")
    .getOrCreate()
)

df = spark.read.text("input.txt")

words = df.select(
    explode(split(lower(col("value")), r"\s+")).alias("word")
)

counts = (
    words
    .where(col("word") != "")
    .groupBy("word")
    .count()
    .orderBy(col("count").desc(), col("word"))
)

counts.show()

spark.stop()
