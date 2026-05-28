FROM apache/spark-py:v3.4.0

USER root
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYSPARK_PYTHON=python3 \
    PYSPARK_DRIVER_PYTHON=python3 \
    PYTHONPATH=/opt/spark/python:/opt/spark/python/lib/py4j-0.10.9.7-src.zip:/opt/spark/python/lib/pyspark.zip

COPY . .
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-numpy \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p outputs \
    && chmod -R 777 /app

CMD ["/opt/spark/bin/spark-submit", "src/kmeans_openfoodfacts.py", "--input", "data/sample_openfoodfacts.csv", "--k", "5", "--output-dir", "outputs/openfoodfacts_kmeans"]
