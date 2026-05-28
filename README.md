# Big Data Lab 5: OpenFoodFacts KMeans

Проект содержит проверочный пример Spark WordCount и модель кластеризации
продуктов OpenFoodFacts на PySpark ML.

## Состав

- `src/wordcount.py` - проверка работоспособности Spark на примере WordCount.
- `src/kmeans_openfoodfacts.py` - обучение KMeans-модели на пищевой ценности продуктов.
- `data/sample_openfoodfacts.csv` - подготовленная выборка OpenFoodFacts.
- `outputs/openfoodfacts_kmeans/` - метрики, примеры кластеров и сохраненная модель.
- `report.md` - отчет о проделанной работе.
- `Dockerfile`, `docker-compose.yml` - запуск лабораторной в контейнере.
- `build_distribution.sh` - сборка zip-дистрибутива.

## Требования

- Python 3.10+
- Java 17+ для локального запуска через `requirements.txt`
- Docker 24+ для контейнерного запуска
- PySpark 4.1.2

Установка зависимостей:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Если на macOS Java 17 установлена через Homebrew, перед запуском Spark можно
выставить окружение так:

```bash
export JAVA_HOME=/opt/homebrew/opt/openjdk@17
export PATH="$JAVA_HOME/bin:$PATH"
```

## Проверка Spark

```bash
.venv/bin/python src/wordcount.py
```

## Запуск в Docker

Docker-образ собирается на базе `apache/spark-py:v3.4.0`, где Spark, PySpark и
Java уже установлены.

Сборка образа:

```bash
docker build -t big-data-lab5-openfoodfacts-kmeans .
```

Проверка Spark WordCount:

```bash
docker run --rm big-data-lab5-openfoodfacts-kmeans /opt/spark/bin/spark-submit src/wordcount.py
```

Обучение модели:

```bash
docker run --rm \
  -v "$PWD/outputs:/app/outputs" \
  big-data-lab5-openfoodfacts-kmeans
```

То же через Docker Compose:

```bash
docker compose run --rm wordcount
docker compose run --rm kmeans
```

## Обучение модели

```bash
.venv/bin/python src/kmeans_openfoodfacts.py \
  --input data/sample_openfoodfacts.csv \
  --k 5 \
  --output-dir outputs/openfoodfacts_kmeans
```

Для слабой машины можно уменьшить объем данных после очистки:

```bash
.venv/bin/python src/kmeans_openfoodfacts.py --sample-fraction 0.2
```

## Сборка дистрибутива

```bash
./build_distribution.sh
```

Готовый архив создается в каталоге `dist/`.
