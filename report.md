# Отчет

## Задача

Настроить Spark, проверить запуск WordCount и обучить KMeans-модель
кластеризации продуктов OpenFoodFacts.

Репозиторий: https://github.com/KrasilnikovAV/big_data_lab5.git

## Среда

- Python 3
- PySpark 4.1.2 для локального запуска
- Docker-образ: `apache/spark-py:v3.4.0`
- Режим Spark: `local[*]`

## Данные

Источник: https://world.openfoodfacts.org/data

В работе используется подготовленная выборка:

```text
data/sample_openfoodfacts.csv
```

## Предобработка

Использованы признаки:

- `energy-kcal_100g`
- `fat_100g`
- `carbohydrates_100g`
- `sugars_100g`
- `proteins_100g`
- `salt_100g`

Выполнено удаление пропусков, приведение признаков к `double`, фильтрация
некорректных значений и масштабирование через `StandardScaler`.

## Проверка Spark

```bash
.venv/bin/python wordcount.py
```

## Обучение

```bash
.venv/bin/python kmeans_openfoodfacts.py \
  --input data/sample_openfoodfacts.csv \
  --k 5 \
  --output-dir outputs/openfoodfacts_kmeans
```

Docker:

```bash
docker build -t big-data-lab5-openfoodfacts-kmeans .
docker run --rm big-data-lab5-openfoodfacts-kmeans /opt/spark/bin/spark-submit wordcount.py
docker run --rm -v "$PWD/outputs:/app/outputs" big-data-lab5-openfoodfacts-kmeans
```

## Результаты

- строк до очистки: 10000
- строк после очистки: 1643
- кластеров: 5
- silhouette score: 0.5682
- модель сохранена в `outputs/openfoodfacts_kmeans/model`

В zip-дистрибутив включены код, выборка данных, Docker-файлы, результаты
обучения и сохраненная модель.
