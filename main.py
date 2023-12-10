from pyspark.sql import SparkSession, Row, functions as F
from pyspark.sql.window import Window
from datetime import timedelta
from io import TextIOWrapper
import zipfile
import shutil
import csv
import os


def createDF(spark, file):
    required_header = {"start_time","end_time","from_station_name","gender","birthyear","tripduration"}
    reader = csv.reader(TextIOWrapper(file))
    header = next(reader)
    if required_header.issubset(header):
        rows = [Row(*line) for line in reader]
        return spark.createDataFrame(rows, header)
    return None

def get_average_trip_duration(df) -> int:
    df2 = df.withColumn("time_diff", F.col("end_time").cast("long") - F.col("start_time").cast("long"))    
    return df2.agg(F.avg("time_diff")).first()[0]

def get_trips_per_day(df):
    df2 = df.withColumn("start_date", F.to_date(F.col("start_time")))
    df2 = df2.groupBy("start_date").count()
    rows = df2.select("start_date", "count").collect()
    return {"day": [str(row["start_date"]) for row in rows],
            "count": [row["count"] for row in rows]}

def get_top_start_station_by_month(spark, df):
    df = df.withColumn("month", F.month(F.col("start_time")))
    station_counts = df.groupBy("month", "from_station_name").count()
    popular_stations = station_counts.groupBy("month").agg(F.max("count").alias("max_count"))
    # result = popular_stations.join(station_counts, (station_counts.month == popular_stations.month) & (station_counts.count == popular_stations.max_count))
    station_counts.createOrReplaceTempView("station_counts")
    popular_stations.createOrReplaceTempView("popular_stations")
    df_res = spark.sql("""
        SELECT sc.month, sc.from_station_name, sc.count
        FROM station_counts sc
        JOIN (
            SELECT month, max_count
            FROM popular_stations
        ) ps ON sc.month = ps.month AND sc.count = ps.max_count
    """)
    rows = df_res.select("month", "from_station_name").collect()
    return {"month" : [row["month"] for row in rows],
        "from_station_name" : [row["from_station_name"] for row in rows]}

def get_top3_station_for_last_2_week(df) -> dict:
    df2 = df.withColumn("start_date", F.to_date(F.col("start_time")))
    
    latest_date = df2.agg(F.max("start_date")).collect()[0][0]
    two_weeks_ago = latest_date - timedelta(weeks=2)
    df2 = df2.filter(df2.start_date >= two_weeks_ago)
    df_count = df2.groupBy("start_date", "from_station_name").count()
    window = Window.partitionBy("start_date").orderBy(F.col("count").desc())
    top_stations = df_count.withColumn("rank", F.rank().over(window)).filter(F.col("rank") <= 3)
    rdd = top_stations.rdd
    mapped_rdd = rdd.map(lambda row: (row.start_date, row.from_station_name))
    grouped_rdd = mapped_rdd.groupByKey().mapValues(list)
    result_dict = grouped_rdd.collectAsMap()
    return result_dict

def get_Men_or_women_drive_longer_on_average(df) -> str:
    df2 = df.groupBy("gender").count()
    return df2.orderBy(F.desc("count")).select("gender").first()["gender"]

def get_top10_age_by_longest_and_shortest_trip(df):
    df2 = df.withColumn("start_date", F.to_date(F.col("start_time")))
    df2 = df2.withColumn("age", F.year("start_date") - df["birthyear"])
    age_duration = df2.groupBy("age").agg(F.mean("tripduration").alias("avg_duration"))
    top_ages = age_duration.sort(F.desc("avg_duration")).limit(10)
    top_ages2 = age_duration.sort(F.asc("avg_duration")).limit(10)
    return {"longest" : [row['age'] for row in top_ages.collect()],
            "shortest" : [row['age'] for row in top_ages2.collect()]}


def getData(spark, df):
    df = df.withColumn("start_time", F.to_timestamp(F.col("start_time"), 'yyyy-MM-dd HH:mm:ss'))
    df = df.withColumn("end_time", F.to_timestamp(F.col("end_time"), 'yyyy-MM-dd HH:mm:ss'))
    return {"1_average_trip_duration" : get_average_trip_duration(df),
        "2_trips_per_day" : get_trips_per_day(df),
        "3_top_start_station_by_month" : get_top_start_station_by_month(spark, df),
        "4_top3_station_for_last_2_week" : get_top3_station_for_last_2_week(df),
        "5_men_or_women_drive_longer_on_average" : get_Men_or_women_drive_longer_on_average(df),
        "6_top10_age_by_longest_and_shortest_trip" : get_top10_age_by_longest_and_shortest_trip(df) }


def create_report(sub_folder, infos):
    os.makedirs(sub_folder)
    for key, value in infos.items():
        file_path = os.path.join(sub_folder, key+".csv")
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            if isinstance(value, dict):
                writer.writerow(list(value.keys()))
                values = list(value.values())
                for row in zip(*values):
                    writer.writerow(row)
            else:
                writer.writerow([value])

def main():
    spark = SparkSession.builder.appName("Exercise6").enableHiveSupport().getOrCreate()
    reports_folder = 'reports'
    zip_file_paths = []
    for root, dirs, files in os.walk("data"):
        zip_file_paths.extend([os.path.join(root, file) for file in files if file.endswith(".zip")])
    
    for zip_file_path in zip_file_paths:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            all_files = zip_ref.namelist()
            csv_file_paths = [file for file in all_files if file.endswith('.csv') and not file.count("__MACOSX")]
            for csv_file_name in csv_file_paths:
                with zip_ref.open(csv_file_name) as file:
                    df = createDF(spark, file)
                    if df is None:
                        break
                    info = getData(spark, df)
                    if os.path.exists(reports_folder):
                        shutil.rmtree(reports_folder)
                    os.makedirs(reports_folder)
                    create_report(os.path.join(reports_folder, file.name), info)


if __name__ == "__main__":
    main()
