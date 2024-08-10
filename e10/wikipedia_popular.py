import sys
from pyspark.sql import SparkSession, functions, types
import re

spark = SparkSession.builder.appName('wikipedia popular').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 5)  # Ensure Python version 3.5 or higher
assert spark.version >= '2.3'  # Ensure Spark version 2.3 or higher

wiki_schema = types.StructType([
    types.StructField('lang', types.StringType()),
    types.StructField('title', types.StringType()),
    types.StructField('views', types.LongType()),
    types.StructField('bytes', types.LongType()),
])

def extract_date_from_path(path):
    date_match = re.search(r'\d{8}-\d{2}', path)
    return date_match.group(0) if date_match else None

def main(in_directory, out_directory):

    data = spark.read.csv(in_directory, schema=wiki_schema, sep=' ').withColumn('filename', functions.input_file_name())

    # Filter data for English language and remove irrelevant pages
    data = data.filter(data['lang'] == 'en')
    data = data.filter(data['title'] != 'Main_Page')
    data = data.filter(~data['title'].startswith('Special:'))

    extract_date_udf = functions.udf(lambda path: extract_date_from_path(path), returnType=types.StringType())
    data = data.withColumn('time', extract_date_udf(data.filename))

    # Cache the DataFrame for optimization
    data = data.cache()

    # Grouping data by time and find the maximum views
    groups = data.groupBy('time')
    max_views = groups.agg(functions.max(data['views']).alias('views'))

    # Joining to find corresponding title for maximum views
    data_join = max_views.join(data, on=['views', 'time'])
    output = data_join.drop('lang', 'bytes', 'filename').select('time', 'title', 'views')

    # Sorting the output DataFrame
    output = output.sort('time', 'title')
    output.show()

    output.write.csv(out_directory + '-wikipedia', mode='overwrite')

if __name__ == '__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    main(in_directory, out_directory)
