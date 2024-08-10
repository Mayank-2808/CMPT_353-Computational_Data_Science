import sys
from pyspark.sql import SparkSession, functions, types, Row
import string, re
import math

spark = SparkSession.builder.appName('wordcount').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 5)
assert spark.version >= '2.3'

def main(in_directory, out_directory):
    
    data = spark.read.text(in_directory)
    wordbreak = r'[%s\s]+' % (re.escape(string.punctuation),)  

    data = data.withColumn('word', functions.explode(functions.split('value', wordbreak)))
    data.cache()

    data = data.withColumn('word', functions.lower(data['word']))
    data = data.select('word')

    data = data.filter(data['word'] != '')
    data = data.groupBy('word').agg(functions.count(data['word']))
    #data.show()

    data = data.sort(functions.desc('count(word)'), functions.asc('word'))
    data.show()
    data.write.csv(out_directory, mode = "overwrite")

if __name__=='__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    main(in_directory, out_directory)
