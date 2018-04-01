from pyspark.ml.fpm import FPGrowth
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import udf, size
from pyspark.sql.types import ArrayType, StringType
from pyspark import SparkContext, SparkConf


def merge(*c):
    merged = sorted(set(c))
    if len(merged) == 1:
        return merged[0]
    else:
        return "[{0}]".format(",".join(merged))


conf = SparkConf().set("local", "false")
sc = SparkContext(appName="PythonStatusAPIDemo", conf=conf)

rdd = sc.textFile('games copy.csv')

d_split = rdd.map(lambda x:x.split(','))
d_frame = d_split.map(lambda x: Row(id=x[0], items=x[11:]))

spark = SparkSession.builder.appName('data').getOrCreate()
df = spark.createDataFrame(d_frame)

fpGrowth = FPGrowth(itemsCol='items', minSupport=0.00001)
model = fpGrowth.fit(df)

df = model.freqItemsets

df = df.withColumn('length', size(df.items))
df = df.orderBy(df.length.desc(), df.freq.desc()).select('items', 'freq')
df.show()


# df = df.select(df.items.cast('array').alias('item'))
# fpGrowth = FPGrowth(itemsCol="items", minSupport=0.05, minConfidence=0.1)
# model = fpGrowth.fit(df)
#
# df = model.freqItemsets
# df = df.withColumn('length', size(df['items']))
# df = df.orderBy(df.freq.desc(), df.length.desc()).select('items', 'freq')
#
# df = df.show()
