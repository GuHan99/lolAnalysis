from pyspark.ml.fpm import FPGrowth
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import size
from pyspark import SparkContext, SparkConf


conf = SparkConf().set("local", "false")
sc = SparkContext(appName="PythonStatusAPIDemo", conf=conf)

rdd = sc.textFile('games.csv')

d_split = rdd.map(lambda x: x.split(','))
d_frame = d_split.map(lambda x: Row(id=x[0], items=x[11:30][:15:3]))
d_frame_2 = d_split.map(lambda x: Row(id=x[0], items=x[36:50][:15:3]))

spark = SparkSession.builder.appName('data').getOrCreate()
df_1 = spark.createDataFrame(d_frame)
df_2 = spark.createDataFrame(d_frame_2)

df = df_1.union(df_2)

fpGrowth = FPGrowth(itemsCol='items', minSupport=0.001)
model = fpGrowth.fit(df)

df = model.freqItemsets

df = df.withColumn('length', size(df.items))
df = df.orderBy(df.length.asc(), df.freq.desc()).select('items', 'freq').collect()
for i in df:
    print(i)

