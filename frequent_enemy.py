from pyspark.ml.fpm import FPGrowth
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import size
from pyspark import SparkContext, SparkConf


conf = SparkConf().set("local", "false")
sc = SparkContext(appName="PythonStatusAPIDemo", conf=conf)

rdd = sc.textFile('games.csv')


def mark(x):
    x[36] = x[36] + '*'
    x[39] = x[39] + '*'
    x[42] = x[42] + '*'
    x[45] = x[45] + '*'
    x[48] = x[48] + '*'
    return x


def slice_cham(x):
    return x[11:30][:15:3] + x[36:50][:15:3]


d_split = rdd.map(lambda x: x.split(','))
d_split = d_split.map(lambda x: mark(x))
d_frame = d_split.map(lambda x: Row(id=x[0], items=slice_cham(x))).collect()

spark = SparkSession.builder.appName('data').getOrCreate()
df = spark.createDataFrame(d_frame)

fpGrowth = FPGrowth(itemsCol='items', minSupport=0.001, minConfidence=0.1)
model = fpGrowth.fit(df)

df = model.freqItemsets

df = df.withColumn('length', size(df.items))
df = df.filter(df.length == 2)
df = df.filter(df.freq > 500)
df = df.orderBy(df.length.asc(), df.freq.desc()).select('items', 'freq').collect()
for i in df:
    print(i)

rules = model.associationRules

rules = rules.withColumn('length', size(rules.antecedent))
rules = rules.orderBy(rules.length.desc(), rules.confidence.desc()).select('antecedent', 'consequent', 'confidence')
rules = rules.collect()

for i in rules:
    print(i)