from pyspark.ml.fpm import FPGrowth
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, size
from pyspark.sql.types import ArrayType, StringType


def merge(*c):
    merged = sorted(set(c))
    if len(merged) == 1:
        return merged[0]
    else:
        return "[{0}]".format(",".join(merged))


spark = SparkSession.builder.appName('data').getOrCreate()
df = spark.read.csv('games.csv', header=True)

merge_udf = udf(merge, ArrayType(StringType()))

df_1 = df.select(merge_udf("t1_champ1id", "t1_champ2id", "t1_champ3id", "t1_champ4id", "t1_champ5id"))
df_1 = df_1.withColumnRenamed('merge(t1_champ1id, t1_champ2id, t1_champ3id, t1_champ4id, t1_champ5id)', 'list_1')
df_2 = df.select(merge_udf("t2_champ1id", "t2_champ2id", "t2_champ3id", "t2_champ4id", "t2_champ5id"))
df_2 = df_2.withColumnRenamed('merge(t2_champ1id, t2_champ2id, t2_champ3id, t2_champ4id, t2_champ5id)', 'list_2')

df_1 = df_1.union(df_2)

df = df_1

df = df.withColumnRenamed('list_1', 'items')

model = FPGrowth(itemsCol='items', minSupport=0.000001)
fpm = model.fit(df)
result = fpm.freqItemsets.show()

# df = df.select(df.items.cast('array').alias('item'))
# fpGrowth = FPGrowth(itemsCol="items", minSupport=0.05, minConfidence=0.1)
# model = fpGrowth.fit(df)
#
# df = model.freqItemsets
# df = df.withColumn('length', size(df['items']))
# df = df.orderBy(df.freq.desc(), df.length.desc()).select('items', 'freq')
#
# df = df.show()
