from pyspark.ml.fpm import FPGrowth
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf


def merge(*c):
    merged = sorted(set(c))
    if len(merged) == 1:
        return merged[0]
    else:
        return "[{0}]".format(",".join(merged))


spark = SparkSession.builder.appName('data').getOrCreate()
df = spark.read.csv('games.csv', header=True)

merge_udf = udf(merge)

df_1 = df.select(merge_udf("t1_champ1id", "t1_champ2id", "t1_champ3id", "t1_champ4id", "t1_champ5id"))
df_1 = df_1.selectExpr('merge(t1_champ1id, t1_champ2id, t1_champ3id, t1_champ4id, t1_champ5id) as list')
df_2 = df.select(merge_udf("t2_champ1id", "t2_champ2id", "t2_champ3id", "t2_champ4id", "t2_champ5id"))
df_2 = df_2.selectExpr('merge(t2_champ1id, t2_champ2id, t2_champ3id, t2_champ4id, t2_champ5id) as list_2')

df_1.show()
df_2.show()
# fpGrowth = FPGrowth(itemsCol="items", minSupport=0.5, minConfidence=0.6)
# model = fpGrowth.fit(df)
#
# # Display frequent itemsets.
# model.freqItemsets.show()
#
# # Display generated association rules.
# model.associationRules.show()
#
# # transform examines the input items against all the association rules and summarize the
# # consequents as prediction
# model.transform(df).show()