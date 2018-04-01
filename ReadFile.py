from pyspark.ml.fpm import FPGrowth
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('data').getOrCreate()
df = spark.read.csv('games.csv', header=True)

fpGrowth = FPGrowth(itemsCol="items", minSupport=0.5, minConfidence=0.6)
model = fpGrowth.fit(df)

# Display frequent itemsets.
model.freqItemsets.show()

# Display generated association rules.
model.associationRules.show()

# transform examines the input items against all the association rules and summarize the
# consequents as prediction
model.transform(df).show()