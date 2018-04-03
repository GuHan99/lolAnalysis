from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession, Row
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import monotonically_increasing_id, udf

maturity_udf = udf(lambda x: x, IntegerType())


spark = SparkSession.builder.master('local').appName('data').getOrCreate()
data_o = spark.read.csv('games.csv', header=True)

data = data_o.select(
    data_o['winner'].cast(IntegerType()), data_o['firstBlood'].cast(IntegerType())
    , data_o['firstTower'].cast(IntegerType())
    , data_o['firstInhibitor'].cast(IntegerType()), data_o['firstBaron'].cast(IntegerType())
    , data_o['firstDragon'].cast(IntegerType())
    , data_o['firstRiftHerald'].cast(IntegerType()))
data = data.withColumn('towerkill', maturity_udf(data_o['t1_towerKills']-data_o['t2_towerKills']))
data = data.withColumn('inhibitorkill', maturity_udf(data_o['t1_inhibitorKills']-data_o['t2_inhibitorKills']))
data = data.withColumn('baronkill', maturity_udf(data_o['t1_baronKills']-data_o['t2_baronKills']))
data = data.withColumn('dragonkill', maturity_udf(data_o['t1_dragonKills']-data_o['t2_dragonKills']))
data = data.withColumn('riftkill', maturity_udf(data_o['t1_riftHeraldKills']-data_o['t2_riftHeraldKills']))
data = data.select(
    data['winner'], data['firstBlood']
    , data['firstTower']
    , data['firstInhibitor'], data['firstBaron']
    , data['firstDragon']
    , data['firstRiftHerald']
    , data['towerkill']
    , data['inhibitorkill']
    , data['baronkill']
    , data['dragonkill']
    , data['riftkill']
)

assembler = VectorAssembler(
    inputCols=['firstBlood', 'firstTower', 'firstInhibitor', 'firstBaron', 'firstDragon',
               'firstRiftHerald', 'towerkill', 'inhibitorkill', 'baronkill', 'dragonkill', 'riftkill'],
    outputCol="features")
output = assembler.transform(data)

data_rdd = data.rdd

data_rdd = data_rdd.map(lambda x: Row(label=x[0], features='features'))

data = spark.createDataFrame(data_rdd)
data = data.withColumn('id', monotonically_increasing_id())
train = data.filter(data.id < 40000)
test = data.filter(data.id >= 40000)

# create the trainer and set its parameters
nb = NaiveBayes(smoothing=0.5, modelType="multinomial")

# train the model
model = nb.fit(train)

# select example rows to display.
predictions = model.transform(test)
predictions.show()

# compute accuracy on the test set
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                              metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test set accuracy = " + str(accuracy))
