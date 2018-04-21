from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession, Row
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.sql.functions import monotonically_increasing_id, udf

spark = SparkSession.builder.master('local').appName('data').getOrCreate()
data = spark.read.csv('games.csv', header=True)

data = data.withColumn('towerkill', data.t1_towerKills-data.t2_towerKills+20)
data = data.withColumn('inhibitorkill', data.t1_inhibitorKills-data.t2_inhibitorKills+20)
data = data.withColumn('baronkill', data.t1_baronKills-data.t2_baronKills+20)
data = data.withColumn('dragonkill', data.t1_dragonKills-data.t2_dragonKills+20)
data = data.withColumn('riftkill', data.t1_riftHeraldKills-data.t2_riftHeraldKills+20)
data = data.withColumn('t-winner', data.winner-1)

data = data.select(
    data['t-winner'].cast(DoubleType()), data['firstBlood'].cast(IntegerType())
    , data['firstTower'].cast(IntegerType())
    , data['firstInhibitor'].cast(IntegerType()), data['firstBaron'].cast(IntegerType())
    , data['firstDragon'].cast(IntegerType())
    , data['firstRiftHerald'].cast(IntegerType())
    , data['towerkill'].cast(IntegerType())
    , data['inhibitorkill'].cast(IntegerType())
    , data['baronkill'].cast(IntegerType())
    , data['dragonkill'].cast(IntegerType())
    , data['riftkill'].cast(IntegerType())

)

data_rdd = data.rdd

data_rdd = data_rdd.map(lambda x: Row(label=x[0], features=Vectors.dense(x[1:12])))

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
