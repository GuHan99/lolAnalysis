from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession, Row
from pyspark.ml.linalg import Vectors
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import monotonically_increasing_id

def features_csv(x):
    result = x[5:11]
    # result = result.append(x[51] - x[26])
    # result = result.append(x[52] - x[27])
    # result = result.append(x[53] - x[28])
    # result = result.append(x[54] - x[29])
    # result = result.append(x[55] - x[30])
    return result


spark = SparkSession.builder.master('local').appName('data').getOrCreate()
data = spark.read.csv('games.csv', header=True)

data = data.select(
    data['winner'].cast(IntegerType()), data['firstBlood'].cast(IntegerType())
    , data['firstTower'].cast(IntegerType())
    , data['firstInhibitor'].cast(IntegerType()), data['firstBaron'].cast(IntegerType())
    , data['firstDragon'].cast(IntegerType())
    , data['firstRiftHerald'].cast(IntegerType()))
data = data.withColumn('towerkill', data['t1_towerKills']-data['t2_towerKills'])
data = data.withColumn('inhibitorkill', data['t1_inhibitorKills']-data['t2_inhibitorKills'])
data = data.withColumn('baronkill', data['t1_baronKills']-data['t2_baronKills'])
data = data.withColumn('dragonkill', data['t1_dragonKills']-data['t2_dragonKills'])
data = data.withColumn('riftkill', data['t1_riftHeraldKills']-data['t2_riftHeraldKills'])
data = data.select(
    data['winner'], data['firstBlood']
    , data['firstTower']
    , data['firstInhibitor'], data['firstBaron']
    , data['firstDragon']
    , data['firstRiftHerald']
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
