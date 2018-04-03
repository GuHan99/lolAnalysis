from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession, Row
from pyspark.ml.linalg import Vectors
from pyspark.sql.types import IntegerType


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

data_rdd = data.rdd

data_rdd = data_rdd.map(lambda x: Row(label=x[0], features=Vectors.dense([x[1], x[2], x[3], x[4], x[5], x[6]])))

data = spark.createDataFrame(data_rdd)

# Split the data into train and test
splits = data.randomSplit([0.6, 0.4], 1234)
train = splits[0]
test = splits[1]

train.show()
test.show()

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
