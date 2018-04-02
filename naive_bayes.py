from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession, Row
from pyspark.ml.linalg import Vectors


def features_csv(x):
    result = x[5:11]
    # result = result.append(x[51] - x[26])
    # result = result.append(x[52] - x[27])
    # result = result.append(x[53] - x[28])
    # result = result.append(x[54] - x[29])
    # result = result.append(x[55] - x[30])
    return result


spark = SparkSession.builder.master('local').appName('data').getOrCreate()
data = spark.read.csv('games.csv', header=True).rdd

data = data.map(lambda x: Row(labels=x[4], features=Vectors.dense(features_csv(x))))

data = spark.createDataFrame(data)

# Split the data into train and test
splits = data.randomSplit([0.6, 0.4], 1234)
train = splits[0]
test = splits[1]

# create the trainer and set its parameters
nb = NaiveBayes(smoothing=1.0, modelType="multinomial")

# train the model
model = nb.fit(train)

# select example rows to display.
predictions = model.transform(test)
predictions.show()

# compute accuracy on the test set
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="winner",
                                              metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test set accuracy = " + str(accuracy))
