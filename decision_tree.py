from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import IntegerType


spark = SparkSession.builder.master('local').appName('data').getOrCreate()
# Load the data stored in LIBSVM format as a DataFrame.
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
# Index labels, adding metadata to the label column.
# Fit on whole dataset to include all labels in index.
labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)
# Automatically identify categorical features, and index them.
# We specify maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = data.randomSplit([0.7, 0.3])

# Train a DecisionTree model.
dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")

# Chain indexers and tree in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt])

# Train model.  This also runs the indexers.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
predictions.select("prediction", "indexedLabel", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g " % (1.0 - accuracy))

treeModel = model.stages[2]
# summary only
print(treeModel)