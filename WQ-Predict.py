#Imports

import findspark
findspark.init()

import sys

from pyspark import SparkContext, SparkConf
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.evaluation import MulticlassMetrics

#Creating Spark Context
conf = SparkConf().setAppName("wine-app")
sc = SparkContext(conf=conf)

#Using Random Forest Model
randomForestModel = RandomForestModel.load(sc, "sp2685-wqModel")

testData = sc.textFile(sys.argv[1])
header = testData.first()
rows = testData.filter(lambda x: x != header)

def parsePoint(line):
    values = [float(x) for x in line.split(';')]
    return LabeledPoint(values[11], values[0:10])

parsedTestData = rows.map(parsePoint)

predictions = randomForestModel.predict(parsedTestData.map(lambda x: x.features))
labelsAndPredictions = parsedTestData.map(lambda lp: lp.label).zip(predictions)

#Calculating F1 Score
metrics = MulticlassMetrics(labelsAndPredictions)
f1Score = metrics.fMeasure()

#F1 Score
print ("F1 score on Test data = ", f1Score)
