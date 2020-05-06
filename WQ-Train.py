#Importing libraries for spark
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark import SparkContext, SparkConf

#Creating spark context 
conf = SparkConf().setAppName("wine-app")
sc = SparkContext("local",conf=conf)

tData = sc.textFile("s3://wq-sp2685/TrainingDataset.csv")
header = tData.first()
rows = tData.filter(lambda x: x != header)

def parsePoint(line):
    values = [float(x) for x in line.split(';')]
    return LabeledPoint(values[11], values[0:10])

parsedTData = rows.map(parsePoint)

#Training data using Random Forest
model = RandomForest.trainClassifier(parsedTData, numClasses=11, categoricalFeaturesInfo={},
                                     numTrees=76, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=18, maxBins=32)

vData = sc.textFile("s3://wq-sp2685/ValidationDataset.csv")
header = vData.first()
rows = vData.filter(lambda x: x != header)

parsedVData = rows.map(parsePoint)

predictions = model.predict(parsedVData.map(lambda x: x.features))
labelsAndPredictions = parsedVData.map(lambda lp: lp.label).zip(predictions)

#Calculating F1 Score on validation data
metrics = MulticlassMetrics(labelsAndPredictions)
f1Score = metrics.weightedFMeasure()

print ("F1 Score on validation data = ", f1Score)

#Saving model
model.save(sc, "s3://wq-sp2685/sp2685-wqModel")