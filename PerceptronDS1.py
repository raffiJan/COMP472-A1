import pandas
import numpy
import csv
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Initialize variables for Dataset 1

infoData1 = pandas.read_csv("Assig1-Dataset/info_1.csv", header=None).to_numpy()
testNLData1 = pandas.read_csv("Assig1-Dataset/test_no_label_1.csv", header=None).to_numpy()
testWLData1 = pandas.read_csv("Assig1-Dataset/test_with_label_1.csv", header=None).to_numpy()
trainData1 = pandas.read_csv("Assig1-Dataset/train_1.csv", header=None).to_numpy()
valData1 = pandas.read_csv("Assig1-Dataset/val_1.csv", header=None).to_numpy()

trainData1X = trainData1[:,:-1]
trainData1Y = trainData1[:,-1]
testData1X = testWLData1[:,:-1]
testData1Y = testWLData1[:,-1]

# Perception Model on DS1

per = Perceptron()
per.fit(trainData1X, trainData1Y)
predictPer1 = per.predict(testData1X)

# Print Output to .csv

##with open("Outputs/PER-DS1.csv", "a") as

## 1. Plot the distribution of the number of the instances in each class.

pandas.DataFrame(numpy.bincount(trainData1Y)).to_csv("Outputs/PER-DS1.csv", mode = "a")

## a) Row number of instance, Index of predicted class of that instance

pandas.DataFrame(predictPer1).to_csv("Outputs/PER-DS1.csv", mode = "a")

## b) Plot the confusion matrix

pandas.DataFrame(confusion_matrix(testData1Y, predictPer1)).to_csv("Outputs/PER-DS1.csv", mode = "a")

## c) The precision, recall, and f1-measure for each class

pandas.DataFrame(precision_score(testData1Y, predictPer1, average = None)).to_csv("Outputs/PER-DS1.csv", mode = "a")
pandas.DataFrame(recall_score(testData1Y, predictPer1, average = None)).to_csv("Outputs/PER-DS1.csv", mode = "a")
pandas.DataFrame(f1_score(testData1Y, predictPer1, average = None)).to_csv("Outputs/PER-DS1.csv", mode = "a")

## d) The accuracy, macro-average f1 and weighted-average f1 of the model

# pandas.DataFrame(accuracy_score(testData1Y, predictPer1).to_csv("Outputs/PER-DS1", mode = "a")
# pandas.DataFrame(f1_score(testData1Y, predictPer1, average = "micro")).to_csv("Outputs/PER-DS1", mode = "a")
# pandas.DataFrame(f1_score(testData1Y, predictPer1, average = "weighted")).to_csv("Outputs/PER-DS1", mode = "a")