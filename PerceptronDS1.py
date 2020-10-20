import pandas
import numpy
import csv
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

# Initialize variables for Dataset 1

infoData1 = pandas.read_csv("dataset/info_1.csv", header=None).to_numpy()
testNLData1 = pandas.read_csv("dataset/test_no_label_1.csv", header=None).to_numpy()
testWLData1 = pandas.read_csv("dataset/test_with_label_1.csv", header=None).to_numpy()
trainData1 = pandas.read_csv("dataset/train_1.csv", header=None).to_numpy()
valData1 = pandas.read_csv("dataset/val_1.csv", header=None).to_numpy()

trainData1X = trainData1[:,:-1]
trainData1Y = trainData1[:,-1]
testData1X = testWLData1[:,:-1]
testData1Y = testWLData1[:,-1]

# Perception Model on DS1

per = Perceptron()
per.fit(trainData1X, trainData1Y)
predictPer1 = per.predict(testData1X)

# Print Output to .csv
## 1. Plot the distribution of the number of the instances in each class.

pandas.DataFrame(numpy.bincount(trainData1Y)).to_csv("output/PER-DS1.csv", mode = "a", header = ["Distribution of class instances in Dataset"])

## a) Row number of instance, Index of predicted class of that instance

pandas.DataFrame(predictPer1).to_csv("output/PER-DS1.csv", mode = "a", header =["Predicted Class"] )

## b) Plot the confusion matrix

pandas.DataFrame(confusion_matrix(testData1Y, predictPer1)).to_csv("output/PER-DS1.csv", mode = "a")

## c) The precision, recall, and f1-measure for each class

pandas.DataFrame(precision_score(testData1Y, predictPer1, average = None)).to_csv("output/PER-DS1.csv", mode = "a", header = ["Precision Score"])
pandas.DataFrame(recall_score(testData1Y, predictPer1, average = None)).to_csv("output/PER-DS1.csv", mode = "a", header = ["Recall Score"])
pandas.DataFrame(f1_score(testData1Y, predictPer1, average = None)).to_csv("output/PER-DS1.csv", mode = "a", header = ["F-1 Measure"])

## d) The accuracy, macro-average f1 and weighted-average f1 of the model

with open("output/PER-DS1.csv", "a") as file:
    writefile = csv.writer(file)
    writefile.writerow("")
    writefile.writerow(["Accuracy: ", accuracy_score(testData1Y, predictPer1)])
    writefile.writerow(["Macro Average F1: ", f1_score(testData1Y, predictPer1, average = "macro")])
    writefile.writerow(["Weighted Average F1: ", f1_score(testData1Y, predictPer1, average = "weighted")])
