import pandas
import numpy
import csv
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

# Initialize variables for Dataset 2

infoData2 = pandas.read_csv("dataset/info_2.csv", header=None).to_numpy()
testNLData2 = pandas.read_csv("dataset/test_no_label_2.csv", header=None).to_numpy()
testWLData2 = pandas.read_csv("dataset/test_with_label_2.csv", header=None).to_numpy()
trainData2 = pandas.read_csv("dataset/train_2.csv", header=None).to_numpy()
valData2 = pandas.read_csv("dataset/val_2.csv", header=None).to_numpy()

trainData2X = trainData2[:,:-1]
trainData2Y = trainData2[:,-1]
testData2X = testWLData2[:,:-1]
testData2Y = testWLData2[:,-1]

# Perception Model on DS2

per = Perceptron()
per.fit(trainData2X, trainData2Y)
predictPer2 = per.predict(testData2X)

# Print Output to .csv
## 1. Plot the distribution of the number of the instances in each class.

pandas.DataFrame(numpy.bincount(trainData2Y)).to_csv("output/PER-DS2.csv", mode = "a", header = ["Distribution of class instances in Dataset"])

## a) Row number of instance, Index of predicted class of that instance

pandas.DataFrame(predictPer2).to_csv("output/PER-DS2.csv", mode = "a", header =["Predicted Class"] )

## b) Plot the confusion matrix

pandas.DataFrame(confusion_matrix(testData2Y, predictPer2)).to_csv("output/PER-DS2.csv", mode = "a")

## c) The precision, recall, and f1-measure for each class

pandas.DataFrame(precision_score(testData2Y, predictPer2, average = None)).to_csv("output/PER-DS2.csv", mode = "a", header = ["Precision Score"])
pandas.DataFrame(recall_score(testData2Y, predictPer2, average = None)).to_csv("output/PER-DS2.csv", mode = "a", header = ["Recall Score"])
pandas.DataFrame(f1_score(testData2Y, predictPer2, average = None)).to_csv("output/PER-DS2.csv", mode = "a", header = ["F-1 Measure"])

## d) The accuracy, macro-average f1 and weighted-average f1 of the model

with open("output/PER-DS2.csv", "a") as file:
    writefile = csv.writer(file)
    writefile.writerow("")
    writefile.writerow(["Accuracy: ", accuracy_score(testData2Y, predictPer2)])
    writefile.writerow(["Macro Average F1: ", f1_score(testData2Y, predictPer2, average = "macro")])
    writefile.writerow(["Weighted Average F1: ", f1_score(testData2Y, predictPer2, average = "weighted")])
