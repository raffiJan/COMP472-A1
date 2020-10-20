from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
import pandas  

ds1_test_label = pandas.read_csv('dataset/test_with_label_1.csv', header=None).to_numpy()
ds1_train = pandas.read_csv('dataset/train_1.csv', header=None).to_numpy()
ds2_test_label = pandas.read_csv('dataset/test_with_label_2.csv', header=None).to_numpy()
ds2_train = pandas.read_csv('dataset/train_2.csv', header=None).to_numpy()

ds1_base_mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', solver='sgd')
ds2_base_mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', solver='sgd')

ds1_train_x = ds1_train[:, :-1]
ds1_train_y = ds1_train[:, -1]
ds1_test_x = ds1_test_label[:, :-1]
ds1_test_y =  ds1_test_label[:, -1]

ds2_train_x = ds2_train[:, :-1]
ds2_train_y = ds2_train[:, -1]
ds2_test_x = ds2_test_label[:, :-1]
ds2_test_y = ds2_test_label[:, -1]

ds1_base_mlp.fit(ds1_train_x, ds1_train_y)
ds2_base_mlp.fit(ds2_train_x, ds2_train_y)

ds1_base_mlp_prediction = ds1_base_mlp.predict(ds1_test_x)
ds2_base_mlp_prediction = ds2_base_mlp.predict(ds2_test_x)

ds1_precision = precision_score(ds1_test_y, ds1_base_mlp_prediction, average=None, zero_division=0)
ds1_recall = recall_score(ds1_test_y, ds1_base_mlp_prediction, average=None)
ds1_accuracy = accuracy_score(ds1_test_y, ds1_base_mlp_prediction)
ds1_confusion = confusion_matrix(ds1_test_y, ds1_base_mlp_prediction)

ds2_precision = precision_score(ds2_test_y, ds2_base_mlp_prediction, average=None, zero_division=0)
ds2_recall = recall_score(ds2_test_y, ds2_base_mlp_prediction, average=None)
ds2_accuracy = accuracy_score(ds2_test_y, ds2_base_mlp_prediction)
ds2_confusion = confusion_matrix(ds2_test_y, ds2_base_mlp_prediction)

pandas.DataFrame(ds1_precision).to_csv("output/baseMLP_ds1.csv", mode='a', header=["precision"])
pandas.DataFrame(ds1_recall).to_csv("output/baseMLP_ds1.csv", mode='a', header=["recall"])
pandas.DataFrame(ds1_confusion).to_csv("output/baseMLP_ds1.csv", mode='a')

pandas.DataFrame(ds2_precision).to_csv("output/baseMLP_ds2.csv", mode='a', header=["precision"])
pandas.DataFrame(ds2_recall).to_csv("output/baseMLP_ds2.csv", mode='a', header=["recall"])
pandas.DataFrame(ds2_confusion).to_csv("output/baseMLP_ds2.csv", mode='a')
