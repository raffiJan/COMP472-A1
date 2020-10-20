import pandas, json, numpy, csv

from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix, f1_score

ds1_train = pandas.read_csv("dataset/train_1.csv", header=None).to_numpy()
ds1_test = pandas.read_csv("dataset/test_with_label_1.csv", header=None).to_numpy()
ds1_val = pandas.read_csv("dataset/val_1.csv", header=None).to_numpy()

ds2_train = pandas.read_csv("dataset/train_2.csv", header=None).to_numpy()
ds2_test = pandas.read_csv("dataset/test_with_label_2.csv", header=None).to_numpy()
ds2_val = pandas.read_csv("dataset/val_2.csv", header=None).to_numpy()

ds1_train_x = ds1_train[:,:-1]
ds1_train_y = ds1_train[:,-1]
ds2_train_x = ds2_train[:, :-1]
ds2_train_y = ds2_train[:, -1]

ds1_test_x = ds1_test[:, :-1]
ds1_test_y = ds1_test[:, -1]
ds2_test_x = ds2_test[:, :-1]
ds2_test_y = ds2_test[:, -1]

ds1_val_x = ds1_val[:,:-1]
ds1_val_y = ds1_val[:, -1]
ds2_val_x = ds2_val[:,:-1]
ds2_val_y = ds2_val[:, -1]

def instance_count():
    pandas.DataFrame(numpy.bincount(ds1_train_y)).to_csv("instances/DS1-Instances.csv", mode = "a", header = ["Distribution of class instances in Dataset 1: TRAINING"])
    pandas.DataFrame(numpy.bincount(ds1_test_y)).to_csv("instances/DS1-Instances.csv", mode = "a", header = ["Distribution of class instances in Dataset 1: TEST"])
    pandas.DataFrame(numpy.bincount(ds1_val_y)).to_csv("instances/DS1-Instances.csv", mode = "a", header = ["Distribution of class instances in Dataset 1: VALIDATION"])

    pandas.DataFrame(numpy.bincount(ds2_train_y)).to_csv("instances/DS2-Instances.csv", mode = "a", header = ["Distribution of class instances in Dataset 2: TRAINING"])
    pandas.DataFrame(numpy.bincount(ds2_test_y)).to_csv("instances/DS2-Instances.csv", mode = "a", header = ["Distribution of class instances in Dataset 2: TEST"])
    pandas.DataFrame(numpy.bincount(ds2_val_y)).to_csv("instances/DS2-Instances.csv", mode = "a", header = ["Distribution of class instances in Dataset 2: VALIDATION"])

# GAUSSIAN NAIVE BAYES CLASSIFIER
def gaussian_nb_ds1():
    gnb = GaussianNB()
    gnb.fit(ds1_train_x, ds1_train_y)
    gnb_ds1_output = gnb.predict(ds1_test_x)
    results_ds1("GNB-DS1", gnb_ds1_output)

def gaussian_nb_ds2():
    gnb = GaussianNB()
    gnb.fit(ds2_train_x, ds2_train_y)
    gnb_ds2_output = gnb.predict(ds2_test_x)
    results_ds2("GNB-DS2", gnb_ds2_output)

# BASELINE DECISION TREE
def base_dt_ds1():
    dt = tree.DecisionTreeClassifier(criterion="entropy")
    dt.fit(ds1_train_x, ds1_train_y)
    dt_ds1_output = dt.predict(ds1_test_x)
    results_ds1("Base-DT-DS1", dt_ds1_output)

def base_dt_ds2():
    dt = tree.DecisionTreeClassifier(criterion="entropy")
    dt.fit(ds2_train_x, ds2_train_y)
    dt_ds2_output = dt.predict(ds2_test_x)
    results_ds2("Base-DT-DS2", dt_ds2_output)

# BETTER PERFORMING DECISION TREE
def best_dt_ds1():
    dt = tree.DecisionTreeClassifier()

    param_grid = { 
        "criterion":["gini", "entropy"],
        "max_depth":[10, None],
        "min_samples_split":[*range(2, 10, 1)],
        "min_impurity_decrease":[*numpy.arange(0.00001, 0.0001, 0.00002)],
        "class_weight":[None, "balanced"]
    }
    
    grid_search = GridSearchCV(dt, param_grid=param_grid, n_jobs=-1)
    grid_search.fit(ds1_val_x, ds1_val_y)
    best_params_string = json.dumps(grid_search.best_params_)
    print(best_params_string)
    best_dt = tree.DecisionTreeClassifier(**grid_search.best_params_)
    best_dt.fit(ds1_train_x, ds1_train_y)
    best_dt_ds1_output = best_dt.predict(ds1_test_x)
    results_ds1("Best-DT-DS1", best_dt_ds1_output)

def best_dt_ds2():
    dt = tree.DecisionTreeClassifier()

    param_grid = { 
        "criterion":["gini", "entropy"],
        "max_depth":[10, None],
        "min_samples_split":[*range(2, 10, 1)],
        "min_impurity_decrease":[*numpy.arange(0.00001, 0.0001, 0.00002)],
        "class_weight":[None, "balanced"]
    }
    
    grid_search = GridSearchCV(dt, param_grid=param_grid, n_jobs=-1)
    grid_search.fit(ds2_val_x, ds2_val_y)
    best_params_string = json.dumps(grid_search.best_params_)
    print(best_params_string)
    best_dt = tree.DecisionTreeClassifier(**grid_search.best_params_)
    best_dt.fit(ds2_train_x, ds2_train_y)
    best_dt_ds2_output = best_dt.predict(ds2_test_x)
    results_ds2("Best-DT-DS2", best_dt_ds2_output)

# PERCEPTRON
def perceptron_ds1():
    per = Perceptron()
    per.fit(ds1_train_x, ds1_train_y)
    per_ds2_output = per.predict(ds1_test_x)
    results_ds1("PER_DS1", per_ds2_output)

def perceptron_ds2():
    per = Perceptron()
    per.fit(ds2_train_x, ds2_train_y)
    per_ds2_output = per.predict(ds2_test_x)
    results_ds2("PER_DS2", per_ds2_output)

# BASELINE MULTI-LAYER PERCEPTRON
def base_mlp_ds1():
    mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', solver='sgd')
    mlp.fit(ds1_train_x, ds1_train_y)
    mlp_ds1_output = mlp.predict(ds1_test_x)
    results_ds1("Base-MLP-DS1", mlp_ds1_output)

def base_mlp_ds2():
    mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', solver='sgd')
    mlp.fit(ds2_train_x, ds2_train_y)
    mlp_ds2_output = mlp.predict(ds2_test_x)
    results_ds2("Base-MLP-DS2", mlp_ds2_output)

# BETTER PERFORMING MULTI-LAYER PERCEPTRON
def best_mlp_ds1():
    mlp = MLPClassifier()

    param_grid = { 
        "hidden_layer_sizes": [(30,50,), (10,10,10,)],
        "activation": ["logistic", "tanh", "relu", "identity"],
        "solver": ["adam", "sgd"]
    }

    grid_search = GridSearchCV(mlp, param_grid=param_grid, n_jobs=-1)
    grid_search.fit(ds1_val_x, ds1_val_y)
    best_params_string = json.dumps(grid_search.best_params_)
    print(best_params_string)
    best_mlp = MLPClassifier(**grid_search.best_params_)
    best_mlp.fit(ds1_train_x, ds1_train_y)
    best_mlp_ds1_output = best_mlp.predict(ds1_test_x)
    results_ds1("Best-MLP-DS1", best_mlp_ds1_output)

def best_mlp_ds2():
    mlp = MLPClassifier()

    param_grid = { 
        "hidden_layer_sizes": [(30,50,), (10,10,10,)],
        "activation": ["logistic", "tanh", "relu", "identity"],
        "solver": ["adam", "sgd"]
    }

    grid_search = GridSearchCV(mlp, param_grid=param_grid, n_jobs=-1)
    grid_search.fit(ds2_val_x, ds2_val_y)
    best_params_string = json.dumps(grid_search.best_params_)
    print(best_params_string)
    best_mlp = MLPClassifier(**grid_search.best_params_)
    best_mlp.fit(ds2_train_x, ds2_train_y)
    best_mlp_ds2_output = best_mlp.predict(ds2_test_x)
    results_ds2("Best-MLP-DS2", best_mlp_ds2_output)

def results_ds1(model, output):
    pandas.DataFrame(output)                                                                .to_csv("output/" + model + ".csv", mode='a', header=["PREDICTED CLASS"] )
    pandas.DataFrame(confusion_matrix(ds1_test_y, output))                                  .to_csv("output/" + model + ".csv", mode='a')
    pandas.DataFrame(precision_score(ds1_test_y, output, average=None, zero_division=0))    .to_csv("output/" + model + ".csv", mode='a', header=["PRECISION"])
    pandas.DataFrame(recall_score(ds1_test_y, output, average=None))                        .to_csv("output/" + model + ".csv", mode='a', header=["RECALL"])
    pandas.DataFrame(f1_score(ds1_test_y, output, average = None))                          .to_csv("output/" + model + ".csv", mode='a', header=["F1-MEASURE"])

    with open("output/" + model + ".csv", "a") as file:
        writefile = csv.writer(file)
        writefile.writerow("")
        writefile.writerow(["Accuracy: ", accuracy_score(ds1_test_y, output)])
        writefile.writerow(["Macro Average F1: ", f1_score(ds1_test_y, output, average = "macro")])
        writefile.writerow(["Weighted Average F1: ", f1_score(ds1_test_y, output, average = "weighted")])

def results_ds2(model, output):
    pandas.DataFrame(output)                                                                .to_csv("output/" + model + ".csv", mode='a', header=["PREDICTED CLASS"] )
    pandas.DataFrame(confusion_matrix(ds2_test_y, output))                                  .to_csv("output/" + model + ".csv", mode='a')
    pandas.DataFrame(precision_score(ds2_test_y, output, average=None, zero_division=0))    .to_csv("output/" + model + ".csv", mode='a', header=["PRECISION"])
    pandas.DataFrame(recall_score(ds2_test_y, output, average=None))                        .to_csv("output/" + model + ".csv", mode='a', header=["RECALL"])
    pandas.DataFrame(f1_score(ds2_test_y, output, average = None))                          .to_csv("output/" + model + ".csv", mode='a', header=["F1-MEASURE"])
    with open("output/" + model + ".csv", "a") as file:
        writefile = csv.writer(file)
        writefile.writerow("")
        writefile.writerow(["Accuracy: ", accuracy_score(ds2_test_y, output)])
        writefile.writerow(["Macro Average F1: ", f1_score(ds2_test_y, output, average = "macro")])
        writefile.writerow(["Weighted Average F1: ", f1_score(ds2_test_y, output, average = "weighted")])

def run_all():
    instance_count()

    gaussian_nb_ds1()
    gaussian_nb_ds2()

    base_dt_ds1()
    base_dt_ds2()

    best_dt_ds1()
    best_dt_ds2()

    perceptron_ds1()
    perceptron_ds2()

    base_mlp_ds1()
    base_mlp_ds2()

    best_mlp_ds1()
    best_mlp_ds2()

run_all()