import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score, confusion_matrix

class KNNClassifier():
    def __init__(self, num_k, typeOfDist=None):
        return

    def fit(self):
        return

    def predict(self):
        return

    def get_accuracy(self):
        return

def KNN(training_set, test_set, K):
    return


if __name__ == '__main__':
    print('test')
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    print("X.shape", X.shape, X.min(), X.max()) # (150, 4) 0.1 7.9
    print("y.shape", y.shape, y.min(), y.max()) # (150,) 0 2 # (Setosa, Versicolour, and Virginica)

    knn = KNNClassifier()
    knn.fit(X)
    y = knn.predict(X) #


