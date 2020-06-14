import copy
import numpy as np
from collections import Counter
from sklearn import datasets
from sklearn.metrics import accuracy_score, confusion_matrix

from dist_utils import L2

class KNNClassifier():
    def __init__(self, num_K, typeOfDist="L2"):
        self._num_K = num_K
        self._typeOfDist = typeOfDist
        return

    def fit(self, X_data, y_label):
        self._X_data = X_data
        self._y_label = y_label
        return

    def predict(self, X, typeOfDist="L2"):
        Preds = [] # (lenOfX, )
        if self._typeOfDist is "L2":
            for X_elem in X:
                listOfDist = []
                for train_sample in self._X_data:
                    _dist = L2([train_sample], [X_elem])
                    listOfDist.append(_dist)

                sorted_ind_list = self._get_sorted_ind(listOfDist)
                nearest_neighbors = [self._y_label[sorted_ind_list[k_ind]] for k_ind in range(self._num_K)]
                pred = self._get_most_frequent(nearest_neighbors)
                Preds.append(pred)

        return np.array(Preds)

    def get_accuracy(self, labels, preds):
        return

    # def get_specificity(self):
    #     return
    #
    # def get_sensitivity(self):
    #     return

    def get_recall(self, labels, preds):
        return

    def _get_sorted_ind(self, items):
        items = np.array(items)
        sorted_items = copy.deepcopy(items)
        sorted_items.sort()

        sorted_ind = []
        for _item in sorted_items:
            ind, = np.where(items==_item)
            sorted_ind.append(ind[0])
        return sorted_ind

    def _get_most_frequent(self, item_list):
        count_dict = dict(Counter(item_list))
        sorted_items = sorted(count_dict.items(), key=lambda x : x[1], reverse=True )
        return sorted_items[0][0]

def KNN(training_set, test_set, K):
    return



if __name__ == '__main__':
    print('test')
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    print("X.shape", X.shape, X.min(), X.max()) # (150, 4) 0.1 7.9
    print("y.shape", y.shape, y.min(), y.max()) # (150,) 0 2 # (Setosa, Versicolour, and Virginica)

    num_K = 3
    knn = KNNClassifier(num_K)
    knn.fit(X, y)
    preds = knn.predict(X) #

    print("label", y, y.shape)
    print("preds", preds, preds.shape)


