import copy
import numpy as np
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import

from custom_exception import CustomException
from dist_utils import L1, L2

class KNNClassifier():
    def __init__(self, num_K, typeOfDist="L2"):
        if num_K%2==0:
            raise CustomException("[!] K must be an odd number.")

        self._num_K = num_K
        self._typeOfDist = typeOfDist
        return

    def fit(self, X_data, y_label):
        self._X_data = X_data
        self._y_label = y_label
        return

    def predict(self, X, typeOfDist="L2"):
        Preds = [] # (lenOfX, )
        if self._typeOfDist is None:
            self._typeOfDist = typeOfDist

        for X_elem in X:
            listOfDist = []
            for train_sample in self._X_data:
                if self._typeOfDist is "L1":
                    _dist = L1([train_sample], [X_elem])
                elif self._typeOfDist is "L2":
                    _dist = L2([train_sample], [X_elem])
                listOfDist.append(_dist)

            sorted_ind_list = self._get_sorted_ind(listOfDist)
            nearest_neighbors = [self._y_label[sorted_ind_list[k_ind]] for k_ind in range(self._num_K)]
            pred = self._get_most_frequent(nearest_neighbors)
            Preds.append(pred)

        return np.array(Preds)

    def get_accuracy(self, labels, preds):
        labels = np.array(labels)
        preds = np.array(preds)
        corr = 0
        for l, p in zip(labels, preds):
            if l==p:
                corr+=1
        return corr/len(labels)

    # def get_specificity(self):
    #     return
    #
    # def get_sensitivity(self):
    #     return

    def get_recall(self, labels, preds):
        """
        :param labels: (N, )
        :param preds: (N, )
        :return:
        """
        labels = np.array(labels)
        preds = np.array(preds)

        recall_dict = dict()
        for label_ind in np.unique(labels):
            ind_label_list = labels[labels==label_ind]
            ind_pred_list = preds[labels==label_ind]
            recall_dict[label_ind] = self.get_accuracy(ind_label_list, ind_pred_list)
        return recall_dict

    def get_confusion_matrix(self, labels, preds):
        """
        rows : labels, cols : preds
        cells : number of samples which is predicted as the label
        :param labels: (N, )
        :param preds: (N, )
        :return:
        """
        labels = np.array(labels)
        preds = np.array(preds)

        conf_matrix = []
        for l in np.unique(labels):
            label_conf_matrix = []
            for expected_p in np.unique(labels):
                ind_preds = preds[labels==l]
                label_conf_matrix.append(len(ind_preds[ind_preds==expected_p]))
            conf_matrix.append(label_conf_matrix)
        return conf_matrix


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
    if K%2==0 or K>5 or K<1:
        raise CustomException("[!] K must be [1, 3, 5]")
    knn = KNNClassifier(K)
    knn.fit(training_set[0], training_set[1])
    predictions = knn.predict(test_set[0])
    return knn.get_accuracy(test_set[1], predictions), knn.get_recall(test_set[1], predictions)



if __name__ == '__main__':
    mode = 'dev' # 'dev', 'test', 'bootstrap'
    if mode == 'dev':
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target

        print("X.shape", X.shape, X.min(), X.max()) # (150, 4) 0.1 7.9
        print("y.shape", y.shape, y.min(), y.max()) # (150,) 0 2 # (Setosa, Versicolour, and Virginica)

        # stratified sampling
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
        print("X_train.shape", X_train.shape)
        print("X_test.shape", X_test.shape)
        print("y_train.shape", y_train.shape, y_train)
        print("y_test.shape", y_test.shape, y_test)

        num_K = 5
        knn = KNNClassifier(num_K)
        knn.fit(X_train, y_train)
        preds = knn.predict(X_test, typeOfDist="L1") #

        print("label", y_test.shape, y_test)
        print("preds", preds.shape, preds)

        # Accuracy
        print("Accuracy : ", knn.get_accuracy(y_test, preds))
        # Recall (Specificity, Sensitivity)
        print("Recall : ", knn.get_recall(y_test, preds))
        # Confusion Matrix
        print("Confusion Matrix\n", knn.get_confusion_matrix(y_test, preds))

    elif mode == 'test':
        # load iris data
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target

        # Stratified Sampling
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
        print("X_train.shape", X_train.shape)
        print("X_test.shape", X_test.shape)
        print("y_train.shape", y_train.shape, y_train)
        print("y_test.shape", y_test.shape, y_test)

        # KNN
        num_K = 1
        ACC, Recall = KNN(training_set=(X_train, y_train), test_set=(X_test, y_test), K=num_K)
        print("Accuracy:", ACC)
        print("Recall", Recall)