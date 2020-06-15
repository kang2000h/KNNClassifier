import numpy as np

from custom_exception import CustomException

def L1(x1, x2):
    """
    :param x1: list of first variables to measure the distance, (N, numOfLength)
    :param x2: list of second variables to measure the distance, (N, numOfLength)
    :return: distance, (N, )
    """
    if len(x1) != len(x2):
        raise CustomException("length of each variable have to be same.")

    x1 = np.array(x1)
    x2 = np.array(x2)
    # for a distance between single variables
    if len(x1)==1 and len(x2)==1:
        res = np.sum(np.abs((x1-x2)))
    else :
        #res = np.sum(np.sqrt((x1-x2)**2), axis=1)
        res = np.sum(np.abs((x1 - x2)), axis=1)
    return res

def L2(x1, x2):
    """
    :param x1: list of first variables to measure the distance, (N, numOfLength)
    :param x2: list of second variables to measure the distance, (N, numOfLength)
    :return: distance, (N, )
    """
    if len(x1) != len(x2):
        raise CustomException("length of each variable have to be same.")

    x1 = np.array(x1)
    x2 = np.array(x2)
    # for a distance between single variables
    if len(x1)==1 and len(x2)==1:
        res = np.sum(np.sqrt((x1-x2)**2))
    else :
        #res = np.sum(np.sqrt((x1-x2)**2), axis=1)
        res = np.sum(np.sqrt((x1 - x2) ** 2), axis=1)
    return res

if __name__ == "__main__":
    print("hello")

    # input variable is a single scalar
    v1 = [[1]]
    v2 = [[1]]
    res = L1(v1, v2)
    print("single scalar variable", res, res.shape)

    v1 = [[1, 2]]
    v2 = [[1, 2]]
    res = L1(v1, v2)
    print("single vector variables", res, res.shape)

    v1 = [[1], [1]]
    v2 = [[1], [2]]
    res = L1(v1, v2)
    print("list of scalar variables", res, res.shape)


    v1 = [[1, 2], [1, 3]]
    v2 = [[1, 2], [2, 4]]
    res = L1(v1, v2)
    print("list of vectors variables", res, res.shape)
