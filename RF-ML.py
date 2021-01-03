import numpy as np
from scipy.io import loadmat, savemat
from sklearn import preprocessing


def knn(k, Ei, X):
    """Find k nearest neighbor of a instance.

    Args:
        k: number of neighbors.
        Ei: a instance.
        X: feature space.

    Returns:

    """
    N, _ = np.shape(X)
    distances = np.zeros(shape=(N,))
    for i in range(N):
        distances[i] = np.dot(Ei - X[i], Ei - X[i])
    distance_index_pair = [(distances[i], i) for i in range(N)]
    distance_index_pair = sorted(distance_index_pair, key=lambda element: element[0], reverse=True)
    indices_neighbors = np.array([distance_index_pair[i][1] for i in range(k)])
    return indices_neighbors


def diff(index_feature, instance, neighbor_instance):
    """Distance between one instance and its neighbor.

    Args:
        index_feature: feature index.
        instance: a instance.
        neighbor_instance: it's neighbor instance.

    Returns: distance between instance and neighbor instance on specific feature.

    """
    return np.abs(instance[index_feature] - neighbor_instance[index_feature])


def dissimilarity_function(Yi, Yj):
    """Multi label dissimilarity function

    Args:
        Yi: the label vector associated with x_i.
        Yj: the label vector associated with x_j.

    Returns: dissimilarity degree between Y_i and Y_j.

    """
    return (np.sum(Yi | Yj) - np.sum(Yi & Yj)) / len(Yi)


def rfml(X, Y, c, k):
    """ReliefF multi label feature selection algorithm.

    Args:
        X: feature space;
        Y: label space;
        c: iteration times;
        k: number of neighbors;

    Returns: feature subset with k features.

    """
    N, M = np.shape(X)
    WdY = 0
    WdX = np.zeros(shape=(M,))
    WdYX = np.zeros(shape=(M,))

    for i in range(c):
        e = np.random.randint(low=0, high=N)
        indices_k = knn(k, X[e], X)
        for z in range(k):
            mld = dissimilarity_function(Y[e], Y[indices_k[z]])
            d = np.power(np.dot(X[e] - X[indices_k[z]], X[e] - X[indices_k[z]]), 0.5)
            WdY += mld * d
            for j in range(M):
                distance = diff(j, X[e], X[indices_k[z]])
                WdX[j] += distance * d
                WdYX[j] += mld * distance * d
    W = WdYX / WdY - (WdX - WdYX) / (c - WdY)
    return W


def get_feature_ranking(W):
    """Get feature ranking from feature weights.

    Args:
        W: feature weight.

    Returns: feature ranking.

    """
    weight_index_pair = [(W[i], i) for i in range(len(W))]
    weight_index_pair = sorted(weight_index_pair, key=lambda element: element[0], reverse=True)
    feature_sorted = [weight_index_pair[i][1] for i in range(len(W))]
    return feature_sorted


if __name__ == "__main__":
    import os

    dataset_name = 'science'
    file = os.path.join(os.path.abspath('..'), 'dataset', dataset_name, dataset_name + '.mat')
    dataset = loadmat(file)
    X_all = dataset['data']
    Y_all = dataset['target']

    indices_file = os.path.join(os.path.abspath('..'), 'dataset', dataset_name, dataset_name + '_10cv_indices.mat')
    cv_indices = loadmat(indices_file)['indices']
    cv_indices = cv_indices.astype(np.int).reshape(cv_indices.shape[0], )
    
    k_folds = 10
    for fold in range(1, k_folds + 1):
        print(dataset_name + '\t:\ttraining is processing for fold ' + str(fold))
        X_train = X_all[cv_indices == fold]
        Y_train = Y_all[cv_indices == fold]
        weight = rfml(X_train, Y_train, c=2 * X_train.shape[0], k=5)
        feature_ranking = get_feature_ranking(weight)
    
        result_file = os.path.join(os.path.abspath('..'), 'fs', 'RFML', dataset_name + '_fold_' + str(fold) + '.mat')
        savemat(result_file, mdict={'indices': feature_ranking})
