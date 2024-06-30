import numpy as np

def davies_bouldin(X, labels):
    k = len(np.unique(labels))
    Y_class = list(set(labels))
    centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        centroids[i, :] = np.mean(X[labels == Y_class[i], :], axis=0)

    S = np.zeros(k)
    for i in range(k):
        S[i] = np.mean(np.sqrt(np.sum((X[labels == Y_class[i], :] - centroids[i, :])**2, axis=1)))

    R = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i != j:
                R[i, j] = (S[i] + S[j]) / np.sqrt(np.sum((centroids[i, :] - centroids[j, :])**2))

    Di = np.max(R, axis=1)
    return np.mean(Di)