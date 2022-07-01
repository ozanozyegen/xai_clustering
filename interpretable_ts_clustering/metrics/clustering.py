from sklearn.metrics import silhouette_score, calinski_harabasz_score, \
    davies_bouldin_score, adjusted_rand_score, adjusted_mutual_info_score
import pandas as pd
import numpy as np


def evaluate_clusters(X, cluster_labels, logger=None):
    results = {}
    for method in [silhouette_score, calinski_harabasz_score,
                   davies_bouldin_score, ]:
        score = method(X, cluster_labels)
        method_name = method.__name__
        result = {method_name: score}
        # print(result)
        results[method_name] = score
        if logger is not None:
            logger.log(result)

    return results
