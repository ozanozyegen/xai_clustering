from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, \
    TimeSeriesResampler
import numpy as np
from sklearn.utils import shuffle
import pickle 

def load_trace(config, return_labels=False):
    
    if config["feat_config"]==None or return_labels:
        X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")
        X_train = X_train[y_train < 4]  # Keep first 3 classes
        y_train = y_train[y_train < 4]
        X_test = X_test[y_test < 4]
        y_test = y_test[y_test < 4]

        X = np.concatenate((X_train, X_test), axis=0)
        # X = TimeSeriesResampler(sz=40).fit_transform(X)
        X = np.squeeze(X, axis=-1)

        y = np.concatenate((y_train, y_test), axis=0)
        y = y - 1  # Cluster numbers start from 0

        X, y = shuffle(X, y, random_state=config['RANDOM_SEED'])

        if return_labels:
            return y
        else:
            return X
        
    else:
        folder = "trace" 
        dataset_name = "trace"
        feature_space = "temporal"
        if config["feat_config"] == "with_feats":
            with open(f"data/{folder}/{dataset_name}_with_features.p", "rb") as f:
                data = pickle.load(f)

        elif config["feat_config"] == "feat_only":
            with open(f"data/{folder}/{dataset_name}_features_{feature_space}.p", "rb") as f:
                data = pickle.load(f)

        return data 