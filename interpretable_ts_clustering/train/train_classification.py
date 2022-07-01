import pickle

import wandb, os
import numpy as np
import json

from sklearn.model_selection import train_test_split

from interpretable_ts_clustering.configs.defaults import Globs
from interpretable_ts_clustering.helpers.wandb_common import restore_wandb_online, wandb_save, convert_wandb_config_to_dict,\
    restore_wandb_online
from interpretable_ts_clustering.helpers.gpu_selection import auto_gpu_selection
from interpretable_ts_clustering.data.loader import data_loader
from interpretable_ts_clustering.models.loader import classification_model_loader
from collections import Counter
import numpy as np
import pandas as pd
import argparse

def downsample_largest_cluster(X, y):
    y_counter = Counter(y)
    print(y_counter)
    largest_cluster = y_counter.most_common()[0][0]
    largest_cluster_samples = y_counter.most_common()[0][1]
    remaining = y_counter.most_common()[1:]
    n_samples_per_cluster_remaining = [i[1] for i in remaining]
    mean_n_samples_remaining = sum(n_samples_per_cluster_remaining)/len(n_samples_per_cluster_remaining)

    indices_most_common = []

    for label_index, label in enumerate(y):
        if label == largest_cluster:
            indices_most_common.append(label_index)

    # delete indices such taht only have mean_n_samples_remaining
    indices_to_delete = np.random.choice(indices_most_common, size= int(largest_cluster_samples-mean_n_samples_remaining))

    X_subset = np.delete(X, indices_to_delete, axis=0)
    y_subset = np.delete(y, indices_to_delete, axis=0)

    print(X_subset.shape)
    print(y_subset.shape)
    return X_subset, y_subset

def remove_low_freq_labels(X, y, threshold=4):
    df = pd.DataFrame(y, columns = ["cluster_label"])
    value_counts = df.stack().value_counts() # Entire DataFrame
    to_remove = value_counts[value_counts <= threshold].index
    df.replace(to_remove, np.nan, inplace=True)
    to_remove_indices = df[df["cluster_label"].isna()].index
    X_new = np.delete(X, to_remove_indices, axis=0)
    y_new = np.delete(y, to_remove_indices, axis=0)

    # remap to sequential labels (bcs of gaps due to removal of labels)
    y_counter = Counter(y_new)
    n_classes = len(y_counter.keys())
    new_unique_labels = list(range(n_classes))

    remapping_dict = {}
    for index, class_key in enumerate(y_counter.keys()):
        remapping_dict[class_key] = new_unique_labels[index]

    for label_index, label in enumerate(y_new):
        y_new[label_index] = remapping_dict[label]

    return X_new, y_new, n_classes

def preprocess_fcn(x_train, x_test, y_train, y_test, nb_classes, data="raw"):
    y_train = (y_train - y_train.min())/(y_train.max()-y_train.min())*(nb_classes-1)
    y_test = (y_test - y_test.min())/(y_test.max()-y_test.min())*(nb_classes-1)
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    x_train_mean = x_train.mean()
    x_train_std = x_train.std()
    x_train = (x_train - x_train_mean)/(x_train_std)
    x_test = (x_test - x_train_mean)/(x_train_std)
    if data == "raw":
        x_train = x_train.reshape(x_train.shape + (1,))
        x_test = x_test.reshape(x_test.shape + (1,))
    else:
        x_train = x_train.reshape(x_train.shape + (1,1))
        x_test = x_test.reshape(x_test.shape + (1,1))
    return x_train, x_test, y_train, y_test

def train_classification(config, clusters, wandb_tags=['classification']):
    wandb_save(is_online=True)
    run = wandb.init(project=Globs.PROJECT_NAME, config=config,
                     entity=Globs.WANDB_ENTITY,
                     tags=wandb_tags, reinit=True)

    with run:
        config = wandb.config
        X, y = data_loader[config['DATASET']](config), clusters

        config["HISTORY_SIZE"] = X.shape[1]

        if config["BALANCING"] == "downsample_largest":
            X, y= downsample_largest_cluster(X,y)

        print(f"Data shape: {X.shape}, {y.shape}")

        if config["DATASET"] != "trace":
            X, y, new_n_classes = remove_low_freq_labels(X, y, threshold=4)
            config["NEW_N_CLASSES"] = new_n_classes
        else:
            config["NEW_N_CLASSES"] = config["N_CLUSTERS"]

        print(f"Data shape after remove low freq: {X.shape}, {y.shape}")

        train_x, test_x, train_y, test_y = train_test_split(X, y,
            test_size=0.3, random_state=config['RANDOM_SEED'], stratify = y)

        if config["CLASSIFICATION_MODEL"] == "fcn":
            # train_x, train_y, test_x, test_y = CachedDatasets().load_dataset("Trace")
            train_x, test_x, train_y, test_y = preprocess_fcn(train_x, test_x, train_y, test_y, nb_classes=config["NEW_N_CLASSES"],
                data="processed")
            config["input_layer_shape"]= train_x.shape[1:]

        model = classification_model_loader[config['CLASSIFICATION_MODEL']](
            convert_wandb_config_to_dict(config))

        if config['CLASSIFICATION_MODEL'] in ["fcn", "lstm"]:
            history = model.train(train_x, train_y)
            config["N_EPOCHS_TRAINED"] = len(history.history['loss'])
        else:
            model.train(train_x, train_y)

        model.log_results(test_x, test_y, wandb, per_cluster=True)

        if config["CLASSIFICATION_MODEL"] not in ["nndtw", "lstm"]:
            pred_y = model.predict(train_x)
            # from interpretable_ts_clustering.visualization.xai import visualize_explanations
            # visualize_explanations(config, model, train_x, train_y,
            #     pred_y, wandb)

        model.save(wandb.run.dir)
        return model
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--stage', type=str, required=True)
    args = parser.parse_args()

    dataset= args.dataset
    stage = args.stage

    fcn_default_config = dict(
        N_FILTERS = [128,256,128],
        PADDING = 'SAME',
        KERNEL_SIZES = [8,5,3],
        BATCH_NORM = True,
        DROPOUT_RATE = 0.2,
        N_EPOCHS = 500,
        BATCH_SIZE = 32,
    )

    lstm_default_config= dict(
        NUM_LAYERS = 2,
        NUM_UNITS = 50,
        DROPOUT_RATE = 0.01,
        N_EPOCHS = 500,
        BATCH_SIZE = 32,
    )

    cluster_model_dict_path = "data/v5_cluster_models.json"
    with open(cluster_model_dict_path, "r") as f:
        cluster_models_dict = json.load(f)

    feat_configs = ["feat_only", None, "with_feats"]

    if dataset == "trace":
        cluster_models = ["kmeans"]
        clustering_init_list = [None]
        n_cluster_list = [3]
        patience= 30
        start_epoch = 150
        balancing = None
        monitor = "val_loss"

    elif dataset in ['pricing_data', "pricing_with_features", "pricing_features_temporal", "pricing_features_statistical"]:
        cluster_models = ["pattern_kmedoid"]
        clustering_init_list = ["random"]
        n_cluster_list = [25, 50, 75, 100]
        patience = 30
        start_epoch = 150
        balancing = "downsample_largest"
        monitor = "val_accuracy"

    elif dataset == "walmart":
        cluster_models = ["kmeans"]
        clustering_init_list = [None]
        n_cluster_list = [10, 20, 30, 40]
        patience = 10
        start_epoch = 20
        balancing = None
        monitor = "val_accuracy"

    elif dataset in  ["electricity_daily", "electricity_hourly"]:
        cluster_models = ["kmeans"]
        clustering_init_list = [None]
        n_cluster_list = [10, 20, 30, 40]
        balancing = None
        start_epoch = 250
        patience = 50
        monitor = "val_accuracy"

    default_config = dict(
        DATASET= dataset,
        RANDOM_SEED=Globs.RANDOM_SEED,
        STAGE=  stage
    )

    knn_metric = "minkowski"
    class_models = ["xgb", "fcn", "nndtw"]

    learning_rates = [0.001]
    optimizers = ["adam"]

    min_delta = 0

    # how many samples in one cluster for plotting
    plot_min_samples = 5
    plot_max_samples = 20

    for class_model_name in class_models:
        for feat_config in feat_configs:
            for optimizer in optimizers:
                for learning_rate in learning_rates:
                    for clustering_init in clustering_init_list:
                        for clustering_model in cluster_models:
                            for n_clusters in n_cluster_list:
                                config = default_config
                                config["BALANCING"] = balancing
                                config["CLUSTERING_MODEL"] = clustering_model
                                config["N_CLUSTERS"] = n_clusters
                                config["NUM_CLASSES"] = n_clusters
                                config["CLUSTER_INIT"] = clustering_init
                                config["MIN_DELTA"] = min_delta
                                config["PATIENCE"] = patience
                                config["feat_config"] = feat_config
                                config["PLOT_MIN_SAMPLES"]= plot_min_samples
                                config["PLOT_MAX_SAMPLES"] = plot_max_samples
                                config["START_EPOCH"] = start_epoch
                                config["MONITOR"] = monitor
                                if class_model_name == "nndtw":
                                    config["N_NEIGHBORS"] = 5
                                    config["KNN_METRIC"] = knn_metric

                                if dataset != "trace":
                                    dataset_key = dataset
                                    # if dataset in ["pricing_with_features", "pricing_features_temporal", "pricing_features_statistical"]:
                                    #     dataset_key = "pricing_with_features"

                                    cluster_model_key = f"{dataset_key}_{clustering_model}_{n_clusters}_{clustering_init}"
                                    if (cluster_model_key in cluster_models_dict.keys()):
                                        cluster_model_id = cluster_models_dict[cluster_model_key]
                                        cluster_filename = 'clusters.npy'
                                        if os.path.exists(cluster_filename):
                                            os.remove(cluster_filename)
                                        cluster_path = wandb.restore(
                                        cluster_filename, run_path=f"{Globs.WANDB_ENTITY}/{Globs.PROJECT_NAME}/{cluster_model_id}")
                                        clusters = np.load(cluster_path.name)

                                    else:
                                        print(f"no cluster model for {cluster_model_key}")
                                        continue

                                else:
                                    cluster_model_id = "trace_default"
                                    with open('data/trace/clusters.npy', 'rb') as f:
                                        clusters = np.load(f)

                                class_model_config = config.copy()

                                class_model_config['CLASSIFICATION_MODEL'] = class_model_name
                                class_model_config['cluster_model_id'] = cluster_model_id

                                if class_model_config['CLASSIFICATION_MODEL'] == "fcn":
                                    class_model_config.update(fcn_default_config)
                                    class_model_config["LR"] = learning_rate
                                    class_model_config["OPTIMIZER"] = optimizer

                                elif class_model_config["CLASSIFICATION_MODEL"] == "lstm":
                                    class_model_config.update(lstm_default_config)

                                    class_model_config["LR"] = learning_rate
                                    class_model_config["OPTIMIZER"] = optimizer

                                model = train_classification(class_model_config, clusters)





