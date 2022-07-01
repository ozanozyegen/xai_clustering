# from typing import Counter
import wandb, os
import numpy as np
import pickle
import json

from collections import Counter
from interpretable_ts_clustering.helpers.wandb_common import wandb_save, convert_wandb_config_to_dict
from interpretable_ts_clustering.data.loader import data_loader
from interpretable_ts_clustering.models.loader import clustering_model_loader
from interpretable_ts_clustering.configs.defaults import Globs, clustering_configs
from interpretable_ts_clustering.metrics.clustering import evaluate_clusters
from interpretable_ts_clustering.visualization.clustering import save_clusters

from interpretable_ts_clustering.data.preprocessing import append_features
import argparse

def train_clustering(config, wandb_tags=['clustering']):
    # if add_features:
    #     wandb_tags.append("extra_features")

    wandb_save(is_online=True)
    run = wandb.init(project=Globs.PROJECT_NAME, config=config,
                     entity=Globs.WANDB_ENTITY,
                     tags=wandb_tags, reinit=True)

    with run:
        config = wandb.config
        print(config)
        X = data_loader[config['DATASET']](config)

        print(f"Data shape: {X.shape}")
        # print(X)
        model = clustering_model_loader[config['CLUSTERING_MODEL']](
            convert_wandb_config_to_dict(config))

        if config["CLUSTERING_MODEL"] == "pattern_kmedoid":
            # use precomputed distances
            if config["DATASET"] == "pricing_with_features":
                print(config["DATASET"])
                dist_path = "data/dist_matrix.p"
            else:
                dist_path = "data/dist_matrix.p"

            with open(dist_path, "rb") as f:
                dist_matrix = pickle.load(f)

            clusters = model.generate_clusters(dist_matrix)
            evaluate_clusters(X, clusters, wandb)
            save_clusters(model, X, clusters, wandb,
            min_samples = config["PLOT_MIN_SAMPLES"],
                max_samples = config["PLOT_MAX_SAMPLES"])
        else:
            clusters = model.generate_clusters(X)
            inertia = model.get_inertia()
            wandb.log({"inertia": inertia})
            evaluate_clusters(X, clusters, wandb)
            save_clusters(model, X, clusters, wandb,
                min_samples = config["PLOT_MIN_SAMPLES"],
                max_samples = config["PLOT_MAX_SAMPLES"])

        model.save(wandb.run.dir)

        np.save(os.path.join(wandb.run.dir, 'clusters.npy'), clusters)

        print(wandb.run.dir)
        run_id = run.id
        run_dir = wandb.run.dir
        print(Counter(clusters))

    return config, run_id, run_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--stage', type=str, required=True)
    args = parser.parse_args()

    pricing_data_list = ["pricing_data", "pricing_with_features", "pricing_features_temporal", "pricing_features_statistical"]

    dataset= args.dataset
    stage = args.stage
    balancing = None

    if dataset == "trace":
        cluster_models = ["kmeans"]
        clustering_init_list = [None]
        n_cluster_list = [3]

    elif dataset in pricing_data_list:
        cluster_models = ["pattern_kmedoid"]
        clustering_init_list = ["random"]
        n_cluster_list = [25, 50, 75, 100]


    elif dataset == "walmart":
        cluster_models = ["kmeans"]
        clustering_init_list = [None]
        n_cluster_list = [10, 20, 30, 40]

    elif dataset in  ["electricity_daily", "electricity_hourly"]:
        cluster_models = ["kmeans"]
        clustering_init_list = [None]
        n_cluster_list = [10, 20, 30, 40]

    n_cluster_list = range(2,15)

    default_config = dict(
        DATASET= dataset,
        RANDOM_SEED=Globs.RANDOM_SEED,
        STAGE = stage
    )

    cluster_model_dict_path = "data/v5_cluster_models.json"
    with open(cluster_model_dict_path, "r") as f:
        cluster_models_dict = json.load(f)

    #how many samples in one cluster for plotting
    plot_min_samples = 5
    plot_max_samples = 20

    for clustering_init in clustering_init_list:
        for clustering_model in cluster_models:
            for n_clusters in n_cluster_list:
                cluster_model_key = f"{dataset}_{clustering_model}_{n_clusters}_{clustering_init}"

                if cluster_model_key not in cluster_models_dict.keys():

                    config = default_config

                    config["BALANCING"] = balancing
                    config["CLUSTERING_MODEL"] = clustering_model
                    config["N_CLUSTERS"] = n_clusters
                    config["NUM_CLASSES"] = n_clusters
                    config["CLUSTER_INIT"] = clustering_init
                    config["PLOT_MIN_SAMPLES"]= plot_min_samples
                    config["PLOT_MAX_SAMPLES"] = plot_max_samples
                    config["feat_config"] = None
                    config_wandb, cluster_model_id, run_dir = train_clustering(config)
                    cluster_models_dict[cluster_model_key] = cluster_model_id

                    with open(cluster_model_dict_path, "w") as f:
                        json.dump(cluster_models_dict, f)

                else:
                    print(f"{cluster_model_key} already exists, skipping")