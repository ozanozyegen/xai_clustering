import json
import os
import numpy as np
import matplotlib.pyplot as plt


def save_clusters(model, X, clusters, wandb, min_samples=None, max_samples=None):
    n_clusters = wandb.config['N_CLUSTERS']
    centroids = model.get_centroids()
    class_dist_dict = dict()
 
    save_dir = os.path.join(wandb.run.dir, 'clusters')
    os.makedirs(save_dir, exist_ok=True)

    for i in range(n_clusters):
        fig = plt.figure()
        idxs = np.where(clusters == i)[0]
        class_dist_dict[i] = len(idxs)

        if idxs.shape[0] >= min_samples: 
            for idx in idxs[:max_samples]:
                plt.plot(X[idx], 'k-', alpha=.2)
            # plt.plot(centroids[i], 'r-')
            plt.title(f'Cluster: {i}, Size: {len(idxs)}')
            fig.savefig(os.path.join(save_dir, f'cluster_{i}.png'))

    json.dump(class_dist_dict, 
        open(os.path.join(wandb.run.dir, 'cluster_dist.json'), 'w'))