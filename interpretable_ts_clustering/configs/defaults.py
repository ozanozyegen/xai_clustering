from interpretable_ts_clustering.helpers.config_helpers import get_permutations


class Globs:
    PROJECT_NAME = "<proj_name>"
    WANDB_ENTITY = '<entity'
    DATASETS = ['pricing_data']
    CLUSTERING_MODELS = ['kmeans']
    CLASSIFICATION_MODELS = ['svm', 'xgb']
    RANDOM_SEED = 0


_clustering = dict(
    DATASET=Globs.DATASETS,
    CLUSTERING_MODEL=Globs.CLUSTERING_MODELS,
    N_CLUSTERS=[20],
    RANDOM_SEED=[Globs.RANDOM_SEED],
)

clustering_configs = get_permutations(_clustering)
