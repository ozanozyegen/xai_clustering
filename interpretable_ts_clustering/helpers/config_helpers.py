import itertools

def get_permutations(hyper_config_dict):
    configs = []
    keys, values = zip(*hyper_config_dict.items())
    for experiment in itertools.product(*values):
        configs.append({key:value for key, value in zip(keys, experiment)})
    return configs