import tsfel
import pandas as pd 
import numpy as np 

def append_features(data):
    
    # extract temporal features 
    cfg_file = tsfel.get_features_by_domain("temporal") 
    data_features = tsfel.time_series_features_extractor(cfg_file, data)  
    
    data_df = pd.DataFrame(data)

    data_features_normalized =  data_features.copy()

    data_features_normalized["mean"] = data_df.mean(axis=1)
    data_features_normalized["max"] = data_df.max(axis=1)
    data_features_normalized["min"] = data_df.min(axis=1)
    data_features_normalized["variance"] = data_df.var(axis=1)
    data_features_normalized["std_dev"] = data_df.std(axis=1)

    # normalize 
    for col in data_features_normalized.columns:
        min_col = data_features_normalized[col].min()
        max_col = data_features_normalized[col].max()
        data_features_normalized[col] =( data_features_normalized[col] - min_col) / (max_col - min_col)

    # drop Nan columns (if max or min is 0)
    data_features_normalized = data_features_normalized.dropna(axis=1, how='any')
    data_features_array = np.array(data_features_normalized)
    data_with_features= np.concatenate([data, data_features_array], axis=1)

    return data_with_features