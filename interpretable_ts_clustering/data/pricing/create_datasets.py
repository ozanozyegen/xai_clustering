import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

from interpretable_ts_clustering.data.pricing.convert_image import arr2imagearr
import pickle

def create_pricing_data(config):
    if config["feat_config"]==None:
        df = pd.read_csv('data/processed/time_series_prices_scaled.csv',
                        index_col=0)
        return df.values.T

    else:
        folder = "pricing"
        dataset_name = "pricing_data"
        feature_space = "temporal"
        if config["feat_config"] == "with_feats":
            with open(f"data/{folder}/{dataset_name}_with_features.p", "rb") as f:
                data = pickle.load(f)

        elif config["feat_config"] == "feat_only":
            with open(f"data/{folder}/{dataset_name}_features_{feature_space}.p", "rb") as f:
                data = pickle.load(f)
        return data

def load_pricing_with_features(config):
    with open("data/prices_with_features.p", "rb") as f:
        data_with_features= pickle.load(f)
    return data_with_features

def load_pricing_features_temp(config):
    with open("data/pricing_features_temporal.p", "rb") as f:
        data_with_features= pickle.load(f)
    return data_with_features

def load_pricing_features_stats(config):
    with open("data/pricing_features_statistical.p", "rb") as f:
        data_with_features= pickle.load(f)
    return data_with_features

def create_walmart_data(config):
    if config["feat_config"]==None:
        walmart_data = pd.read_csv("./data/walmart/train.csv")
        walmart_data["store_dept"] = walmart_data["Store"].astype(str) + "_" + walmart_data["Dept"].astype(str)
        # standard = 143 timesteps
        valid_stores = walmart_data["store_dept"].value_counts().index[np.where(walmart_data["store_dept"].value_counts()==143)]
        walmart_data = walmart_data[walmart_data["store_dept"].isin(valid_stores.values)]
        train_data = pd.pivot_table(walmart_data, index="store_dept", columns='Date', values='Weekly_Sales')
        train_data = train_data.apply(lambda data: (data - data.min())/ (data.max() - data.min()), axis=1)
        walmart_train_data = np.array(train_data)

    else:
        folder = "walmart"
        dataset_name = "walmart"
        feature_space = "temporal"
        if config["feat_config"] == "with_feats":
            with open(f"data/{folder}/{dataset_name}_with_features.p", "rb") as f:
                walmart_train_data = pickle.load(f)

        elif config["feat_config"] == "feat_only":
            with open(f"data/{folder}/{dataset_name}_features_{feature_space}.p", "rb") as f:
                walmart_train_data = pickle.load(f)
    return walmart_train_data

def create_electricity_data(config):
    elect_data_root_path = "data/elect_data/elect"
    # with open(, 'rb') as f:
    if config["feat_config"]==None:
        if config["DATASET"]=="electricity_daily":
            data = np.load(f'{elect_data_root_path}/data_daily.npy')
        elif config["DATASET"]=="electricity_hourly":
            data = np.load(f'{elect_data_root_path}/data_hourly.npy')
    else:
        folder = "electricity"
        dataset_name = config["DATASET"]
        feature_space = "temporal"
        if config["feat_config"] == "with_feats":
            with open(f"data/{folder}/{dataset_name}_with_features.p", "rb") as f:
                data = pickle.load(f)

        elif config["feat_config"] == "feat_only":
            with open(f"data/{folder}/{dataset_name}_features_{feature_space}.p", "rb") as f:
                data = pickle.load(f)
    return data

def create_imagearr_dataset(image_shape=(224, 224)):
    df = pd.read_csv('data/processed/time_series_prices_scaled.csv',
                     index_col=0)
    num_series = df.shape[1]
    data = np.empty((num_series, image_shape[0], image_shape[1]))
    for sample_idx in tqdm(range(num_series)):
        data[sample_idx] = arr2imagearr(df.iloc[:, sample_idx], image_shape)

    # arr = create_imagearr_dataset()
    np.save('data/processed/pricing_imagearr.npy', data)


def create_pricing_vgg_data():
    model = VGG16(weights='imagenet', include_top=False)
    # deeper layer, more abstract features
    # takes image, returns bunch of features before final layer
    # vgg model without last layer
    arr = np.load('data/processed/pricing_imagearr.npy')
    rgb_batch = np.repeat(arr[:, :, :, np.newaxis], 3, -1)
    x = preprocess_input(rgb_batch)
    features = model.predict(x)
    flattened_features = features.reshape(features.shape[0], -1)
    np.save('data/processed/pricing_vggarr.npy', flattened_features)


if __name__ == "__main__":
    # Generate the processed datasets
    create_imagearr_dataset()
    create_pricing_vgg_data()