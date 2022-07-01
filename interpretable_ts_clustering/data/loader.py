from interpretable_ts_clustering.data.pricing.preprocessing import load_pricing_imagearr_data, \
    load_pricing_vgg_data
from interpretable_ts_clustering.data.pricing.create_datasets import create_pricing_data,  load_pricing_with_features, \
     load_pricing_features_temp, load_pricing_features_stats, \
     create_walmart_data, create_electricity_data
from interpretable_ts_clustering.data.trace.preprocessing import load_trace


data_loader = dict(
    pricing_data=create_pricing_data,
    pricing_image=load_pricing_imagearr_data,
    pricing_vgg=load_pricing_vgg_data,
    trace=load_trace,
    pricing_with_features = load_pricing_with_features,
    pricing_features_temporal = load_pricing_features_temp,
    pricing_features_statistical = load_pricing_features_stats,
    walmart = create_walmart_data,
    electricity_hourly = create_electricity_data,
    electricity_daily = create_electricity_data
)
