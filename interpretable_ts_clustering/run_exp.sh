# python3 train/train_clustering.py --dataset "walmart" --stage "version6"
# python3 train/train_clustering.py --dataset "pricing_data" --stage "version6"
# python3 train/train_clustering.py --dataset "electricity_hourly" --stage "version6"
# python3 train/train_classification.py --dataset "trace" --stage "version6"
# python3 train/train_classification.py --dataset "walmart" --stage "version6"
# python3 train/train_classification.py --dataset "pricing_data" --stage "version6"
# python3 train/train_classification.py --dataset "electricity_hourly" --stage "version6"

python3 train/train_clustering.py --dataset "walmart" --stage "decide_cls_size"
python3 train/train_clustering.py --dataset "electricity_hourly" --stage "decide_cls_size"
python3 train/train_clustering.py --dataset "pricing_data" --stage "decide_cls_size"