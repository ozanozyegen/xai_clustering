from abc import ABC, abstractmethod
from turtle import color
from typing import Union
import os, pickle
from skimage.measure import block_reduce
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from interpretable_ts_clustering.models.classification import DecisionTreeWrapper, XGBWrapper
import tensorflow as tf
from interpretable_ts_clustering.visualization.shap_features import shap_method_dict

def visualize_explanations(config, model, X, y_true, y_pred, wandb):
    """ Saves visualizations of local explanations """
    if config['CLASSIFICATION_MODEL'] == 'fcn':
        cam_explainer = CAMExplainer(config, model)
        cam_explainer.compute_contributions(X)
        cam_explainer.visualize_contributions(y_true, y_pred, wandb)

    elif config['CLASSIFICATION_MODEL'] == 'xgb':
        shap_explainer = SHAPExplainer(config, model)
        shap_explainer.compute_contributions(X, method_name='treeshap')
        shap_explainer.visualize_contributions(y_true, y_pred, wandb)
        feat_explainer = FeatImpExplainer(config, model)
        feat_explainer.compute_contributions()
        feat_explainer.visualize_contributions(wandb)

class Explainer(ABC):
    def __init__(self, config: dict, model) -> None:
        super().__init__()
        self.config = config
        self.model = model

    @abstractmethod
    def compute_contributions(self, X):
        pass

    def visualize_global_feat_imp(self, feat_imps, title, wandb=None):
        history_size = len(feat_imps)
        x_range = list(range(0, history_size))
        # x_ticks = list(range(-history_size, 0))
        x_ticks = list(range(0, history_size))
        fig, ax = plt.subplots()
        ax.plot(x_range, feat_imps)
        ax.plot(x_range, self.aggregate_windows(feat_imps), 'r-')
        ax.set_ylabel(f'{title} Feature Importance')
        ax.set_xlabel('History')
        plt.xticks(x_range, x_ticks)
        plt.locator_params(axis='x', nbins=10)
        plt.title(f'{title} Feature Importance')
        if wandb is not None:
            save_dir = os.path.join(wandb.run.dir, 'global_explanations')
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, f'{title}_feature_importance.png'))
        else:
            plt.show()

    def save(self, SAVE_DIR: str, file_name='contributions.pickle'):
        file_path = os.path.join(SAVE_DIR, file_name)
        pickle.dump(self.config, open(file_path, 'wb'))

    def load(self, SAVE_DIR: str, file_name='contributions.pickle'):
        file_path = os.path.join(SAVE_DIR, file_name)
        self.contributions = pickle.load(open(file_path, 'rb'))

    @staticmethod
    def aggregate_windows(x: np.ndarray, split_size=10):
        """ Creates windows of mean values
        x (np.ndarray): 1D array
        split_size (int): split window size
        """
        total_window_size = len(x)
        # https://stackoverflow.com/questions/15956309/averaging-over-every-n-elements-of-a-numpy-array
        x_reduced = block_reduce(x, block_size=(split_size,),
                                    func=np.mean,
                                    cval=np.mean(x))

        x_repeat = np.repeat(x_reduced, split_size)
        x_trimmed = x_repeat[:total_window_size]
        return x_trimmed

class GiniTreeExplainer(Explainer):
    def __init__(self, config: dict, model: Union[XGBWrapper, DecisionTreeWrapper]) -> None:
        super().__init__(config, model)

    def compute_contributions(self, ):
        # contributions = (features,)
        self.contributions = self.model.model.feature_importances_

    def visualize_contributions(self, wandb=None):
        self.visualize_global_feat_imp(self.contributions, 'gini', wandb)

class CAMExplainer(Explainer):
    def __init__(self, config: dict, model) -> None:
        super().__init__(config, model)

    def compute_contributions(self, X):
        grad_model = tf.keras.Model(
        inputs=[self.model.model.inputs],
        outputs=[self.model.model.layers[10].output,
        self.model.model.output])

        with tf.GradientTape() as tape:
            (conv_outputs, predictions) = grad_model(X)
            loss = predictions[:,0]
            grads = tape.gradient(loss, conv_outputs)
            casted_conv_outputs = tf.cast(conv_outputs > 0, "float32")
            casted_grads = tf.cast(grads > 0, "float32")

            guided_grads = casted_conv_outputs * casted_grads * grads

            weights = tf.reduce_mean(guided_grads, axis=(0, 1))
            cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1)

        # samples = X
        samples =  np.squeeze(X)
        self.contributions = (cam, samples)

    def visualize_contributions(self, y_true, y_pred, wandb=None):
        labels = np.unique(y_true)
        cam, samples = self.contributions

        history_size = samples.shape[1]
        x_range = list(range(0, history_size))
        # x_ticks = list(range(-history_size, 0))
        x_ticks = list(range(0, history_size))

        # Visualize single sample contributions
        for i in range(20):
            if y_true[i] != y_pred[i]:
                continue

            sample_contributions = np.abs(cam.numpy()[i])
            y_class = y_true[i]

            y_range_series = [np.min(samples) - 0.1, np.max(samples) + 0.1]
            y_range_contrib = [np.min(cam), np.max(cam)]
            fig, ax1 = plt.subplots()
            ax1.plot(x_range, samples[i], color='blue')
            ax1.set_ylabel('Time Series')
            ax1.set_ylim(y_range_series)

            plt.xticks(x_range, x_ticks)
            plt.locator_params(axis='x', nbins=10)
            plt.xlabel('History')
            plt.title(f'Sample: {i} from Cluster: {y_class}')

            ax2 = ax1.twinx()
            ax2.fill_between(x_range, [0 for _ in range(len(x_range))],
                sample_contributions,
                alpha=0.5, edgecolor='red', facecolor='red')
            ax2.set_ylabel('Contributions')
            ax2.set_ylim(y_range_contrib)
            fig.tight_layout()

            if wandb is not None:
                save_dir = os.path.join(wandb.run.dir, 'cam_local_explanations')
                os.makedirs(save_dir, exist_ok=True)
                fig.savefig(os.path.join(save_dir,
                         f'sample_{i}_cluster_{y_class}.png'))
            else:
                plt.show()

        # aggregate local explanations
        n_samples = self.config["PLOT_MAX_SAMPLES"]

        for label in labels:
            label_samples_idxs = np.where(y_pred == label)[0][:n_samples]
            sample_contributions = np.abs(cam.numpy()[label_samples_idxs])
            mean_absolute_contributions = np.mean((sample_contributions), axis=0)
            fig, axs = plt.subplots(2, 1, sharex=True)
            plt.xticks(x_range, x_ticks)
            plt.locator_params(axis='x', nbins=10)

            for sample_contrib in sample_contributions:
                axs[0].plot(sample_contrib, "k-", alpha=.2)
            axs[0].plot(self.aggregate_windows(mean_absolute_contributions), "r-")
            # ax.plot(mean_absolute_contributions, "r-")
            axs[0].set_ylabel('CAM values')
            axs[0].set_title(f'Mean feature importance for cluster: {label}')

            for sample in samples[label_samples_idxs]:
                axs[1].plot(sample, "k-", alpha=.2)
            axs[1].set_ylabel('Feature value')
            axs[1].set_title(f'Samples from cluster: {label}')

            if wandb is not None:
                save_dir = os.path.join(wandb.run.dir, 'cam_aggregate_local_explanations')
                os.makedirs(save_dir, exist_ok=True)
                fig.savefig(os.path.join(save_dir,
                         f'cluster_{label}.png'))
            else:
                plt.show()

class FeatImpExplainer(Explainer):
    def __init__(self, config: dict, model) -> None:
        super().__init__(config, model)
    def compute_contributions(self):
        self.feat_imps = self.model.model.feature_importances_
    def visualize_contributions(self, wandb=None):
        plt.plot(self.feat_imps, color="r")
        plt.xlabel("Timestep")
        plt.xlabel("Feature Importance")

        if wandb is not None:
            save_dir = os.path.join(wandb.run.dir, 'feature_importances')
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir,
                        f'feat_imp.png'))
        else:
            plt.show()

class SHAPExplainer(Explainer):
    def __init__(self, config: dict, model) -> None:
        super().__init__(config, model)

    def compute_contributions(self, X, method_name: str,
                              nsamples=100000, random_state=0):
        shap_val, samples = shap_method_dict[method_name](self.model.model, X,
                                                          nsamples,
                                                          random_state)
        self.contributions = (shap_val, samples)

    def visualize_contributions(self, y_true, y_pred,wandb=None,  nsamples=None):
        shap_val, samples = self.contributions
        history_size = samples.shape[1]
        x_range = list(range(0, history_size))
        # x_ticks = list(range(-history_size, 0))
        x_ticks = list(range(0, history_size))
        # Visualize global explanation
        global_shap_vals = np.abs(shap_val).sum(axis=0).sum(axis=0)
        # taking sum over all samples
        self.visualize_global_feat_imp(global_shap_vals, 'shap', wandb)
        # Visualize single sample contributions

        for i in range(20):
            if y_true[i] != y_pred[i]:
                continue

            sample_contributions = shap_val[y_true[i], i, :]
            y_class = y_true[i]

            y_range_series = [np.min(samples) - 0.1, np.max(samples) + 0.1]
            y_range_contrib = [np.min(shap_val), np.max(shap_val)]
            fig, ax1 = plt.subplots()
            ax1.plot(x_range, samples[i], color='blue')
            ax1.set_ylabel('Time Series')
            ax1.set_ylim(y_range_series)

            plt.xticks(x_range, x_ticks)
            plt.locator_params(axis='x', nbins=10)
            plt.xlabel('History')
            plt.title(f'Sample: {i} from Cluster: {y_class}')

            ax2 = ax1.twinx()
            ax2.fill_between(x_range, [0 for _ in range(len(x_range))],
                sample_contributions,
                alpha=0.5, edgecolor='blue', facecolor='blue')
            ax2.set_ylabel('Contributions')
            ax2.set_ylim(y_range_contrib)
            fig.tight_layout()

            if wandb is not None:
                save_dir = os.path.join(wandb.run.dir, 'local_explanations')
                os.makedirs(save_dir, exist_ok=True)
                fig.savefig(os.path.join(save_dir,
                         f'sample_{i}_cluster_{y_class}.png'))
            else:
                plt.show()

        # Visualize window aggregate contributions

        labels = np.unique(y_true)
        for label in labels:
            # go by cluster
            label_samples_idxs = np.where(y_pred == label)[0]
            sample_contributions = np.abs(shap_val[label, label_samples_idxs])
            mean_absolute_contributions = np.mean(np.abs(sample_contributions), axis=0)

            fig, ax = plt.subplots()
            plt.xticks(x_range, x_ticks)
            plt.locator_params(axis='x', nbins=10) # is 10 good for any scenario

            for sample_contrib in sample_contributions:
                ax.plot(sample_contrib, "k-", alpha=.2)
            ax.plot(self.aggregate_windows(mean_absolute_contributions), "r-")
            # ax.plot(mean_absolute_contributions, "r-")
            ax.set_ylabel('Feature Importances')
            ax.set_title(f'Mean feature importance for cluster: {label}')

            if wandb is not None:
                save_dir = os.path.join(wandb.run.dir, 'aggregate_local_explanations')
                os.makedirs(save_dir, exist_ok=True)
                fig.savefig(os.path.join(save_dir,
                         f'cluster_{label}.png'))
            else:
                plt.show()

        # Side by side aggregate figures
        labels = np.unique(y_true)
        for label in labels:
            label_samples_idxs = np.where(y_pred == label)[0]
            sample_contributions = np.abs(shap_val[label, label_samples_idxs])
            mean_absolute_contributions = np.mean(np.abs(sample_contributions), axis=0)

            fig, axs = plt.subplots(2, 1, sharex=True)
            plt.xticks(x_range, x_ticks)
            plt.locator_params(axis='x', nbins=10)

            for sample_contrib in sample_contributions:
                axs[0].plot(sample_contrib, "k-", alpha=.2)
            axs[0].plot(self.aggregate_windows(mean_absolute_contributions), "r-")
            # ax.plot(mean_absolute_contributions, "r-")
            axs[0].set_ylabel('Feature Importances')
            axs[0].set_title(f'Mean feature importance for cluster: {label}')

            for sample in samples[label_samples_idxs]:
                axs[1].plot(sample, "k-", alpha=.2)
            axs[1].set_ylabel('Feature value')
            axs[1].set_title(f'Samples from cluster: {label}')

            if wandb is not None:
                save_dir = os.path.join(wandb.run.dir, 'aggregate_local_explanations')
                os.makedirs(save_dir, exist_ok=True)
                fig.savefig(os.path.join(save_dir,
                         f'cluster_{label}.png'))
            else:
                plt.show()

def normalize_data(data):
    return ((data - np.min(data)) / (np.max(data) - np.min(data)))

def get_top_features(mean_absolute_contributions, n_top, extra_features, extra_feat_start, label, mode="default"):
    def index_to_string(result, extra_feat_start, extra_features, extra_only):
        if extra_only:
            feat_string = extra_features[result]
        else:
            if result>=extra_feat_start:
                feat_index = result - extra_feat_start
                feat_string = extra_features[feat_index]
            else:
                feat_string = f"timestep_{result}"
        return feat_string

    if mode == "extra_only":
        feat_impt = mean_absolute_contributions[extra_feat_start:]
        extra_only = True
    elif mode == "ts_only":
        feat_impt = mean_absolute_contributions[:extra_feat_start]
        extra_only = False
    elif mode=="default":
        feat_impt = mean_absolute_contributions
        extra_only=False

    ind = np.argpartition(feat_impt, -n_top)[-n_top:]
    top_sorted = ind[np.argsort(feat_impt[ind])] #top to bottom

    top_features = {
        "index": top_sorted,
        "value": feat_impt[top_sorted]
    }

    top_features["index_string"] = [index_to_string(feat, extra_feat_start, extra_features, extra_only) for feat in top_features["index"]]

    undefined_counter = 0
    for index, i in enumerate(top_sorted):
        if top_features["value"][index] == 0:
            undefined_counter += 1
            top_features["index_string"][index] = f"zero_{undefined_counter}"

    top_features["label"] = label
    return top_features

def plot_top_features(features_list, class_config, save_dir, model, mode):
    rows = len(features_list)
    fig, axs = plt.subplots(rows, 1, figsize=(15,10))
    for index, top_features in enumerate(features_list):
        axs[index].barh(top_features["index_string"], top_features["value"])
        axs[index].set_ylabel(top_features["label"])

    plt.tight_layout()
    fig.savefig(f"{save_dir}/{class_config['DATASET']}_barplots_{model}_{mode}")

def plot_xgb(train_x, class_config, class_model, save_dir, extra_features, n_top = 10, fontsize = 20, tick_fontsize=18):
    shap_explainer = SHAPExplainer(class_config, class_model)
    shap_explainer.compute_contributions(train_x, method_name='treeshap')
    feat_explainer = FeatImpExplainer(class_config, class_model)
    feat_explainer.compute_contributions()
    feat_imps = feat_explainer.feat_imps
    shap_val, shap_samples = shap_explainer.contributions

    feat_impt_dict = {
        "treeshap": shap_val,
        "feat_imp": feat_imps
    }

    with open(f"{save_dir}/{class_config['DATASET']}_xgb_xai_dict.p", "wb") as f:
        pickle.dump(feat_impt_dict, f)

    history_size = shap_samples.shape[1]
    x_range = list(range(0, history_size))
    x_ticks = list(range(0, history_size))

    global_shap_vals = np.abs(shap_val).sum(axis=0).sum(axis=0)

    fig, axs = plt.subplots(3,1, sharex="row", figsize=(15,15))

    plt.xticks(x_range, x_ticks)
    plt.locator_params(axis='x', nbins=10) # is 10 good for any scenario
    for index, i in enumerate(axs):
        axs[index].tick_params(axis='both', which='major', labelsize=tick_fontsize)

    axs[0].plot(global_shap_vals, "r")
    axs[0].set_ylabel("TreeSHAP", fontsize = fontsize)
    axs[2].plot(feat_imps, "b")
    axs[2].set_ylabel("Tree feature importance",  fontsize = fontsize)
    axs[1].set_ylabel("Feature values",  fontsize = fontsize)
    for sample in shap_samples[:20]:
        axs[1].plot(sample, "k-", alpha=.2)

    plt.tight_layout()
    fig.savefig(f"{save_dir}/{class_config['DATASET']}_xgb_shap_feat.pdf", format="pdf")

    modes = ["default", "extra_only", "ts_only"]
    extra_feat_start = class_config["HISTORY_SIZE"] - extra_features.shape[0]
    for mode in modes:
        shap_features = get_top_features(global_shap_vals, n_top, extra_features, extra_feat_start, label="shap", mode=mode)
        feat_imp_features = get_top_features(feat_imps, n_top, extra_features, extra_feat_start, label="feat_imp", mode=mode)
        features_list = [shap_features, feat_imp_features]
        plot_top_features(features_list, class_config, save_dir, model="xgb", mode=mode)

def plot_fcn(train_x, train_y, class_config, class_model, save_dir, extra_features, n_top = 10, fontsize = 20, tick_fontsize=18):
    pred_y = class_model.predict(train_x).argmax(axis=1)
    y_pred = pred_y
    labels = np.unique(y_pred)

    cam_explainer = CAMExplainer(class_config, class_model)
    cam_explainer.compute_contributions(train_x)
    shap_explainer = SHAPExplainer(class_config, class_model)
    shap_explainer.compute_contributions(train_x, method_name='gradshap')
    shap_val, shap_samples = shap_explainer.contributions
    cam_val, cam_samples = cam_explainer.contributions

    feat_impt_dict = {
        "gradshap": shap_val,
        "gradcam": cam_val
    }

    with open(f"{save_dir}/{class_config['DATASET']}_fcn_xai_dict.p", "wb") as f:
        pickle.dump(feat_impt_dict, f)

    print(f"shap val shape: {shap_val.shape}")
    print(f"cam val shape: {cam_val.shape}")
    global_shap_vals = normalize_data(np.abs(shap_val).sum(axis=0).sum(axis=0))
    global_cam_vals = normalize_data(np.abs(cam_val).sum(axis=0))

    pred_y = class_model.predict(train_x).argmax(axis=1)
    history_size = shap_samples.shape[1]
    x_range = list(range(0, history_size))

    y_true = train_y
    n_samples = class_config["PLOT_MAX_SAMPLES"]

    extra_feat_start = class_config["HISTORY_SIZE"] - extra_features.shape[0]

    indi_list = [None, True]

    modes = ["default", "extra_only", "ts_only"]
    for mode in modes:
        gradshap_top_features = get_top_features(global_shap_vals, n_top, extra_features, extra_feat_start, label="gradshap", mode=mode)
        cam_top_features = get_top_features(global_cam_vals, n_top, extra_features, extra_feat_start, label="cam", mode=mode)
        features_list = [cam_top_features, gradshap_top_features]
        plot_top_features(features_list, class_config, save_dir, model="fcn", mode=mode)

    for label in labels:
        # go by cluster
        # print(np.where(y_pred == label))
        label_samples_idxs = list(np.where(y_pred == label)[0])

        for i in label_samples_idxs:
            if y_true[i] != y_pred[i]:
                label_samples_idxs.remove(i)

        label_samples_idxs = np.array(label_samples_idxs)

        if len(label_samples_idxs) != 0:

            sample_contributions = np.abs(shap_val[label, label_samples_idxs])
            # sample_contributions = np.array([normalize_data(sample_contrib) for sample_contrib in sample_contributions])
            mean_absolute_contributions = np.mean(np.abs(sample_contributions), axis=0)

            cam_sample_contributions = np.abs(cam_val.numpy()[label_samples_idxs])
            # cam_sample_contributions = np.array([normalize_data(sample_contrib) for sample_contrib in cam_sample_contributions])
            cam_mean_absolute_contributions = np.mean((cam_sample_contributions), axis=0)

            for plot_individual_samples in indi_list:
                print(f"plot_individual: {plot_individual_samples}")
                fig, axs = plt.subplots(3,1, sharex=True, figsize=(15,15))
                plt.xticks(x_range, x_range)
                plt.locator_params(axis='x', nbins=10) # is 10 good for any scenario

                for index, i in enumerate(axs):
                    axs[index].tick_params(axis='both', which='major', labelsize=tick_fontsize)

                if plot_individual_samples:
                    for sample_contrib in sample_contributions:
                        axs[0].plot(sample_contrib, "k-", alpha=.2)
                    for cam_sample_contrib in cam_sample_contributions:
                        axs[2].plot(cam_sample_contrib, "k-", alpha=.2)

                axs[0].plot(shap_explainer.aggregate_windows(mean_absolute_contributions), "r-")
                axs[0].set_ylabel('GradientSHAP values', fontsize = fontsize)

                axs[2].plot(cam_explainer.aggregate_windows(cam_mean_absolute_contributions), "r-")
                axs[2].set_ylabel('Grad-CAM values', fontsize = fontsize)

                for sample in shap_samples[label_samples_idxs]:
                    axs[1].plot(sample, "k-", alpha=.2)
                axs[1].set_ylabel('Feature values', fontsize = fontsize)

                plt.tight_layout()

                if plot_individual_samples:
                    fig.savefig(f"{save_dir}/{class_config['DATASET']}_samples_fcn_plot_{label}.pdf", format="pdf")
                else:
                    fig.savefig(f"{save_dir}/{class_config['DATASET']}_fcn_plot_{label}.pdf", format = "pdf")
