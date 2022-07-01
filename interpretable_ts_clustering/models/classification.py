from abc import ABC, abstractmethod
from gc import callbacks
import tensorflow as tf
import os, pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, \
    f1_score, accuracy_score, classification_report, precision_score, recall_score, precision_recall_fscore_support
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

from tensorflow.keras.layers import Input, Reshape, LSTM, Dense, Conv1D,\
    BatchNormalization, Dropout, Activation, GlobalAveragePooling1D, GlobalAveragePooling2D, Flatten
from tensorflow.keras.models import Model
from tensorflow import keras
# from interpretable_ts_clustering.visualization.cam import cam_graph
from fastdtw import fastdtw
from sklearn.neighbors import KNeighborsClassifier

from collections import Counter
import json
import wandb

import matplotlib.pyplot as plt

def create_lstm(config):
    inp = Input(shape=(config['HISTORY_SIZE']))
    x = Reshape((config['HISTORY_SIZE'],1))(inp)

    if config['NUM_LAYERS']>1:
        for _ in range(config['NUM_LAYERS']-1):
            x = LSTM(units=config['NUM_UNITS'], dropout=config['DROPOUT_RATE'],
                return_sequences=True, activation= "tanh")(x)
                # default is tanh
    # have at least 1 layer
    x = LSTM(units=config['NUM_UNITS'], dropout=config['DROPOUT_RATE'],
            return_sequences=False, activation= "tanh")(x)

    out = Dense(config['NUM_CLASSES'], activation='softmax')(x)
    model = Model(inp, out)

    optimizer_dict = {
        "adam": keras.optimizers.Adam(learning_rate=config["LR"]),
        "sgd": keras.optimizers.SGD(learning_rate=config["LR"]),
    }
    opt = optimizer_dict[config["OPTIMIZER"]]

    model.compile(optimizer=opt , loss='sparse_categorical_crossentropy',
        metrics='accuracy')
    return model

def create_fcn(config):
    inp = Input(shape=(config['HISTORY_SIZE']))
    res = Reshape((config['HISTORY_SIZE'],1))(inp)
    x = Conv1D(filters=config['N_FILTERS'][0],
        kernel_size=config['KERNEL_SIZES'][0],
        padding=config['PADDING'])(res)
    if config['BATCH_NORM']:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(config['DROPOUT_RATE'])(x)
    x = Conv1D(filters=config['N_FILTERS'][1],
        kernel_size=config['KERNEL_SIZES'][1],
        padding=config['PADDING'])(x)
    if config['BATCH_NORM']:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(config['DROPOUT_RATE'])(x)
    x = Conv1D(filters=config['N_FILTERS'][2],
        kernel_size=config['KERNEL_SIZES'][2],
        padding=config['PADDING'])(x)

    print("conv: ", x.shape)

    if config['BATCH_NORM']:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # x = GlobalAveragePooling1D()(x)
    # print("pooling", x.shape)

    x = Flatten()(x)
    print("flatten: ", x.shape)

    out = Dense(config['NEW_N_CLASSES'],
        activation='softmax')(x)

    print("softmax: ", out.shape)
    model = Model(inp, out)

    optimizer_dict = {
        "adam": keras.optimizers.Adam(learning_rate=config["LR"]),
        "sgd": keras.optimizers.SGD(learning_rate=config["LR"]),
    }
    opt = optimizer_dict[config["OPTIMIZER"]]

    model.compile(optimizer=opt , loss='sparse_categorical_crossentropy',
        metrics='accuracy')

    return model

class BaseModel(ABC):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        self.model = None
    @abstractmethod
    def train(self,):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def save(self, SAVE_DIR):
        pass

    @abstractmethod
    def load(self, SAVE_DIR):
        pass

    def log_results(self, test_x, test_y, logger, per_cluster=None):
        pred_y = self.predict(test_x)
        # wandb_conf_matrix = logger.plot.confusion_matrix(
        #     y_true=y, preds=pred_y,
        #     class_names=list(range(self.config['N_CLUSTERS']))
        # )

        # sample_counter = Counter(pred_y)
        precision, recall, fscore, support = precision_recall_fscore_support(test_y, pred_y, average='weighted')

        logger.log({
            'acc': accuracy_score(test_y, pred_y),
            'f1': f1_score(test_y, pred_y, average='weighted'),
            'macro_f1': f1_score(test_y, pred_y, average='macro'),
            'micro_f1': f1_score(test_y, pred_y, average='micro'),
            "support": support,
            "precision": precision,
            "recall": recall,
            "fscore": fscore
            # 'conf_matrix': wandb_conf_matrix
        })

        if per_cluster:
            report = classification_report(y_true=test_y, y_pred = pred_y, output_dict=True)
            report_path = os.path.join(logger.run.dir, "per_cluster_results.json")
            report_dict = dict(report)

            with open(report_path, 'w') as fp:
                json.dump(report_dict, fp)

        # cluster_labels = list(range(self.config["NEW_N_CLASSES"]))
        # if self.config['CLASSIFICATION_MODEL'] == 'fcn':
        #     results = cam_graph(self.model, test_x, test_y,
        #         cluster_labels, self.config)
        #     np.save(os.path.join(logger.run.dir, 'cam.npy'), results)

        # if self.config["CLASSIFICATION_MODEL"]=="fcn":

        # logger.log(dict(report))

class CustomStopper(keras.callbacks.EarlyStopping):
    def __init__(self, *args, **kwargs):
        self.start_epoch = kwargs.pop('start_epoch')
        super(CustomStopper, self).__init__(*args, **kwargs)
        # self.start_epoch = kwargs["start_epoch"]

    # def __init__(self, monitor='val_loss',
    #          min_delta=0, patience=0, verbose=0, mode='auto', restore_best_weights=True, start_epoch = 100): # add argument for starting epoch
    #     super(CustomStopper, self).__init__()
    #     self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.start_epoch:
            super().on_epoch_end(epoch, logs)

class KerasModel(BaseModel):
    def __init__(self, config: dict):
        super().__init__(config)

    def predict(self, X):
        return self.model.predict(X).argmax(axis=1)

    def predict_proba(self, X):
        return self.model.predict(X)

    def save(self, SAVE_DIR):
        pickle.dump(self.config, open(os.path.join(SAVE_DIR, 'config.pickle'), 'wb'))
        self.model.save(os.path.join(SAVE_DIR, 'model.h5'))

    def load(self, SAVE_DIR):
        self.config = pickle.load(open(os.path.join(SAVE_DIR, 'config.pickle'), 'rb'))
        self.model = tf.keras.models.load_model(os.path.join(SAVE_DIR, 'model.h5'))

    def create_callbacks(self, wandb=None):
        """ Creates a list of callbacks for the Keras Model"""
        callbacks = []
        callbacks.append(CustomStopper(
            start_epoch= self.config["START_EPOCH"],
            monitor=self.config["MONITOR"],
            patience=self.config["PATIENCE"],
            min_delta = self.config["MIN_DELTA"],
            restore_best_weights=True))

        # callbacks.append(ModelCheckpoint(
        #     filepath='model-best.h5',
        #     monitor='val_accuracy', save_best_only=True
        #     )
        if wandb is not None:
            callbacks.append(wandb.keras.WandbCallback(monitor=self.config["MONITOR"]))
        return callbacks

    def train(self, X, y):
        # callbacks = [EarlyStopping(patience=20, restore_best_weights=True)]
        callbacks = self.create_callbacks(wandb=wandb)
        history = self.model.fit(X, y, validation_split = 0.1, epochs= self.config["N_EPOCHS"], callbacks= callbacks, batch_size = self.config["BATCH_SIZE"])
        return history

class FCNWrapper(KerasModel):
    def __init__(self, config: dict,):
        super().__init__(config)
        self.model = create_fcn(config)

    def train(self, X, y):
        # callbacks = [EarlyStopping(patience=20, restore_best_weights=True)]
        # callbacks = [keras.callbacks.ReduceLROnPlateau(monitor = 'loss', factor=0.5,
        #               patience=20, min_lr=0.0001)]

        # callbacks = []
        # callbacks.append(EarlyStopping(monitor='val_loss',
        #     min_delta=0.01,
        #     patience=self.config.get('patience', 20),
        #     restore_best_weights=True))
        callbacks = self.create_callbacks(wandb=wandb)
        history = self.model.fit(X, y, validation_split = 0.3, epochs= self.config["N_EPOCHS"], callbacks= callbacks, batch_size = self.config["BATCH_SIZE"])
        return history

    def log_results(self, test_x, test_y, logger, per_cluster=None):
        pred_y = self.predict(test_x)
        # wandb_conf_matrix = logger.plot.confusion_matrix(
        #     y_true=y, preds=pred_y,
        #     class_names=list(range(self.config['N_CLUSTERS']))
        # )

        # sample_counter = Counter(pred_y)
        precision, recall, fscore, support = precision_recall_fscore_support(test_y, pred_y, average='weighted')

        logger.log({
            'acc': accuracy_score(test_y, pred_y),
            'f1': f1_score(test_y, pred_y, average='weighted'),
            'macro_f1': f1_score(test_y, pred_y, average='macro'),
            'micro_f1': f1_score(test_y, pred_y, average='micro'),
            "support": support,
            "precision": precision,
            "recall": recall,
            "fscore": fscore
            # 'conf_matrix': wandb_conf_matrix
        })

        if per_cluster:
            report = classification_report(y_true=test_y, y_pred = pred_y, output_dict=True)
            report_path = os.path.join(logger.run.dir, "per_cluster_results.json")
            report_dict = dict(report)

            with open(report_path, 'w') as fp:
                json.dump(report_dict, fp)

    def visualize_gradcam(self, test_x, test_y, n_samples, wandb=None):
        get_last_conv = keras.backend.function([self.model.layers[0].input], [self.model.layers[-3].output])
        conv_outputs = get_last_conv([test_x])[0]

        pred_y = self.predict(test_x)

        print(pred_y)
        print(test_y)
        with tf.GradientTape() as tape:
            scce = tf.keras.losses.SparseCategoricalCrossentropy(dtype="float32")
            loss = scce(test_y, pred_y)

            grads = tape.gradient(loss, conv_outputs)

        casted_conv_outputs = tf.cast(conv_outputs > 0, "float32")
        casted_grads = tf.cast(grads > 0, "float32")
        guided_grads = casted_conv_outputs * casted_grads * grads

        conv_outputs = conv_outputs[0]
        guided_grads = guided_grads[0]

        weights = tf.reduce_mean(guided_grads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1)
        print(cam.shape)
        import sys
        sys.exit()

    def visualize(self, test_x, test_y, n_samples, wandb=None):
        # cam
        get_last_conv = keras.backend.function([self.model.layers[0].input], [self.model.layers[-3].output])
        last_conv = get_last_conv([test_x])[0]

        get_softmax = keras.backend.function([self.model.layers[0].input], [self.model.layers[-1].output])
        softmax = get_softmax(([test_x]))[0]
        softmax_weight = self.model.get_weights()[-2]
        CAM = np.dot(last_conv, softmax_weight)

        # pp = PdfPages('CAM.pdf')
        if wandb:
            save_dir = os.path.join(wandb.run.dir, 'cam')
            os.makedirs(save_dir, exist_ok=True)

        labels = np.unique(test_y)

        for label in labels:
            label_samples_idxs = np.where(test_y == label)[0]
            plt.figure(figsize=(13, 7));
            for label_id in label_samples_idxs[:n_samples]:
                CAM = (CAM - CAM.min(axis=1, keepdims=True)) / (CAM.max(axis=1, keepdims=True) - CAM.min(axis=1, keepdims=True))
                c = np.exp(CAM) / np.sum(np.exp(CAM), axis=1, keepdims=True)
                plt.plot(test_x[label_id].squeeze());
                plt.scatter(np.arange(len(test_x[label_id])), test_x[label_id].squeeze(), cmap='hot_r', c=c[label_id, :, :, int(test_y[label_id])].squeeze(), s=100);
                plt.title(
                    'True label:' + str(test_y[label_id]) + '   likelihood of label ' + str(test_y[label_id]) + ': ' + str(softmax[label_id][int(test_y[label_id])]))
                plt.colorbar();

            if wandb:
                plt.savefig(os.path.join(save_dir, f'cam_{label}.png'))
            else:
                plt.show()

class LSTMWrapper(KerasModel):
    def __init__(self, config: dict):
        super().__init__(config)
        self.model = create_lstm(config)

class VggWrapper(KerasModel):
    def __init__(self, config: dict):
        super().__init__(config)
        self.model = VGG16(include_top=True, weights='imagenet')
        self._compile_model()

    def _compile_model(self,):
        optimizer = tf.keras.optimizers.Adam()
        self.model.compile(optimizer=optimizer,
            loss='sparse_categorical_crossentropy', metrics='acc')

    def train(self, X, y, val_X, val_y, wandb=None):
        if X.ndim == 2:
            X = self.adjust_input(X)
            val_X = self.adjust_input(val_X)

        self.model.fit(X, y,
            validation_data=(val_X, val_y),
            batch_size=self.config.get('batch_size', 8),
            epochs=self.config.get('epochs', 200),
            callbacks=self.create_callbacks(wandb))

    def predict(self, X, proba=False):
        if X.ndim == 2:
            X = self.adjust_input(X)
        pred_proba = self.model.predict(X)
        if proba:
            return pred_proba
        else:
            return pred_proba.argmax(axis=1)

    @staticmethod
    def adjust_input(X: np.ndarray, image_shape=(224, 224)):
        """ Adjust input from flattened form (batch_size, features) to
            to 3D (batch_size, height, width, channel) form
        """
        if X.ndim == 2:
            X = X.reshape(X.shape[0], image_shape[0], image_shape[1])
        rgb_batch = np.repeat(X[:, :, :, np.newaxis], 3, -1)
        x = preprocess_input(rgb_batch)
        return x



class SkModel(BaseModel):
    def __init__(self, config: dict):
        super().__init__(config)

    def train(self, X, y, wandb=None):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, SAVE_DIR):
        pickle.dump(self.config, open(os.path.join(SAVE_DIR, 'config.pickle'), 'wb'))
        pickle.dump(self.model, open(os.path.join(SAVE_DIR, 'model.h5'), 'wb'))

    def load(self, SAVE_DIR):
        self.config = pickle.load(open(os.path.join(SAVE_DIR, 'config.pickle'), 'rb'))
        self.model = pickle.load(open(os.path.join(SAVE_DIR, 'model.h5'), 'rb'))


class SVMWrapper(SkModel):
    def __init__(self, config: dict):
        super().__init__(config)
        self.model = SVC(random_state=config['RANDOM_SEED'])


class DecisionTreeWrapper(SkModel):
    def __init__(self, config: dict):
        super().__init__(config)
        self.model = DecisionTreeClassifier(random_state=config['RANDOM_SEED'])


class XGBWrapper(SkModel):
    def __init__(self, config: dict):
        super().__init__(config)
        self.model = XGBClassifier(random_state=config['RANDOM_SEED'],
            use_label_encoder=False)

class KNNWrapper(SkModel):
    def __init__(self, config: dict):
        super().__init__(config)
        metric = config["KNN_METRIC"]

        if metric == "dtw":
            metric = self.dtw

        self.model = KNeighborsClassifier(metric = metric, n_neighbors=config["N_NEIGHBORS"]) # no random state

    def dtw(self, t1, t2):
        return fastdtw(t1,t2)[0]