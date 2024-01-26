import os

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.python.client import device_lib
import keras
import keras.backend as K
from keras import regularizers
from keras.callbacks import Callback, LearningRateScheduler
from keras.layers import *
from keras.metrics import *
from keras.models import Model
from keras.optimizers.legacy import Adam

import vessl


class AucScoreMonitor(Callback):
    def __init__(
        self,
        fold,
        val_data,
        val_target,
        checkpoint_file,
        min_lr=1e-5,
        reduce_lr_patience=2,
        early_stop_patience=4,
        factor=0.1,
    ):
        self.fold = fold
        self.val_data = val_data
        self.val_target = val_target
        self.checkpoint_file = checkpoint_file
        self.reduce_lr_patience = reduce_lr_patience
        self.early_stop_patience = early_stop_patience
        self.best_val_score = 0
        self.epoch_num = 0
        self.factor = factor
        self.unimproved_lr_counter = 0
        self.unimproved_stop_counter = 0
        self.min_lr = min_lr

    def on_train_begin(self, logs={}):
        self.val_scores = []

    def on_epoch_end(self, epoch, logs={}):
        val_pred = self.model.predict(self.val_data).reshape((-1,))
        val_score = roc_auc_score(self.val_target, val_pred)
        # clip pred
        self.val_scores.append(val_score)

        # vessl.log
        vessl.log(
            step=epoch,
            payload={
                "auc_score": val_score,
                # "fold": self.fold
            },
        )

        # print(self.val_target, '\n', val_pred)
        print("Epoch {} val_score: {:.5f}".format(self.epoch_num, val_score))
        self.epoch_num += 1

        if val_score > self.best_val_score:
            print(
                "Val Score improve from {:5f} to {:5f}".format(
                    self.best_val_score, val_score
                )
            )
            self.best_val_score = val_score
            self.unimproved_lr_counter = 0
            self.unimproved_stop_counter = 0
            if self.checkpoint_file is not None:
                print("Saving file to", self.checkpoint_file)
                self.model.save_weights(self.checkpoint_file)
        else:
            if val_score < self.best_val_score:
                print("no improve from {:.5f}".format(self.best_val_score))
                self.unimproved_lr_counter += 1
                self.unimproved_stop_counter += 1

        if (
            self.reduce_lr_patience is not None
            and self.unimproved_lr_counter >= self.reduce_lr_patience
        ):
            current_lr = K.eval(self.model.optimizer.lr)
            if current_lr > self.min_lr:
                print(
                    "Reduce LR from {:.6f} to {:.6f}".format(
                        current_lr, current_lr * self.factor
                    )
                )
                K.set_value(self.model.optimizer.lr, current_lr * self.factor)
                # self.model.load_weights(self.checkpoint_file)
            else:
                pass

            self.unimproved_lr_counter = 0

        if (
            self.early_stop_patience is not None
            and self.unimproved_stop_counter >= self.early_stop_patience
        ):
            print("Early Stop Criteria Meet")
            self.model.stop_training = True

        return


class DataGenerator(keras.utils.Sequence):
    def __init__(
        self,
        X,
        y,
        batch_size=32,
        positive_rate=1.0,
        negative_rate=1.0,
    ):
        #'Initialization'
        self.batch_size = batch_size
        self.X = X
        self.y = y
        self.positive_rate = positive_rate
        self.negative_rate = negative_rate
        self.on_epoch_end()

    def __len__(self):
        #'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.resampled_y) / self.batch_size))

    def __getitem__(self, index):
        #'Generate one batch of data'
        # Generate indexes of the batch
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, len(self.resampled_y))
        indexes = np.arange(len(self.resampled_y))[start:end]

        # Generate data
        return self.resampled_X[indexes, :], self.resampled_y[indexes]

    def on_epoch_end(self):
        # resample + shuffle
        feat_len = 200
        X_p = self.X[self.y == 1]
        X_n = self.X[self.y == 0]

        pos_size = int(self.positive_rate * X_p.shape[0])
        X_p_new = np.zeros((pos_size, X_p.shape[1])).astype(np.float32)
        neg_size = int(self.negative_rate * X_n.shape[0])
        X_n_new = np.zeros((neg_size, X_n.shape[1])).astype(np.float32)

        for f in range(feat_len):
            pos_idx = np.random.choice(
                np.arange(X_p.shape[0]), size=pos_size, replace=True
            )
            X_p_new[:, f] = X_p[pos_idx, f]

            neg_idx = np.random.choice(
                np.arange(X_n.shape[0]), size=neg_size, replace=True
            )
            X_n_new[:, f] = X_n[neg_idx, f]

        self.resampled_X = np.vstack([X_p_new, X_n_new])
        self.resampled_y = np.array([1] * pos_size + [0] * neg_size)

        seq = np.random.choice(
            np.arange(len(self.resampled_y)), size=len(self.resampled_y), replace=False
        )
        self.resampled_X = self.resampled_X[seq]
        self.resampled_y = self.resampled_y[seq]


def build_model():
    # share components
    inputs = Input(shape=(200, 1))

    main = inputs
    main = Dense(64, activation="relu")(main)
    main = Dense(32, activation="relu")(main)
    main = Flatten()(main)

    out = Dense(1, activation="sigmoid")(main)  # 1 class to be classified

    model = Model(inputs, out)
    model.regularizers = [regularizers.l2(0.0001)]

    model.compile(optimizer=Adam(lr=0.001, clipnorm=1.0), loss="binary_crossentropy")

    return model


if __name__ == "__main__":
    data_path = os.environ.get("DATA_PATH", "data")
    # model_path = os.environ.get("MODEL_PATH", "model_weights")

    train = pd.read_csv(os.path.join(data_path, "train.csv.zip"))

    special_cols = [col for col in train.columns if train[col].dtype != np.float64]
    feature_cols = [col for col in train.columns if col not in special_cols]
    target = train.target.values

    train_df = train[feature_cols]

    # check gpu
    print(device_lib.list_local_devices())

    # configs for NN
    seed = 0
    train_epochs = 50
    batch_size = 32
    cpu_count = 4
    n_classses = 1
    fold_num = 4
    model_prefix = "nn-aug-v5"

    fold = 0

    for tr_ix, val_ix in KFold(fold_num, shuffle=True, random_state=seed).split(
        target, target
    ):
        fold += 1

        print("fold = {}".format(fold))

        tr = train_df.values[tr_ix, :]
        tr_y = target[tr_ix]

        val = train_df.values[val_ix, :]
        val_y = target[val_ix]

        model = build_model()
        # file_name = f"{model_prefix}_fold_{fold}.hdf5"
        # file_path = os.path.join(model_path, file_name)

        lrs = [0.001] * 15 + [0.0001] * 25 + [0.00001] * 10
        lr_schd = LearningRateScheduler(lambda ep: lrs[ep], verbose=1)
        wmlog_loss_monitor = AucScoreMonitor(
            fold,
            val,
            val_y,
            checkpoint_file=None,
            reduce_lr_patience=None,
            # early_stop_patience=None,
            factor=None,
        )  # calculate weighted m log loss per epoch

        training_generator = DataGenerator(
            tr,
            tr_y,
            batch_size=batch_size,
            positive_rate=2.0,
            negative_rate=1.0,
        )
        history = model.fit_generator(
            generator=training_generator,
            validation_data=(val, val_y),
            use_multiprocessing=False,
            workers=1,
            epochs=len(lrs),
            verbose=0,
            callbacks=[
                lr_schd,
                wmlog_loss_monitor,
            ],
        )
        # model.save_weights(file_path)

    # generate oof
    # train_oof = np.zeros((train.shape[0],))
    # train_aucs = []
    # model = build_model()

    # fold = 0

    # for tr_ix, val_ix in KFold(fold_num, shuffle=True, random_state=seed).split(
    #     target, target
    # ):
    #     fold += 1
    #     val = train_df.values[val_ix, :]
    #     val_y = target[val_ix]

    #     file_path = f"model_weights/{model_prefix}_fold_{fold}.hdf5"

    #     # Predict val + test oofs
    #     model.load_weights(file_path)  # load weight with best validation score

    #     pred = model.predict(val, batch_size=batch_size).reshape((len(val_ix),))
    #     train_oof[val_ix] += pred
    #     val_auc = roc_auc_score(target[val_ix], pred)
    #     train_aucs.append(val_auc)
    #     print("val acc = {:.5f}".format(val_auc))

    # full_auc = roc_auc_score(target, train_oof)
    # print(
    #     "CV Mean = {:.5f}, Std = {:.5f}, Overall AUC = {:.5f}".format(
    #         np.mean(train_aucs), np.std(train_aucs), full_auc
    #     )
    # )
