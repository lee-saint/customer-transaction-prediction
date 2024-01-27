import os

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.client import device_lib
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
        val_data,
        val_target,
        checkpoint_file,
        min_lr=1e-5,
        reduce_lr_patience=2,
        early_stop_patience=4,
        factor=0.1,
    ):
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
                self.model.save(self.checkpoint_file)
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
    model_path = os.environ.get("MODEL_PATH", "model_weights")

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
    model_prefix = "nn"

    tr, val, tr_y, val_y = train_test_split(
        train_df, target, test_size=0.2, random_state=seed
    )

    model = build_model()
    file_name = f"{model_prefix}.keras"
    file_path = os.path.join(model_path, file_name)

    lrs = [0.001] * 15 + [0.0001] * 25 + [0.00001] * 10
    lr_schd = LearningRateScheduler(lambda ep: lrs[ep], verbose=1)
    wmlog_loss_monitor = AucScoreMonitor(
        val,
        val_y,
        checkpoint_file=None,
        reduce_lr_patience=None,
        # early_stop_patience=None,
        factor=None,
    )

    history = model.fit(
        # generator=training_generator,
        x=tr,
        y=tr_y,
        batch_size=batch_size,
        shuffle=True,
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
    model.save(file_path, save_format="keras")
