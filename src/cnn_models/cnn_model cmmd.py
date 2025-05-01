import os
import json
import tensorflow as tf
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy, CategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import config
from cnn_models.basic_cnn     import create_basic_cnn_model
from cnn_models.densenet121   import create_densenet121_model
from cnn_models.inceptionv3   import create_inceptionv3_model
from cnn_models.mobilenet_v2  import create_mobilenet_model
from cnn_models.resnet50      import create_resnet50_model
from cnn_models.vgg19         import create_vgg19_model
from cnn_models.vgg19_common  import create_vgg19_model_common

from data_visualisation.csv_report import generate_csv_report, generate_csv_metadata
from data_visualisation.plots      import plot_training_results, plot_confusion_matrix, plot_comparison_chart
from data_visualisation.roc_curves import plot_roc_curve_binary, plot_roc_curve_multiclass


class CnnModel:

    def __init__(self, model_name: str, num_classes: int):
        self.model_name  = model_name
        self.num_classes = num_classes
        self.history     = None
        self.prediction  = None

        # instantiate chosen architecture
        if model_name == "VGG":
            self._model = create_vgg19_model(num_classes)
        elif model_name == "VGG-common":
            self._model = create_vgg19_model_common(num_classes)
        elif model_name == "ResNet":
            self._model = create_resnet50_model(num_classes)
        elif model_name == "Inception":
            self._model = create_inceptionv3_model(num_classes)
        elif model_name == "DenseNet":
            self._model = create_densenet121_model(num_classes)
        elif model_name == "MobileNet":
            self._model = create_mobilenet_model(num_classes)
        elif model_name == "CNN":
            self._model = create_basic_cnn_model(num_classes)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def train_model(self, X_train, X_val, y_train, y_val, class_weights) -> None:
        """
        2-phase train: freeze → fit → plot, then unfreeze → fit → plot.
        Skip any phase whose max_epoch == 0.
        """
        # Xác định layer chỉ train FC
        layer_idx = 1 if self.model_name == "VGG" else 0

        # --- Phase 1: frozen layers ---
        if config.max_epoch_frozen > 0:
            # freeze
            self._model.layers[layer_idx].trainable = False
            if config.verbose_mode:
                print(f"[INFO] Phase 1: freezing '{self._model.layers[layer_idx].name}' for {config.max_epoch_frozen} epochs")
            # compile & fit
            self.compile_model(config.learning_rate)
            self._fit(X_train, X_val, y_train, y_val, class_weights, frozen=True)
            # plot only if history not empty
            if self.history and self.history.history:
                plot_training_results(self.history, "Initial_training", is_frozen_layers=True)

        # --- Phase 2: fine-tune all layers ---
        if config.max_epoch_unfrozen > 0:
            # unfreeze
            self._model.layers[layer_idx].trainable = True
            if config.verbose_mode:
                print(f"[INFO] Phase 2: unfreezing all layers for {config.max_epoch_unfrozen} epochs")
            # compile & fit
            self.compile_model(1e-5)
            self._fit(X_train, X_val, y_train, y_val, class_weights, frozen=False)
            # plot only if history not empty
            if self.history and self.history.history:
                plot_training_results(self.history, "Fine_tuning_training", is_frozen_layers=False)


    def compile_model(self, lr) -> None:
        """
        Compile model with appropriate loss & metric:
          - Binary datasets: CBIS-DDSM, mini-MIAS-binary, CMMD
          - Multi-class: mini-MIAS
        """
        if config.dataset in ["CBIS-DDSM", "mini-MIAS-binary", "CMMD", "CMMD_binary"]:
            self._model.compile(
                optimizer=Adam(lr),
                loss=BinaryCrossentropy(),
                metrics=[BinaryAccuracy()]
            )
        else:
            self._model.compile(
                optimizer=Adam(lr),
                loss=CategoricalCrossentropy(),
                metrics=[CategoricalAccuracy()]
            )

    def _fit(self, X_train, X_val, y_train, y_val, class_weights, frozen: bool):
        """
        Internal fit method handling both tf.data.Dataset and NumPy inputs.
        """
        # set epochs & patience based on phase
        if frozen:
            epochs  = config.max_epoch_frozen
            patience = max(1, epochs // 10)
        else:
            epochs  = config.max_epoch_unfrozen
            patience = max(1, epochs // 10)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', patience=max(1, patience//2))
        ]

        # Binary datasets branch
        if config.dataset in ["CBIS-DDSM", "mini-MIAS-binary", "CMMD", "CMMD_binary"]:
            if isinstance(X_train, tf.data.Dataset):
                # CBIS-DDSM pipeline
                self.history = self._model.fit(
                    X_train,
                    validation_data=X_val,
                    class_weight=class_weights,
                    epochs=epochs,
                    callbacks=callbacks
                )
            else:
                # NumPy arrays (mini-MIAS-binary, CMMD)
                steps     = len(X_train) // config.batch_size
                val_steps = len(X_val)   // config.batch_size
                self.history = self._model.fit(
                    x=X_train,
                    y=y_train,
                    batch_size=config.batch_size,
                    steps_per_epoch=steps,
                    validation_data=(X_val, y_val),
                    validation_steps=val_steps,
                    epochs=epochs,
                    callbacks=callbacks
                )
            return  # end binary

        # Multi-class branch (mini-MIAS)
        steps     = len(X_train) // config.batch_size
        val_steps = len(X_val)   // config.batch_size
        self.history = self._model.fit(
            x=X_train,
            y=y_train,
            batch_size=config.batch_size,
            steps_per_epoch=steps,
            validation_data=(X_val, y_val),
            validation_steps=val_steps,
            epochs=epochs,
            callbacks=callbacks
        )

    def evaluate_model(self,
                       X_test: np.ndarray,
                       y_true: np.ndarray,
                       label_encoder: LabelEncoder,
                       classification_type: str,
                       runtime) -> None:
        """
        Evaluate on X_test / y_true:
         - compute self.prediction = model.predict(X_test)
         - generate CSV, confusion matrix, ROC, comparison chart
        """
        # 1) Run prediction on the actual test images:
        self.prediction = self._model.predict(
            x=X_test.astype("float32"),
            batch_size=config.batch_size
        )

        # 2) Invert labels if needed
        if label_encoder.classes_.size == 2:
            y_true_inv = y_true
            y_pred_inv = np.round_(self.prediction, 0)
        else:
            y_true_inv = label_encoder.inverse_transform(np.argmax(y_true, axis=1))
            y_pred_inv = label_encoder.inverse_transform(np.argmax(self.prediction, axis=1))

        # 3) Accuracy
        accuracy = accuracy_score(y_true_inv, y_pred_inv)
        print(f"Accuracy = {accuracy:.4f}\n")

        # 4) CSV report
        generate_csv_report(y_true_inv, y_pred_inv, label_encoder, accuracy)
        generate_csv_metadata(runtime)

        # 5) Confusion matrix
        cm = confusion_matrix(y_true_inv, y_pred_inv)
        plot_confusion_matrix(cm, "d", label_encoder, False)
        cmn = cm.astype("float") / cm.sum(axis=1)[:, None]
        cmn[np.isnan(cmn)] = 0
        plot_confusion_matrix(cmn, ".2f", label_encoder, True)

        # 6) ROC curve
        if label_encoder.classes_.size == 2:
            plot_roc_curve_binary(y_true, self.prediction)
        else:
            plot_roc_curve_multiclass(y_true, self.prediction, label_encoder)

        # # 7) Comparison chart
        # with open("data_visualisation/other_paper_results.json") as f:
        #     data = json.load(f)
        # key = "mini-MIAS" if config.dataset == "mini-MIAS-binary" else config.dataset
        # df = pd.DataFrame.from_records(data[key][classification_type],
        #                                columns=["paper", "accuracy"])
        # new = pd.DataFrame({"paper": "Dissertation", "accuracy": accuracy}, index=[0])
        # df = pd.concat([new, df]).reset_index(drop=True)
        # df["accuracy"] = pd.to_numeric(df["accuracy"])
        # plot_comparison_chart(df)

        # Comparison chart – only if we have data for this dataset
        try:
            with open('data_visualisation/other_paper_results.json') as f:
                data = json.load(f)
            key = "mini-MIAS" if config.dataset == "mini-MIAS-binary" else config.dataset
            if key in data and classification_type in data[key]:
                df = pd.DataFrame.from_records(
                    data[key][classification_type],
                    columns=["paper", "accuracy"]
                )
                new = pd.DataFrame(
                    {"paper": "Dissertation", "accuracy": accuracy},
                    index=[0]
                )
                df = pd.concat([new, df]).reset_index(drop=True)
                df["accuracy"] = pd.to_numeric(df["accuracy"])
                plot_comparison_chart(df)
            else:
                if config.verbose_mode:
                    print(f"[WARN] No comparison-data for dataset '{config.dataset}' / type '{classification_type}' – skipping chart.")
        except FileNotFoundError:
            if config.verbose_mode:
                print("[WARN] other_paper_results.json not found – skipping comparison chart.")


    def save_model(self) -> None:
        os.makedirs("../saved_models", exist_ok=True)
        self._model.save(
            f"../saved_models/"
            f"dataset-{config.dataset}_type-{config.mammogram_type}_"
            f"model-{config.model}_lr-{config.learning_rate}_"
            f"b-{config.batch_size}_e1-{config.max_epoch_frozen}_"
            f"e2-{config.max_epoch_unfrozen}_roi-{config.is_roi}_{config.name}.h5"
        )

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, m):
        self._model = m


def test_model_evaluation(y_true, predictions, label_encoder: LabelEncoder, cls_type: str, runtime) -> None:
    """
    Standalone evaluation function (identical behavior to CnnModel.evaluate_model).
    """
    pass
