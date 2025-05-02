import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy, CategoricalAccuracy
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

    def compile_model(self, learning_rate: float) -> None:
        """
        Compile with appropriate loss & metric:
         - 2 classes → BinaryCrossentropy + BinaryAccuracy
         - >2 classes → CategoricalCrossentropy + CategoricalAccuracy
        """
        # if self.num_classes == 2:
        #     self._model.compile(
        #         optimizer=Adam(learning_rate),
        #         loss=BinaryCrossentropy(),
        #         metrics=[BinaryAccuracy()]
        #     )
        # else:
        #     self._model.compile(
        #         optimizer=Adam(learning_rate),
        #         loss=CategoricalCrossentropy(),
        #         metrics=[CategoricalAccuracy()]
        #     )
        # Khởi tạo optimizer và alias để callback tìm attribute 'lr'
        opt = Adam(learning_rate=learning_rate)
        # Thiết lập alias cho legacy callbacks
        setattr(opt, 'lr', opt.learning_rate)

        if self.num_classes == 2:
            self._model.compile(
                optimizer=opt,
                loss=BinaryCrossentropy(),
                metrics=[BinaryAccuracy()]
            )
        else:
            self._model.compile(
                optimizer=opt,
                loss=CategoricalCrossentropy(),
                metrics=[CategoricalAccuracy()]
            )


    def train_model(self, X_train, X_val, y_train, y_val, class_weights) -> None:
        """
        Two-phase training for pretrained backbones; single-phase for custom CNN.
        """
        if self.model_name != "CNN":
            # Phase 1: freeze base
            if config.max_epoch_frozen > 0:
                for layer in self._model.layers:
                    layer.trainable = False
                self.compile_model(config.learning_rate)
                self._fit(X_train, X_val, y_train, y_val, class_weights,
                          epochs=config.max_epoch_frozen, frozen=True)
                plot_training_results(self.history, "Initial_training", is_frozen_layers=True)

            # Phase 2: unfreeze all
            if config.max_epoch_unfrozen > 0:
                for layer in self._model.layers:
                    layer.trainable = True
                self.compile_model(1e-5)
                self._fit(X_train, X_val, y_train, y_val, class_weights,
                          epochs=config.max_epoch_unfrozen, frozen=False)
                plot_training_results(self.history, "Fine_tuning_training", is_frozen_layers=False)

        else:
            # Custom CNN from scratch
            self.compile_model(config.learning_rate)
            self._fit(X_train, X_val, y_train, y_val, class_weights,
                      epochs=config.max_epoch_unfrozen, frozen=False)
            plot_training_results(self.history, "CNN_training", is_frozen_layers=False)

    def _fit(self, X_train, X_val, y_train, y_val, class_weights, epochs, frozen):
        # patience = max(1, epochs // 10)
        # callbacks = [
        #     EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
        #     ReduceLROnPlateau(monitor='val_loss', patience=max(1, patience//2))
        # ]
        # Dùng thẳng config.py để điều khiển callback

        es = EarlyStopping(
            monitor='val_loss',
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
        rlrp = ReduceLROnPlateau(
            monitor='val_loss',
            patience=config.reduce_lr_patience,
            factor=config.reduce_lr_factor,
            min_lr=config.min_learning_rate,
            verbose=1
        )
        callbacks = [es, rlrp]
        if isinstance(X_train, tf.data.Dataset):
            self.history = self._model.fit(
                X_train,
                validation_data=X_val,
                class_weight=class_weights,
                epochs=epochs,
                callbacks=callbacks
            )
        else:
            self.history = self._model.fit(
                x=X_train, y=y_train,
                validation_data=(X_val, y_val),
                batch_size=config.batch_size,
                epochs=epochs,
                class_weight=class_weights,
                callbacks=callbacks
            )

    # ... (rest of evaluate_model, save_model, etc. unchanged) ...

    # def evaluate_model(self,
    #                    X_test: np.ndarray,
    #                    y_true: np.ndarray,
    #                    label_encoder: LabelEncoder,
    #                    classification_type: str,
    #                    runtime) -> None:
    #     """
    #     Evaluate on X_test / y_true:
    #      - compute self.prediction = model.predict(X_test)
    #      - generate CSV, confusion matrix, ROC, comparison chart
    #     """
    #     # 1) Run prediction on the actual test images:
    #     self.prediction = self._model.predict(
    #         x=X_test.astype("float32"),
    #         batch_size=config.batch_size
    #     )

    #     # 2) Invert labels if needed
    #     if label_encoder.classes_.size == 2:
    #         y_true_inv = y_true
    #         y_pred_inv = np.round_(self.prediction, 0)
    #     else:
    #         y_true_inv = label_encoder.inverse_transform(np.argmax(y_true, axis=1))
    #         y_pred_inv = label_encoder.inverse_transform(np.argmax(self.prediction, axis=1))
    #     # 3) Accuracy
    #     accuracy = accuracy_score(y_true_inv, y_pred_inv)
    #     print(f"Accuracy = {accuracy:.4f}\n")

    #     # 4) CSV report
    #     generate_csv_report(y_true_inv, y_pred_inv, label_encoder, accuracy)
    #     generate_csv_metadata(runtime)

    #     # 5) Confusion matrix
    #     cm = confusion_matrix(y_true_inv, y_pred_inv)
    #     plot_confusion_matrix(cm, "d", label_encoder, False)
    #     cmn = cm.astype("float") / cm.sum(axis=1)[:, None]
    #     cmn[np.isnan(cmn)] = 0
    #     plot_confusion_matrix(cmn, ".2f", label_encoder, True)

    #     # 6) ROC curve
    #     if label_encoder.classes_.size == 2:
    #         plot_roc_curve_binary(y_true, self.prediction)
    #     else:
    #         plot_roc_curve_multiclass(y_true, self.prediction, label_encoder)
    def evaluate_model(self, X_test, y_true, label_encoder, classification_type, runtime):
        """
        Evaluate on X_test / y_true:
        - compute self.prediction = model.predict(X_test)
        - generate CSV, confusion matrix, ROC, comparison chart
        """
        # 1) Chạy dự đoán
        self.prediction = self._model.predict(
            x=X_test.astype("float32"),
            batch_size=config.batch_size
        )

        # 2) Chuyển ngược nhãn
        if label_encoder.classes_.size == 2:
            # NHỊ PHÂN: y_true đã là 1-D array [0/1], prediction là xác suất
            y_true_inv = y_true
            # làm tròn về 0 hoặc 1, rồi flatten về 1-D
            y_pred_inv = np.round(self.prediction).astype(int).flatten()
        else:
            # ĐA LỚP: y_true là one-hot, prediction là softmax → argmax
            y_true_inv = label_encoder.inverse_transform(np.argmax(y_true, axis=1))
            y_pred_inv = label_encoder.inverse_transform(np.argmax(self.prediction, axis=1))

        # 3) Tính accuracy
        acc = accuracy_score(y_true_inv, y_pred_inv)
        print(f"Accuracy = {acc:.4f}\n")

        # 4) Báo cáo CSV
        generate_csv_report(y_true_inv, y_pred_inv, label_encoder, acc)
        generate_csv_metadata(runtime)

        # 5) Confusion matrix
        cm = confusion_matrix(y_true_inv, y_pred_inv)
        plot_confusion_matrix(cm, 'd', label_encoder)
        plot_confusion_matrix(cm.astype('float')/cm.sum(axis=1)[:, None], '.2f', label_encoder)
        # in ma trận nhầm lẫn thô
        # plot_confusion_matrix(cm, 'd', label_encoder)

        # # in ma trận nhầm lẫn đã chuẩn hoá
        # cm_norm = cm.astype('float') / cm.sum(axis=1)[:, None]
        # plot_confusion_matrix(cm_norm, '.2f', label_encoder)
        # # 6) ROC curve
        # if label_encoder.classes_.size == 2:
        #     plot_roc_curve_binary(y_true, self.prediction)
        # else:
        #     plot_roc_curve_multiclass(y_true, self.prediction, label_encoder)

    # 7) So sánh với các paper khác (nếu có)
    # ... giữ nguyên phần này ...

        # # 7) Comparison chart
        # with open("data_visualisation/other_paper_results.json") as f:
        #     data = json.load(f)
        # key = "mini-MIAS" if config.dataset == "mini-MIAS-binary" else config.dataset
        # df = pd.DataFrame.from_records(data[key][classification_type],
        #                                columns=["paper", "accuracy"])
        # new = pd.DataFrame({"paper": "Dissertation", "accuracy": acc}, index=[0])
        # df = pd.concat([new, df]).reset_index(drop=True)
        # df["accuracy"] = pd.to_numeric(df["accuracy"])
        # # plot_comparison_chart(df)

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
                    {"paper": "Dissertation", "accuracy": acc},
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
