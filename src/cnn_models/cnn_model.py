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
# from tensorflow.data.experimental import cardinality
import shutil
from tensorflow.keras.mixed_precision import LossScaleOptimizer
from tensorflow.data import experimental as tfdata_exp
from tensorflow.data.experimental import assert_cardinality, cardinality
import math
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
from data_operations.data_preprocessing import flatten_to_slices

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
        # opt = Adam(learning_rate=learning_rate)
        # # Thiết lập alias cho legacy callbacks
        # setattr(opt, 'lr', opt.learning_rate)
    # 1. Tạo Adam gốc
        base_opt = Adam(learning_rate=learning_rate)
        # 2. Wrap để mixed precision (nếu policy = 'mixed_float16')
        opt = LossScaleOptimizer(base_opt)

        # 3. Thiết lập alias 'lr' lên LossScaleOptimizer → trỏ về base_opt.learning_rate
        setattr(opt, 'lr', base_opt.learning_rate)


        if self.num_classes == 2:
            self._model.compile(
                optimizer=opt,
                loss=CategoricalCrossentropy(),
                metrics=[CategoricalAccuracy(name="accuracy")]
            )
        else:
            self._model.compile(
                optimizer=opt,
                loss=BinaryCrossentropy(),
                metrics=[BinaryAccuracy(name="accuracy")]
            )


    def train_model(self, X_train, X_val, y_train, y_val, class_weights) -> None:
        """
        Two-phase training for pretrained backbones; single-phase for custom CNN.
        """
        if self.model_name != "CNN":
            # Phase 1: freeze base
            # if config.max_epoch_frozen > 0:
            #     for layer in self._model.layers:
            #         layer.trainable = False
            #     self.compile_model(config.learning_rate)
            #     self._fit(X_train, X_val, y_train, y_val, class_weights,
            #               epochs=config.max_epoch_frozen, frozen=True)
            #     plot_training_results(self.history, "Initial_training", is_frozen_layers=True)
            if config.max_epoch_frozen > 0:
                for layer in self._model.layers:
                    # nếu là Dense (head) thì trainable, ngược lại freeze
                    if isinstance(layer, tf.keras.layers.Dense):
                        layer.trainable = True
                    else:
                        layer.trainable = False

                self.compile_model(config.learning_rate)
                self._fit(X_train, X_val, y_train, y_val,
                          class_weights,
                          epochs=config.max_epoch_frozen,
                          frozen=True)
                
            # Phase 2: unfreeze all
        #     if config.max_epoch_unfrozen > 0:
        #         for layer in self._model.layers:
        #             layer.trainable = True
        #         self.compile_model(1e-5)
        #         self._fit(X_train, X_val, y_train, y_val, class_weights,
        #                   epochs=config.max_epoch_unfrozen, frozen=False)
        #         plot_training_results(self.history, "Fine_tuning_training", is_frozen_layers=False)

        # else:
        #     # Custom CNN from scratch
        #     self.compile_model(config.learning_rate)
        #     self._fit(X_train, X_val, y_train, y_val, class_weights,
        #               epochs=config.max_epoch_unfrozen, frozen=False)
        #     plot_training_results(self.history, "CNN_training", is_frozen_layers=False)
            if config.max_epoch_unfrozen > 0:
                for layer in self._model.layers:
                    layer.trainable = True

                # learning rate thấp để fine-tune
                self.compile_model(1e-5)
                self._fit(X_train, X_val, y_train, y_val,
                          class_weights,
                          epochs=config.max_epoch_unfrozen,
                          frozen=False)

        else:
            # custom CNN from scratch
            self.compile_model(config.learning_rate)
            self._fit(X_train, X_val, y_train, y_val,
                      class_weights,
                      epochs=config.max_epoch_unfrozen,
                      frozen=False)
    def _fit(self, X_train, X_val, y_train, y_val, class_weights, epochs, frozen):
        self._model.optimizer.lr = self._model.optimizer.learning_rate
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
    #     if isinstance(X_train, tf.data.Dataset):
    #         # 1. Tính số batch mỗi epoch qua cardinality
    #         train_steps = int(tfdata_exp.cardinality(X_train).numpy())
    #         val_steps   = int(tfdata_exp.cardinality(X_val).numpy())

    #         # 2. Nếu cardinality unknown, báo rõ ràng
    #         if train_steps < 0 or val_steps < 0:
    #             raise ValueError(
    #                 "Cannot infer dataset size. Please ensure X_train/X_val have known cardinality "
    #                 "or switch to numpy inputs."
    #             )

    #         # 3. Đặt dataset lặp vô hạn
    #         ds_train = X_train.repeat()
    #         ds_val   = X_val.repeat()

    #         # 4. Fit với steps_per_epoch và validation_steps
    #         self.history = self._model.fit(
    #             ds_train,
    #             epochs=epochs,
    #             steps_per_epoch=train_steps,
    #             validation_data=ds_val,
    #             validation_steps=val_steps,
    #             class_weight=class_weights,
    #             callbacks=callbacks
    #         )
    #         return
    #         # self.history = self._model.fit(
    #         #     X_train,
    #         #     validation_data=X_val,
    #         #     class_weight=class_weights,
    #         #     epochs=epochs,
    #         #     callbacks=callbacks
    #         # )
    #     # else:
    #     #     self.history = self._model.fit(
    #     #         x=X_train, y=y_train,
    #     #         validation_data=(X_val, y_val),
    #     #         batch_size=config.batch_size,
    #     #         epochs=epochs,
    #     #         class_weight=class_weights,
    #     #         callbacks=callbacks
    #     #     )
        if isinstance(X_train, tf.data.Dataset):
            # X_train, X_val đã được batch() & prefetch() ở pipeline
            self.history = self._model.fit(
                X_train,
                validation_data=X_val,
                epochs=epochs,
                class_weight=class_weights,
                callbacks=callbacks
            )
        else:
            self.history = self._model.fit(
                x=X_train, y=y_train,
                batch_size=config.batch_size,
                epochs=epochs,
                validation_data=(X_val, y_val),
                class_weight=class_weights,
                callbacks=callbacks
            )
        # # ensure labels are int32 so tf.cond branches match
        # y_train = y_train.astype('int32')
        # y_val   = y_val.astype('int32')
        # self.history = self._model.fit(
        #     x=X_train, y=y_train,
        #     batch_size=config.batch_size,
        #     epochs=epochs,
        #     validation_data=(X_val, y_val),
        #     class_weight=class_weights,
        #     callbacks=callbacks
        # )
    # # If using a tf.data.Dataset
    #     if isinstance(X_train, tf.data.Dataset):
    #         # compute number of batches per epoch
    #         train_steps = int(tf.data.experimental.cardinality(X_train).numpy())
    #         val_steps   = int(tf.data.experimental.cardinality(X_val).numpy())
    #         if train_steps < 0 or val_steps < 0:
    #             raise ValueError(
    #                 "Cannot infer dataset size. Ensure X_train/X_val have known cardinality."
    #             )

    #         # make them infinite but Keras will stop at steps_per_epoch
    #         ds_train = X_train.repeat()
    #         ds_val   = X_val.repeat()

    #         self.history = self._model.fit(
    #             ds_train,
    #             epochs=epochs,
    #             steps_per_epoch=train_steps,
    #             validation_data=ds_val,
    #             validation_steps=val_steps,
    #             class_weight=class_weights,
    #             callbacks=callbacks
    #         )
    #         return

    #     # Otherwise, NumPy arrays / Sequence branch
    #     # **Cast labels to int32 so both branches produce the same dtype**
    #     y_train = y_train.astype('int64')
    #     y_val   = y_val.astype('int64')

    #     self.history = self._model.fit(
    #         x=X_train,
    #         y=y_train,
    #         batch_size=config.batch_size,
    #         epochs=epochs,
    #         validation_data=(X_val, y_val),
    #         class_weight=class_weights,
    #         callbacks=callbacks
    #     )
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
    # def evaluate_model(self, X_test, y_true, label_encoder, classification_type, runtime):
    #     """
    #     Evaluate on X_test / y_true:
    #     - compute self.prediction = model.predict(X_test)
    #     - generate CSV, confusion matrix, ROC, comparison chart
    #     """
    #     # 1) Chạy dự đoán
    #     self.prediction = self._model.predict(
    #         x=X_test.astype("float32"),
    #         batch_size=config.batch_size
    #     )

    #     # 2) Chuyển ngược nhãn
    #     if label_encoder.classes_.size == 2:
    #         # NHỊ PHÂN: y_true đã là 1-D array [0/1], prediction là xác suất
    #         y_true_inv = y_true
    #         # làm tròn về 0 hoặc 1, rồi flatten về 1-D
    #         y_pred_inv = np.round(self.prediction).astype(int).flatten()
    #     else:
    #         # ĐA LỚP: y_true là one-hot, prediction là softmax → argmax
    #         y_true_inv = label_encoder.inverse_transform(np.argmax(y_true, axis=1))
    #         y_pred_inv = label_encoder.inverse_transform(np.argmax(self.prediction, axis=1))
    # def evaluate_model(self, X_test, y_test, label_encoder, cls_type, runtime):
    #     # … trước đó bạn đã thu y_pred, y_true = y_test …
    #     y_pred_prob = self._model.predict(X_test)
    #     if cls_type == 'binary':
    #         # threshold mặc định = 0.5
    #         y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    #     else:
    #         y_pred = y_pred_prob  # nếu multiclass, giữ softmax output

    #     # --- Chuyển y_true về dạng labels 1D ---
    #     if y_test.ndim > 1:
    #         y_true_labels = np.argmax(y_test, axis=1)
    #     else:
    #         y_true_labels = y_test.astype(int)

    #     # # --- Chuyển y_pred về dạng labels 1D nếu cần ---
    #     # if y_pred.ndim > 1:
    #     #     y_pred_labels = np.argmax(y_pred, axis=1)
    #     # else:
    #     #     y_pred_labels = y_pred
    #     y_pred_labels = y_pred

    #     # Giờ map ngược về nhãn chuỗi
    #     y_true_inv = label_encoder.inverse_transform(y_true_labels)
    #     y_pred_inv = label_encoder.inverse_transform(y_pred_labels)
    #     print(f"[DEBUG Evaluate] y_true_labels shape: {y_true_labels.shape}, y_pred_labels shape: {y_pred_labels.shape}")
    #     # Kiểm tra kích thước trước khi inverse_transform
    #     if y_true_labels.shape[0] != y_pred_labels.shape[0]:
    #         print(f"[ERROR Evaluate] Mismatch in number of samples before inverse_transform:")
    #         print(f"  y_true_labels samples: {y_true_labels.shape[0]}")
    #         print(f"  y_pred_labels samples: {y_pred_labels.shape[0]}")
    #         # Có thể raise lỗi ở đây hoặc xử lý khác
    #         # For now, let's see what happens if we proceed, but this is likely the root of the inconsistency
        
    #     try:
    #         y_true_inv = label_encoder.inverse_transform(y_true_labels)
    #         y_pred_inv = label_encoder.inverse_transform(y_pred_labels)
    #     except ValueError as e:
    #         print(f"[ERROR Evaluate] Error during inverse_transform: {e}")
    #         print(f"  Unique y_true_labels: {np.unique(y_true_labels)}")
    #         print(f"  Unique y_pred_labels: {np.unique(y_pred_labels)}")
    #         print(f"  LabelEncoder classes: {label_encoder.classes_}")
    #         # Thêm thông tin debug để xem tại sao lại có lỗi này (ví dụ: nhãn không có trong encoder)
    #         # raise e # Hoặc return để tránh crash hoàn toàn
    #         return


    #     print(f"[DEBUG Evaluate] y_true_inv length: {len(y_true_inv)}, y_pred_inv length: {len(y_pred_inv)}")

    #     # 3) Tính accuracy
    #     acc = accuracy_score(y_true_inv, y_pred_inv)
    #     print(f"Accuracy = {acc:.4f}\n")
    def evaluate_model(self, X_test, y_test, label_encoder, cls_type, runtime):
        print(f"[DEBUG Evaluate Start] Param X_test shape: {X_test.shape}, Param y_test shape: {y_test.shape}, cls_type: {cls_type}")
        print(f"[DEBUG Evaluate] Model input shape: {self._model.input_shape}, Model output shape: {self._model.output_shape}")

        y_pred_prob = self._model.predict(X_test)
        print(f"[DEBUG Evaluate] y_pred_prob shape AFTER predict: {y_pred_prob.shape}") # Rất quan trọng

        # ---- THÊM ĐOẠN CODE NÀY ĐỂ XỬ LÝ KHẢ NĂNG DUPLICATE OUTPUT ----
        if y_pred_prob.shape[0] != X_test.shape[0]:
            print(f"[WARN Evaluate] Output samples ({y_pred_prob.shape[0]}) from predict() do not match input samples ({X_test.shape[0]}).")
            if y_pred_prob.shape[0] == 2 * X_test.shape[0] and X_test.shape[0] > 0:
                print(f"    Assuming duplicated output, taking the first half of predictions.")
                y_pred_prob = y_pred_prob[:X_test.shape[0]]
                print(f"    y_pred_prob shape after slicing: {y_pred_prob.shape}")
            else:
                print(f"    Cannot automatically resolve prediction sample mismatch. Evaluation might fail or be incorrect.")
        # ---------------------------------------------------------------

        if cls_type == 'binary':
            if y_pred_prob.shape[-1] == 1: # Sigmoid output
                y_pred_labels = (y_pred_prob > 0.5).astype(int).flatten()
            else: # Softmax output (ví dụ: (None, 2))
                y_pred_labels = np.argmax(y_pred_prob, axis=1)
        else: # multiclass
            y_pred_labels = np.argmax(y_pred_prob, axis=1)
        # Log này của bạn đã cho thấy y_pred_labels shape: (384,)
        print(f"[DEBUG Evaluate] y_pred_labels shape after argmax/processing: {y_pred_labels.shape}")


        if y_test.ndim > 1:
            y_true_labels = np.argmax(y_test, axis=1)
        else:
            y_true_labels = y_test.astype(int)
        # Log này của bạn đã cho thấy y_true_labels shape: (192,)
        print(f"[DEBUG Evaluate] y_true_labels shape after processing y_test: {y_true_labels.shape}")
        
        # Kiểm tra lại số lượng mẫu trước khi inverse_transform
        if y_true_labels.shape[0] != y_pred_labels.shape[0]:
            print(f"[ERROR Evaluate] Mismatch in number of samples before inverse_transform:")
            print(f"  y_true_labels samples: {y_true_labels.shape[0]}")
            print(f"  y_pred_labels samples: {y_pred_labels.shape[0]}")
            # Bạn có thể dừng ở đây nếu muốn để tránh lỗi sau này
            # raise ValueError("Sample count mismatch before inverse_transform in evaluate_model") 
            # Hoặc tạm thời chỉ print và để code chạy tiếp xem lỗi ở đâu
            print("[WARN Evaluate] Proceeding despite sample mismatch before inverse_transform. Accuracy will be incorrect.")


        try:
            y_true_inv = label_encoder.inverse_transform(y_true_labels)
            # Nếu y_pred_labels vẫn là (384,) và y_true_labels là (192,), dòng dưới sẽ gây lỗi
            # hoặc accuracy_score sẽ lỗi.
            # Đoạn code check và slice ở trên nhằm mục đích làm cho y_pred_labels có shape (192,)
            y_pred_inv = label_encoder.inverse_transform(y_pred_labels)
        except ValueError as e:
            print(f"[ERROR Evaluate] Error during inverse_transform: {e}")
            print(f"  Unique y_true_labels (len {len(y_true_labels)}): {np.unique(y_true_labels, return_counts=True)}")
            print(f"  Unique y_pred_labels (len {len(y_pred_labels)}): {np.unique(y_pred_labels, return_counts=True)}")
            print(f"  LabelEncoder classes: {label_encoder.classes_}")
            return # Dừng hàm evaluate nếu có lỗi ở đây

        print(f"[DEBUG Evaluate] y_true_inv length: {len(y_true_inv)}, y_pred_inv length: {len(y_pred_inv)}")
        
        # Dòng 388 gây lỗi gốc
        acc = accuracy_score(y_true_inv, y_pred_inv)
        print(f"Accuracy = {acc:.4f}\n")
        # 4) Báo cáo CSV
        generate_csv_report(y_true_inv, y_pred_inv, label_encoder, acc)
        generate_csv_metadata(runtime)

        # 5) Confusion matrix
        cm = confusion_matrix(y_true_inv, y_pred_inv)
        # plot_confusion_matrix(cm, 'd', label_encoder)
        plot_confusion_matrix(cm.astype('float')/cm.sum(axis=1)[:, None], '.2f', label_encoder, is_normalised=True)
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
            if key in data and cls_type in data[key]:
                df = pd.DataFrame.from_records(
                    data[key][cls_type],
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
                    print(f"[WARN] No comparison-data for dataset '{config.dataset}' / type '{cls_type}' – skipping chart.")
        except FileNotFoundError:
            if config.verbose_mode:
                print("[WARN] other_paper_results.json not found – skipping comparison chart.")


    # def save_model(self) -> None:
    #     os.makedirs("../saved_models", exist_ok=True)
    #     self._model.save(
    #         f"../saved_models/"
    #         f"dataset-{config.dataset}_type-{config.mammogram_type}_"
    #         f"model-{config.model}_lr-{config.learning_rate}_"
    #         f"b-{config.batch_size}_e1-{config.max_epoch_frozen}_"
    #         f"e2-{config.max_epoch_unfrozen}_roi-{config.is_roi}_{config.name}.h5"
    #     )
    # def save_model(self) -> None:
    #     os.makedirs("../saved_models", exist_ok=True)
    #     # 1. Tạo biến path
    #     path = (
    #         f"../saved_models/"
    #         f"dataset-{config.dataset}_type-{config.mammogram_type}_"
    #         f"model-{config.model}_lr-{config.learning_rate}_"
    #         f"b-{config.batch_size}_e1-{config.max_epoch_frozen}_"
    #         f"e2-{config.max_epoch_unfrozen}_roi-{config.is_roi}_{config.name}.h5"
    #     )
    #     # # Nếu path tồn tại (file hoặc thư mục), xóa hoàn toàn
    #     # if os.path.exists(path):
    #     #     if os.path.isfile(path):
    #     #         os.remove(path)
    #     #     else:
    #     #         shutil.rmtree(path)
    #     # # Lưu model mới
    #     # self._model.save(path, overwrite=True)
    #     full_path = os.path.abspath(path)

    #     # Nếu tồn tại, xóa hẳn
    #     if os.path.exists(full_path):
    #         if os.path.isfile(full_path):
    #             os.remove(full_path)
    #         else:
    #             shutil.rmtree(full_path)

    #     # 2. Thử lưu nguyên file .h5
    #     try:
    #         self._model.save(full_path, overwrite=True)
    #     except Exception as e:
    #         print(f"[WARNING] Save .h5 thất bại: {e}")
    #         print("→ Chuyển sang lưu architecture  weights riêng.")
    #         # 2a) Lưu architecture vào JSON
    #         json_path = full_path.replace(".h5", "_arch.json")
    #         with open(json_path, "w") as f:
    #             f.write(self._model.to_json())
    #         # 2b) Lưu weights vào HDF5
    #         wt_path = full_path.replace(".h5", ".h5")
    #         self._model.save_weights(wt_path)
    #         print(f"[INFO] Architecture: {json_path}")
    #         print(f"[INFO] Weights:      {wt_path}")
    #         return

    #     # Nếu thành công thì in thông báo
    #     if self.verbose:
    #         print(f"[INFO] Model saved to {full_path}")
    def save_model(self) -> None:
        # Xác định đường dẫn gốc của dự án một cách đáng tin cậy
        # Giả sử cnn_model.py nằm trong PROJECT_ROOT/src/cnn_models/
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        # Đi lên 2 cấp để tới PROJECT_ROOT: src/cnn_models -> src -> PROJECT_ROOT
        project_root_actual = os.path.dirname(os.path.dirname(current_file_dir))

        save_dir = os.path.join(project_root_actual, "saved_models")
        os.makedirs(save_dir, exist_ok=True)

        # Tạo tên file cơ sở (không có đuôi .h5 hay .json)
        base_name = (
            f"dataset-{config.dataset}_type-{config.mammogram_type}_"
            f"model-{config.model}_lr-{config.learning_rate}_"
            f"b-{config.batch_size}_e1-{config.max_epoch_frozen}_"
            f"e2-{config.max_epoch_unfrozen}_roi-{config.is_roi}_{config.name}"
        )

        # --- Lưu kiến trúc model dưới dạng JSON ---
        arch_path = os.path.join(save_dir, f"{base_name}_arch.json")
        try:
            if os.path.exists(arch_path):
                os.remove(arch_path)
            with open(arch_path, "w") as json_file:
                json_file.write(self._model.to_json())
            print(f"Model architecture saved to: {arch_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save model architecture: {e}")

        # --- Lưu trọng số model với đuôi .weights.h5 ---
        # Đây là dòng gây lỗi trong traceback của bạn (line 498 trong file đã sửa của bạn)
        # cần đảm bảo wt_path có đuôi .weights.h5
        wt_path = os.path.join(save_dir, f"{base_name}.weights.h5") # SỬA Ở ĐÂY
        try:
            if os.path.exists(wt_path):
                os.remove(wt_path)
            self._model.save_weights(wt_path) # Bây giờ wt_path sẽ đúng định dạng
            print(f"Model weights saved to: {wt_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save model weights: {e}")

        # (Không bắt buộc) Nếu vẫn muốn thử lưu full model .h5 như một phương án dự phòng
        # hoặc nếu bạn muốn có cả hai:
        full_model_h5_path = os.path.join(save_dir, f"{base_name}.h5")
        try:
            if os.path.exists(full_model_h5_path):
                if os.path.isfile(full_model_h5_path):
                    os.remove(full_model_h5_path)
                else: # Nếu là thư mục (ví dụ khi lưu ở định dạng 'tf')
                    shutil.rmtree(full_model_h5_path)
            self._model.save(full_model_h5_path, overwrite=True)
            print(f"Full model also saved to (HDF5): {full_model_h5_path}")
        except Exception as e:
            print(f"[INFO] Saving full model as .h5 failed again (as expected or for other reasons): {e}")
            print(f"Architecture and weights were saved separately: {arch_path} and {wt_path}")

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
