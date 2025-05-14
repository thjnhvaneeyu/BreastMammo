#!/usr/bin/env python3
# main_cnn.py

import os
import sys
import time
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import config
from data_operations.data_preprocessing import (
    import_minimias_dataset,
    import_cbisddsm_training_dataset,
    import_cbisddsm_testing_dataset,
    import_cmmd_dataset,
    import_inbreast_roi_dataset,
    import_inbreast_full_dataset
)
from cnn_models.cnn_model import CnnModel
import argparse
from data_operations.data_preprocessing import dataset_stratified_split
from data_operations.data_preprocessing import make_class_weights
from tensorflow.keras import mixed_precision
from utils import load_trained_model
# === Bật mixed-precision toàn cục ===
policy = mixed_precision.Policy("mixed_float16")
mixed_precision.set_global_policy(policy)

# Project imports
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
DATA_ROOT_BREAST = '/kaggle/input/breastdata'
DATA_ROOT_CMMD = '/kaggle/input/cmmddata/CMMD'

def main():
    # 1) CLI args
    parser = argparse.ArgumentParser(description="Mammogram DL pipeline")
    parser.add_argument("-d", "--dataset",
                        choices=["mini-MIAS","mini-MIAS-binary","CBIS-DDSM","CMMD","INbreast"],
                        required=True,
                        help="Dataset to use")
    parser.add_argument("-mt", "--mammogram_type",
                        choices=["calc","mass","all"], default="all",
                        help="For CBIS-DDSM only")
    parser.add_argument("-m", "--model",
                        choices=["CNN","VGG","VGG-common","ResNet","Inception","DenseNet","MobileNet"],
                        required=True,
                        help="Model backbone")
    parser.add_argument("-r", "--runmode",
                        choices=["train","test"], default="train",
                        help="train or test")
    parser.add_argument("-lr", "--learning_rate", type=float,
                        default=config.learning_rate, help="Learning rate")
    parser.add_argument("-b", "--batch_size", type=int,
                        default=config.batch_size, help="Batch size")
    parser.add_argument("-e1", "--max_epoch_frozen", type=int,
                        default=config.max_epoch_frozen, help="Frozen epochs")
    parser.add_argument("-e2", "--max_epoch_unfrozen", type=int,
                        default=config.max_epoch_unfrozen, help="Unfrozen epochs")
    parser.add_argument("--roi", action="store_true",
                        help="Use ROI mode for INbreast / mini-MIAS")
    parser.add_argument("--augment", action="store_true",
                        help="Apply augmentation transforms")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose logging")
    parser.add_argument("-n", "--name", default=config.name,
                        help="Experiment name")
    args = parser.parse_args()

    # 2) Override config from args
    config.dataset            = args.dataset
    config.mammogram_type     = args.mammogram_type
    config.model              = args.model
    config.run_mode           = args.runmode
    config.learning_rate      = args.learning_rate
    config.batch_size         = args.batch_size
    config.max_epoch_frozen   = args.max_epoch_frozen
    config.max_epoch_unfrozen = args.max_epoch_unfrozen
    config.is_roi             = args.roi
    config.augment_data       = args.augment
    config.verbose_mode       = args.verbose
    config.name               = args.name
    
    if config.verbose_mode:
        print(f"[DEBUG] Config: dataset={config.dataset}, model={config.model}, roi={config.is_roi}, augment={config.augment_data}")

    # 2) Load & preprocess
    le = LabelEncoder()
    X_train = X_test = y_train = y_test = None
    ds_train = ds_val = None
    class_weights = None               # <<< thêm dòng này

    if config.dataset in ["mini-MIAS","mini-MIAS-binary"]:
        X, y = import_minimias_dataset(os.path.join(DATA_ROOT_BREAST, config.dataset), le)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

    elif config.dataset=="CBIS-DDSM":
        X_train, y_train = import_cbisddsm_training_dataset(le)
        X_test,  y_test  = import_cbisddsm_testing_dataset(le)

    elif config.dataset.upper() == "CMMD":
        # --- Load & stratified split CMMD (80/20 on y) ---
        d = DATA_ROOT_CMMD
        X, y = import_cmmd_dataset(d, le)
        X_train, X_test, y_train, y_test = dataset_stratified_split(
            0.2, X, y
        )

        # --- Determine number of classes ---
        num_classes = y_train.shape[1] if y_train.ndim > 1 else 2

        # --- Compute class weights only for binary case ---
        if num_classes == 2:
            # if one-hot, convert to label vector
            labels = y_train.argmax(axis=1) if y_train.ndim > 1 else y_train
            class_weights = make_class_weights(labels)
        else:
            class_weights = None

        # downstream training / eval will use NumPy arrays
        ds_train = ds_val = None

    elif config.dataset.upper()=="INBREAST":
        data_dir = os.path.join(DATA_ROOT_BREAST, "INbreast", "INbreast")
        if config.is_roi:
            ds, class_weights, num_classes, num_samples = import_inbreast_roi_dataset(
                data_dir, le, target_size=(
                     config.INBREAST_IMG_SIZE["HEIGHT"],
                     config.INBREAST_IMG_SIZE["WIDTH"]
                 ), csv_path="/kaggle/input/breastdata/INbreast/INbreast/INbreast.csv" 
            )
            ds = ds.shuffle(buffer_size=num_samples)
            split = int(0.8 * num_samples)
            ds_train = ds.take(split).batch(config.batch_size).prefetch(tf.data.AUTOTUNE)
            ds_val   = ds.skip(split).batch(config.batch_size).prefetch(tf.data.AUTOTUNE)
        else:
            X, y = import_inbreast_full_dataset(
                data_dir, le,
                target_size=(config.INBREAST_IMG_SIZE["HEIGHT"],
                             config.INBREAST_IMG_SIZE["WIDTH"])
            )
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )
            class_weights = make_class_weights(y_train)
            if y.ndim>1: y = np.argmax(y,axis=1)
            # drop Normal
            normal_idx = np.where(le.classes_=="Normal")[0][0]
            mask = (y!=normal_idx)
            X, y = X[mask], y[mask]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )
            class_weights = make_class_weights(y_train)
    else:
        raise ValueError("Unsupported dataset")

    # 3) Build & compile model via CnnModel
    # Determine num_classes
    if ds_train is not None:
        # num_classes = 2
        pass
    elif y_train is not None: # Dữ liệu NumPy
        num_classes = y_train.shape[1] if y_train.ndim > 1 else len(np.unique(y_train))
    else:
        print("[WARNING] num_classes may not be correctly determined before model loading in test mode if y_train is None.")
        num_classes = 2 # Hoặc giá trị mặc định nào đó
    cnn = CnnModel(config.model, num_classes)
    # Let CnnModel.compile_model handle loss & metrics
    cnn.compile_model(config.learning_rate)
    # 4) If in test-mode: load saved .h5 and evaluate immediately
# # 4) Chế độ TEST
#     if config.run_mode.lower() == "test":
#         print("[INFO] Running in TEST mode.")
#         print(f"[INFO] Using dataset: {config.dataset}, Model: {config.model}, ROI: {config.is_roi}")
#         fname = (
#             f"dataset-{config.dataset}_type-{config.mammogram_type}"
#             f"_model-{config.model}_lr-{config.learning_rate}"
#             f"_b-{config.batch_size}_e1-{config.max_epoch_frozen}"
#             f"_e2-{config.max_epoch_unfrozen}"
#             f"_roi-{config.is_roi}_{config.name}.h5"
#         )
#         path = os.path.join(PROJECT_ROOT, "saved_models", fname)
#         if not os.path.exists(path):
#             raise FileNotFoundError(f"Model file not found: {path}")
#         cnn._model = tf.keras.models.load_model(path)
#         cls_type = 'binary' if num_classes==2 else 'multiclass'
#         cnn.evaluate_model(X_test, y_test, le, cls_type, time.time())
#         return

#     # 5) TRAIN mode
#     if ds_train is not None:
#         # INbreast ROI
#         # cnn.train_model(ds_train, ds_val, None, None, class_weights)
# # main.py, ROI‐mode branch:
#         cnn.compile_model(config.learning_rate)
#         cnn.train_model(
#             ds_train,
#             ds_val,
#             y_train=None,
#             y_val=None,
#             class_weights=class_weights
#         )
#     else:
#         # numpy arrays
#         cnn.train_model(X_train, X_test, y_train, y_test, class_weights)

#     # 6) Save trained model
#     cnn.save_model()

#     # 7) Evaluate on test set
#     cls_type = 'binary' if num_classes==2 else 'multiclass'
#     cnn.evaluate_model(X_test, y_test, le, cls_type, time.time())

# if __name__ == "__main__":
#     main()

    if config.run_mode.lower() == "test":
        print("[INFO] Loading model for evaluation...")
        # Hàm load_trained_model đã được sửa trong utils.py để tự compile
        loaded_keras_model = load_trained_model(compile_model_on_load=True)

        if loaded_keras_model is None:
            raise FileNotFoundError(
                f"Không thể tải model dựa trên cấu hình hiện tại. "
                "Vui lòng kiểm tra log từ 'load_trained_model' trong utils.py."
            )

        # Kiểm tra và cập nhật num_classes của cnn_instance nếu cần
        output_neurons_loaded = loaded_keras_model.output_shape[-1]
        actual_num_classes_from_loaded_model = output_neurons_loaded if output_neurons_loaded > 1 else 2
        
        if cnn.num_classes != actual_num_classes_from_loaded_model:
            print(f"[INFO] CnnModel was initialized with {cnn.num_classes} classes, "
                  f"but loaded model has {actual_num_classes_from_loaded_model} output classes. "
                  f"Re-initializing CnnModel for consistency.")
            cnn = CnnModel(config.model, actual_num_classes_from_loaded_model)
        
        cnn.model = loaded_keras_model # Gán model đã tải và compile

        # Đảm bảo dữ liệu test có sẵn
        if X_test is None or y_test is None:
            print(f"[ERROR] Dữ liệu test (X_test hoặc y_test) là None cho dataset {config.dataset}. Không thể đánh giá.")
            return
        if le.classes_.size == 0:
            print("[ERROR] LabelEncoder (le) chưa được fit. Không thể đánh giá chính xác tên lớp.")
            # Cân nhắc fit le trên y_test nếu y_test là nhãn chữ, hoặc đảm bảo le được load/fit đúng.
            # Ví dụ đơn giản: le.fit(y_test) nếu y_test chứa nhãn số 0,1... và bạn muốn map số đó ra chữ.
            # Điều này phụ thuộc vào y_test là dạng số hay chữ, và evaluate_model kỳ vọng gì.
            # Giả sử evaluate_model cần le đã fit để inverse_transform nhãn số về chữ.
            # Và y_test truyền vào evaluate_model là nhãn số.
            try:
                unique_test_labels = np.unique(y_test)
                if np.issubdtype(unique_test_labels.dtype, np.number): # Nếu y_test là số
                    # Cần nhãn chữ để fit cho LabelEncoder hoạt động đúng nếu mục tiêu là map số -> chữ
                    # Nếu không có nhãn chữ, tạo nhãn giả:
                    if len(unique_test_labels) == 2 : le.fit(['Class_0', 'Class_1'])
                    elif len(unique_test_labels) > 2 : le.fit([f'Class_{i}' for i in unique_test_labels])
                    else: print("[WARNING] Cannot fit LabelEncoder on y_test as it's empty or has only one class after potential filtering.")
                else: # Nếu y_test là chữ (ít khả năng ở giai đoạn này)
                    le.fit(y_test)
                print(f"[INFO] LabelEncoder refitted on y_test unique values. Classes: {le.classes_}")
            except Exception as e_le_fit:
                print(f"[WARNING] Could not fit LabelEncoder on y_test: {e_le_fit}. Evaluation report might lack proper class names.")


        cls_type_eval = 'binary' if cnn.num_classes == 2 else 'multiclass'
        print(f"[INFO] Evaluating with: X_test shape {X_test.shape}, y_test shape {y_test.shape}, num_classes {cnn_instance.num_classes}, cls_type {cls_type_eval}")
        
        cnn.evaluate_model(X_test, y_test, le, cls_type_eval, time.time())
        return # Kết thúc test mode

    # --- Chế độ TRAIN ---
    # Hàm train_model của CnnModel sẽ tự xử lý việc compile với learning rate phù hợp cho từng giai đoạn
    print("[INFO] Running in TRAIN mode.")
    if ds_train is not None: # INbreast ROI (sử dụng tf.data.Dataset)
        print(f"[INFO] Training with INbreast ROI (tf.data.Dataset). Class weights: {class_weights}")
        cnn.train_model(
            ds_train,
            ds_val,
            y_train=None, # y_train, y_val không cần thiết khi dùng Dataset
            y_val=None,
            class_weights=class_weights # Truyền class_weights đã tính
        )
    elif X_train is not None and y_train is not None: # Dữ liệu NumPy
        print(f"[INFO] Training with NumPy arrays. X_train shape: {X_train.shape}. Class weights: {class_weights}")
        cnn.train_model(X_train, X_test, y_train, y_test, class_weights)
    else:
        print("[ERROR] Dữ liệu huấn luyện không có sẵn (X_train/y_train hoặc ds_train là None).")
        return

    # Lưu model sau khi huấn luyện
    print("[INFO] Training complete. Saving model...")
    cnn.save_model()

    # Đánh giá model trên tập test sau khi huấn luyện
    print("[INFO] Evaluating model on test set after training...")
    if X_test is None or y_test is None:
        print(f"[ERROR] Dữ liệu test (X_test hoặc y_test) là None sau khi huấn luyện. Không thể đánh giá.")
        return
    # le đã được fit trong quá trình tải dữ liệu
    cls_type_eval_after_train = 'binary' if cnn.num_classes == 2 else 'multiclass'
    cnn.evaluate_model(X_test, y_test, le, cls_type_eval_after_train, time.time()) # Sử dụng lại thời gian hiện tại

if __name__ == "__main__":
    start_time_main = time.time()
    main()
    print(f"Total execution time of main.py: {time.time() - start_time_main:.2f} seconds.")



