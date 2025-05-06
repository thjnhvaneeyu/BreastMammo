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
            # --- ROI‐mode: tf.data.Dataset on‐the‐fly ---
            # ds = import_inbreast_roi_dataset(
            #     data_dir, le,
            #     target_size=(
            #         config.INBREAST_IMG_SIZE["HEIGHT"],
            #         config.INBREAST_IMG_SIZE["WIDTH"]
            #     )
            # )
            ds, class_weights, num_classes, num_samples = import_inbreast_roi_dataset(
                data_dir, le, target_size=(
                     config.INBREAST_IMG_SIZE["HEIGHT"],
                     config.INBREAST_IMG_SIZE["WIDTH"]
                 ), csv_path="/kaggle/input/breastdata/INbreast/INbreast/INbreast.csv" 
            )
            # ds = ds.unbatch()
            # Shuffle + split
            ds = ds.shuffle(buffer_size=num_samples)
            split = int(0.8 * num_samples)
            ds_train = ds.take(split).batch(config.batch_size).prefetch(tf.data.AUTOTUNE)
            ds_val   = ds.skip(split).batch(config.batch_size).prefetch(tf.data.AUTOTUNE)

            # 1) Tính class_weights
            # labels = [int(l) for _, l in ds_train.unbatch().as_numpy_iterator()]
            # class_weights = make_class_weights(np.array(labels))

            # 2) Trích xuất X_test, y_test từ ds_val để dùng cho evaluate
            X_test_list, y_test_list = [], []
            for img, lbl in ds_val.unbatch().as_numpy_iterator():
                # chuyển Tensor → numpy
                X_test_list.append(img)
                y_test_list.append(int(lbl))
            X_test = np.stack(X_test_list, axis=0)
            y_test = np.array(y_test_list)
        else:
            X, y = import_inbreast_full_dataset(
                data_dir, le,
                target_size=(config.INBREAST_IMG_SIZE["HEIGHT"],
                             config.INBREAST_IMG_SIZE["WIDTH"])
            )
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
        num_classes = 2
    else:
        num_classes = y_train.shape[1] if y_train.ndim>1 else len(np.unique(y_train))
    cnn = CnnModel(config.model, num_classes)
    # Let CnnModel.compile_model handle loss & metrics
    cnn.compile_model(config.learning_rate)
    # 4) If in test-mode: load saved .h5 and evaluate immediately
    if config.run_mode.lower() == "test":
        # construct filename exactly as save_model() does
        fname = (
            f"dataset-{config.dataset}_type-{config.mammogram_type}"
            f"_model-{config.model}_lr-{config.learning_rate}"
            f"_b-{config.batch_size}_e1-{config.max_epoch_frozen}"
            f"_e2-{config.max_epoch_unfrozen}"
            f"_roi-{config.is_roi}_{config.name}.h5"
        )
        path = os.path.join(PROJECT_ROOT, "saved_models", fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        cnn._model = tf.keras.models.load_model(path)
        cls_type = 'binary' if num_classes==2 else 'multiclass'
        cnn.evaluate_model(X_test, y_test, le, cls_type, time.time())
        return

    # 5) TRAIN mode
    if ds_train is not None:
        # INbreast ROI
        cnn.train_model(ds_train, ds_val, None, None, class_weights)
    else:
        # numpy arrays
        cnn.train_model(X_train, X_test, y_train, y_test, class_weights)

    # 6) Save trained model
    cnn.save_model()

    # 7) Evaluate on test set
    cls_type = 'binary' if num_classes==2 else 'multiclass'
    cnn.evaluate_model(X_test, y_test, le, cls_type, time.time())

if __name__ == "__main__":
    main()

