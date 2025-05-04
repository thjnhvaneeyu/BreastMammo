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

# Project imports
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
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

def make_class_weights(y):
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return dict(zip(classes, weights))

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

    if config.dataset in ["mini-MIAS","mini-MIAS-binary"]:
        X, y = import_minimias_dataset(os.path.join(config.DATA_ROOT_BREAST, config.dataset), le)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=config.RANDOM_SEED
        )

    elif config.dataset=="CBIS-DDSM":
        X_train, y_train = import_cbisddsm_training_dataset(le)
        X_test,  y_test  = import_cbisddsm_testing_dataset(le)

    elif config.dataset=="CMMD":
        X, y = import_cmmd_dataset(config.DATA_ROOT_CMMD, le)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=config.RANDOM_SEED
        )

    elif config.dataset.upper()=="INBREAST":
        data_dir = os.path.join(config.DATA_ROOT_BREAST, "INbreast", "INbreast")
        if config.is_roi:
            ds = import_inbreast_roi_dataset(
                data_dir, le,
                target_size=(config.INBREAST_IMG_SIZE["HEIGHT"],
                             config.INBREAST_IMG_SIZE["WIDTH"])
            )
            ds = ds.shuffle(1000)
            split = int(0.8*1000)
            ds_train = ds.take(split).batch(config.batch_size)
            ds_val   = ds.skip(split).batch(config.batch_size)
            # extract labels for class_weights
            labels = [int(l) for _,l in ds_train.unbatch().as_numpy_iterator()]
            class_weights = make_class_weights(np.array(labels))
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
                X, y, test_size=0.2, stratify=y, random_state=config.RANDOM_SEED
            )
            class_weights = make_class_weights(y_train)
    else:
        raise ValueError("Unsupported dataset")

    # 3) Build & compile model via CnnModel
    # Determine num_classes
    if ds_train is not None:
        num_classes = 2
    else:
        num_classes = len(np.unique(y_train))
    cnn = CnnModel(config.model, num_classes)
    # Let CnnModel.compile_model handle loss & metrics
    cnn.compile_model(config.learning_rate)

    # 4) Train
    if ds_train is not None:
        cnn.train_model(ds_train, ds_val, None, None, class_weights)
    else:
        cnn.train_model(X_train, X_test, y_train, y_test, class_weights)

    # 5) Save & Evaluate
    cnn.save_model()
    cls_type = 'binary' if num_classes==2 else 'multiclass'
    cnn.evaluate_model(X_test, y_test, le, cls_type, time.time())

if __name__=="__main__":
    main()
