# src/main.py

import os
import sys
import time
import argparse

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# allow imports from project root
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
DATA_ROOT_BREAST = '/kaggle/input/breastdata'
DATA_ROOT_CMMD = '/kaggle/input/cmmddata/CMMD'

import config
from data_operations.data_transformations import generate_image_transforms
from data_operations import data_preprocessing, data_transformations, dataset_feed
from cnn_models.cnn_model import CnnModel
from tensorflow.keras import layers, models, applications, optimizers

def build_cnn(input_shape, num_classes):
    """Simple grayscale CNN."""
    m = models.Sequential(name="CustomCNN")
    m.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
    m.add(layers.MaxPooling2D((2,2)))
    m.add(layers.Conv2D(64, (3,3), activation='relu'))
    m.add(layers.MaxPooling2D((2,2)))
    m.add(layers.Flatten())
    m.add(layers.Dropout(0.5))
    m.add(layers.Dense(64, activation='relu'))
    if num_classes == 2:
        m.add(layers.Dense(1, activation='sigmoid'))
    else:
        m.add(layers.Dense(num_classes, activation='softmax'))
    return m

def build_pretrained_model(model_name, input_shape, num_classes):
    """Pretrained ImageNet base + GAP→Dropout→Dense."""
    mn = model_name.lower()
    if mn.startswith("vgg"):
        base = applications.VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
    elif mn == "resnet":
        base = applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    elif mn == "inception":
        base = applications.InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    elif mn == "densenet":
        base = applications.DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
    elif mn == "mobilenet":
        base = applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    base.trainable = False
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    if num_classes == 2:
        out = layers.Dense(1, activation='sigmoid')(x)
    else:
        out = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs=base.input, outputs=out, name=model_name), base

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

    # 3) Load & preprocess data
    # le = LabelEncoder()
    # X_train = X_test = y_train = y_test = None

    # if config.dataset in ["mini-MIAS","mini-MIAS-binary"]:
    #     d = os.path.join("/kaggle/input/breastdata", config.dataset)
    #     X, y = data_preprocessing.import_minimias_dataset(d, le)
    #     X_train, X_test, y_train, y_test = data_preprocessing.dataset_stratified_split(0.2, X, y)

    # elif config.dataset == "CBIS-DDSM":
    #     X_train, y_train = data_preprocessing.import_cbisddsm_training_dataset(le)
    #     X_test,  y_test  = data_preprocessing.import_cbisddsm_testing_dataset(le)

    # elif config.dataset == "CMMD":
    #     d = os.path.join("/kaggle/input/breastdata", "CMMD-binary")
    #     # if you reverted to the original 3-arg loader:
    #     #     X, y = data_preprocessing.import_cmmd_dataset(d, csv_path, le)
    #     X, y = data_preprocessing.import_cmmd_dataset(d, le)
    #     X_train, X_test, y_train, y_test = data_preprocessing.dataset_stratified_split(0.2, X, y)

    # # elif config.dataset == "INbreast":
    # #     d = os.path.join("/kaggle/input/breastdata", "INbreast")
    # #     X, y = data_preprocessing.import_inbreast_dataset(d, le)
    # #     X_train, X_test, y_train, y_test = data_preprocessing.dataset_stratified_split(0.2, X, y)
    # elif config.dataset.upper() == "INBREAST":
    #     data_dir = os.path.join("/kaggle/input/breastdata","INbreast")
    #     X, y = data_preprocessing.import_inbreast_dataset(data_dir, le)
    #     X_train, X_test, y_train, y_test = data_preprocessing.dataset_stratified_split(0.2, X, y)
    #     if getattr(config, "augment_data", False):
    #         X_train, y_train = data_transformations.generate_image_transforms(X_train, y_train)

    # else:
    #     raise ValueError(f"Unsupported dataset: {config.dataset}")

    le = LabelEncoder()
    X_train = X_test = y_train = y_test = None

    if config.dataset in ["mini-MIAS", "mini-MIAS-binary"]:
        d = os.path.join(DATA_ROOT_BREAST, config.dataset)
        X, y = data_preprocessing.import_minimias_dataset(d, le)
        X_train, X_test, y_train, y_test = data_preprocessing.dataset_stratified_split(0.2, X, y)

    elif config.dataset == "CBIS-DDSM":
        X_train, y_train = data_preprocessing.import_cbisddsm_training_dataset(le)
        X_test, y_test = data_preprocessing.import_cbisddsm_testing_dataset(le)

    elif config.dataset == "CMMD":
        # Nếu muốn dùng CMMD-binary:
        # d = os.path.join(DATA_ROOT, "CMMD-binary")
        # Nếu muốn dùng CMMD gốc (ảnh và clinical):
        # d = os.path.join(DATA_ROOT, "CMMD", "CMMD")
        d = DATA_ROOT_CMMD
        X, y = data_preprocessing.import_cmmd_dataset(d, le)
        X_train, X_test, y_train, y_test = data_preprocessing.dataset_stratified_split(0.2, X, y)

    elif config.dataset.upper() == "INBREAST":
        data_dir = os.path.join(DATA_ROOT_BREAST, "INbreast")
        X, y = data_preprocessing.import_inbreast_dataset(data_dir, le)
        X_train, X_test, y_train, y_test = data_preprocessing.dataset_stratified_split(0.2, X, y)

    else:
        raise ValueError(f"Unsupported dataset: {config.dataset}")

    # 4) Augmentation if requested
    if config.augment_data and X_train is not None:
        X_train, y_train = generate_image_transforms(X_train, y_train)

    # 5) Determine number of classes
    num_classes = 2 if y_train.ndim == 1 else y_train.shape[1]
    if config.verbose_mode:
        print(f"[INFO] Number of classes = {num_classes}")

    # 6) Test mode
    if config.run_mode == "test":
        # load model and evaluate
        model_fname = f"{config.dataset}_{config.model}.h5"
        model_path = os.path.join(PROJECT_ROOT, "saved_models", model_fname)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Cannot find model: {model_path}")
        keras_model = tf.keras.models.load_model(model_path)
        cnn = CnnModel(config.model, num_classes)
        cnn._model = keras_model

        # prediction & evaluation
        if config.dataset == "CBIS-DDSM":
            ds = dataset_feed.create_dataset(X_test, y_test)
            cnn.make_prediction(ds)
        else:
            cnn.make_prediction(X_test)
        cnn.evaluate_model(y_test, le, "binary", time.time() - start_time)
        return

    # 7) Build & compile a fresh model for training
    #   convert grayscale→RGB if necessary for pretrained
    if config.model != "CNN" and isinstance(X_train, np.ndarray) and X_train.ndim==4 and X_train.shape[-1]==1:
        X_train = np.repeat(X_train, 3, axis=-1)
        X_test  = np.repeat(X_test,  3, axis=-1)

    if config.model == "CNN":
        in_shape = X_train.shape[1:]
        keras_model = build_cnn(in_shape, num_classes)
    else:
        # figure out correct height/width
        if config.dataset == "CMMD":
            h,w = config.CMMD_IMG_SIZE.values()
        else:
            size_attr = f"{config.model.upper()}_IMG_SIZE"
            h,w = getattr(config, size_attr).values()
        keras_model, _ = build_pretrained_model(config.model, (h,w,3), num_classes)

    loss_fn = "binary_crossentropy" if num_classes==2 else "categorical_crossentropy"
    keras_model.compile(
        loss=loss_fn,
        optimizer=optimizers.Adam(lr=config.learning_rate),
        metrics=["accuracy"]
    )

    # 8) Wrap and train
    cnn = CnnModel(config.model, num_classes)
    cnn._model = keras_model

    if config.dataset == "CBIS-DDSM":
        ds_train = dataset_feed.create_dataset(X_train, y_train)
        ds_val   = dataset_feed.create_dataset(X_test,  y_test)
        cnn.train_model(ds_train, ds_val, y_train, y_test, class_weights=None)
    else:
        cnn.train_model(X_train, X_test, y_train, y_test, class_weights=None)
        start_time = time.time()

    # 9) Save & evaluate
    cnn.save_model()

    runtime  = time.time() - start_time
    cls_type = 'B-M' if num_classes==2 else 'N-B-M'
    # cnn.save_weights()
    # cnn.make_prediction(X_test)
    cnn.evaluate_model(X_test, y_test, le, cls_type, runtime)

    if config.verbose_mode:
        print(f"[DONE] Training + evaluation completed in {time.time()-start_time:.1f}s")

if __name__ == "__main__":
    main()
