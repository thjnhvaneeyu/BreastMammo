# src/main.py

import os
import sys
import time
import argparse

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, applications, optimizers
from sklearn.preprocessing import LabelEncoder

# allow imports from project root
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import config
from data_operations import data_preprocessing, data_transformations, dataset_feed
from cnn_models.cnn_model import CnnModel


def build_cnn(input_shape, num_classes):
    """Simple grayscale CNN."""
    model = models.Sequential(name="CustomCNN")
    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation='relu'))
    if num_classes == 2:
        model.add(layers.Dense(1, activation='sigmoid'))
    else:
        model.add(layers.Dense(num_classes, activation='softmax'))
    return model


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

    return models.Model(inputs=base.input, outputs=out, name=model_name.capitalize()), base


def main():
    # 1) CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--dataset",      required=True, help="mini-MIAS, mini-MIAS-binary, CBIS-DDSM, CMMD")
    parser.add_argument("-mt","--mammogram_type", default="all")
    parser.add_argument("-m","--model",        required=True, help="CNN, VGG, VGG-common, ResNet, Inception, DenseNet, MobileNet")
    parser.add_argument("-r","--runmode",      default="train", choices=["train","test"])
    parser.add_argument("-lr","--learning_rate", type=float, default=0.001)
    parser.add_argument("-b","--batch_size",     type=int,   default=2)
    parser.add_argument("-e1","--max_epoch_frozen",   type=int, default=100)
    parser.add_argument("-e2","--max_epoch_unfrozen", type=int, default=50)
    parser.add_argument("-roi","--is_roi",     action="store_true")
    parser.add_argument("-v","--verbose",      action="store_true")
    parser.add_argument("-n","--name",         default="")
    parser.add_argument("--phase",              choices=["1","2","all"], default="all")
    args = parser.parse_args()
    phase = args.phase

    # 2) Update config
    config.dataset            = args.dataset
    config.mammogram_type     = args.mammogram_type
    config.model              = args.model
    config.batch_size         = args.batch_size
    config.learning_rate      = args.learning_rate
    config.max_epoch_frozen   = args.max_epoch_frozen
    config.max_epoch_unfrozen = args.max_epoch_unfrozen
    config.is_roi             = args.is_roi
    config.verbose_mode       = args.verbose
    config.NAME               = args.name

    # 3) Default image sizes
    config.ROI_IMG_SIZE        = getattr(config, "ROI_IMG_SIZE",        {"HEIGHT":224,"WIDTH":224})
    config.MINI_MIAS_IMG_SIZE  = getattr(config, "MINI_MIAS_IMG_SIZE",  {"HEIGHT":224,"WIDTH":224})
    config.VGG_IMG_SIZE        = getattr(config, "VGG_IMG_SIZE",        {"HEIGHT":512,"WIDTH":512})
    config.RESNET_IMG_SIZE     = getattr(config, "RESNET_IMG_SIZE",     {"HEIGHT":224,"WIDTH":224})
    config.INCEPTION_IMG_SIZE  = getattr(config, "INCEPTION_IMG_SIZE",  {"HEIGHT":224,"WIDTH":224})
    config.DENSE_NET_IMG_SIZE  = getattr(config, "DENSE_NET_IMG_SIZE",  {"HEIGHT":224,"WIDTH":224})
    config.MOBILE_NET_IMG_SIZE = getattr(config, "MOBILE_NET_IMG_SIZE", {"HEIGHT":224,"WIDTH":224})
    config.CMMD_IMG_SIZE       = getattr(config, "CMMD_IMG_SIZE",       {"HEIGHT":224,"WIDTH":224})
    config.RANDOM_SEED         = getattr(config, "RANDOM_SEED",         42)

    # 4) Load data
    label_encoder = LabelEncoder()
    X_train = X_test = y_train = y_test = None

    if config.dataset in ["mini-MIAS","mini-MIAS-binary"]:
        data_dir = os.path.join("..","data",config.dataset)
        X, y = data_preprocessing.import_minimias_dataset(data_dir, label_encoder)
        X_train, X_test, y_train, y_test = data_preprocessing.dataset_stratified_split(0.2, X, y)
        X_train, y_train = data_transformations.generate_image_transforms(X_train, y_train)

    elif config.dataset == "CBIS-DDSM":
        X_train, y_train = data_preprocessing.import_cbisddsm_training_dataset(label_encoder)
        X_test,  y_test  = data_preprocessing.import_cbisddsm_testing_dataset(label_encoder)

    elif config.dataset.upper() == "CMMD":
        data_dir = os.path.join("..","data","CMMD-binary")
        X, y = data_preprocessing.import_cmmd_dataset(data_dir, label_encoder)
        X_train, X_test, y_train, y_test = data_preprocessing.dataset_stratified_split(0.2, X, y)
        if getattr(config, "augment_data", False):
            X_train, y_train = data_transformations.generate_image_transforms(X_train, y_train)

    else:
        raise ValueError(f"Unsupported dataset: {config.dataset}")

    # 5) Num classes
    if y_train is not None:
        num_classes = 2 if y_train.ndim==1 else y_train.shape[1]
    else:
        num_classes = len(label_encoder.classes_)
    if config.verbose_mode:
        print(f"[INFO] Classes: {num_classes}")

    # 6) Model filename
    model_filename = f"{config.dataset}"
    if config.dataset == "CBIS-DDSM":
        model_filename += f"_{config.mammogram_type}"
    model_filename += f"_{config.model}"
    if config.NAME:
        model_filename += f"_{config.NAME}"
    model_filename += ".h5"
    model_path = os.path.join("..","saved_models", model_filename)

    # 7) Test mode
    if args.runmode == "test":
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        model = tf.keras.models.load_model(model_path)
        loss_fn = "binary_crossentropy" if num_classes==2 else "categorical_crossentropy"
        model.compile(loss=loss_fn,
                      optimizer=optimizers.Adam(lr=config.learning_rate),
                      metrics=["accuracy"])
        if config.dataset == "CBIS-DDSM":
            ds = dataset_feed.create_dataset(X_test, y_test)
            res = model.evaluate(ds, verbose=1)
        else:
            res = model.evaluate(X_test, y_test,
                                 batch_size=config.batch_size,
                                 verbose=1)
        print(f"[TEST] Loss={res[0]:.4f}, Acc={res[1]*100:.2f}%")
        return

    # 8) Build & compile
    use_pre = (config.model != "CNN")
    if use_pre and isinstance(X_train, np.ndarray) and X_train.ndim==4 and X_train.shape[-1]==1:
        X_train = np.repeat(X_train,3,axis=-1)
        X_test  = np.repeat(X_test,3,axis=-1)

    if config.model == "CNN":
        in_shape = X_train.shape[1:] if isinstance(X_train,np.ndarray) else (512,512,1)
        model = build_cnn(in_shape, num_classes)
    else:
        if config.dataset.upper().startswith("CMMD"):
            h,w = config.CMMD_IMG_SIZE["HEIGHT"], config.CMMD_IMG_SIZE["WIDTH"]
        elif config.model=="VGG-common":
            h,w = config.VGG_IMG_SIZE["HEIGHT"], config.VGG_IMG_SIZE["WIDTH"]
            config.model = "VGG"
        else:
            h = getattr(config, f"{config.model.upper()}_IMG_SIZE")["HEIGHT"]
            w = getattr(config, f"{config.model.upper()}_IMG_SIZE")["WIDTH"]
        model, _ = build_pretrained_model(config.model, (h,w,3), num_classes)

    loss_fn = "binary_crossentropy" if num_classes==2 else "categorical_crossentropy"
    model.compile(loss=loss_fn,
                  optimizer=optimizers.Adam(lr=config.learning_rate),
                  metrics=["accuracy"])

    # 9) Train & plot via CnnModel
    cnn = CnnModel(config.model, num_classes)
    cnn._model = model

    start = time.time()
    if config.dataset == "CBIS-DDSM":
        tr = dataset_feed.create_dataset(X_train, y_train)
        val = dataset_feed.create_dataset(X_test,  y_test)
        cnn.train_model(tr, val, y_train, y_test, class_weights=None)
    else:
        cnn.train_model(X_train, X_test, y_train, y_test, class_weights=None)

    cnn.save_model()

    runtime  = time.time() - start
    cls_type = 'B-M' if num_classes==2 else 'N-B-M'
    # cnn.evaluate_model(y_test, label_encoder, cls_type, runtime)
    cnn.evaluate_model(X_test, y_test, label_encoder, cls_type, runtime)

if __name__ == "__main__":
    main()
