import os, sys
import time
from cnn_models.cnn_model import CnnModel, test_model_evaluation
# Add project root (parent of this script) to module search path
# so we can import config, data_preprocessing, etc. even when running from src/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, applications, optimizers

import config
from data_operations import data_preprocessing, data_transformations, dataset_feed
from cnn_models.cnn_model import CnnModel

def build_cnn(input_shape, num_classes):
    """Xây dựng mô hình CNN đơn giản cho ảnh grayscale."""
    model = models.Sequential(name="CustomCNN")
    # Ví dụ kiến trúc CNN: 2 convolution + pool + dense
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
    """
    Xây dựng mô hình pre-trained (VGG, MobileNet, v.v.) với weights ImageNet và 
    thêm tầng Dense.
    """
    base = None
    model_name = model_name.lower()
    if model_name.startswith("vgg"):
        # Sử dụng VGG19
        base = applications.VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == "resnet":
        base = applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == "inception":
        base = applications.InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == "densenet":
        base = applications.DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == "mobilenet":
        base = applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    base.trainable = False  # Đóng băng base model
    # Thêm các layer đầu ra
    inputs = base.input
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)  # Dropout 20%
    if num_classes == 2:
        outputs = layers.Dense(1, activation='sigmoid')(x)
    else:
        outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=inputs, outputs=outputs, name=model_name.capitalize())
    return model, base

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True,
                        help="mini‑MIAS, mini‑MIAS‑binary, CBIS‑DDSM, CMMD")
    parser.add_argument("-mt", "--mammogram_type", default="all")
    parser.add_argument("-m", "--model", required=True,
                        help="CNN, VGG, VGG-common, ResNet, Inception, DenseNet, MobileNet")
    parser.add_argument("-r", "--runmode", default="train", choices=["train","test"])
    parser.add_argument("-lr","--learning_rate", type=float, default=0.001)
    parser.add_argument("-b","--batch_size", type=int, default=2)
    parser.add_argument("-e1","--max_epoch_frozen", type=int, default=100)
    parser.add_argument("-e2","--max_epoch_unfrozen", type=int, default=50)
    parser.add_argument("-roi","--is_roi", action="store_true")
    parser.add_argument("-v","--verbose", action="store_true")
    parser.add_argument("-n","--name", default="")
    parser.add_argument("--phase", choices=["1","2","all"], default="all")
    args = parser.parse_args()
    phase = args.phase

    # đổ vào config
    for attr in ["dataset","mammogram_type","model",
                 "batch_size","learning_rate",
                 "max_epoch_frozen","max_epoch_unfrozen",
                 "is_roi","verbose","name"]:
        setattr(config, attr if attr!="name" else "NAME", getattr(args, attr))


    # Ghi các tham số vào config
    config.dataset = args.dataset
    config.mammogram_type = args.mammogram_type
    config.model = args.model
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.MAX_EPOCH_FROZEN = args.max_epoch_frozen
    config.MAX_EPOCH_UNFROZEN = args.max_epoch_unfrozen
    config.is_roi = args.is_roi
    config.verbose_mode = args.verbose
    config.NAME = args.name

    # Định nghĩa trước một số kích thước ảnh cho từng mô hình (nếu chưa có trong config)
    # Giả định các biến trong config đã tồn tại; nếu không, ta tạo tạm:
    if not hasattr(config, "ROI_IMG_SIZE"):
        config.ROI_IMG_SIZE = {"HEIGHT": 224, "WIDTH": 224}
    if not hasattr(config, "MINI_MIAS_IMG_SIZE"):
        config.MINI_MIAS_IMG_SIZE = {"HEIGHT": 224, "WIDTH": 224}
    if not hasattr(config, "VGG_IMG_SIZE"):
        config.VGG_IMG_SIZE = {"HEIGHT": 512, "WIDTH": 512}
    if not hasattr(config, "RESNET_IMG_SIZE"):
        config.RESNET_IMG_SIZE = {"HEIGHT": 224, "WIDTH": 224}
    if not hasattr(config, "INCEPTION_IMG_SIZE"):
        config.INCEPTION_IMG_SIZE = {"HEIGHT": 224, "WIDTH": 224}
    if not hasattr(config, "DENSE_NET_IMG_SIZE"):
        config.DENSE_NET_IMG_SIZE = {"HEIGHT": 224, "WIDTH": 224}
    if not hasattr(config, "MOBILE_NET_IMG_SIZE"):
        config.MOBILE_NET_IMG_SIZE = {"HEIGHT": 224, "WIDTH": 224}
    if not hasattr(config, "RANDOM_SEED"):
        config.RANDOM_SEED = 42

    # Chuẩn bị bộ encoder cho nhãn
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()

    # Biến lưu dataset
    X_train = y_train = X_test = y_test = None

    # Load dữ liệu theo dataset
    if config.dataset in ["mini-MIAS", "mini-MIAS-binary"]:
        # Đường dẫn thư mục ảnh mini-MIAS (giả sử cấu trúc dữ liệu đã được chuẩn bị)
        data_dir = os.path.join("..", "data", config.dataset)
        if config.verbose_mode:
            print(f"[INFO] Loading dataset {config.dataset} from {data_dir} ...")
        X, y = data_preprocessing.import_minimias_dataset(data_dir, label_encoder)
        # Chia train/test
        split = 0.2  # 20% test
        X_train, X_test, y_train, y_test = data_preprocessing.dataset_stratified_split(split, X, y)
        # Augmentation (chỉ áp dụng cho mini-MIAS)
        if config.dataset == "mini-MIAS" or config.dataset == "mini-MIAS-binary":
            if config.verbose_mode:
                print("[INFO] Augmenting training data ...")
            # Generate transforms để tăng mẫu (chủ yếu cho binary)
            # (Hàm generate_image_transforms gốc chỉ hỗ trợ mini-MIAS, an toàn để gọi ở đây)
            X_train, y_train = data_transformations.generate_image_transforms(X_train, y_train)
    elif config.dataset == "CBIS-DDSM":
        if config.verbose_mode:
            print(f"[INFO] Loading CBIS-DDSM dataset (type={config.mammogram_type}) ...")
        # Import training và testing từ CSV (hàm sẽ mã hóa nhãn cho tập train)
        train_paths, y_train = data_preprocessing.import_cbisddsm_training_dataset(label_encoder)
        test_paths, y_test = data_preprocessing.import_cbisddsm_testing_dataset(label_encoder)
        # Để đảm bảo encoder không mất tính nhất quán, ta có thể fit trên train (đã làm trong import_cbisddsm_training_dataset)
        # Lưu lại list file cho train/test
        X_train = train_paths
        X_test = test_paths
    elif config.dataset.upper() == "CMMD" or config.dataset == "CMMD":
        data_dir = os.path.join("/home/neeyuhuynh/Desktop/Breast-Cancer-Detection-Mammogram-Deep-Learning-Publication/data/CMMD-binary")  # thư mục chứa các thư mục lớp của CMMD
        if config.verbose_mode:
            print(f"[INFO] Loading CMMD dataset from {data_dir} ...")
        # Xử lý cả trường hợp nhập "CMMD" hoặc "cmmd"
        X, y = data_preprocessing.import_cmmd_dataset(data_dir, label_encoder)
        # Chia stratified train/test
        split = 0.2
        X_train, X_test, y_train, y_test = data_preprocessing.dataset_stratified_split(split, X, y)
        if config.augment_data:
            if config.verbose_mode:
                print("[INFO] Augmenting CMMD training data ...")
            X_train, y_train = data_transformations.generate_image_transforms(X_train, y_train)
        # (Không gọi generate_image_transforms cho CMMD để tránh xung đột)
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset}")

    # Đến đây ta đã có X_train, y_train, X_test, y_test (với CBIS-DDSM, X_train, X_test là list path, y_train, y_test đã encoded)
    # Kiểm tra số lớp
    num_classes = None
    if y_train is not None:
        # y_train có thể là one-hot (ndim=2) hoặc nhãn số (ndim=1) tùy binary/multi
        if y_train.ndim == 1:
            # LabelEncoder cho binary trả về array 0/1
            num_classes = 2
        else:
            num_classes = y_train.shape[1]
    else:
        # Trường hợp chỉ test mode
        num_classes = len(label_encoder.classes_)

    if config.verbose_mode:
        print(f"[INFO] Number of classes: {num_classes}")
        if num_classes <= 2:
            classes = label_encoder.classes_ if hasattr(label_encoder, "classes_") else None
            print(f"[INFO] Classes: {classes if classes is not None else 'binary'}")

    # Nếu runmode là test, chỉ cần load model và đánh giá
    model = None
    model_filename = f"{config.dataset}_{config.model}"
    if config.dataset == "CBIS-DDSM":
        model_filename = f"{config.dataset}_{config.mammogram_type}_{config.model}"
    if config.NAME:
        model_filename += f"_{config.NAME}"
    model_filename += ".h5"
    model_path = os.path.join("..", "saved_models", model_filename)

    if args.runmode == "test":
        # Đảm bảo mô hình được load
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No saved model found at {model_path}. Train the model first or check the name.")
        if config.verbose_mode:
            print(f"[INFO] Loading model from {model_path} ...")
        model = tf.keras.models.load_model(model_path)
        # Compile lại với thông số loss, metrics (phòng trường hợp chưa compile sẵn)
        if num_classes == 2:
            model.compile(loss="binary_crossentropy", optimizer=optimizers.Adam(lr=config.learning_rate), metrics=["accuracy"])
        else:
            model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=config.learning_rate), metrics=["accuracy"])

        # Chuẩn bị dữ liệu test để evaluate
        if config.dataset == "CBIS-DDSM":
            # Tạo tf.data dataset cho test từ đường dẫn
            test_dataset = dataset_feed.create_dataset(X_test, y_test)
            results = model.evaluate(test_dataset, verbose=1)
        else:
            # Với mini-MIAS hoặc CMMD, X_test, y_test đã là numpy array
            results = model.evaluate(X_test, y_test, batch_size=config.batch_size, verbose=1)
        print(f"[TEST] Loss={results[0]:.4f}, Accuracy={results[1]*100:.2f}%")
        return  # Kết thúc ở đây khi test mode

    # Nếu runmode là train:
    # Xử lý chuyển ảnh 1 kênh thành 3 kênh cho mô hình pre-trained (ngoại trừ CNN)
    use_pretrained = config.model not in ["CNN"]
    if use_pretrained:
        # Với CBIS-DDSM: dữ liệu train/test dạng tf.data sẽ được xử lý kênh trong dataset_feed, 
        # nhưng với tập nhỏ (mini-MIAS, CMMD) ta convert ngay mảng numpy:
        if isinstance(X_train, np.ndarray) and X_train.ndim == 4 and X_train.shape[-1] == 1:
            # Lặp kênh 3 lần
            X_train = np.repeat(X_train, 3, axis=-1)
            X_test = np.repeat(X_test, 3, axis=-1)
        # Đối với CBIS (list path), ta điều chỉnh parse_function để decode 3 kênh (trong dataset_feed)
        # (Giả định dataset_feed.parse_function đã được sửa để xét config.model và decode_png channels=3 nếu cần)
        # Nếu chưa, có thể xử lý ảnh CBIS trước khi feed, nhưng điều đó không hiệu quả - bỏ qua do ngoài scope.

    # Xây dựng mô hình theo config.model
    if config.model == "CNN":
        # Input shape dựa trên dữ liệu train
        if isinstance(X_train, np.ndarray):
            in_shape = X_train.shape[1:]  # (H,W,channels)
        else:
            # Trường hợp CBIS với CNN: parse_function mặc định pad 512x512, 1 kênh
            in_shape = (512, 512, 1)
        model = build_cnn(in_shape, num_classes)
    else:
        # Xác định input size theo model
        model_name = config.model
        if config.dataset.upper().startswith("CMMD-binary"):
            input_h = config.CMMD_IMG_SIZE["HEIGHT"]
            input_w = config.CMMD_IMG_SIZE["WIDTH"]
        elif model_name == "VGG" or model_name == "Inception":
            input_h = config.MINI_MIAS_IMG_SIZE["HEIGHT"]
            input_w = config.MINI_MIAS_IMG_SIZE["WIDTH"]
        elif model_name == "VGG-common":
            input_h = config.VGG_IMG_SIZE["HEIGHT"]
            input_w = config.VGG_IMG_SIZE["WIDTH"]
            model_name = "VGG"  # dùng VGG19 nhưng khác kích thước
        elif model_name == "ResNet":
            input_h = config.RESNET_IMG_SIZE["HEIGHT"]
            input_w = config.RESNET_IMG_SIZE["WIDTH"]
        elif model_name == "DenseNet":
            input_h = config.DENSE_NET_IMG_SIZE["HEIGHT"]
            input_w = config.DENSE_NET_IMG_SIZE["WIDTH"]
        elif model_name == "MobileNet":
            input_h = config.MOBILE_NET_IMG_SIZE["HEIGHT"]
            input_w = config.MOBILE_NET_IMG_SIZE["WIDTH"]
        else:
            # Mặc định nếu khác (nhưng các trường hợp đã liệt kê đủ)
            input_h = X_train.shape[1] if isinstance(X_train, np.ndarray) else 224
            input_w = X_train.shape[2] if isinstance(X_train, np.ndarray) else 224

        # input_shape = (input_h, input_w, 3)  # mô hình pre-trained dùng 3 kênh
        input_shape = (input_h, input_w, 3 if config.model!="CNN" else 1)
        model, base_model = build_pretrained_model(model_name, input_shape, num_classes)

    # Compile mô hình trước khi train
    if num_classes == 2:
        model.compile(loss="binary_crossentropy", optimizer=optimizers.Adam(lr=config.learning_rate), metrics=["accuracy"])
    else:
        model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=config.learning_rate), metrics=["accuracy"])

    # Huấn luyện mô hình
    phase1_weights = os.path.join(
    "..", "saved_models",
    f"{config.dataset}_{config.model}_{config.NAME}_phase1.h5"
)
    if config.model != "CNN":
        # Giai đoạn 1: train với base model đóng băng (đã freeze ở build_pretrained_model)
        epochs1 = config.MAX_EPOCH_FROZEN
        if phase in ("1", "all"):
        # if epochs1 > 0:
            if config.verbose_mode:
                print(f"[INFO] Phase 1: Training top layers for {epochs1} epochs...")
            if config.dataset == "CBIS-DDSM":
                # Dùng tf.data pipeline
                train_dataset = dataset_feed.create_dataset(X_train, y_train)
                # (Có thể tạo một dataset cho validation từ X_test, y_test nếu muốn theo dõi)
                model.fit(train_dataset, epochs=epochs1, verbose=1)
            else:
                model.fit(X_train, y_train, batch_size=config.batch_size, epochs=epochs1, verbose=1)
            model.save_weights(phase1_weights)
            if config.verbose_mode:
                print(f"[INFO] Saved phase 1 weights to {phase1_weights}")
        # Giai đoạn 2: unfreeze và train fine-tune
        # epochs2 = config.MAX_EPOCH_UNFROZEN
        # if epochs2 > 0:
        #     if config.verbose_mode:
        #         print(f"[INFO] Phase 2: Fine-tuning all layers for {epochs2} epochs...")
        #     # Mở khóa toàn bộ base model
        #     for layer in model.layers:
        #         layer.trainable = True
        #     # (Hoặc: base_model.trainable = True, rồi compile lại)
        #     model.compile(loss=model.loss, optimizer=optimizers.Adam(lr=config.learning_rate), metrics=["accuracy"])
        #     if config.dataset == "CBIS-DDSM":
        #         train_dataset = dataset_feed.create_dataset(X_train, y_train)
        #         model.fit(train_dataset, epochs=epochs2, verbose=1)
        #     else:
        #         model.fit(X_train, y_train, batch_size=config.batch_size, epochs=epochs2, verbose=1)
        epochs2 = config.MAX_EPOCH_UNFROZEN
        if phase in ("2", "all"):
            # rebuild fresh architecture
            model, base_model = build_pretrained_model(model_name, input_shape, num_classes)
            # load the phase‑1 top‑only weights
            model.load_weights(phase1_weights)
            if config.verbose_mode:
                print(f"[INFO] Loaded phase 1 weights from {phase1_weights}")
            # unfreeze all layers
            for layer in model.layers:
                layer.trainable = True
            # recompile for fine‑tuning
            model.compile(
                loss="binary_crossentropy" if num_classes==2 else "categorical_crossentropy",
                optimizer=optimizers.Adam(lr=config.learning_rate),
                metrics=["accuracy"]
            )
            if config.verbose_mode:
                print(f"[INFO] Phase 2: Fine‑tuning all layers for {epochs2} epochs…")
            if config.dataset == "CBIS-DDSM":
                ds = dataset_feed.create_dataset(X_train, y_train)
                model.fit(ds, epochs=epochs2, verbose=1)
            else:
                model.fit(X_train, y_train,
                          batch_size=config.batch_size,
                          epochs=epochs2,
                          verbose=1)

    else:
        # Mô hình CNN tự tạo - train toàn bộ luôn (e1+e2 epoch)
        total_epochs = config.MAX_EPOCH_FROZEN + config.MAX_EPOCH_UNFROZEN
        if total_epochs <= 0:
            total_epochs = 1  # đảm bảo train ít nhất 1 epoch nếu người dùng nhập 0
        if config.verbose_mode:
            print(f"[INFO] Training CNN from scratch for {total_epochs} epochs...")
        if config.dataset == "CBIS-DDSM":
            train_dataset = dataset_feed.create_dataset(X_train, y_train)
            model.fit(train_dataset, epochs=total_epochs, verbose=1)
        else:
            model.fit(X_train, y_train, batch_size=config.batch_size, epochs=total_epochs, verbose=1)

    # Lưu model đã huấn luyện
    os.makedirs(os.path.join("..", "saved_models"), exist_ok=True)
    model.save(model_path)

    if config.verbose_mode:
        print(f"[INFO] Model saved to {model_path}")

    # Đánh giá trên tập test sau khi train (có thể thực hiện luôn hoặc để user chạy -r test riêng)
    # Ở đây ta sẽ in kết quả để tiện theo dõi trong cùng một lần chạy train.
    if config.dataset == "CBIS-DDSM":
        test_dataset = dataset_feed.create_dataset(X_test, y_test)
        loss, acc = model.evaluate(test_dataset, verbose=0)
    else:
        loss, acc = model.evaluate(X_test, y_test, batch_size=config.batch_size, verbose=0)
    print(f"[RESULT] Final Test Loss = {loss:.4f}, Test Accuracy = {acc*100:.2f}%")
# --- TRAIN & EVAL VIA CnnModel ---
    # cnn = CnnModel(config.model, num_classes)
    # cnn._model = model        # inject model đã build & compile

    # start = time.time()
    # if config.dataset == "CBIS-DDSM":
    #     train_ds = dataset_feed.create_dataset(X_train, y_train)
    #     val_ds   = dataset_feed.create_dataset(X_test,  y_test)
    #     cnn.train_model(train_ds, val_ds, y_train, y_test, class_weights=None)
    # else:
    #     cnn.train_model(X_train, X_test, y_train, y_test, class_weights=None)

    # cnn.save_model()          # lưu h5 giống trước

    # runtime = time.time() - start
    # cls_type = 'B-M' if num_classes == 2 else 'N-B-M'
    # cnn.evaluate_model(y_test, label_encoder, cls_type, runtime)


if __name__ == "__main__":
    main()
