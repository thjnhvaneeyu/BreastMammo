#!/usr/bin/env python3
# main2.py
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
from data_operations.data_preprocessing import dataset_stratified_split, generate_image_transforms
from data_operations.data_preprocessing import make_class_weights
from tensorflow.keras import mixed_precision
from utils import load_trained_model
import config # File config.py của bạn
from cnn_models.cnn_model import CnnModel # Lớp CnnModel của bạn
from data_operations.data_transformations import (
    generate_image_transforms
)
from utils import create_label_encoder, print_cli_arguments, print_error_message, print_num_gpus_available, print_runtime, set_random_seeds, load_trained_model
from data_operations.data_preprocessing import make_class_weights # Hàm tính class weight

# === Bật mixed-precision toàn cục ===
policy = mixed_precision.Policy("mixed_float16")
mixed_precision.set_global_policy(policy)

# Project imports
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
DATA_ROOT_BREAST = '/kaggle/input/breastdata'
DATA_ROOT_CMMD = '/kaggle/input/cmmddata/CMMD'

def remove_pectoral_muscle(image_array: np.ndarray) -> np.ndarray:
    print("      [INFO] Pectoral muscle removal (placeholder)...")
    return image_array

def parse_xml_for_pectoral_muscle(xml_path: str) -> list:
    pectoral_coords = []
    # ... (Logic parse XML của bạn) ...
    # Ví dụ:
    # try:
    #     tree = ET.parse(xml_path)
    #     root = tree.getroot()
    #     for roi_tag in root.findall('.//ROI[ROIType="PECTORAL_MUSCLE"]'): # Giả định cấu trúc
    #         for point_tag in roi_tag.findall('.//Point'):
    #             x = float(point_tag.find('x').text)
    #             y = float(point_tag.find('y').text)
    #             pectoral_coords.append((x, y))
    # except Exception as e:
    #     print(f"      [WARNING] Could not parse pectoral muscle from XML {xml_path}: {e}")
    if pectoral_coords:
        print(f"      [INFO] Found {len(pectoral_coords)} points for pectoral muscle in {os.path.basename(xml_path)}")
    return pectoral_coords

def clean_image_with_pectoral_removal(image_array: np.ndarray, xml_path: str) -> np.ndarray:
    # print(f"Attempting to clean image with pectoral removal, XML: {xml_path}")
    pectoral_coords = parse_xml_for_pectoral_muscle(xml_path)
    if pectoral_coords:
        # print(f"      [INFO] Using XML coordinates for pectoral muscle removal (logic to be implemented).")
        # Ví dụ: tạo mask và xóa, hoặc truyền coords vào hàm remove_pectoral_muscle nếu nó hỗ trợ
        # Hiện tại vẫn gọi hàm remove_pectoral_muscle chung
        image_cleaned = remove_pectoral_muscle(image_array.copy())
    else:
        # print(f"      [INFO] No XML coordinates for pectoral muscle, using automatic removal.")
        image_cleaned = remove_pectoral_muscle(image_array.copy())
    return image_cleaned

def elastic_transform_placeholder(image, alpha, sigma, random_state=None):
    print(f"      [INFO] Elastic Transform (placeholder alpha={alpha}, sigma={sigma})...")
    return image
# =====================================================================

# (Hàm load_inbreast_data_with_pectoral_removal sẽ tương tự như hàm
# load_inbreast_data_no_pectoral_removal ở main.py, nhưng có thêm bước
# clean_image_with_pectoral_removal và truyền các tham số augmentation)

def load_inbreast_data_with_pectoral_removal(
    data_dir: str,
    label_encoder: LabelEncoder,
    use_roi_patches: bool,
    target_size: tuple,
    enable_elastic: bool = False, elastic_alpha_val: float = 34.0, elastic_sigma_val: float = 4.0,
    enable_mixup: bool = False, mixup_alpha_val: float = 0.2,
    enable_cutmix: bool = False, cutmix_alpha_val: float = 1.0
):
    print(f"\n[INFO] Loading INbreast data {'with ROI patches' if use_roi_patches else 'as full images'} (Pectoral Removal ENABLED)...")
    print(f"  Elastic: {enable_elastic}, MixUp: {enable_mixup}, CutMix: {enable_cutmix}")

    dicom_dir = os.path.join(data_dir, "AllDICOMs")
    roi_dir = os.path.join(data_dir, "AllROI")
    xml_dir = os.path.join(data_dir, "AllXML")
    csv_path = os.path.join(data_dir, "INbreast.csv")

    if not os.path.exists(csv_path): raise FileNotFoundError(f"INbreast.csv not found at {csv_path}")
    if not os.path.isdir(dicom_dir): raise NotADirectoryError(f"DICOM directory not found: {dicom_dir}")
    if use_roi_patches and not os.path.isdir(roi_dir): raise NotADirectoryError(f"ROI directory not found: {roi_dir}")
    if not os.path.isdir(xml_dir): print(f"[WARNING] XML directory for pectoral muscle not found: {xml_dir}. Automatic removal will be used.")

    df = pd.read_csv(csv_path, sep=';')
    df.columns = [c.strip() for c in df.columns]

    all_images_data_list = []
    all_labels_data_list = [] # Sẽ chứa nhãn text trước, sau đó encode một lần
    processed_files_count = 0

    for index, row in df.iterrows():
        file_name_from_csv = str(row['File Name']).strip() # Đây là Patient ID hoặc File ID, ví dụ '22678622'
        dicom_fn_pattern = file_name_from_csv + "*.dcm" # Tìm các file .dcm bắt đầu bằng ID này
        
        # Tìm file DICOM thực tế trong thư mục AllDICOMs
        # Thường thì sẽ có dạng {ID}.dcm hoặc {ID}_{view}_{laterality}.dcm
        # Chúng ta sẽ tìm file đầu tiên khớp
        actual_dicom_files = tf.io.gfile.glob(os.path.join(dicom_dir, dicom_fn_pattern))
        if not actual_dicom_files:
            # Thử tìm không có hậu tố (trường hợp tên file trong CSV là tên file đầy đủ không có .dcm)
            actual_dicom_files = tf.io.gfile.glob(os.path.join(dicom_dir, file_name_from_csv + ".dcm"))

        if not actual_dicom_files:
            # print(f"  [WARNING] No DICOM file found for CSV entry: {file_name_from_csv} with pattern {dicom_fn_pattern}. Skipping.")
            continue
        
        dicom_path = actual_dicom_files[0] # Lấy file đầu tiên tìm được
        base_dicom_name = os.path.splitext(os.path.basename(dicom_path))[0] # Tên file không có .dcm, ví dụ 22678622 hoặc 22678622_R_CC

        xml_path = os.path.join(xml_dir, base_dicom_name + ".xml") # XML thường khớp với tên file DICOM
        # Nếu không có file xml khớp hoàn toàn, thử khớp với phần ID gốc từ CSV
        if not os.path.exists(xml_path):
            xml_path_alt = os.path.join(xml_dir, file_name_from_csv + ".xml")
            if os.path.exists(xml_path_alt):
                xml_path = xml_path_alt
            # else:
                # print(f"    [WARNING] XML file not found for {base_dicom_name} or {file_name_from_csv}. Pectoral removal might be suboptimal.")


        try:
            dicom_data = pydicom.dcmread(dicom_path)
            image_array = dicom_data.pixel_array.astype(np.float32)
            image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array) + 1e-8)
            
            # Bước 2: Cắt cơ ngực
            if os.path.exists(xml_path):
                image_cleaned = clean_image_with_pectoral_removal(image_array, xml_path)
            else:
                # print(f"    [INFO] XML file {xml_path} not found for {base_dicom_name}, using_automatic pectoral removal.")
                image_cleaned = remove_pectoral_muscle(image_array.copy())

            birad_value_csv = str(row['Bi-Rads']).strip()
            current_label_text = None
            for label_name_map, birad_list_map in config.INBREAST_BIRADS_MAPPING.items():
                standardized_birad_list_map = [val.replace("BI-RADS", "").strip() for val in birad_list_map]
                if birad_value_csv in standardized_birad_list_map:
                    current_label_text = label_name_map
                    break
            
            if current_label_text is None or current_label_text == "Normal":
                continue # Bỏ qua nếu không có nhãn hoặc là Normal

            # --- Bắt đầu phần xử lý ảnh/ROI và augmentation ---
            images_for_current_dicom = []
            labels_for_current_dicom = []

            if use_roi_patches:
                roi_file_pattern_search = os.path.join(roi_dir, base_dicom_name + "*.roi")
                # Nếu không thấy, thử tìm với ID gốc từ CSV (trường hợp tên ROI file chỉ chứa ID)
                if not tf.io.gfile.glob(roi_file_pattern_search):
                     roi_file_pattern_search = os.path.join(roi_dir, file_name_from_csv + "*.roi")

                matching_roi_files = tf.io.gfile.glob(roi_file_pattern_search)
                if not matching_roi_files: continue

                for roi_path_single in matching_roi_files:
                    temp_birad_map_roi = {base_dicom_name: birad_value_csv, file_name_from_csv: birad_value_csv}
                    coords, _ = data_preprocessing.load_roi_and_label(roi_path_single, temp_birad_map_roi)
                    if coords is None or not coords: continue

                    xs = [p[0] for p in coords]; ys = [p[1] for p in coords]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    h_img, w_img = image_cleaned.shape[:2]
                    x_min, y_min = max(0, x_min), max(0, y_min)
                    x_max, y_max = min(w_img - 1, x_max), min(h_img - 1, y_max)

                    if x_min >= x_max or y_min >= y_max: continue
                    roi_patch = image_cleaned[y_min:y_max+1, x_min:x_max+1]
                    if roi_patch.size == 0: continue
                    
                    resized_patch = cv2.resize(roi_patch, target_size, interpolation=cv2.INTER_AREA)
                    # ... (logic xử lý kênh cho resized_patch như đã làm) ...
                    if config.model != "CNN": # Pretrained models need 3 channels
                        if resized_patch.ndim == 2: resized_patch = cv2.cvtColor(resized_patch, cv2.COLOR_GRAY2RGB)
                        elif resized_patch.shape[-1] == 1: resized_patch = cv2.cvtColor(resized_patch, cv2.COLOR_GRAY2RGB)
                    elif resized_patch.ndim == 3 and resized_patch.shape[-1] == 3 and config.model == "CNN": # CNN needs 1 channel
                        resized_patch = cv2.cvtColor(resized_patch, cv2.COLOR_RGB2GRAY)
                        resized_patch = np.expand_dims(resized_patch, axis=-1)
                    
                    resized_patch = resized_patch.astype(np.float32)
                    if np.max(resized_patch) > 1.0: # Re-normalize if needed after resize/cvtColor
                        resized_patch = (resized_patch - np.min(resized_patch)) / (np.max(resized_patch) - np.min(resized_patch) + 1e-8)

                    images_for_current_dicom.append(resized_patch)
                    labels_for_current_dicom.append(current_label_text)
            else: # Full image
                resized_full_image = cv2.resize(image_cleaned, target_size, interpolation=cv2.INTER_AREA)
                # ... (logic xử lý kênh cho resized_full_image) ...
                if config.model != "CNN":
                    if resized_full_image.ndim == 2: resized_full_image = cv2.cvtColor(resized_full_image, cv2.COLOR_GRAY2RGB)
                    elif resized_full_image.shape[-1] == 1: resized_full_image = cv2.cvtColor(resized_full_image, cv2.COLOR_GRAY2RGB)
                elif resized_full_image.ndim == 3 and resized_full_image.shape[-1] == 3 and config.model == "CNN":
                    resized_full_image = cv2.cvtColor(resized_full_image, cv2.COLOR_RGB2GRAY)
                    resized_full_image = np.expand_dims(resized_full_image, axis=-1)

                resized_full_image = resized_full_image.astype(np.float32)
                if np.max(resized_full_image) > 1.0:
                     resized_full_image = (resized_full_image - np.min(resized_full_image)) / (np.max(resized_full_image) - np.min(resized_full_image) + 1e-8)
                
                images_for_current_dicom.append(resized_full_image)
                labels_for_current_dicom.append(current_label_text)

            # Áp dụng augmentation (Elastic, MixUp, CutMix) nếu có ảnh để xử lý
            if images_for_current_dicom:
                images_np = np.array(images_for_current_dicom, dtype=np.float32)
                
                # Chuyển nhãn text sang one-hot float32 cho generate_image_transforms
                le_temp = LabelEncoder().fit(["Benign", "Malignant"]) # Fit cục bộ
                labels_numeric_temp = le_temp.transform(labels_for_current_dicom)
                labels_one_hot_temp = tf.keras.utils.to_categorical(labels_numeric_temp, num_classes=2).astype(np.float32)

                # Gọi hàm augmentation chung
                aug_images_np, aug_labels_np = data_transformations.generate_image_transforms(
                    images_np, labels_one_hot_temp, # Truyền nhãn one-hot
                    apply_elastic=enable_elastic, elastic_alpha=elastic_alpha_val, elastic_sigma=elastic_sigma_val,
                    apply_mixup=enable_mixup, mixup_alpha=mixup_alpha_val,
                    apply_cutmix=enable_cutmix, cutmix_alpha=cutmix_alpha_val
                )
                all_images_data_list.extend(list(aug_images_np))
                all_labels_data_list.extend(list(aug_labels_np)) # aug_labels_np đã là one-hot/mixed
                processed_files_count += 1

        except InvalidDicomError: continue
        except Exception as e:
            print(f"  [ERROR] Failed to process DICOM file {dicom_path} or its ROIs: {e}")
            # import traceback; traceback.print_exc()
            continue
    
    print(f"[INFO] Total DICOM-level entries processed and augmented: {processed_files_count}")
    if not all_images_data_list:
        print("[ERROR] No data was loaded after processing. Check paths and logic.")
        return np.array([]), np.array([])

    final_images_array = np.array(all_images_data_list, dtype=np.float32)
    final_labels_array = np.array(all_labels_data_list, dtype=np.float32) # Đã là one-hot/mixed

    # Không cần encode lại bằng label_encoder.transform() ở đây nữa
    # vì generate_image_transforms đã trả về nhãn ở dạng one-hot (hoặc mixed)
    # và LabelEncoder chính (truyền vào hàm này) sẽ được dùng ở cuối pipeline huấn luyện để lấy tên lớp.

    return final_images_array, final_labels_array
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

    if config.augment_data and config.dataset == "INbreast": # Chỉ augment cho INbreast
        print(f"Applying extended augmentation for INbreast (multiplier, CutMix, MixUp)...")
        # Đảm bảo y_train là one-hot cho generate_image_transforms nếu nó mong đợi one-hot
        # Hàm generate_image_transforms phiên bản mới đã tự xử lý việc này bên trong.
        
        # Kiểm tra và đảm bảo X_train có channel dimension nếu là ảnh xám
        if X_train.ndim == 3: # (N, H, W)
            X_train = np.expand_dims(X_train, axis=-1) # (N, H, W, 1)

        X_train, y_train = generate_image_transforms(X_train, y_train)
        # y_train trả về từ generate_image_transforms (phiên bản mới) sẽ là one-hot (hoặc mixed one-hot)
        # hoặc scalar binary tùy thuộc vào logic cuối cùng của hàm đó.
        # Nếu model của bạn yêu cầu scalar labels cho binary, bạn có thể cần argmax ở đây.
        # Ví dụ: if num_classes == 2 and y_train.ndim > 1: y_train = np.argmax(y_train, axis=1)
        
        # Cập nhật lại class_weights nếu y_train thay đổi đáng kể về phân phối
        # (Hàm make_class_weights cần y_train ở dạng scalar)
        if y_train.ndim > 1 and y_train.shape[1] > 1: # Nếu y_train là one-hot
            y_train_for_weights = np.argmax(y_train, axis=1)
        else:
            y_train_for_weights = y_train
        class_weights = make_class_weights(y_train_for_weights)
        print(f"Re-calculated class weights after INbreast augmentation: {class_weights}")


    elif config.augment_data: # Augmentation cơ bản cho các dataset khác
        print(f"Applying basic augmentation for {config.dataset}...")
        if X_train is not None and y_train is not None:
            if X_train.ndim == 3: # (N, H, W)
                 X_train = np.expand_dims(X_train, axis=-1) # (N, H, W, 1)
            X_train, y_train = generate_image_transforms(X_train, y_train)
            # Cập nhật class_weights
            if y_train.ndim > 1 and y_train.shape[1] > 1:
                y_train_for_weights = np.argmax(y_train, axis=1)
            else:
                y_train_for_weights = y_train
            class_weights = make_class_weights(y_train_for_weights)
            print(f"Re-calculated class weights after basic augmentation for {config.dataset}: {class_weights}")
        else:
            print(f"Skipping augmentation for {config.dataset} as X_train or y_train is None (possibly using tf.data pipeline like CBIS-DDSM).")
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
        loaded_keras_model = load_trained_model()

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
        print(f"[INFO] Evaluating with: X_test shape {X_test.shape}, y_test shape {y_test.shape}, num_classes {cnn.num_classes}, cls_type {cls_type_eval}")
        
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



