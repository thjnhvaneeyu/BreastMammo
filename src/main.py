#!/usr/bin/env python3
# main.py (Không cắt cơ ngực - Đã cập nhật CLI và logic gọi hàm load)

import os
import sys
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import cv2
import pydicom
from pydicom.errors import InvalidDicomError
from collections import Counter # Ưu tiên 2
from imblearn.over_sampling import SMOTE # Ưu tiên 2
import config
# Đảm bảo các hàm này được import từ đúng vị trí
from data_operations.data_preprocessing import (
    make_class_weights,
    dataset_stratified_split,
    import_minimias_dataset,
    import_cbisddsm_training_dataset,
    import_cbisddsm_testing_dataset,
    import_cmmd_dataset,
    load_inbreast_data_no_pectoral_removal
)
from data_operations.data_transformations import (
    elastic_transform,
    generate_image_transforms
)
from cnn_models.cnn_model import CnnModel
import argparse
from tensorflow.keras import mixed_precision
from utils import (load_trained_model, print_num_gpus_available,
                   set_random_seeds, print_cli_arguments, print_runtime)

policy = mixed_precision.Policy("mixed_float16")
mixed_precision.set_global_policy(policy)

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

DATA_ROOT_BREAST = getattr(config, 'DATA_ROOT_BREAST', '/kaggle/input/breastdata')
# DATA_ROOT_CMMD is now expected to be handled by cli_args.data_dir if dataset is CMMD, or define similarly if fixed.

# Hàm main_logic giờ sẽ nhận cli_args đã được parse
def main_logic(cli_args):
    print_num_gpus_available()
    set_random_seeds()

    # Cập nhật config từ cli_args (đặt lên đầu để các module khác có thể dùng config đã cập nhật)
    config.dataset            = cli_args.dataset
    config.mammogram_type     = cli_args.mammogram_type
    config.model              = cli_args.model_name_arg
    config.run_mode           = cli_args.runmode
    config.learning_rate      = cli_args.learning_rate
    config.batch_size         = cli_args.batch_size
    config.max_epoch_frozen   = cli_args.max_epoch_frozen
    config.max_epoch_unfrozen = cli_args.max_epoch_unfrozen
    config.is_roi             = cli_args.roi # Note: was cli_args.use_roi for INbreast, now standardized to cli_args.roi
    config.augment_data       = cli_args.apply_elastic or cli_args.apply_mixup or cli_args.apply_cutmix
    config.verbose_mode       = cli_args.verbose_logging
    config.name               = cli_args.name
        # Cập nhật config cho các tham số mới
    config.USE_FOCAL_LOSS = cli_args.use_focal_loss
    config.FOCAL_LOSS_ALPHA = cli_args.focal_alpha
    config.FOCAL_LOSS_GAMMA = cli_args.focal_gamma
    config.APPLY_SMOTE = cli_args.apply_smote
    config.INBREAST_MANUAL_WEIGHT_BOOST = cli_args.manual_weight_boost
    config.ELASTIC_ALPHA = cli_args.elastic_alpha # Từ CLI
    config.ELASTIC_SIGMA = cli_args.elastic_sigma # Từ CLI
    config.MIXUP_ALPHA = cli_args.mixup_alpha     # Từ CLI
    config.CUTMIX_ALPHA = cli_args.cutmix_alpha   # Từ CLI
    print_cli_arguments()

    if config.verbose_mode:
        print(f"[DEBUG] CLI Args received: {cli_args}")
        print(f"[DEBUG] Config set: dataset={config.dataset}, model={config.model}, roi={config.is_roi}")
        print(f"  Augmentation flags: Elastic={cli_args.apply_elastic}, MixUp={cli_args.apply_mixup}, CutMix={cli_args.apply_cutmix}")

    # --- Dynamic LabelEncoder Initialization ---
    le = LabelEncoder()
    dataset_config_prefix = config.dataset.upper().replace("-", "_") # e.g., INBREAST, MINI_MIAS, CBIS_DDSM, CMMD
    target_classes_attr = f"{dataset_config_prefix}_TARGET_CLASSES"
    default_target_classes = ["Benign", "Malignant"] # Fallback if specific classes not in config

    if hasattr(config, target_classes_attr):
        target_classes = getattr(config, target_classes_attr)
        print(f"[INFO] Using target classes for {config.dataset} from config: {target_classes_attr}")
    else:
        target_classes = default_target_classes
        print(f"[WARNING] Attribute '{target_classes_attr}' not found in config. Using default target classes: {target_classes} for {config.dataset}.")
    
    le.fit(target_classes)
    print(f"[INFO] Main LabelEncoder classes set for {config.dataset}: {le.classes_}")
    # num_classes will be refined based on loaded data for each dataset below.
    # Initialize with a value that will be checked/updated.
    num_classes = len(le.classes_) 

    X_np = y_np = None
    X_train_np = X_val_np = X_test_np = None
    y_train_np = y_val_np = y_test_np = None
    class_weights = None

    current_data_dir = cli_args.data_dir # Centralize data directory from CLI

    # --- Xử lý cho INbreast ---
    if config.dataset.upper() == "INBREAST":
        target_size_inbreast = (config.INBREAST_IMG_SIZE["HEIGHT"], config.INBREAST_IMG_SIZE["WIDTH"])
        
        # Gọi hàm load_inbreast_data_no_pectoral_removal đã sửa đổi
        # Nó sẽ không nhận các tham số augmentation nữa
        X_np_loaded, y_np_loaded_text = load_inbreast_data_no_pectoral_removal(
            data_dir=current_data_dir, # Sửa lại nếu đường dẫn của bạn khác
            label_encoder=le, # Truyền le, nhưng hàm load sẽ không fit nó nữa
            use_roi_patches=config.is_roi,
            target_size=target_size_inbreast
        )

        if X_np_loaded is None or X_np_loaded.size == 0 or y_np_loaded_text is None or y_np_loaded_text.size == 0:
            print(f"[ERROR main_logic] Load_inbreast_data returned empty data. Exiting.")
            return
        else:
            print("[DEBUG MAIN] X_np_loaded is None or empty!")

        # Fit LabelEncoder với tất cả các nhãn text đã tải và xác định num_classes
        le.fit(y_np_loaded_text)
        num_classes = len(le.classes_)
        if num_classes < 2:
            print(f"[ERROR main_logic] Not enough classes found after loading INbreast: {le.classes_}. Need at least 2. Exiting.")
            return
        print(f"[INFO main_logic] LabelEncoder fitted for INbreast. Classes: {le.classes_}, num_classes: {num_classes}")

        # Mã hóa nhãn text sang dạng số
        y_np_loaded_numeric = le.transform(y_np_loaded_text)

        # Chuyển sang one-hot encoding nếu model yêu cầu (ví dụ: CategoricalCrossentropy)
        # CNNModel của bạn có vẻ dùng CategoricalCrossentropy cho binary nên cần one-hot
        y_np_loaded_one_hot = tf.keras.utils.to_categorical(y_np_loaded_numeric, num_classes=num_classes)
        
        # Chia dữ liệu thành train/validation/test (ví dụ 60/20/20)
        # Sử dụng y_np_loaded_numeric (nhãn số 1D) để stratify
        X_train_val, X_test_np, y_train_val_one_hot, y_test_np_encoded = train_test_split(
            X_np_loaded, y_np_loaded_one_hot,
            test_size=0.2, # 20% cho test
            stratify=y_np_loaded_numeric, # Dùng nhãn số để stratify
            random_state=config.RANDOM_SEED,
            shuffle=True
        )
        # Lấy lại nhãn số cho y_train_val để stratify lần nữa
        y_train_val_numeric = np.argmax(y_train_val_one_hot, axis=1)

        X_train_np, X_val_np, y_train_np_encoded, y_val_np_encoded = train_test_split(
            X_train_val, y_train_val_one_hot,
            test_size=0.25, # 0.25 * 0.8 = 0.2 (20% tổng cho validation)
            stratify=y_train_val_numeric,
            random_state=config.RANDOM_SEED,
            shuffle=True
        )
        # y_train_np = y_train_np_encoded
        # y_val_np = y_val_np_encoded
        # y_test_np = y_test_np_encoded # Cập nhật luôn y_test_np cho nhất quán
        y_train_np = y_train_np_encoded.astype(np.float32)
        y_val_np = y_val_np_encoded.astype(np.float32)
        # Tương tự cho y_test_np
        y_test_np = y_test_np_encoded.astype(np.float32)
        print(f"[INFO main_logic] INbreast data split: Train {X_train_np.shape[0]}, Val {X_val_np.shape[0]}, Test {X_test_np.shape[0]}")

        # --- ÁP DỤNG TĂNG CƯỜNG DỮ LIỆU CHỈ CHO TẬP HUẤN LUYỆN ---
        if config.augment_data: # config.augment_data giờ sẽ là cờ chung cho các phép aug này
            print(f"\n[INFO main_logic] Applying data augmentation to INbreast training set...")
            # generate_image_transforms nhận cờ cho từng loại augmentation
            X_train_np, y_train_np_encoded = generate_image_transforms(
                X_train_np, y_train_np_encoded, # y_train_np_encoded đã là one-hot
                apply_elastic=cli_args.apply_elastic, 
                elastic_alpha=cli_args.elastic_alpha, 
                elastic_sigma=cli_args.elastic_sigma,
                apply_mixup=cli_args.apply_mixup, 
                mixup_alpha=cli_args.mixup_alpha,
                apply_cutmix=cli_args.apply_cutmix, 
                cutmix_alpha=cli_args.cutmix_alpha
            )
            print(f"  Shapes after training set augmentation: X_train={X_train_np.shape}, y_train={y_train_np_encoded.shape}")
            y_train_np = y_train_np_encoded.astype(np.float32) # Hoặc y_train_np = y_train_np_encoded

        # Tính class_weights SAU KHI augmentation tập train (nếu không dùng SMOTE)
        # Hoặc dựa trên tập train gốc nếu SMOTE sẽ xử lý cân bằng
        if not config.APPLY_SMOTE:
            y_train_for_weights = np.argmax(y_train_np_encoded, axis=1) if y_train_np_encoded.ndim > 1 else y_train_np_encoded
            class_weights = make_class_weights(y_train_for_weights, num_classes)
            print(f"[INFO main_logic] Calculated class_weights (SMOTE is OFF): {class_weights}")
        
        # --- ÁP DỤNG SMOTE (NẾU CÓ) CHỈ CHO TẬP HUẤN LUYỆN ---
        if config.APPLY_SMOTE:
            print(f"\n[INFO main_logic] Applying SMOTE to INbreast training data...")
            # SMOTE cần X_train_np có shape (n_samples, n_features)
            # và y_train_np_encoded ở dạng nhãn số 1D
            
            original_y_train_shape_for_smote = y_train_np_encoded.shape
            y_train_labels_for_smote = np.argmax(y_train_np_encoded, axis=1) if y_train_np_encoded.ndim > 1 else y_train_np_encoded.astype(int)
            
            print(f"  Original y_train distribution before SMOTE: {Counter(y_train_labels_for_smote)}")
            original_X_train_shape_for_smote = X_train_np.shape
            X_train_reshaped_for_smote = X_train_np.reshape(original_X_train_shape_for_smote[0], -1)

            smote_instance = SMOTE(random_state=config.RANDOM_SEED)
            try:
                X_train_smote, y_train_smote_labels = smote_instance.fit_resample(X_train_reshaped_for_smote, y_train_labels_for_smote)
                print(f"  y_train distribution after SMOTE: {Counter(y_train_smote_labels)}")
                
                X_train_np = X_train_smote.reshape(-1, *original_X_train_shape_for_smote[1:])
                y_train_np_encoded = tf.keras.utils.to_categorical(y_train_smote_labels, num_classes=num_classes)
                y_train_np = y_train_np_encoded.astype(np.float32)
                
                print(f"  Shapes after SMOTE: X_train_np={X_train_np.shape}, y_train_np_encoded={y_train_np_encoded.shape}")
                # y_train_np = y_train_np_encoded.astype(np.float32) 
                print("  Setting class_weights to None after SMOTE.")
                class_weights = None # SMOTE đã cân bằng, không cần class weights nữa
            except ValueError as e_smote:
                print(f"[ERROR main_logic] SMOTE failed: {e_smote}. Proceeding without SMOTE.")
                # Nếu SMOTE lỗi, tính class_weights dựa trên dữ liệu trước SMOTE
                if class_weights is None: # Chỉ tính nếu chưa được tính (ví dụ SMOTE là False ngay từ đầu)
                    y_train_for_weights = np.argmax(y_train_np_encoded, axis=1) if y_train_np_encoded.ndim > 1 else y_train_np_encoded
                    class_weights = make_class_weights(y_train_for_weights, num_classes)
                    print(f"  Calculated class_weights (SMOTE failed/skipped): {class_weights}")
        
        # --- ĐIỀU CHỈNH CLASS_WEIGHTS THỦ CÔNG (ƯU TIÊN 3) ---
        # Logic này sẽ chạy nếu class_weights không phải None (tức là SMOTE OFF hoặc SMOTE lỗi)
        # HOẶC nếu SMOTE ON nhưng bạn vẫn muốn boost (manual_weight_boost > 1.0)
        if (class_weights is not None and config.INBREAST_MANUAL_WEIGHT_BOOST > 1.0 and not config.APPLY_SMOTE) or \
           (config.APPLY_SMOTE and config.INBREAST_MANUAL_WEIGHT_BOOST > 1.0):
            
            print(f"\n[INFO main_logic] Attempting manual class weight adjustment for INbreast.")
            # Nếu SMOTE đã chạy và class_weights là None, tạo base weights để boost
            if class_weights is None and config.APPLY_SMOTE:
                print("  SMOTE was applied, class_weights is None. Creating base weights (1.0) for manual boost.")
                class_weights = {i: 1.0 for i in range(num_classes)}

            if class_weights is not None: # Kiểm tra lại sau khi có thể đã được tạo ở trên
                print(f"  Current weights before manual boost: {class_weights}")
                try:
                    malignant_class_index = le.transform(['Malignant'])[0] # Lấy index của 'Malignant'
                    print(f"  LabelEncoder classes: {list(le.classes_)}, Malignant index: {malignant_class_index}")

                    if malignant_class_index in class_weights:
                        original_malignant_weight = class_weights.get(malignant_class_index, 1.0)
                        if config.INBREAST_MANUAL_WEIGHT_BOOST > 1.0:
                            class_weights[malignant_class_index] = original_malignant_weight * config.INBREAST_MANUAL_WEIGHT_BOOST
                            print(f"  Manually boosted Malignant class ({malignant_class_index}) weight. New weight: {class_weights[malignant_class_index]:.2f}")
                        print(f"  Final class_weights for INbreast after manual adjustment: {class_weights}")
                    else:
                        print(f"[WARNING main_logic] Malignant class index {malignant_class_index} not found in class_weights: {class_weights}. Manual boost not applied.")
                except Exception as e_cw_manual:
                    print(f"[ERROR main_logic] Failed to manually adjust class weights: {e_cw_manual}")
            else:
                print("[INFO main_logic] Manual class weight boost skipped as class_weights is None (and no base was created if SMOTE was OFF).")

    # --- Xử lý cho mini-MIAS ---
    elif config.dataset.upper() in ["MINI-MIAS", "MINI-MIAS-BINARY"]:
        # Assuming import_minimias_dataset returns X, y (scalar labels)
        # And current_data_dir is expected to point to the root of mini-MIAS data structured as needed by the import function.
        # If mini-MIAS path is fixed, use os.path.join(DATA_ROOT_BREAST, config.dataset) or adjust current_data_dir logic.
        X_np, y_scalar_labels = import_minimias_dataset(current_data_dir, le) # Pass le for consistent encoding
        if X_np.size == 0: print(f"[ERROR] No {config.dataset} data loaded. Exiting."); return

        # Determine num_classes from loaded data
        num_classes = len(np.unique(y_scalar_labels))
        if num_classes == 1: num_classes = 2 # Assume binary if only one class label present in data
        if num_classes != len(le.classes_):
             print(f"[INFO] Updated num_classes for {config.dataset} from loaded data to: {num_classes} (LE initially had {len(le.classes_)})")

        # Convert scalar labels to one-hot encoding using the determined num_classes
        y_np = tf.keras.utils.to_categorical(y_scalar_labels, num_classes=num_classes)

        # Data splitting (similar to INbreast)
        y_stratify = y_scalar_labels # Use scalar labels for stratification
        unique_labels_stratify = np.unique(y_stratify)
        stratify_param = y_stratify if len(unique_labels_stratify) >= 2 else None
        
        X_train_val, X_test_np, y_train_val, y_test_np = train_test_split(
            X_np, y_np, test_size=0.2, stratify=stratify_param, random_state=config.RANDOM_SEED, shuffle=True
        )
        
        y_train_val_stratify_inner = np.argmax(y_train_val, axis=1) if y_train_val.ndim > 1 and y_train_val.shape[1] > 1 else y_train_val # Use one-hot for argmax if needed
        unique_labels_stratify_inner = np.unique(y_train_val_stratify_inner)
        stratify_param_inner = y_train_val_stratify_inner if len(unique_labels_stratify_inner) >=2 else None

        X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
            X_train_val, y_train_val, test_size=0.25, stratify=stratify_param_inner, random_state=config.RANDOM_SEED, shuffle=True
        )
        
        y_train_for_weights = np.argmax(y_train_np, axis=1) if y_train_np.ndim > 1 and y_train_np.shape[1] > 1 else y_train_np
        if y_train_for_weights.size > 0: class_weights = make_class_weights(y_train_for_weights, num_classes)
        else: class_weights = None

    # --- Xử lý cho CBIS-DDSM ---
    elif config.dataset.upper() == "CBIS-DDSM":
        # Assuming import functions return X_train, y_train_scalar, X_test, y_test_scalar
        # And current_data_dir is used by these import functions.
        # This section assumes your import functions might directly provide splits.
        # If they return full X, y, then adapt splitting logic from INbreast/mini-MIAS.
        print(f"[INFO] Loading CBIS-DDSM. Data directory: {current_data_dir}")
        # The import functions should handle 'le' for encoding.
        # Let's assume they return scalar labels that need one-hot encoding.
        _X_train_np, _y_train_scalar = import_cbisddsm_training_dataset(le, data_root_dir=current_data_dir)
        _X_test_np,  _y_test_scalar  = import_cbisddsm_testing_dataset(le, data_root_dir=current_data_dir)

        if _X_train_np.size == 0 or _X_test_np.size == 0:
            print("[ERROR] CBIS-DDSM training or testing data not loaded. Exiting."); return

        # Determine num_classes from training data (or combined, ensure consistency)
        num_classes = len(np.unique(_y_train_scalar))
        if num_classes == 1: num_classes = 2
        if num_classes != len(le.classes_):
             print(f"[INFO] Updated num_classes for CBIS-DDSM from loaded data to: {num_classes} (LE initially had {len(le.classes_)})")

        y_train_np = tf.keras.utils.to_categorical(_y_train_scalar, num_classes=num_classes)
        y_test_np = tf.keras.utils.to_categorical(_y_test_scalar, num_classes=num_classes)
        X_train_np = _X_train_np
        X_test_np = _X_test_np
        
        # Create validation set from training set
        y_train_stratify = _y_train_scalar
        unique_labels_stratify = np.unique(y_train_stratify)
        stratify_param_train = y_train_stratify if len(unique_labels_stratify) >=2 else None

        X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
            X_train_np, y_train_np, test_size=0.2, # e.g., 20% of original train for validation
            stratify=stratify_param_train, random_state=config.RANDOM_SEED, shuffle=True
        )
        
        y_train_for_weights = np.argmax(y_train_np, axis=1) if y_train_np.ndim > 1 and y_train_np.shape[1] > 1 else y_train_np
        if y_train_for_weights.size > 0: class_weights = make_class_weights(y_train_for_weights, num_classes)
        else: class_weights = None

    #     def load_images_from_paths(paths_list, target_h, target_w):
    #         images_out = []
    #         for p in paths_list:
    #             img = load_img(p, color_mode="grayscale", target_size=(target_h, target_w))
    #             img_arr = img_to_array(img) / 255.0
    #             # Xử lý kênh cho model
    #             if config.model != "CNN": # Cần 3 kênh
    #                 img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB) if img_arr.shape[-1] == 1 else img_arr
    #             # Nếu CNN mà ảnh là RGB (hiếm), thì chuyển về xám
    #             elif config.model == "CNN" and img_arr.shape[-1] == 3:
    #                 img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
    #                 img_arr = np.expand_dims(img_arr, axis=-1)

    #             images_out.append(img_arr)
    #         return np.array(images_out, dtype=np.float32)

    #     cbis_target_h = config.VGG_IMG_SIZE["HEIGHT"] # Ví dụ, hoặc lấy từ config theo model
    #     cbis_target_w = config.VGG_IMG_SIZE["WIDTH"]
    #     X_train_np_all = load_images_from_paths(X_train_paths, cbis_target_h, cbis_target_w)
    #     y_train_one_hot_all = tf.keras.utils.to_categorical(y_train_numeric, num_classes=num_classes)
        
    #     X_test_np = load_images_from_paths(X_test_paths, cbis_target_h, cbis_target_w)
    #     y_test_np_encoded = tf.keras.utils.to_categorical(y_test_numeric, num_classes=num_classes)

    #     X_train_np, X_val_np, y_train_np_encoded, y_val_np_encoded = train_test_split(
    #         X_train_np_all, y_train_one_hot_all, test_size=0.2, stratify=y_train_numeric, random_state=config.RANDOM_SEED
    #     )
    #     if config.augment_data:
    #         print(f"\n[INFO main_logic] Applying data augmentation to {config.dataset} training set...")
    #         # (Tương tự gọi generate_image_transforms)
    #         X_train_np, y_train_np_encoded = generate_image_transforms(
    #             X_train_np, y_train_np_encoded,
    #             apply_elastic=cli_args.apply_elastic, elastic_alpha=cli_args.elastic_alpha, elastic_sigma=cli_args.elastic_sigma,
    #             apply_mixup=cli_args.apply_mixup, mixup_alpha=cli_args.mixup_alpha,
    #             apply_cutmix=cli_args.apply_cutmix, cutmix_alpha=cli_args.cutmix_alpha
    #         )
    #     y_train_for_weights = np.argmax(y_train_np_encoded, axis=1) if y_train_np_encoded.ndim > 1 else y_train_np_encoded
    #     class_weights = make_class_weights(y_train_for_weights, num_classes)
        
    # else:
    #     print(f"[ERROR main_logic] Dataset '{config.dataset}' not supported in this simplified flow. Exiting.")
    #     return
    elif config.dataset.upper() == "CMMD":
        # Assuming import_cmmd_dataset returns X, y (scalar labels)
        # And current_data_dir points to CMMD root.
        X_np, y_scalar_labels = import_cmmd_dataset(current_data_dir, le) # Pass le
        if X_np.size == 0: print("[ERROR] No CMMD data loaded. Exiting."); return

        num_classes = len(np.unique(y_scalar_labels))
        if num_classes == 1: num_classes = 2
        if num_classes != len(le.classes_):
             print(f"[INFO] Updated num_classes for CMMD from loaded data to: {num_classes} (LE initially had {len(le.classes_)})")
        
        y_np = tf.keras.utils.to_categorical(y_scalar_labels, num_classes=num_classes)
        
        # Data splitting
        y_stratify = y_scalar_labels
        unique_labels_stratify = np.unique(y_stratify)
        stratify_param = y_stratify if len(unique_labels_stratify) >= 2 else None
        
        X_train_val, X_test_np, y_train_val, y_test_np = train_test_split(
            X_np, y_np, test_size=0.2, stratify=stratify_param, random_state=config.RANDOM_SEED, shuffle=True
        )
        
        y_train_val_stratify_inner = np.argmax(y_train_val, axis=1) if y_train_val.ndim > 1 and y_train_val.shape[1] > 1 else y_train_val
        unique_labels_stratify_inner = np.unique(y_train_val_stratify_inner)
        stratify_param_inner = y_train_val_stratify_inner if len(unique_labels_stratify_inner) >=2 else None

        X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
            X_train_val, y_train_val, test_size=0.25, stratify=stratify_param_inner, random_state=config.RANDOM_SEED, shuffle=True
        )
        # if config.augment_data:
        #     print(f"\n[INFO main_logic] Applying data augmentation to {config.dataset} training set...")
        #     X_train_np, y_train_np_encoded = generate_image_transforms(
        #         X_train_np, y_train_np_encoded,
        #         apply_elastic=cli_args.apply_elastic, elastic_alpha=cli_args.elastic_alpha, elastic_sigma=cli_args.elastic_sigma,
        #         apply_mixup=cli_args.apply_mixup, mixup_alpha=cli_args.mixup_alpha,
        #         apply_cutmix=cli_args.apply_cutmix, cutmix_alpha=cli_args.cutmix_alpha
        #     )
        # y_train_for_weights = np.argmax(y_train_np_encoded, axis=1) if y_train_np_encoded.ndim > 1 else y_train_np_encoded
        # class_weights = make_class_weights(y_train_for_weights, num_classes)
        y_labels_for_class_weights_calc = None # Biến tạm để lưu nhãn dùng tính class weights

        if config.augment_data:
            print(f"\n[INFO main_logic] Applying data augmentation to {config.dataset} training set...")
            # generate_image_transforms trả về y_train_np_encoded là nhãn sau augmentation
            X_train_np, y_train_np_encoded_after_aug = generate_image_transforms( # Đổi tên biến để rõ ràng
                X_train_np, y_train_np, # y_train_np ở đây là one-hot từ split
                apply_elastic=cli_args.apply_elastic, elastic_alpha=cli_args.elastic_alpha, elastic_sigma=cli_args.elastic_sigma,
                apply_mixup=cli_args.apply_mixup, mixup_alpha=cli_args.mixup_alpha,
                apply_cutmix=cli_args.apply_cutmix, cutmix_alpha=cli_args.cutmix_alpha
            )
            # Cập nhật y_train_np để phản ánh dữ liệu sau augmentation cho các bước sau (nếu cần)
            y_train_np = y_train_np_encoded_after_aug # Gán lại y_train_np nếu có augmentation
            y_labels_for_class_weights_calc = y_train_np_encoded_after_aug # Dùng nhãn sau aug để tính weights
        else:
            # Nếu không có augmentation, nhãn để tính class_weights chính là y_train_np (one-hot gốc)
            y_labels_for_class_weights_calc = y_train_np

        # Bây giờ tính y_train_for_weights dựa trên y_labels_for_class_weights_calc
        # Đảm bảo y_labels_for_class_weights_calc không phải là None trước khi truy cập
        if y_labels_for_class_weights_calc is not None:
            # Chuyển về nhãn số (1D) nếu nó đang là one-hot (2D)
            y_train_for_weights = np.argmax(y_labels_for_class_weights_calc, axis=1) if y_labels_for_class_weights_calc.ndim > 1 and y_labels_for_class_weights_calc.shape[1] > 1 else y_labels_for_class_weights_calc
            
            # Kiểm tra nếu y_train_for_weights vẫn là 2D (ví dụ: (N,1)), thì flatten nó
            if y_train_for_weights.ndim > 1 and y_train_for_weights.shape[1] == 1:
                y_train_for_weights = y_train_for_weights.flatten()

            if y_train_for_weights.size > 0:
                class_weights = make_class_weights(y_train_for_weights, num_classes)
            else:
                class_weights = None
                print("[WARN CMMD] y_train_for_weights is empty after processing, cannot calculate class_weights.")
        else:
            class_weights = None
            print("[WARN CMMD] Labels for class weight calculation (y_labels_for_class_weights_calc) is None.")


# Trong main.py, sau khi có y_train_np, y_val_np, y_test_np
        print(f"\n[INFO] Class distribution for {config.dataset}:")
        if y_train_np is not None:
            # Nếu y_train_np là one-hot, cần argmax để lấy nhãn số
            y_train_labels = np.argmax(y_train_np, axis=1) if y_train_np.ndim > 1 and y_train_np.shape[1] > 1 else y_train_np
            print(f"  Training set: {Counter(y_train_labels)}")
        if y_val_np is not None:
            y_val_labels = np.argmax(y_val_np, axis=1) if y_val_np.ndim > 1 and y_val_np.shape[1] > 1 else y_val_np
            print(f"  Validation set: {Counter(y_val_labels)}")
        if y_test_np is not None:
            y_test_labels = np.argmax(y_test_np, axis=1) if y_test_np.ndim > 1 and y_test_np.shape[1] > 1 else y_test_np
            print(f"  Test set: {Counter(y_test_labels)}")
# Làm tương tự cho val, test và cho CMMD
        y_train_for_weights = np.argmax(y_train_np, axis=1) if y_train_np.ndim > 1 and y_train_np.shape[1] > 1 else y_train_np
        if y_train_for_weights.size > 0: class_weights = make_class_weights(y_train_for_weights, num_classes)

        else: class_weights = None
        print("[WARN CMMD] y_train_for_weights is empty, cannot calculate class_weights.")

    else:
        print(f"[ERROR] Dataset '{config.dataset}' is not supported or logic is not implemented. Exiting.")
        return

    # --- Common logic post data loading and splitting ---
    if num_classes == 0 or (X_train_np is not None and X_train_np.size == 0): # Check num_classes specifically
        print(f"[ERROR] Number of classes is {num_classes} or no training data available for {config.dataset}. Cannot build or train model.")
        return
    
    print(f"[INFO] Final configuration for {config.dataset}: num_classes for model={num_classes}, class_weights={class_weights}")
    print(f"  Shapes: Train X:{X_train_np.shape if X_train_np is not None else 'None'}, Y:{y_train_np.shape if y_train_np is not None else 'None'}")
    print(f"          Val   X:{X_val_np.shape if X_val_np is not None else 'None'}, Y:{y_val_np.shape if y_val_np is not None else 'None'}")
    print(f"          Test  X:{X_test_np.shape if X_test_np is not None else 'None'}, Y:{y_test_np.shape if y_test_np is not None else 'None'}")


    cnn = CnnModel(config.model, num_classes) # Use the determined num_classes
    cnn.compile_model(config.learning_rate)

    start_time_train = time.time()
    if config.run_mode.lower() == "train":
        print(f"[INFO] Running in TRAIN mode for {config.dataset}.")
        # if X_train_np is not None and y_train_np is not None and X_train_np.size > 0 :
        if X_train_np is not None:
            print(f"[DEBUG DTYPE PRE-FIT] X_train_np.dtype: {X_train_np.dtype}")
        if y_train_np is not None:
            print(f"[DEBUG DTYPE PRE-FIT] y_train_np.dtype: {y_train_np.dtype}")
        if X_val_np is not None:
            print(f"[DEBUG DTYPE PRE-FIT] X_val_np.dtype: {X_val_np.dtype}")
        if y_val_np is not None:
            print(f"[DEBUG DTYPE PRE-FIT] y_val_np.dtype: {y_val_np.dtype}")
            cnn.train_model(X_train_np, X_val_np, y_train_np, y_val_np, class_weights)
        else: print(f"[ERROR] Training data for {config.dataset} is empty or None. Cannot train."); return
        print_runtime("Model training", time.time() - start_time_train)
        cnn.save_model() # Consider making model name dataset-specific

        print(f"[INFO] Evaluating model on test set for {config.dataset} after training...")
        if X_test_np is not None and y_test_np is not None and X_test_np.size > 0:
            cls_type_eval = 'binary' if num_classes == 2 else 'multiclass'
            cnn.evaluate_model(X_test_np, y_test_np, le, cls_type_eval, time.time() - start_time_train)
        else: print(f"[ERROR] Test data for {config.dataset} is empty or None. Cannot evaluate.")

    elif config.run_mode.lower() == "test":
        print(f"[INFO] Running in TEST mode for {config.dataset}.")
        # Model path might need to be dataset-specific if you train separate models
        # loaded_keras_model = load_trained_model(model_name_prefix=f"{config.dataset}_{config.model}") # Example modification
        loaded_keras_model = load_trained_model() # Example modification
        if loaded_keras_model is None:
            print(f"[ERROR] Failed to load trained model. Attempted with prefix: {config.dataset}_{config.model}")
            raise FileNotFoundError("Failed to load trained model for testing.")
        
        output_neurons = loaded_keras_model.output_shape[-1]
        actual_num_classes_model = output_neurons if output_neurons > 1 else 2 # Handle single neuron for binary

        # If the loaded model's classes differ from what data expects, it could be an issue.
        # Here, we re-initialize CnnModel's num_classes based on the loaded model.
        if cnn.num_classes != actual_num_classes_model :
            print(f"[INFO] Re-initializing CnnModel for loaded model with {actual_num_classes_model} classes (was {cnn.num_classes}).")
            cnn = CnnModel(config.model, actual_num_classes_model) # Use model's actual classes
        cnn.model = loaded_keras_model

        if X_test_np is None or y_test_np is None or X_test_np.size == 0:
            print(f"[ERROR] Test data for {config.dataset} is None or empty. Cannot evaluate."); return
        
        # Ensure `le` used for evaluation matches the classes the model was trained on.
        # This might require saving/loading `le` with the model or ensuring consistent class definitions.
        cls_type_eval = 'binary' if cnn.num_classes == 2 else 'multiclass'
        print(f"[DEBUG main_logic] Before final evaluation:")
        print(f"  X_test_np shape: {X_test_np.shape if X_test_np is not None else 'None'}")
        print(f"  y_test_np shape: {y_test_np.shape if y_test_np is not None else 'None'}")
        print(f"  Number of classes in LabelEncoder: {len(le.classes_) if le.classes_ is not None else 'N/A'}")
        print(f"  LabelEncoder classes: {le.classes_ if le.classes_ is not None else 'N/A'}")
        print(f"  cls_type_eval: {cls_type_eval}") # Đây là cls_type_eval_after_train trong code bạn gửi

        cnn.evaluate_model(X_test_np, y_test_np, le, cls_type_eval, time.time() - start_time_train)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mammogram DL pipeline")
    # Adjusted choices for dataset to include others, assuming they are supported by data loading logic.
    parser.add_argument("--data_dir", type=str, help="Path to the root dataset directory (e.g., INbreast/INbreast, CMMD_data_root, etc.). Structure inside depends on the dataset.")
    parser.add_argument("-d", "--dataset", choices=["INbreast", "mini-MIAS", "mini-MIAS-binary", "CBIS-DDSM", "CMMD"], required=True, help="Dataset to use.")
    parser.add_argument("--model_name_arg", "--model", dest="model_name_arg", choices=["CNN","VGG","VGG-common","ResNet","Inception","DenseNet","MobileNet"], required=True, help="Model backbone")
    parser.add_argument("-r", "--runmode", choices=["train","test"], default="train", help="train or test")
    parser.add_argument("-lr", "--learning_rate", type=float, default=config.learning_rate, help="Learning rate")
    parser.add_argument("-b", "--batch_size", type=int, default=config.batch_size, help="Batch size")
    parser.add_argument("-e1", "--max_epoch_frozen", type=int, default=config.max_epoch_frozen, help="Frozen epochs")
    parser.add_argument("-e2", "--max_epoch_unfrozen", type=int, default=config.max_epoch_unfrozen, help="Unfrozen epochs")
    parser.add_argument("--roi", action="store_true", help="Use ROI patches (mainly for INbreast, adapt if other datasets use ROIs).")
    
    parser.add_argument("--apply_elastic", action='store_true', help="Apply Elastic Transform.")
    parser.add_argument("--elastic_alpha", type=float, default=34.0, help="Alpha for Elastic Transform.") # Default from original INbreast call
    parser.add_argument("--elastic_sigma", type=float, default=4.0, help="Sigma for Elastic Transform.") # Default from original INbreast call
    
    parser.add_argument("--apply_mixup", action='store_true', help="Apply MixUp augmentation.")
    parser.add_argument("--mixup_alpha", type=float, default=0.2, help="Alpha for MixUp.")
    
    parser.add_argument("--apply_cutmix", action='store_true', help="Apply CutMix augmentation.")
    parser.add_argument("--cutmix_alpha", type=float, default=1.0, help="Alpha for CutMix.")
    parser.add_argument("--use_focal_loss", 
                        action="store_true",  # Tạo cờ boolean, nếu có mặt thì là True
                        default=False,        # Giá trị mặc định nếu cờ không có mặt
                        help="Use Focal Loss instead of CategoricalCrossentropy (primarily for INbreast binary).")
    parser.add_argument("--focal_alpha", 
                        type=float, 
                        default=0.25, # Giá trị mặc định từ config hoặc theo ý bạn
                        help="Alpha parameter for Focal Loss.")

    parser.add_argument("--focal_gamma", 
                        type=float, 
                        default=2.0,  # Giá trị mặc định từ config hoặc theo ý bạn
                        help="Gamma parameter for Focal Loss.")
    parser.add_argument("--apply_smote", 
                        action="store_true",  # Tạo cờ boolean, nếu có mặt thì là True
                        default=False,        # Giá trị mặc định nếu cờ không có mặt
                        help="Apply SMOTE to the training data (primarily for INbreast).")
    parser.add_argument("--manual_weight_boost", 
                        type=float, 
                        default=1.0,  # Giá trị mặc định (1.0 nghĩa là không boost)
                        help="Factor to manually boost the weight of the Malignant class for INbreast (e.g., 2.0). Active if SMOTE is off or boost > 1.0.")

    parser.add_argument("--verbose_logging", action="store_true", default=config.verbose_mode, help="Verbose logging.")
    parser.add_argument("-n", "--name", default=config.name, help="Experiment name.")
    # Added mammogram_type for datasets other than INbreast, though INbreast sets it to "all"
    parser.add_argument("--mammogram_type", default="all", help="Mammogram type (e.g., CC, MLO), relevant if dataset loader uses it.")


    args = parser.parse_args()

    # --- Data Directory Validation ---
    if not args.data_dir:
        # Try to use a global default if data_dir not provided
        if args.dataset.upper() == "INBREAST" and DATA_ROOT_BREAST and os.path.isdir(os.path.join(DATA_ROOT_BREAST, "INbreast", "INbreast")):
            args.data_dir = os.path.join(DATA_ROOT_BREAST, "INbreast", "INbreast")
            print(f"[INFO] Using default DATA_ROOT_BREAST for INbreast. Effective data_dir: {args.data_dir}")
        # Add similar default logic for other datasets if applicable, e.g., CMMD_DATA_ROOT
        # elif args.dataset.upper() == "CMMD" and DATA_ROOT_CMMD_CLI_ARG_OR_CONFIG: # Example
        #     args.data_dir = DATA_ROOT_CMMD_CLI_ARG_OR_CONFIG
        #     print(f"[INFO] Using default for CMMD. Effective data_dir: {args.data_dir}")
        else:
            parser.error("--data_dir is required if a suitable default is not available for the selected dataset.")

    # Specific validation for INbreast structure if it's the selected dataset
    if args.dataset.upper() == "INBREAST":
        if not (os.path.isdir(os.path.join(args.data_dir, "AllDICOMs")) and os.path.exists(os.path.join(args.data_dir, "INbreast.csv"))):
            # Try to adjust if a parent directory was given
            potential_path = os.path.join(args.data_dir, "INbreast", "INbreast")
            if os.path.isdir(potential_path) and os.path.exists(os.path.join(potential_path, "INbreast.csv")):
                args.data_dir = potential_path
                print(f"[INFO] Adjusted INbreast data_dir to: {args.data_dir}")
            else:
                 parser.error(f"The provided --data_dir '{args.data_dir}' for INbreast does not point to the 'INbreast/INbreast' subfolder or is missing key components. It should contain 'AllDICOMs' and 'INbreast.csv'.")
    
    # For other datasets, you might add specific path validation here if needed.
    # For example, check if args.data_dir is a valid directory:
    elif not os.path.isdir(args.data_dir):
        parser.error(f"The provided --data_dir '{args.data_dir}' is not a valid directory for dataset {args.dataset}.")


    # Cập nhật config từ args (moved to top of main_logic, this is redundant here if main_logic uses cli_args directly)
    # We are passing cli_args to main_logic, so config updates happen there.
    # This section can be removed if all config updates are consolidated at the start of main_logic.
    # For clarity, ensuring config is set based on final args values before main_logic is called:
    config.dataset = args.dataset
    config.model = args.model_name_arg
    config.mammogram_type     = "all" if args.dataset.upper() == "INBREAST" else args.mammogram_type
    config.run_mode           = args.runmode
    config.learning_rate      = args.learning_rate
    config.batch_size         = args.batch_size
    config.max_epoch_frozen   = args.max_epoch_frozen
    config.max_epoch_unfrozen = args.max_epoch_unfrozen
    config.is_roi             = args.roi # Standardized
    config.augment_data       = args.apply_elastic or args.apply_mixup or args.apply_cutmix
    config.verbose_mode       = args.verbose_logging
    config.name               = args.name
    config.ELASTIC_ALPHA      = args.elastic_alpha
    config.ELASTIC_SIGMA      = args.elastic_sigma
    config.MIXUP_ALPHA        = args.mixup_alpha
    config.CUTMIX_ALPHA       = args.cutmix_alpha
    config.USE_FOCAL_LOSS     = args.use_focal_loss
    config.FOCAL_LOSS_ALPHA   = args.focal_alpha
    config.FOCAL_LOSS_GAMMA   = args.focal_gamma
    config.APPLY_SMOTE        = args.apply_smote
    config.INBREAST_MANUAL_WEIGHT_BOOST = args.manual_weight_boost
    # Cờ augment_data chung có thể được suy ra từ các cờ cụ thể
    config.augment_data = args.apply_elastic or args.apply_mixup or args.apply_cutmix or args.apply_smote

    start_time_main = time.time()
    main_logic(args)
    print(f"Total execution time of main.py: {time.time() - start_time_main:.2f} seconds.")        
