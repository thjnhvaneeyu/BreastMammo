import os

from imutils import paths
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical  # nếu bạn dùng to_categorical trong full‐mode
from PIL import Image
import pydicom
import cv2
import re
import config
from data_operations.data_transformations import generate_image_transforms
import xml.etree.ElementTree as ET
from pydicom.errors import InvalidDicomError
from sklearn.utils.class_weight import compute_class_weight
from typing import List, Tuple, Optional, Dict
from sklearn.preprocessing import LabelEncoder

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def make_class_weights(y, num_classes=None): # Đổi tên tham số
    y_processed = y  # Khởi tạo y_processed

    # Xử lý y đầu vào để có được mảng nhãn 1D (y_processed)
    if isinstance(y, np.ndarray) and y.ndim > 1 and y.shape[1] > 1:
        y_processed = np.argmax(y, axis=1)
    elif isinstance(y, list) and y and isinstance(y[0], np.ndarray): # Thêm kiểm tra y không rỗng
        try:
            y_processed = np.array([np.argmax(vec) for vec in y])
        except:
            y_processed = np.array(y).ravel()
    elif isinstance(y, np.ndarray):
        y_processed = y.ravel()
    else:
        y_processed = np.array(y)

    # Xử lý trường hợp y_processed rỗng sau các bước trên
    if y_processed.size == 0:
        if num_classes is not None and num_classes > 0:
            # Trả về trọng số bằng nhau (ví dụ: 1.0) cho tất cả các lớp nếu biết tổng số lớp
            return {i: 1.0 for i in range(num_classes)}
        return {} # Trả về dictionary rỗng nếu không có thông tin

    # Xác định các lớp thực sự có mặt trong y_processed
    present_classes = np.unique(y_processed)
    
    # Tính toán trọng số cho các lớp hiện có
    try:
        weights_array = compute_class_weight(class_weight="balanced", classes=present_classes, y=y_processed)
        # Tạo dictionary trọng số với key là kiểu int
        calculated_weights = {int(cls): w for cls, w in zip(present_classes, weights_array)}
    except ValueError as e:
        # Trường hợp compute_class_weight báo lỗi (ví dụ: chỉ có 1 lớp trong một batch nhỏ)
        print(f"[WARN] make_class_weights: Could not compute class weights via sklearn ({e}). Defaulting to 1.0 for present classes.")
        calculated_weights = {int(cls): 1.0 for cls in present_classes}


    if num_classes is not None:
        # Nếu num_classes_for_weights được cung cấp, đảm bảo dictionary cuối cùng có đủ các mục.
        # Các lớp không có trong y_processed (và do đó không có trong calculated_weights)
        # sẽ được gán trọng số mặc định là 1.0.
        final_class_weights = {i: 1.0 for i in range(num_classes)}
        final_class_weights.update(calculated_weights) # Ghi đè bằng trọng số đã tính cho các lớp hiện có
        return final_class_weights
    else:
        # Nếu không cung cấp num_classes_for_weights, chỉ trả về trọng số cho các lớp hiện có.
        return calculated_weights

def make_class_weights_in(y) -> Dict[int, float]:
    # đảm bảo y là 1-D numpy array
    y_arr = np.asarray(y).ravel()
    # các lớp duy nhất
    classes = np.unique(y_arr)
    # compute_class_weight chỉ chấp nhận y dạng 1-D array
    weights = compute_class_weight("balanced", classes=classes, y=y_arr)
    # trả về dict int→float
    return {int(c): float(w) for c, w in zip(classes, weights)}

# def load_inbreast_data_no_pectoral_removal(
#     data_dir: str, # Đường dẫn đầy đủ đến thư mục INbreast (ví dụ: .../INbreast/INbreast)
#     label_encoder: LabelEncoder, # LabelEncoder đã được fit ở hàm main_logic
#     use_roi_patches: bool,
#     target_size: tuple,
#     # Tham số augmentation từ CLI
#     enable_elastic: bool = False, elastic_alpha_val: float = 34.0, elastic_sigma_val: float = 4.0,
#     enable_mixup: bool = False, mixup_alpha_val: float = 0.2,
#     enable_cutmix: bool = False, cutmix_alpha_val: float = 1.0
# ):
#     print(f"\n[INFO] Loading INbreast data (No Pectoral Removal) {'with ROI patches' if use_roi_patches else 'as full images'}...")
#     print(f"  Target Size: {target_size}")
#     print(f"  Augmentation - Elastic: {enable_elastic} (alpha:{elastic_alpha_val}, sigma:{elastic_sigma_val})")
#     print(f"  Augmentation - MixUp: {enable_mixup} (alpha:{mixup_alpha_val})")
#     print(f"  Augmentation - CutMix: {enable_cutmix} (alpha:{cutmix_alpha_val})")

#     dicom_dir = os.path.join(data_dir, "AllDICOMs")
#     roi_dir = os.path.join(data_dir, "AllROI")
#     csv_path = os.path.join(data_dir, "INbreast.csv")

#     if not os.path.exists(csv_path): raise FileNotFoundError(f"INbreast.csv not found at {csv_path}")
#     if not os.path.isdir(dicom_dir): raise NotADirectoryError(f"DICOM directory not found: {dicom_dir}")
#     if use_roi_patches and not os.path.isdir(roi_dir): raise NotADirectoryError(f"ROI directory not found: {roi_dir}")

#     df = pd.read_csv(csv_path, sep=';')
#     df.columns = [c.strip() for c in df.columns]
#     birad_map = {str(row['File Name']).strip(): str(row['Bi-Rads']).strip() for _, row in df.iterrows()}

#     all_images_data_accumulator = [] # List để chứa các NumPy arrays của ảnh
#     all_labels_accumulator = []      # List để chứa các NumPy arrays của nhãn (đã qua one-hot/mixed)

#     processed_dicom_count = 0

#     for index, row in df.iterrows():
#         file_id_csv = str(row['File Name']).strip()
        
#         dicom_path = None
#         # Thử tìm file DICOM bằng file_id_csv trực tiếp, sau đó là với pattern
#         direct_dicom_path = os.path.join(dicom_dir, file_id_csv + ".dcm")
#         if os.path.exists(direct_dicom_path):
#             dicom_path = direct_dicom_path
#         else:
#             for name_pattern_match in tf.io.gfile.glob(os.path.join(dicom_dir, file_id_csv + "*.dcm")):
#                 dicom_path = name_pattern_match
#                 break
        
#         if not dicom_path:
#             # print(f"  [DEBUG] No DICOM found for CSV File Name: {file_id_csv}")
#             continue

#         base_dicom_name_no_ext = os.path.splitext(os.path.basename(dicom_path))[0]

#         try:
#             dicom_data = pydicom.dcmread(dicom_path)
#             image_array_original = dicom_data.pixel_array.astype(np.float32)
#             if config.verbose_mode:
#                 print(f"  [DEBUG DICOM Initial] Path: {dicom_path}, Initial Shape: {image_array_original.shape}, Ndim: {image_array_original.ndim}")

#             # Bước 1: Đảm bảo có được một khung hình ảnh xám 2D (H, W) duy nhất
#             single_gray_frame_2d = None
#             if image_array_original.ndim == 2: # Đã là ảnh xám 2D
#                 single_gray_frame_2d = image_array_original
#             elif image_array_original.ndim == 3:
#                 if image_array_original.shape[-1] == 3: # Ảnh màu (H, W, 3)
#                     if config.verbose_mode: print(f"    [DEBUG DICOM Frame] Color (H,W,3) detected. Converting to grayscale.")
#                     # Chuẩn hóa về [0,1] trước khi cvtColor nếu cần (vì cvtColor hoạt động tốt nhất với float [0,1] hoặc uint8)
#                     temp_for_cvt = image_array_original.astype(np.float32)
#                     if np.max(temp_for_cvt) > 1.0: # Giả sử >1 là chưa chuẩn hóa về [0,1]
#                         min_val_rgb, max_val_rgb = np.min(temp_for_cvt), np.max(temp_for_cvt)
#                         if max_val_rgb - min_val_rgb > 1e-8:
#                             temp_for_cvt = (temp_for_cvt - min_val_rgb) / (max_val_rgb - min_val_rgb)
#                         else:
#                             temp_for_cvt = np.zeros_like(temp_for_cvt)
#                     single_gray_frame_2d = cv2.cvtColor(temp_for_cvt, cv2.COLOR_RGB2GRAY)
#                 elif image_array_original.shape[-1] == 1: # Ảnh xám (H, W, 1)
#                     if config.verbose_mode: print(f"    [DEBUG DICOM Frame] Grayscale (H,W,1) detected. Squeezing.")
#                     single_gray_frame_2d = image_array_original.squeeze(axis=-1)
#                 # Kiểm tra xem có phải là (num_frames, H, W) không dựa trên NumberOfFrames
#                 elif hasattr(dicom_data, 'NumberOfFrames') and dicom_data.NumberOfFrames > 1 and image_array_original.shape[0] == dicom_data.NumberOfFrames:
#                     if config.verbose_mode: print(f"    [DEBUG DICOM Frame] Multi-frame grayscale (Frames,H,W) detected. Taking frame 0. Frames: {dicom_data.NumberOfFrames}")
#                     single_gray_frame_2d = image_array_original[0] # Lấy khung hình đầu tiên
#                 else: # Các trường hợp 3D không rõ ràng khác
#                     if config.verbose_mode: print(f"    [WARNING DICOM Frame] Ambiguous 3D shape {image_array_original.shape}. Assuming first slice is a 2D frame.")
#                     single_gray_frame_2d = image_array_original[0] # Cẩn thận với giả định này
#             elif image_array_original.ndim == 4: # Ảnh đa khung, đa kênh (Frames, H, W, C)
#                 if hasattr(dicom_data, 'NumberOfFrames') and dicom_data.NumberOfFrames > 1 and image_array_original.shape[0] == dicom_data.NumberOfFrames:
#                     if config.verbose_mode: print(f"    [DEBUG DICOM Frame] Multi-frame 4D (Frames,H,W,C) detected. Taking frame 0. Shape: {image_array_original.shape}")
#                     first_frame = image_array_original[0] # first_frame là (H,W,C)
#                     if first_frame.shape[-1] == 3: # Nếu frame đầu là màu
#                         if config.verbose_mode: print(f"      [DEBUG DICOM Frame] Frame 0 is color. Converting to grayscale.")
#                         temp_for_cvt_4d = first_frame.astype(np.float32)
#                         if np.max(temp_for_cvt_4d) > 1.0:
#                             min_val_rgb4d, max_val_rgb4d = np.min(temp_for_cvt_4d), np.max(temp_for_cvt_4d)
#                             if max_val_rgb4d - min_val_rgb4d > 1e-8:
#                                 temp_for_cvt_4d = (temp_for_cvt_4d - min_val_rgb4d) / (max_val_rgb4d - min_val_rgb4d)
#                             else: temp_for_cvt_4d = np.zeros_like(temp_for_cvt_4d)
#                         single_gray_frame_2d = cv2.cvtColor(temp_for_cvt_4d, cv2.COLOR_RGB2GRAY)
#                     elif first_frame.shape[-1] == 1: # Nếu frame đầu là xám (H,W,1)
#                         if config.verbose_mode: print(f"      [DEBUG DICOM Frame] Frame 0 is grayscale (H,W,1). Squeezing.")
#                         single_gray_frame_2d = first_frame.squeeze(axis=-1)
#                     else:
#                         print(f"    [ERROR DICOM Frame] Unhandled channel count in 4D frame: {first_frame.shape}. Skipping {dicom_path}.")
#                         continue
#                 else:
#                     print(f"    [ERROR DICOM Frame] Ambiguous 4D shape or NumberOfFrames mismatch: {image_array_original.shape}. Skipping {dicom_path}.")
#                     continue
#             else:
#                 print(f"  [ERROR DICOM Frame] Unhandled DICOM ndim: {image_array_original.ndim} for path {dicom_path}. Skipping.")
#                 continue

#             if single_gray_frame_2d is None:
#                 print(f"  [ERROR DICOM Frame] Could not extract single 2D gray frame from {dicom_path}. Skipping.")
#                 continue
            
#             if single_gray_frame_2d.ndim != 2:
#                 print(f"  [CRITICAL ERROR DICOM Frame] single_gray_frame_2d is NOT 2D! Shape: {single_gray_frame_2d.shape} for {dicom_path}. Skipping.")
#                 continue

#             # Bước 2: Chuẩn hóa ảnh xám 2D về [0,1]
#             min_val, max_val = np.min(single_gray_frame_2d), np.max(single_gray_frame_2d)
#             if max_val - min_val > 1e-8:
#                 image_normalized_2d = (single_gray_frame_2d - min_val) / (max_val - min_val)
#             else:
#                 image_normalized_2d = np.zeros_like(single_gray_frame_2d)
#             image_array_original = np.clip(image_normalized_2d, 0.0, 1.0).astype(np.float32)
                    
#         #     # Chuẩn hóa ảnh gốc về [0,1]
#         #     # min_val, max_val = np.min(image_array_original), np.max(image_array_original)
#         #     # if max_val - min_val > 1e-8:
#         #     #     image_array_original = (image_array_original - min_val) / (max_val - min_val)
#         #     # else:
#         #     #     image_array_original = np.zeros_like(image_array_original)
#         #     # image_array_original = np.clip(image_array_original, 0.0, 1.0)
#         #     # # Kiểm tra nếu là ảnh đa khung và chọn khung hình đầu tiên
#         #     # if image_array_original.ndim == 3:
#         #     #     # Giả định rằng nếu chiều đầu tiên nhỏ hơn các chiều khác, đó là số khung hình
#         #     #     # Đây là một heuristic, bạn có thể cần điều chỉnh tùy theo đặc điểm cụ thể của dữ liệu
#         #     #     if image_array_original.shape[0] < image_array_original.shape[1] and \
#         #     #        image_array_original.shape[0] < image_array_original.shape[2]:
#         #     #         if config.verbose_mode: # Sử dụng biến verbose_mode từ config của bạn
#         #     #             print(f"  [INFO] DICOM {base_dicom_name_no_ext if 'base_dicom_name_no_ext' in locals() else file_id_csv} has shape {image_array_original.shape}. Assuming multi-frame, taking the first frame.")
#         #     #         image_array_original = image_array_original[0] 
#         #     #     # Nếu không, và chiều cuối cùng là 1 hoặc 3 (đã là ảnh HWC rồi), thì không cần làm gì
#         #     #     elif image_array_original.shape[-1] == 1 or image_array_original.shape[-1] == 3:
#         #     #         pass # Đã ở dạng (H, W, C) mong muốn
#         #     #     else: # Trường hợp 3 chiều không xác định rõ
#         #     #         if config.verbose_mode:
#         #     #             print(f"  [WARNING] DICOM {base_dicom_name_no_ext if 'base_dicom_name_no_ext' in locals() else file_id_csv} has an ambiguous 3D shape {image_array_original.shape}. Attempting to use first slice if it's small, otherwise this might lead to errors.")
#         #     #         # Heuristic bổ sung: nếu chiều cuối cùng không phải là kênh màu điển hình, và chiều đầu nhỏ, vẫn lấy frame đầu
#         #     #         if image_array_original.shape[-1] not in [1,3] and image_array_original.shape[0] < 10: # ví dụ < 10 frames
#         #     #              image_array_original = image_array_original[0]
#         #     processed_2d_image = None
#         #     if image_array_original.ndim == 3:
#         #         # Trường hợp 1: (num_frames, H, W) - Ảnh xám đa khung
#         #         if image_array_original.shape[0] < image_array_original.shape[1] and \
#         #         image_array_original.shape[0] < image_array_original.shape[2] and \
#         #         image_array_original.shape[0] < 10: # Heuristic: số khung thường nhỏ
#         #             if config.verbose_mode:
#         #                 print(f"    [DEBUG DICOM Load] Multi-frame grayscale detected. Taking frame 0.")
#         #             processed_2d_image = image_array_original[0] # Lấy khung hình đầu tiên
#         #         # Trường hợp 2: (H, W, 1) - Ảnh xám đơn khung nhưng có chiều kênh
#         #         elif image_array_original.shape[-1] == 1:
#         #             if config.verbose_mode:
#         #                 print(f"    [DEBUG DICOM Load] Grayscale with channel dim detected. Squeezing.")
#         #             processed_2d_image = image_array_original.squeeze(axis=-1)
#         #         # Trường hợp 3: (H, W, 3) - Ảnh màu (ít gặp trong mammography nhưng cần xử lý)
#         #         elif image_array_original.shape[-1] == 3:
#         #             if config.verbose_mode:
#         #                 print(f"    [DEBUG DICOM Load] Color image (H,W,3) detected. Converting to grayscale.")
#         #             # Chuyển sang float 0-1 trước khi convert nếu chưa
#         #             if np.max(image_array_original) > 1.0: # Giả sử > 1 là chưa chuẩn hóa
#         #                 temp_norm = (image_array_original - np.min(image_array_original)) / (np.max(image_array_original) - np.min(image_array_original) + 1e-8)
#         #                 processed_2d_image = cv2.cvtColor(temp_norm.astype(np.float32), cv2.COLOR_RGB2GRAY)
#         #             else:
#         #                 processed_2d_image = cv2.cvtColor(image_array_original.astype(np.float32), cv2.COLOR_RGB2GRAY)
#         #         else: # Các trường hợp 3D không xác định rõ
#         #             if config.verbose_mode:
#         #                 print(f"    [WARNING DICOM Load] Ambiguous 3D shape {image_array_original.shape}. Attempting to take first slice/frame.")
#         #             processed_2d_image = image_array_original[0] # Thử lấy slice đầu tiên

#         #     elif image_array_original.ndim == 2: # Ảnh xám 2D (H,W)
#         #         if config.verbose_mode:
#         #             print(f"    [DEBUG DICOM Load] Grayscale 2D image detected.")
#         #         processed_2d_image = image_array_original

#         #     elif image_array_original.ndim == 4: # Ví dụ (num_frames, H, W, C)
#         #         if config.verbose_mode:
#         #             print(f"    [DEBUG DICOM Load] 4D DICOM detected ({image_array_original.shape}). Taking frame 0.")
#         #         first_frame = image_array_original[0]
#         #         if first_frame.shape[-1] == 3: # Nếu frame đầu là màu
#         #             if config.verbose_mode:
#         #                 print(f"      [DEBUG DICOM Load] Frame 0 is color. Converting to grayscale.")
#         #             if np.max(first_frame) > 1.0:
#         #                 temp_norm_frame = (first_frame - np.min(first_frame)) / (np.max(first_frame) - np.min(first_frame) + 1e-8)
#         #                 processed_2d_image = cv2.cvtColor(temp_norm_frame.astype(np.float32), cv2.COLOR_RGB2GRAY)
#         #             else:
#         #                 processed_2d_image = cv2.cvtColor(first_frame.astype(np.float32), cv2.COLOR_RGB2GRAY)
#         #         elif first_frame.shape[-1] == 1: # Nếu frame đầu là grayscale (H,W,1)
#         #             if config.verbose_mode:
#         #                 print(f"      [DEBUG DICOM Load] Frame 0 is grayscale (H,W,1). Squeezing.")
#         #             processed_2d_image = first_frame.squeeze(axis=-1)
#         #         else: # Không xác định
#         #             print(f"    [ERROR DICOM Load] Unhandled 4D frame shape: {first_frame.shape}. Skipping DICOM.")
#         #             continue
#         #     else:
#         #         print(f"  [ERROR DICOM Load] Unhandled DICOM shape: {image_array_original.shape}. Skipping DICOM.")
#         #         continue

#         #     if processed_2d_image is None: # Nếu không xử lý được
#         #         print(f"  [ERROR DICOM Load] Could not derive a 2D image from DICOM {dicom_path}. Skipping.")
#         #         continue

#         #     # Bây giờ image_array_original nên là 2D (ảnh xám) hoặc (H,W,3) nếu DICOM gốc là màu
#         #     # Chuẩn hóa ảnh gốc về [0,1]
#         #     min_val, max_val = np.min(image_array_original), np.max(image_array_original)
#         #     if max_val - min_val > 1e-8:
#         #         image_array_original = (image_array_original - min_val) / (max_val - min_val)
#         #     else:
#         #         image_array_original = np.zeros_like(image_array_original)
#         #     image_array_original = np.clip(image_array_original, 0.0, 1.0)
            
#             birad_value_csv = str(row['Bi-Rads']).strip()
#             current_label_text = None
#             for label_text, birad_code_list in config.INBREAST_BIRADS_MAPPING.items():
#                 standardized_birad_codes = [val.replace("BI-RADS", "").strip() for val in birad_code_list]
#                 if birad_value_csv in standardized_birad_codes:
#                     current_label_text = label_text
#                     break
            
#             if current_label_text is None or current_label_text == "Normal":
#                 continue

#             images_for_this_entry = [] # Các patch/ảnh từ một file DICOM
#             labels_for_this_entry_text = [] # Nhãn text tương ứng

#             if use_roi_patches:
#                 roi_file_pattern_1 = os.path.join(roi_dir, base_dicom_name_no_ext + "*.roi")
#                 roi_file_pattern_2 = os.path.join(roi_dir, file_id_csv + "*.roi")
#                 matching_roi_files = tf.io.gfile.glob(roi_file_pattern_1)
#                 if not matching_roi_files: matching_roi_files = tf.io.gfile.glob(roi_file_pattern_2)
#                 if not matching_roi_files: continue

#                 for roi_path_single in matching_roi_files:
#                     coords, roi_label_text_from_func = load_roi_and_label(roi_path_single, birad_map)
#                     if coords is None or not coords or roi_label_text_from_func != current_label_text:
#                         continue

#                     xs = [p[0] for p in coords]; ys = [p[1] for p in coords]
#                     x_min, x_max = int(min(xs)), int(max(xs))
#                     y_min, y_max = int(min(ys)), int(max(ys))
#                     h_img, w_img = image_array_original.shape[:2]
#                     x_min, y_min = max(0, x_min), max(0, y_min)
#                     x_max, y_max = min(w_img - 1, x_max), min(h_img - 1, y_max)

#                     if x_min >= x_max or y_min >= y_max: continue
#                     roi_patch = image_array_original[y_min:y_max+1, x_min:x_max+1]
#                     if roi_patch.size == 0: continue
                    
#                     resized_patch = cv2.resize(roi_patch, target_size, interpolation=cv2.INTER_AREA)
                    
#                     # # --- Xử lý kênh cho ROI patch ---
#                     # if config.model != "CNN":
#                     #     if resized_patch.ndim == 2: resized_patch = cv2.cvtColor(resized_patch, cv2.COLOR_GRAY2RGB)
#                     #     elif resized_patch.ndim == 3 and resized_patch.shape[-1] == 1: resized_patch = cv2.cvtColor(resized_patch, cv2.COLOR_GRAY2RGB)
#                     # elif config.model == "CNN":
#                     #     if resized_patch.ndim == 3 and resized_patch.shape[-1] == 3:
#                     #         resized_patch = cv2.cvtColor(resized_patch, cv2.COLOR_RGB2GRAY)
#                     #         resized_patch = np.expand_dims(resized_patch, axis=-1)
#                     #     elif resized_patch.ndim == 2:
#                     #          resized_patch = np.expand_dims(resized_patch, axis=-1)
#                     if resized_patch.ndim == 3 and resized_patch.shape[-1] == 1:
#                         resized_patch = resized_patch.squeeze(axis=-1) # Chuyển (H,W,1) thành (H,W)
                    
#                     if config.model != "CNN": # Cần 3 kênh
#                         if resized_patch.ndim == 2: # Ảnh xám (H,W)
#                             resized_patch = cv2.cvtColor(resized_patch, cv2.COLOR_GRAY2RGB) # Thành (H,W,3)
#                         # Nếu đã là (H,W,3) thì không làm gì
#                     elif config.model == "CNN": # Cần 1 kênh
#                         if resized_patch.ndim == 3 and resized_patch.shape[-1] == 3: # Ảnh màu (H,W,3)
#                             resized_patch = cv2.cvtColor(resized_patch, cv2.COLOR_RGB2GRAY) # Thành (H,W)
#                         if resized_patch.ndim == 2: # Đảm bảo có channel dim
#                             resized_patch = np.expand_dims(resized_patch, axis=-1) # Thành (H,W,1)
                                        
#                     resized_patch = resized_patch.astype(np.float32)
#                     min_rp, max_rp = np.min(resized_patch), np.max(resized_patch)
#                     if max_rp - min_rp > 1e-8: resized_patch = (resized_patch - min_rp) / (max_rp - min_rp)
#                     else: resized_patch = np.zeros_like(resized_patch)
#                     resized_patch = np.clip(resized_patch, 0.0, 1.0)

#                     images_for_this_entry.append(resized_patch)
#                     labels_for_this_entry_text.append(current_label_text)
            
#             else: # Full image
#                 # resized_full_image = cv2.resize(image_array_original, target_size, interpolation=cv2.INTER_AREA)
#                 # if config.model != "CNN":
#                 #     if resized_full_image.ndim == 2: resized_full_image = cv2.cvtColor(resized_full_image, cv2.COLOR_GRAY2RGB)
#                 #     elif resized_full_image.ndim == 3 and resized_full_image.shape[-1] == 1: resized_full_image = cv2.cvtColor(resized_full_image, cv2.COLOR_GRAY2RGB)
#                 # elif config.model == "CNN":
#                 #     if resized_full_image.ndim == 3 and resized_full_image.shape[-1] == 3:
#                 #         resized_full_image = cv2.cvtColor(resized_full_image, cv2.COLOR_RGB2GRAY)
#                 #         resized_full_image = np.expand_dims(resized_full_image, axis=-1)
#                 #     elif resized_full_image.ndim == 2: resized_full_image = np.expand_dims(resized_full_image, axis=-1)
#                 resized_full_image = cv2.resize(image_array_original, target_size, interpolation=cv2.INTER_AREA)
                
#                 # --- Xử lý kênh cho full image ---
#                 # Đảm bảo resized_full_image là 2D (ảnh xám) trước khi cvtColor nếu nó đang là (H,W,1)
#                 if resized_full_image.ndim == 3 and resized_full_image.shape[-1] == 1:
#                     resized_full_image = resized_full_image.squeeze(axis=-1)
#                 output_image_for_model = None
#                 if config.model != "CNN": # Cần 3 kênh cho MobileNet
#                     if resized_full_image.ndim == 2: # Ảnh xám (H,W)
#                         resized_full_image = cv2.cvtColor(resized_full_image, cv2.COLOR_GRAY2RGB) # Thành (H,W,3)
#                     else: # Lỗi logic nếu không phải 2D
#                         print(f"    [ERROR DICOM Channelize] Expected 2D image for GRAY2RGB, got {resized_full_image.shape}. Skipping {dicom_path}")
#                         continue
#                     # Nếu đã là (H,W,3) thì không làm gì
#                 elif config.model == "CNN": # Cần 1 kênh
#                     if resized_full_image.ndim == 3 and resized_full_image.shape[-1] == 3: # Ảnh màu (H,W,3)
#                         resized_full_image = cv2.cvtColor(resized_full_image, cv2.COLOR_RGB2GRAY) # Thành (H,W)
#                     if resized_full_image.ndim == 2: # Đảm bảo có channel dim
#                         resized_full_image = np.expand_dims(resized_full_image, axis=-1) # Thành (H,W,1)
                                
#                 resized_full_image = resized_full_image.astype(np.float32)
#                 min_rfi, max_rfi = np.min(resized_full_image), np.max(resized_full_image)
#                 if max_rfi - min_rfi > 1e-8 : resized_full_image = (resized_full_image - min_rfi) / (max_rfi - min_rfi)
#                 else: resized_full_image = np.zeros_like(resized_full_image)
#                 resized_full_image = np.clip(resized_full_image, 0.0, 1.0)
#     # --- XỬ LÝ KÊNH CUỐI CÙNG TRƯỚC KHI THÊM VÀO LIST ---
#     # final_processed_image hiện tại là ảnh xám 2D (H, W) đã được resize và chuẩn hóa 0-1

#                 # output_image_for_model = None
#                 # if config.model != "CNN": # Cần 3 kênh (ví dụ: MobileNet)
#                 #     # if final_processed_image.ndim == 2: # Chắc chắn là 2D ở đây
#                 #     output_image_for_model = cv2.cvtColor(resized_full_image, cv2.COLOR_GRAY2RGB) # Thành (H,W,3)
#                 #     # else: ERROR, không nên xảy ra nếu logic trên đúng
#                 # else: # config.model == "CNN", cần 1 kênh
#                 #     # if final_processed_image.ndim == 2:
#                 #     output_image_for_model = np.expand_dims(resized_full_image, axis=-1) # Thành (H,W,1)
#                 #     # else: ERROR

#                 # if output_image_for_model is None:
#                 #     print(f"    [ERROR DICOM Process] Could not finalize image channels for {dicom_path}. Skipping.")
#                 #     continue

#                 # Chuẩn hóa lại lần cuối sau tất cả các bước (đặc biệt nếu cvtColor thay đổi dải giá trị)
#                 output_image_for_model = output_image_for_model.astype(np.float32)
#                 min_f, max_f = np.min(output_image_for_model), np.max(output_image_for_model)
#                 if max_f - min_f > 1e-8:
#                     output_image_for_model = (output_image_for_model - min_f) / (max_f - min_f)
#                 else:
#                     output_image_for_model = np.zeros_like(output_image_for_model)
#                 output_image_for_model = np.clip(output_image_for_model, 0.0, 1.0)

#                 if config.verbose_mode:
#                     print(f"    [DEBUG DICOM Process] Final shape for model {config.model}: {output_image_for_model.shape} for {dicom_path}")

#                 images_for_this_entry.append(output_image_for_model)
#                 # labels_for_this_entry_text.append(current_label_text) # Thêm label tương ứng

#                 images_for_this_entry.append(resized_full_image)
#                 labels_for_this_entry_text.append(current_label_text)

#             if images_for_this_entry:
#                 images_np_current_entry = np.array(images_for_this_entry, dtype=np.float32)
#                 labels_numeric_current_entry = label_encoder.transform(labels_for_this_entry_text) # Dùng LE chính
#                 labels_one_hot_for_aug = tf.keras.utils.to_categorical(labels_numeric_current_entry, num_classes=len(label_encoder.classes_)).astype(np.float32)

#                 aug_images_np, aug_labels_np = generate_image_transforms(
#                     images_np_current_entry, labels_one_hot_for_aug,
#                     apply_elastic=enable_elastic, elastic_alpha=elastic_alpha_val, elastic_sigma=elastic_sigma_val,
#                     apply_mixup=enable_mixup, mixup_alpha=mixup_alpha_val,
#                     apply_cutmix=enable_cutmix, cutmix_alpha=cutmix_alpha_val
#                 )
#                 all_images_data_accumulator.extend(list(aug_images_np))
#                 all_labels_accumulator.extend(list(aug_labels_np))
#                 processed_dicom_count += 1

#         except InvalidDicomError: continue
#         except Exception as e:
#             print(f"  [ERROR] Failed to process DICOM entry {file_id_csv} (path: {dicom_path}): {e}")
#             # import traceback; traceback.print_exc()
#             continue
#     if all_images_data_accumulator:
#         first_shape = all_images_data_accumulator[0].shape
#         for i, img_arr in enumerate(all_images_data_accumulator):
#             if img_arr.shape != first_shape:
#                 print(f"[CRITICAL ERROR] Inconsistent shape found in all_images_data_accumulator at index {i}. Shape: {img_arr.shape}, Expected: {first_shape}")
#                 # Có thể bạn muốn dừng hoặc xử lý lỗi ở đây thay vì để np.array báo lỗi
#     # else:
#     #    print("[WARN] all_images_data_accumulator is empty before final np.array conversion.")
    
#     print(f"[INFO] Total unique DICOM files processed and augmented: {processed_dicom_count}")
#     # if not all_images_data_accumulator:
#     #     return np.array([]), np.array([])

#     # final_images_array = np.array(all_images_data_accumulator, dtype=np.float32)
#     # final_labels_array = np.array(all_labels_accumulator, dtype=np.float32)
#     # return final_images_array, final_labels_array
#     if all_images_data_accumulator:
#         first_shape = all_images_data_accumulator[0].shape
#         for i, img_arr_check in enumerate(all_images_data_accumulator): # Đổi tên biến lặp
#             if img_arr_check.shape != first_shape:
#                 print(f"[CRITICAL ERROR] Inconsistent shape found in all_images_data_accumulator at index {i}. Shape: {img_arr_check.shape}, Expected: {first_shape}. DICOMs processed: {processed_dicom_count}")
#                 # return np.array([]), np.array([]) # Có thể trả về rỗng ngay lập tức
#     else:
#         print("[WARN] all_images_data_accumulator is empty before final np.array conversion.")
    
#     print(f"[INFO] Total unique DICOM entries successfully processed and added to accumulator: {processed_dicom_count}")
#     if not all_images_data_accumulator or not all_labels_accumulator : # Thêm kiểm tra labels
#         print("[ERROR load_inbreast] Final accumulator is empty. Returning empty arrays.")
#         return np.array([]), np.array([])

#     final_images_array = np.array(all_images_data_accumulator, dtype=np.float32)
#     final_labels_array = np.array(all_labels_accumulator, dtype=np.float32) # y_np từ hàm load

#     if config.verbose_mode:
#         print(f"[INFO load_inbreast] Returning final arrays: X_shape={final_images_array.shape}, y_shape={final_labels_array.shape}")
#     return final_images_array, final_labels_array
def load_inbreast_data_no_pectoral_removal(
    data_dir: str,
    label_encoder: LabelEncoder,
    use_roi_patches: bool,
    target_size: tuple,
    enable_elastic: bool = False, elastic_alpha_val: float = 34.0, elastic_sigma_val: float = 4.0,
    enable_mixup: bool = False, mixup_alpha_val: float = 0.2,
    enable_cutmix: bool = False, cutmix_alpha_val: float = 1.0
):
    print(f"\n[INFO] Loading INbreast data (No Pectoral Removal) {'with ROI patches' if use_roi_patches else 'as full images'}...")
    print(f"  Target Size: {target_size}")
    print(f"  Augmentation - Elastic: {enable_elastic} (alpha:{elastic_alpha_val}, sigma:{elastic_sigma_val})")
    print(f"  Augmentation - MixUp: {enable_mixup} (alpha:{mixup_alpha_val})")
    print(f"  Augmentation - CutMix: {enable_cutmix} (alpha:{cutmix_alpha_val})")

    dicom_dir = os.path.join(data_dir, "AllDICOMs")
    roi_dir = os.path.join(data_dir, "AllROI")
    csv_path = os.path.join(data_dir, "INbreast.csv")

    if not os.path.exists(csv_path): raise FileNotFoundError(f"INbreast.csv not found at {csv_path}")
    if not os.path.isdir(dicom_dir): raise NotADirectoryError(f"DICOM directory not found: {dicom_dir}")
    if use_roi_patches and (not os.path.isdir(roi_dir) or not os.listdir(roi_dir)): # Kiểm tra ROI dir có file không
        print(f"[WARNING load_inbreast] ROI directory '{roi_dir}' not found or empty. Falling back to full image mode.")
        use_roi_patches = False # Chuyển sang full image nếu ROI không hợp lệ

    df = pd.read_csv(csv_path, sep=';')
    df.columns = [c.strip() for c in df.columns]
    # Tạo birad_map: key là PatientID (phần số trước dấu _ nếu có trong tên file DICOM)
    birad_map = {}
    for _, row in df.iterrows():
        file_name_csv = str(row['File Name']).strip()
        # Giả sử file_name_csv có thể là '22678622' hoặc '22678622_xxxx_MG_L_CC_ANON'
        # Chúng ta muốn lấy phần số ID bệnh nhân làm key chính
        patient_id_from_csv = file_name_csv.split('_')[0]
        birad_map[patient_id_from_csv] = str(row['Bi-Rads']).strip()

    all_images_data_accumulator = []
    all_labels_accumulator = [] # Sẽ chứa nhãn one-hot sau khi encode và augment
    processed_dicom_count = 0
    
    # Bước A: Thu thập tất cả các nhãn text hợp lệ để fit LabelEncoder một lần
    all_valid_label_texts_for_fitting_le = []
    # (Vòng lặp này chỉ để xác định các lớp, không đọc ảnh)
    for _, row in df.iterrows():
        file_id_csv = str(row['File Name']).strip()
        patient_id_key = file_id_csv.split('_')[0]
        birad_value_csv = birad_map.get(patient_id_key)

        if birad_value_csv is None: continue

        current_label_text = None
        for label_text_map, birad_code_list_map in config.INBREAST_BIRADS_MAPPING.items():
            standardized_birad_codes = [val.replace("BI-RADS", "").strip() for val in birad_code_list_map]
            if birad_value_csv in standardized_birad_codes:
                current_label_text = label_text_map
                break
        
        if current_label_text is not None and current_label_text != "Normal":
            all_valid_label_texts_for_fitting_le.append(current_label_text)

    if not all_valid_label_texts_for_fitting_le:
        print("[ERROR load_inbreast] No valid labels (Benign/Malignant) found in CSV to fit LabelEncoder. Returning empty arrays.")
        return np.array([]), np.array([])
    
    unique_valid_labels = sorted(list(set(all_valid_label_texts_for_fitting_le)))
    if not unique_valid_labels:
        print("[ERROR load_inbreast] No unique valid labels to fit LabelEncoder. Returning empty.")
        return np.array([]), np.array([])
    
    label_encoder.fit(unique_valid_labels) # Fit LabelEncoder với các lớp thực sự có
    if config.verbose_mode:
        print(f"[INFO load_inbreast] LabelEncoder fitted with classes: {list(label_encoder.classes_)}")
    
    num_model_classes = len(label_encoder.classes_)
    if num_model_classes < 2:
        print(f"[ERROR load_inbreast] After filtering 'Normal', only {num_model_classes} class(es) remain: {list(label_encoder.classes_)}. Need at least 2 for classification.")
        return np.array([]), np.array([])

    # Bước B: Vòng lặp xử lý từng file DICOM
    for index, row in df.iterrows():
        file_id_csv = str(row['File Name']).strip()
        patient_id_key = file_id_csv.split('_')[0] # Dùng patient_id để tra cứu Bi-Rads

        dicom_path = None
        # Ưu tiên tìm DICOM bằng patient_id_key nếu file_id_csv không phải là tên file đầy đủ
        # Điều này giả định tên file DICOM thường bắt đầu bằng patient_id_key
        # Ví dụ: 22678622.dcm hoặc 22678622_xxxxx.dcm
        found_dicom_files = tf.io.gfile.glob(os.path.join(dicom_dir, patient_id_key + "*.dcm"))
        if found_dicom_files:
            dicom_path = found_dicom_files[0] # Lấy file đầu tiên khớp
        else: # Thử tìm bằng file_id_csv đầy đủ nếu nó có thể là tên file
            direct_dicom_path = os.path.join(dicom_dir, file_id_csv + ".dcm")
            if os.path.exists(direct_dicom_path):
                dicom_path = direct_dicom_path
        
        if not dicom_path:
            if config.verbose_mode: print(f"  [DEBUG DICOM Find] No DICOM found for CSV File Name/Patient ID: {file_id_csv}/{patient_id_key}")
            continue

        birad_value_csv = birad_map.get(patient_id_key)
        if birad_value_csv is None: continue

        current_label_text = None
        for label_text_map, birad_code_list_map in config.INBREAST_BIRADS_MAPPING.items():
            standardized_birad_codes = [val.replace("BI-RADS", "").strip() for val in birad_code_list_map]
            if birad_value_csv in standardized_birad_codes:
                current_label_text = label_text_map
                break
        
        if current_label_text is None or current_label_text == "Normal":
            continue
        
        try:
            dicom_data = pydicom.dcmread(dicom_path)
            image_array_original_unnormalized = dicom_data.pixel_array.astype(np.float32)
            
            if config.verbose_mode:
                print(f"  [DEBUG DICOM Initial] Path: {dicom_path}, Initial Shape: {image_array_original_unnormalized.shape}, Ndim: {image_array_original_unnormalized.ndim}")

            # Bước 1: Đảm bảo có được một khung hình ảnh xám 2D (H, W) duy nhất
            single_gray_frame_2d = None
            if image_array_original_unnormalized.ndim == 2:
                single_gray_frame_2d = image_array_original_unnormalized
            elif image_array_original_unnormalized.ndim == 3:
                if image_array_original_unnormalized.shape[-1] == 3: # (H, W, 3) Color
                    temp_for_cvt = image_array_original_unnormalized # Không cần chuẩn hóa ở đây, cvtColor sẽ xử lý
                    single_gray_frame_2d = cv2.cvtColor(temp_for_cvt, cv2.COLOR_RGB2GRAY)
                elif image_array_original_unnormalized.shape[-1] == 1: # (H, W, 1) Grayscale with channel
                    single_gray_frame_2d = image_array_original_unnormalized.squeeze(axis=-1)
                elif hasattr(dicom_data, 'NumberOfFrames') and dicom_data.NumberOfFrames > 1 and image_array_original_unnormalized.shape[0] == dicom_data.NumberOfFrames: # (Frames, H, W) Grayscale multi-frame
                    single_gray_frame_2d = image_array_original_unnormalized[0]
                else: # Ambiguous 3D, try first slice
                    single_gray_frame_2d = image_array_original_unnormalized[0]
            elif image_array_original_unnormalized.ndim == 4: # (Frames, H, W, C)
                if hasattr(dicom_data, 'NumberOfFrames') and dicom_data.NumberOfFrames > 1 and image_array_original_unnormalized.shape[0] == dicom_data.NumberOfFrames:
                    first_frame = image_array_original_unnormalized[0] # (H,W,C)
                    if first_frame.shape[-1] == 3: # Color frame
                        temp_for_cvt_4d = first_frame
                        single_gray_frame_2d = cv2.cvtColor(temp_for_cvt_4d, cv2.COLOR_RGB2GRAY)
                    elif first_frame.shape[-1] == 1: # Grayscale frame (H,W,1)
                        single_gray_frame_2d = first_frame.squeeze(axis=-1)
                    else: print(f"    [ERROR DICOM Frame] Unhandled 4D frame channel for {dicom_path}. Skipping."); continue
                else: print(f"    [ERROR DICOM Frame] Ambiguous 4D shape for {dicom_path}. Skipping."); continue
            else: print(f"  [ERROR DICOM Frame] Unhandled DICOM ndim {image_array_original_unnormalized.ndim} for {dicom_path}. Skipping."); continue

            if single_gray_frame_2d is None or single_gray_frame_2d.ndim != 2:
                print(f"  [CRITICAL ERROR DICOM Frame] Could not get valid 2D frame from {dicom_path}. Shape was {single_gray_frame_2d.shape if single_gray_frame_2d is not None else 'None'}. Skipping.")
                continue
            
            # Bước 2: Chuẩn hóa ảnh xám 2D về [0,1]
            min_val, max_val = np.min(single_gray_frame_2d), np.max(single_gray_frame_2d)
            current_image_processed_2d = np.zeros_like(single_gray_frame_2d, dtype=np.float32)
            if max_val - min_val > 1e-8:
                current_image_processed_2d = (single_gray_frame_2d - min_val) / (max_val - min_val)
            current_image_processed_2d = np.clip(current_image_processed_2d, 0.0, 1.0)

            images_for_this_entry_processed = [] # Ảnh sau khi resize và xử lý kênh
            
            # Xử lý ROI hoặc Full Image, kết quả là image_to_finalize_channels (ảnh xám 2D đã resize)
            image_to_finalize_channels = None
            if use_roi_patches:
                # (Logic crop ROI từ current_image_processed_2d -> roi_patch (2D))
                # Đây là nơi bạn cần tích hợp hàm load_roi_and_label để lấy coords
                # Dựa vào base_dicom_name_no_ext hoặc file_id_csv để tìm ROI file
                base_name_for_roi = os.path.splitext(os.path.basename(dicom_path))[0] # e.g. "22614074_..."
                roi_file_pattern = os.path.join(roi_dir, base_name_for_roi + "*.roi") 
                # Hoặc thử với patient_id_key nếu tên file ROI chỉ có ID
                if not tf.io.gfile.glob(roi_file_pattern):
                    roi_file_pattern = os.path.join(roi_dir, patient_id_key + "*.roi")

                matching_roi_files = tf.io.gfile.glob(roi_file_pattern)
                if not matching_roi_files:
                    if config.verbose_mode: print(f"    [DEBUG ROI] No ROI file found for {dicom_path} with pattern {roi_file_pattern}. Skipping this DICOM for ROI processing.");
                    continue # Bỏ qua DICOM này nếu không có ROI file tương ứng

                for roi_path_single in matching_roi_files:
                    coords, roi_label_text_from_func = load_roi_and_label(roi_path_single, birad_map) # birad_map đã được tạo ở trên
                    if coords is None or not coords or roi_label_text_from_func != current_label_text:
                        if config.verbose_mode and coords is not None: print(f"      [DEBUG ROI] ROI label mismatch or no coords for {roi_path_single}. Expected {current_label_text}, got {roi_label_text_from_func}.")
                        continue
                    
                    xs_roi = [p[0] for p in coords]; ys_roi = [p[1] for p in coords_roi]
                    x_min_r, x_max_r = int(min(xs_roi)), int(max(xs_roi))
                    y_min_r, y_max_r = int(min(ys_roi)), int(max(ys_roi))
                    h_img_r, w_img_r = current_image_processed_2d.shape[:2] # Dùng ảnh 2D đã xử lý
                    x_min_r, y_min_r = max(0, x_min_r), max(0, y_min_r)
                    x_max_r, y_max_r = min(w_img_r - 1, x_max_r), min(h_img_r - 1, y_max_r)

                    if x_min_r < x_max_r and y_min_r < y_max_r:
                        roi_patch_from_2d = current_image_processed_2d[y_min_r:y_max_r+1, x_min_r:x_max_r+1]
                        if roi_patch_from_2d.size > 0:
                            image_to_finalize_channels = cv2.resize(roi_patch_from_2d, target_size, interpolation=cv2.INTER_AREA)
                        else:
                            if config.verbose_mode: print(f"      [WARN ROI] Empty ROI patch for {roi_path_single}. Skipping.");
                            continue
                    else:
                        if config.verbose_mode: print(f"      [WARN ROI] Invalid ROI coordinates for {roi_path_single}. Skipping.");
                        continue
                    
                    # Sau khi có image_to_finalize_channels (ảnh ROI đã resize), xử lý kênh và thêm vào list
                    if image_to_finalize_channels is not None and image_to_finalize_channels.ndim == 2:
                        # (Logic xử lý kênh và thêm vào images_for_this_entry_processed tương tự như nhánh full image)
                        # ...
                        # images_for_this_entry_processed.append(final_model_input_image)
                        # labels_for_this_entry_text_local.append(current_label_text)
                        pass # Sẽ xử lý kênh chung ở dưới
                    else: continue # Bỏ qua ROI này nếu có lỗi
            else: # Full image
                image_to_finalize_channels = cv2.resize(current_image_processed_2d, target_size, interpolation=cv2.INTER_AREA)

            if image_to_finalize_channels is None or image_to_finalize_channels.ndim != 2:
                print(f"    [ERROR DICOM Resize/ROI] Image after resize/ROI is not 2D. Shape: {image_to_finalize_channels.shape if image_to_finalize_channels is not None else 'None'}. Skipping {dicom_path}")
                continue
            
            # Bước 3: Xử lý kênh màu cuối cùng
            final_model_input_image = None
            if config.model != "CNN": # Cần 3 kênh
                final_model_input_image = cv2.cvtColor(image_to_finalize_channels, cv2.COLOR_GRAY2RGB)
            else: # CNN cần 1 kênh
                final_model_input_image = np.expand_dims(image_to_finalize_channels, axis=-1)
            
            # Bước 4: Chuẩn hóa lại giá trị pixel về [0,1] sau các phép biến đổi kênh/resize
            final_model_input_image = final_model_input_image.astype(np.float32)
            min_f, max_f = np.min(final_model_input_image), np.max(final_model_input_image)
            if max_f - min_f > 1e-8:
                final_model_input_image = (final_model_input_image - min_f) / (max_f - min_f)
            else:
                final_model_input_image = np.zeros_like(final_model_input_image)
            final_model_input_image = np.clip(final_model_input_image, 0.0, 1.0)

            if config.verbose_mode:
                print(f"    [DEBUG DICOM Finalize] Final input shape for model {config.model}: {final_model_input_image.shape} from {dicom_path}")
            
            images_for_this_entry_processed.append(final_model_input_image)
            # labels_for_this_entry_text_local.append(current_label_text) # Sẽ thêm sau augment

            # Bước 5: Augmentation (chỉ áp dụng nếu có ảnh và các cờ được bật)
            if images_for_this_entry_processed:
                current_images_np = np.array(images_for_this_entry_processed, dtype=np.float32)
                # Tạo nhãn one-hot cho các ảnh này
                try:
                    current_labels_numeric = label_encoder.transform([current_label_text] * len(current_images_np))
                except ValueError as e_le_transform_loop:
                     print(f"    [ERROR LabelTransformLoop] Failed to transform label '{current_label_text}' for {dicom_path}. LE classes: {list(label_encoder.classes_)}. Error: {e_le_transform_loop}. Skipping this entry.")
                     continue

                current_labels_one_hot = tf.keras.utils.to_categorical(current_labels_numeric, num_classes=num_model_classes).astype(np.float32)

                should_apply_detailed_aug = enable_elastic or enable_mixup or enable_cutmix
                if should_apply_detailed_aug:
                    if config.verbose_mode: print(f"    [DEBUG Augment] Applying detailed augmentations for {dicom_path} (Num images in batch: {current_images_np.shape[0]})")
                    
                    # generate_image_transforms cần được điều chỉnh để nó không cố gắng cân bằng lớp bên trong nữa
                    # mà chỉ áp dụng các phép biến đổi đã chọn lên batch ảnh đầu vào.
                    # Hoặc, bạn có thể áp dụng từng phép biến đổi ở đây nếu generate_image_transforms quá phức tạp.
                    
                    aug_images_batch, aug_labels_batch = generate_image_transforms(
                        current_images_np, current_labels_one_hot, # Truyền nhãn one-hot
                        apply_elastic=enable_elastic, elastic_alpha=elastic_alpha_val, elastic_sigma=elastic_sigma_val,
                        apply_mixup=enable_mixup, mixup_alpha=mixup_alpha_val,
                        apply_cutmix=enable_cutmix, cutmix_alpha=cutmix_alpha_val
                        # Bỏ qua các tham số liên quan đến cân bằng lớp nếu generate_image_transforms đã được sửa
                    )
                    all_images_data_accumulator.extend(list(aug_images_batch))
                    all_labels_accumulator.extend(list(aug_labels_batch))
                else:
                    all_images_data_accumulator.extend(list(current_images_np))
                    all_labels_accumulator.extend(list(current_labels_one_hot))
                
                processed_dicom_count += len(images_for_this_entry_processed) # Đếm số ảnh gốc (trước aug chi tiết) được xử lý

        except InvalidDicomError:
            if config.verbose_mode: print(f"  [WARNING] Invalid DICOM file skipped: {dicom_path}")
            continue
        except Exception as e_outer:
            print(f"  [ERROR General Loop] Failed to process DICOM entry {file_id_csv} (path: {dicom_path}): {type(e_outer).__name__} - {e_outer}")
            # import traceback; traceback.print_exc()
            continue
            
    if config.verbose_mode and all_images_data_accumulator:
        first_shape = all_images_data_accumulator[0].shape
        for i, img_arr_check in enumerate(all_images_data_accumulator):
            if img_arr_check.shape != first_shape:
                print(f"[CRITICAL ERROR Inhomogeneity] Inconsistent shape found in accumulator at index {i}. Shape: {img_arr_check.shape}, Expected: {first_shape}.")
                # Nên dừng ở đây hoặc có cơ chế xử lý/loại bỏ các ảnh không đồng nhất
                # return np.array([]), np.array([]) 
    elif not all_images_data_accumulator:
       print("[WARN load_inbreast] all_images_data_accumulator is empty before final np.array conversion.")
    
    print(f"[INFO] Total DICOM-derived image instances (after potential ROI splitting, before detailed aug if any): {processed_dicom_count}")
    print(f"[INFO] Total images in accumulator (after detailed aug if any): {len(all_images_data_accumulator)}")

    if not all_images_data_accumulator or not all_labels_accumulator :
        print("[ERROR load_inbreast] Final accumulator is empty or labels are missing. Returning empty arrays.")
        return np.array([]), np.array([])

    final_images_array = np.array(all_images_data_accumulator, dtype=np.float32)
    final_labels_array = np.array(all_labels_accumulator, dtype=np.float32)

    if config.verbose_mode:
        print(f"[INFO load_inbreast] Returning final arrays: X_shape={final_images_array.shape}, y_shape={final_labels_array.shape}")
    return final_images_array, final_labels_array
def import_minimias_dataset(data_dir: str, label_encoder) -> (np.ndarray, np.ndarray):
    """
    Import the dataset by pre-processing the images and encoding the labels.
    Originally written as a group for the common pipeline. Later amended by Adam Jaamour.
    :param data_dir: Directory to the mini-MIAS images.
    :param label_encoder: The label encoder.
    :return: Two NumPy arrays, one for the processed images and one for the encoded labels.
    """
    # Initialise variables.
    images = list()
    labels = list()

    if not config.is_roi:
        # Loop over the image paths and update the data and labels lists with the pre-processed images & labels.
        print("Loading whole images")
        for image_path in list(paths.list_images(data_dir)):
            images.append(preprocess_image(image_path))
            labels.append(image_path.split(os.path.sep)[-2])  # Extract label from path.
    else:
        # Use the CSV file to get the images and their labels, and crop the images around the specified ROI.
        print("Loading cropped ROI images")
        images, labels = crop_roi_image(data_dir)

    # Convert the data and labels lists to NumPy arrays.
    images = np.array(images, dtype="float32")  # Convert images to a batch.
    labels = np.array(labels)

    # Encode labels.
    labels = encode_labels(labels, label_encoder)
    return images, labels

def import_cmmd_dataset(data_dir: str, label_encoder, target_size=None) -> (np.ndarray, np.ndarray):
    """
    Import CMMD dataset (binary classification) by loading images and encoding labels.
    :param data_dir: Thư mục chứa ảnh CMMD đã xử lý, gồm hai folder con cho từng lớp.
    :param label_encoder: Bộ encoder để mã hóa label.
    :param target_size: Tuple (H, W) mong muốn; nếu None sẽ chọn tự động theo mô hình.
    :return: images (NumPy array) và labels (NumPy array) đã mã hóa.
    """
    images = []
    labels = []
    # Xác định kích thước ảnh dựa trên mô hình nếu chưa chỉ định
    if target_size is None:
        if config.model == "CNN" or config.is_roi:
            target_size = (config.ROI_IMG_SIZE['HEIGHT'], config.ROI_IMG_SIZE['WIDTH'])
        elif config.model == "VGG" or config.model == "Inception":
            target_size = (config.MINI_MIAS_IMG_SIZE['HEIGHT'], config.MINI_MIAS_IMG_SIZE['WIDTH'])
        elif config.model == "VGG-common":
            target_size = (config.VGG_IMG_SIZE['HEIGHT'], config.VGG_IMG_SIZE['WIDTH'])
        elif config.model == "ResNet":
            target_size = (config.RESNET_IMG_SIZE['HEIGHT'], config.RESNET_IMG_SIZE['WIDTH'])
        elif config.model == "MobileNet":
            target_size = (config.MOBILE_NET_IMG_SIZE['HEIGHT'], config.MOBILE_NET_IMG_SIZE['WIDTH'])
        elif config.model == "DenseNet" or config.model == "Inception":
            target_size = (config.INCEPTION_IMG_SIZE['HEIGHT'], config.INCEPTION_IMG_SIZE['WIDTH'])
        else:
            target_size = (224, 224)  # mặc định
    # Duyệt qua từng thư mục lớp (benign, malignant)
    for label_folder in sorted(os.listdir(data_dir)):
        label_path = os.path.join(data_dir, label_folder)
        if not os.path.isdir(label_path):
            continue  # bỏ qua nếu không phải thư mục
        for image_file in sorted(os.listdir(label_path)):
            image_path = os.path.join(label_path, image_file)
            if not os.path.isfile(image_path) or not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            # try:
            #     # Load ảnh; đảm bảo ảnh grayscale 1 kênh
            #     image = load_img(image_path, color_mode="grayscale", target_size=target_size)
            #     image_array = img_to_array(image) / 255.0  # chuyển thành mảng và chuẩn hóa [0,1]
            #     images.append(image_array)
            #     labels.append(label_folder)  # label chính là tên thư mục
            try:
                image = load_img(image_path, color_mode="grayscale", target_size=target_size)
                image_array = img_to_array(image) # Shape (H, W, 1)

                # ===== THAY ĐỔI CHO CMMD KHI DÙNG MODEL 3 KÊNH =====
                if config.model != "CNN": # Nếu model không phải CNN (ví dụ MobileNet, VGG, ResNet)
                    # Chuyển ảnh xám (H,W,1) sang (H,W,3) bằng cách lặp kênh
                    image_array_rgb = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB) # image_array là float 0-255
                    image_array = image_array_rgb 
                # ====================================================

                image_array = image_array / 255.0 # Chuẩn hóa sau khi đã có đúng số kênh
                images.append(image_array)
                labels.append(label_folder)
            except Exception as e:
                print(f"Error processing file {image_path}: {e}")
    if len(images) == 0 or len(labels) == 0:
        raise ValueError("No images or labels found in CMMD dataset directory!")
    # Chuyển sang numpy array
    # images = np.array(images, dtype="float32")
    # labels = np.array(labels)
    # # Mã hóa label (benign/malignant) thành số và one-hot nếu cần
    # labels = encode_labels(labels, label_encoder)
    # return images, labels
    images = np.array(images, dtype="float32")
    labels = np.array(labels)
    # Mã hóa label (benign/malignant) thành số và one-hot nếu cần
    labels = encode_labels(labels, label_encoder)
    # --- Augmentation nếu được bật ---
    if config.augment_data:
        images, labels = generate_image_transforms(images, labels)
    return images, labels

def import_inbreast_full_dataset(
    data_dir: str,
    label_encoder,
    target_size=None,
    csv_path="/kaggle/input/breastdata/INbreast/INbreast/INbreast.csv"   # <-- đường dẫn tới file CSV của bạn
):
    """
    Load toàn bộ DICOM trong AllDICOMs, đọc BI-RADS từ CSV, trả về (X: np.ndarray, y: np.ndarray).
    CSV phải có cột 'File Name' khớp với prefix của .dcm (ví dụ '22678622'), và cột 'Bi-Rads'.
    """
    # 1) Đọc CSV và build map: pid_base -> birad_val
    df = pd.read_csv(csv_path, sep=';')
    # đảm bảo cột đúng tên, strip space
    df.columns = [c.strip() for c in df.columns]
    birad_map = dict(zip(
        df['File Name'].astype(str).str.strip(),
        df['Bi-Rads'].astype(str).str.strip()
    ))
    # 2) Build list samples (dcm_path, label_name)
    samples = []
    dicom_dir = os.path.join(data_dir, "AllDICOMs")
    for fn in sorted(os.listdir(dicom_dir)):
        if not fn.lower().endswith(".dcm"):
            continue
        # pid_base lấy trước dấu '_' hoặc trước '.dcm'
        pid = fn.split("_", 1)[0]
        birad = birad_map.get(pid)
        if birad is None:
            continue
        # map birad (string) sang class name
        label_name = None
        for cls, vals in config.INBREAST_BIRADS_MAPPING.items():
            # vals ví dụ ["BI-RADS 2","BI-RADS 3"], ta strip về số
            normalized = [v.replace("BI-RADS", "").strip() for v in vals]
            if str(birad) in normalized:
                label_name = cls
                break
        if label_name:
            samples.append((os.path.join(dicom_dir, fn), label_name))

    if not samples:
        raise ValueError(f"No valid samples found in {dicom_dir} using CSV {csv_path}")

    # 3) Load image + build arrays
    X_list, y_list = [], []
    for dcm_path, label_name in samples:
        try:
            ds = pydicom.dcmread(dcm_path, force=True)
        except InvalidDicomError:
            continue
        arr = ds.pixel_array.astype(np.float32)
        arr -= arr.min()
        if arr.max()>0: arr /= arr.max()
        H, W = target_size or (
            config.INBREAST_IMG_SIZE["HEIGHT"],
            config.INBREAST_IMG_SIZE["WIDTH"]
        )
        if (arr.shape[0], arr.shape[1]) != (H, W):
            arr = cv2.resize(arr, (W, H))
        X_list.append(arr[..., np.newaxis])
        y_list.append(label_name)

    if not X_list:
        raise ValueError("All DICOMs were invalid or skipped.")

    X = np.stack(X_list, axis=0).astype(np.float32)

    # 4) Fit & transform labels
    y_enc = label_encoder.fit_transform(y_list)
    if label_encoder.classes_.size > 2:
        y_enc = to_categorical(y_enc)

    return X, y_enc


def load_roi_and_label(
    roi_path: str,
    birad_map: Dict[str,str]
) -> Tuple[Optional[List[Tuple[int,int]]], Optional[str]]:
    """
    Đọc file .roi, trả về:
      - coords: List[(x,y)] vùng ROI
      - label_name: 'Benign' / 'Malignant' / 'Normal'
    Nếu không tìm được coords hoặc nhãn, trả về (None, None).
    """
    # 1) load text, bỏ qua lỗi decode
    raw = open(roi_path, 'rb').read().decode('utf-8', errors='ignore')

    # 2) regex tìm tất cả {x, y}
    pts = re.findall(r'\{\s*([\d\.]+)\s*,\s*([\d\.]+)\s*\}', raw)
    if not pts:
        return None, None

    # 3) chuyển float → int, bỏ cặp mặc định {75,19}
    # coords: List[Tuple[int,int]] = []
    coords = []
    for xs, ys in pts:
        x, y = float(xs), float(ys)
        if abs(x-75.0)<1e-6 and abs(y-19.0)<1e-6:
            continue
        coords.append((int(x),int(y)))
    if not coords:
        return None, None

    # 4) tìm PID từ tên file .roi
    fn       = os.path.basename(roi_path)
    pid_base = os.path.splitext(fn)[0].split('_',1)[0]

    # 5) lấy giá trị BI-RADS gốc từ birad_map
    birad_val = birad_map.get(pid_base)
    if birad_val is None or not birad_val.strip():
        return None, None
    birad_val = birad_val.strip()

    # 6) tìm nhãn cuối cùng qua config.INBREAST_BIRADS_MAPPING
    label_name: Optional[str] = None
    for cls, raw_vals in config.INBREAST_BIRADS_MAPPING.items():
        # chuẩn hoá: xóa "BI-RADS" và khoảng trắng
        normalized = [v.replace("BI-RADS","").strip() for v in raw_vals]
        if birad_val in normalized:
            label_name = cls
            break

    # 7) nếu không map được hoặc là Normal, bỏ luôn
    if label_name is None or label_name == "Normal":
        return None, None

    return coords, label_name

def flatten_to_slices(ds):
    # ds: mỗi phần tử là (volume, label) với volume.shape = (32, H, W, 1)
    return ds.flat_map(lambda vol, lab:
        tf.data.Dataset.from_tensor_slices(
            (vol, tf.repeat(lab, tf.shape(vol)[0]))
        )
    )

def import_inbreast_roi_dataset(
    data_dir: str,
    label_encoder,
    target_size: Tuple[int,int]=None,
    csv_path: str="/kaggle/input/breastdata/INbreast/INbreast/INbreast.csv"
):
    """
    Load & crop ROI on-the-fly:
     - Đọc CSV BI-RADS để build birad_map: pid → Bi-Rads
     - Duyệt AllROI/*.roi, parse coords + label qua load_roi_and_label
     - Crop DICOM, resize, normalize
     - Encode nhãn thành int / one-hot
     - Trả về tf.data.Dataset
    """
    # --- 0) đọc CSV, build PID→Bi-Rads ---
    df = pd.read_csv(csv_path, sep=';')
    df.columns = [c.strip() for c in df.columns]
    birad_map: Dict[str,str] = {}
    for fn, val in zip(df['File Name'], df['Bi-Rads']):
        pid = str(fn).strip().split('_',1)[0]
        birad_map[pid] = str(val).strip()

    # --- 1) scan thư mục ROI ---
    samples: List[Tuple[str, List[Tuple[int,int]], str]] = []
    dicom_dir = os.path.join(data_dir, "AllDICOMs")
    roi_dir   = os.path.join(data_dir, "AllROI")

    print("[DEBUG] roi_dir =", roi_dir, "contains:", os.listdir(roi_dir)[:5], "...")
    for roi_fn in sorted(os.listdir(roi_dir)):
        if not roi_fn.lower().endswith(".roi"):
            continue
        roi_path = os.path.join(roi_dir, roi_fn)

        coords, label_name = load_roi_and_label(roi_path, birad_map)
        # print("   → ROI", roi_fn, "gives", len(coords or []), "points and label=", label_name)
        if coords is None:
            continue

        # pid = roi_fn.split('.',1)[0]
        # dcm_fp = os.path.join(dicom_dir, f"{pid}.dcm")
        # if not os.path.exists(dcm_fp):
        #     continue

        # samples.append((dcm_fp, coords, label_name))
        pid = os.path.splitext(roi_fn)[0]
        # 3) tìm DICOM file bất kỳ bắt đầu bằng pid + '_'
        matches = [f for f in os.listdir(dicom_dir)
                if f.startswith(pid + "_") and f.lower().endswith(".dcm")]
        if not matches:
            continue
        dcm_fp = os.path.join(dicom_dir, matches[0])

        samples.append((dcm_fp, coords, label_name))
    if not samples:
        raise ValueError(f"No ROI samples found in {roi_dir} (sau khi lọc coords & labels)")

    labels_str = [lbl for _,_,lbl in samples]
    label_encoder.fit(labels_str)
    labels_int = label_encoder.transform(labels_str)             # array shape=(N,)
    class_weights = make_class_weights_in(labels_int)
    classes     = list(label_encoder.classes_)                   # ['Benign','Malignant']
    num_classes = len(classes)

    # map text→int ngay
    label_to_idx = {cls:i for i,cls in enumerate(classes)}

    def _gen():
        for dcm_fp, coords, lbl_txt in samples:
            try:
                ds = pydicom.dcmread(dcm_fp, force=True)
            except:
                continue
            arr = ds.pixel_array.astype(np.float32)
            arr = (arr - arr.min())/(arr.max() - arr.min() + 1e-8)
            xs, ys = zip(*coords)
            x0, x1 = max(0, min(xs)), min(arr.shape[1], max(xs))
            y0, y1 = max(0, min(ys)), min(arr.shape[0], max(ys))
            roi = arr[y0:y1, x0:x1]
            H, W = target_size or (
                config.INBREAST_IMG_SIZE["HEIGHT"],
                config.INBREAST_IMG_SIZE["WIDTH"]
            )
            roi = cv2.resize(roi, (W, H), interpolation=cv2.INTER_AREA)
            yield roi[..., None], np.int32(label_to_idx[lbl_txt])

    # 2) Định nghĩa signature
    H, W = target_size or (
        config.INBREAST_IMG_SIZE["HEIGHT"],
        config.INBREAST_IMG_SIZE["WIDTH"]
    )
    sig = (
        tf.TensorSpec((H, W, 1), tf.float32),
        tf.TensorSpec((),        tf.int32)
    )
    ds = tf.data.Dataset.from_generator(_gen, output_signature=sig)

    # 3) Nếu multi-class (>2), one-hot một lần duy nhất
    if num_classes > 2:
        ds = ds.map(
            lambda x, y: (x, tf.one_hot(y, num_classes)),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    num_samples = len(samples)
    print(f"[DEBUG] ROI dataset ready: N={num_samples}, classes={num_classes}, class_weights={class_weights}")
    return ds, class_weights, num_classes, num_samples

def calculate_class_weights(y_train, label_encoder):
    """
    Compute balanced class weights for imbalanced data.
    """
    if label_encoder.classes_.size > 2:
        flat = label_encoder.inverse_transform(np.argmax(y_train, axis=1))
    else:
        flat = y_train
    weights = class_weight.compute_class_weight('balanced',
                                                np.unique(flat),
                                                flat)
    return {i: w for i, w in enumerate(weights)}

def import_cbisddsm_training_dataset(label_encoder):
    """
    Import the dataset getting the image paths (downloaded on BigTMP) and encoding the labels.
    Originally written as a group for the common pipeline. Later amended by Adam Jaamour.
    :param label_encoder: The label encoder.
    :return: Two arrays, one for the image paths and one for the encoded labels.
    """
    print("Importing CBIS-DDSM training set")
    cbis_ddsm_path = str()
    if config.mammogram_type == "calc":
        cbis_ddsm_path = "../data/CBIS-DDSM/calc-training.csv"
    elif config.mammogram_type == "mass":
        cbis_ddsm_path = "../data/CBIS-DDSM/mass-training.csv"
    else:
        cbis_ddsm_path = "../data/CBIS-DDSM/training.csv"
    df = pd.read_csv(cbis_ddsm_path)
    list_IDs = df['img_path'].values
    labels = encode_labels(df['label'].values, label_encoder)
    return list_IDs, labels


def import_cbisddsm_testing_dataset(label_encoder):
    """
    Import the testing dataset getting the image paths (downloaded on BigTMP) and encoding the labels.
    :param label_encoder: The label encoder.
    :return: Two arrays, one for the image paths and one for the encoded labels.
    """
    print("Importing CBIS-DDSM testing set")
    cbis_ddsm_path = str()
    if config.mammogram_type == "calc":
        cbis_ddsm_path = "../data/CBIS-DDSM/calc-test.csv"
    elif config.mammogram_type == "mass":
        cbis_ddsm_path = "../data/CBIS-DDSM/mass-test.csv"
    else:
        cbis_ddsm_path = "../data/CBIS-DDSM/testing.csv"
    df = pd.read_csv(cbis_ddsm_path)
    list_IDs = df['img_path'].values
    labels = encode_labels(df['label'].values, label_encoder)
    return list_IDs, labels

def preprocess_image(image_path: str) -> np.ndarray:
    """
    Pre-processing steps:
        * Load the input image in grayscale mode (1 channel),
        * resize it to fit the CNN model input,
        * transform it to an array format,
        * normalise the pixel intensities.
    Originally written as a group for the common pipeline. Later amended by Adam Jaamour.
    :param image_path: The path to the image to preprocess.
    :return: The pre-processed image in NumPy array format.
    """
    # Resize if using full image.
    if not config.is_roi:
        if config.model == "VGG" or config.model == "Inception":
            target_size = (config.MINI_MIAS_IMG_SIZE['HEIGHT'], config.MINI_MIAS_IMG_SIZE["WIDTH"])
        elif config.model == "VGG-common":
            target_size = (config.VGG_IMG_SIZE['HEIGHT'], config.VGG_IMG_SIZE["WIDTH"])
        elif config.model == "MobileNet":
            target_size = (config.MOBILE_NET_IMG_SIZE['HEIGHT'], config.MOBILE_NET_IMG_SIZE["WIDTH"])
        elif config.model == "CNN":
            target_size = (config.ROI_IMG_SIZE['HEIGHT'], config.ROI_IMG_SIZE["WIDTH"])
        image = load_img(image_path, color_mode="grayscale", target_size=target_size)

    # Do not resize if using cropped ROI image.
    else:
        image = load_img(image_path, color_mode="grayscale")

    image = img_to_array(image)
    image /= 255.0
    return image


def encode_labels(labels_list: np.ndarray, label_encoder) -> np.ndarray:
    """
    Encode labels using one-hot encoding.
    Originally written as a group for the common pipeline. Later amended by Adam Jaamour.
    :param label_encoder: The label encoder.
    :param labels_list: The list of labels in NumPy array format.
    :return: The encoded list of labels in NumPy array format.
    """
    labels = label_encoder.fit_transform(labels_list)
    if label_encoder.classes_.size == 2:
        return labels
    else:
        return to_categorical(labels)


def dataset_stratified_split(split: float, dataset: np.ndarray, labels: np.ndarray) -> \
        (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Partition the data into training and testing splits. Stratify the split to keep the same class distribution in both
    sets and shuffle the order to avoid having imbalanced splits.
    Originally written as a group for the common pipeline. Later amended by Adam Jaamour.
    :param split: Dataset split (e.g. if 0.2 is passed, then the dataset is split in 80%/20%).
    :param dataset: The dataset of pre-processed images.
    :param labels: The list of labels.
    :return: the training and testing sets split in input (X) and label (Y).
    """
    train_X, test_X, train_Y, test_Y = train_test_split(dataset,
                                                        labels,
                                                        test_size=split,
                                                        stratify=labels,
                                                        random_state=config.RANDOM_SEED,
                                                        shuffle=True)
    return train_X, test_X, train_Y, test_Y


def calculate_class_weights(y_train, label_encoder):
    """
    Tính toán trọng số lớp cho datasets không cân bằng.
    """
    if label_encoder.classes_.size != 2:
        y_train = label_encoder.inverse_transform(np.argmax(y_train, axis=1))

    # Tính trọng số cân bằng
    weights = class_weight.compute_class_weight("balanced",
                                            np.unique(y_train),
                                            y_train)
    class_weights = dict(enumerate(weights))
    
    # Thay đổi từ return None thành return class_weights
    return class_weights

def crop_roi_image(data_dir):
    """
    Crops the images from the mini-MIAS dataset.
    Function originally written by Shuen-Jen and amended by Adam Jaamour.
    """
    images = list()
    labels = list()

    csv_dir = data_dir
    images_dir = data_dir.split("_")[0] + "_png"

    df = pd.read_csv('/'.join(csv_dir.split('/')[:-1]) + '/data_description.csv', header=None)

    for row in df.iterrows():
        # Skip normal cases.
        if str(row[1][4]) == 'nan':
            continue
        if str(row[1][4]) == '*NOT':
            continue

        # Process image.
        image = preprocess_image(images_dir + '/' + row[1][0] + '.png')

        # Abnormal case: crop around tumour.
        y2 = 0
        x2 = 0
        if row[1][2] != 'NORM':
            y1 = image.shape[1] - int(row[1][5]) - 112
            if y1 < 0:
                y1 = 0
                y2 = 224
            if y2 != 224:
                y2 = image.shape[1] - int(row[1][5]) + 112
                if y2 > image.shape[1]:
                    y2 = image.shape[1]
                    y1 = image.shape[1] - 224
            x1 = int(row[1][4]) - 112
            if x1 < 0:
                x1 = 0
                x2 = 224
            if x2 != 224:
                x2 = int(row[1][4]) + 112
                if x2 > image.shape[0]:
                    x2 = image.shape[0]
                    x1 = image.shape[0] - 224

        # Normal case: crop around centre of image.
        else:
            y1 = int(image.shape[1] / 2 - 112)
            y2 = int(image.shape[1] / 2 + 112)
            x1 = int(image.shape[0] / 2 - 112)
            x2 = int(image.shape[0] / 2 + 112)

        # Get label from CSV file.
        label = "normal"
        if str(row[1][3]) == 'B':
            label = "benign"
        elif str(row[1][3]) == 'M':
            label = "malignant"

        # Append image and label to lists.
        images.append(image[y1:y2, x1:x2, :])
        labels.append(label)

    return images, labels
