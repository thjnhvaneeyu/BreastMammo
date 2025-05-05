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
from data_operations.data_transformations import load_roi_and_label
import xml.etree.ElementTree as ET
from pydicom.errors import InvalidDicomError
from sklearn.utils.class_weight import compute_class_weight

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

    # X = np.array(images, dtype="float32")
    # y = label_encoder.fit_transform(labels)
    # if label_encoder.classes_.size > 2:
    #     y = to_categorical(y)

    # if config.augment_data:
    #     X, y = generate_image_transforms(X, y)

#     # return X, y
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
            try:
                # Load ảnh; đảm bảo ảnh grayscale 1 kênh
                image = load_img(image_path, color_mode="grayscale", target_size=target_size)
                image_array = img_to_array(image) / 255.0  # chuyển thành mảng và chuẩn hóa [0,1]
                images.append(image_array)
                labels.append(label_folder)  # label chính là tên thư mục
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

# # def import_cmmd_dataset(data_dir: str, metadata_csv: str, label_encoder, target_size=None) -> (np.ndarray, np.ndarray):
# #     """
# #     Import CMMD dataset and optionally augment to balance classes.
# #     """
# #     # --- phần load ảnh và encode labels giống như trước ---
# #     # (dùng bản gốc bạn đã có)
# #     df = pd.read_csv(metadata_csv)
# #     images, labels = [], []
# #     for _, row in df.iterrows():
# #         fn = row['filename']; lbl = row['label']
# #         path = os.path.join(data_dir, fn)
# #         if not os.path.isfile(path):
# #             continue
# #         img = Image.open(path).convert("L")
# #         arr = np.array(img, dtype="float32") / 255.0
# #         if arr.ndim == 2:
# #             arr = arr[..., np.newaxis]
# #         images.append(arr)
# #         labels.append(lbl)

# #     X = np.stack(images, axis=0)
# #     y = np.array(label_encoder.fit_transform(labels))
# #     # Với multi-class thì one-hot
# #     if label_encoder.classes_.size > 2:
# #         y = to_categorical(y)

# #     # --- Tăng cường dữ liệu ngay tại đây nếu bật config.augment_data ---
# #     if config.augment_data:
# #         X, y = generate_image_transforms(X, y)

# #     return X, y


# def import_inbreast_dataset(data_dir: str, label_encoder, target_size=None):
#     import os, numpy as np, cv2, pydicom
#     import xml.etree.ElementTree as ET
#     from tensorflow.keras.utils import to_categorical
#     import config

#     dicom_dir = os.path.join(data_dir, "AllDICOMs")
#     xml_dir   = os.path.join(data_dir, "AllXML")
#     roi_dir   = os.path.join(data_dir, "AllROI")

#     # 1) Determine resize
#     if target_size is None:
#         ts = config.INBREAST_IMG_SIZE
#         target_size = (ts["HEIGHT"], ts["WIDTH"])

#     # 2) Decide whether to use ROI
#     use_roi = False
#     if config.is_roi and os.path.isdir(roi_dir) and any(
#         f.lower().endswith((".png",".jpg",".jpeg",".roi"))
#         for f in os.listdir(roi_dir)
#     ):
#         use_roi = True
#     elif config.is_roi:
#         print(f"[WARN] ROI folder empty or missing: {roi_dir}, using full DICOM")

#     X_list, y_list = [], []

#     def parse_birads(xml_path, pid):
#         tree = ET.parse(xml_path)
#         root = tree.getroot()

#         # 1) Try <bi_rads>
#         bi_el = root.find(".//bi_rads")
#         birad = bi_el.text.strip() if bi_el is not None and bi_el.text else None

#         # 2) Scan for any “BI-RADS …”
#         if birad is None:
#             for e in root.iter():
#                 txt = e.text.strip() if e.text else ""
#                 if txt.startswith("BI-RADS"):
#                     birad = txt
#                     break

#         if birad is None:
#             print(f"[WARN] No BI-RADS in XML '{pid}.xml', skipping")
#             return None

#         # 3) Map via config.BI_RADS_MAPPING
#         for cls, vals in config.BI_RADS_MAPPING.items():
#             if birad in vals:
#                 return 0 if cls == "Benign" else 1

#         print(f"[WARN] BI-RADS '{birad}' not in mapping, skipping")
#         print(ET.tostring(root, encoding='unicode'))
#         return None

#     # -- ROI branch --
#     if use_roi:
#         for fn in sorted(os.listdir(roi_dir)):
#             name, ext = os.path.splitext(fn)
#             pid        = name.split("_",1)[0]
#             xml_path   = os.path.join(xml_dir, pid + ".xml")
#             dicom_path = os.path.join(dicom_dir, pid + ".dcm")
#             if not (os.path.exists(xml_path) and os.path.exists(dicom_path)):
#                 continue

#             # load ROI or crop from DICOM
#             if ext in (".png",".jpg",".jpeg"):
#                 img = cv2.imread(os.path.join(roi_dir, fn), cv2.IMREAD_GRAYSCALE)
#             elif ext == ".roi":
#                 coords = []
#                 with open(os.path.join(roi_dir, fn)) as f:
#                     for line in f:
#                         parts = line.split()
#                         if len(parts)>=2:
#                             x,y = map(int, map(float, parts[:2]))
#                             coords.append((x,y))
#                 if not coords:
#                     continue
#                 ds = pydicom.dcmread(dicom_path)
#                 arr = ds.pixel_array.astype(np.float32)
#                 arr -= arr.min()
#                 if arr.max()>0: arr /= arr.max()
#                 xs, ys = zip(*coords)
#                 x0,x1 = max(0,min(xs)), min(arr.shape[1],max(xs))
#                 y0,y1 = max(0,min(ys)), min(arr.shape[0],max(ys))
#                 img = arr[y0:y1, x0:x1]
#             else:
#                 continue

#             img = cv2.resize(img, (target_size[1], target_size[0]))
#             img = img.astype(np.float32)/255.0
#             lbl = parse_birads(xml_path, pid)
#             if lbl is None:
#                 continue

#             X_list.append(img[...,None])
#             y_list.append(lbl)

#         if not X_list:
#             print("[WARN] ROI branch loaded 0 images, falling back to full DICOM")
#             use_roi = False

#     # -- Full DICOM branch --
#     if not use_roi:
#         for fn in sorted(os.listdir(dicom_dir)):
#             if not fn.lower().endswith(".dcm"):
#                 continue
#             pid      = os.path.splitext(fn)[0].split("_",1)[0]
#             xml_path = os.path.join(xml_dir, pid + ".xml")
#             if not os.path.exists(xml_path):
#                 continue

#             ds  = pydicom.dcmread(os.path.join(dicom_dir, fn))
#             img = ds.pixel_array.astype(np.float32)
#             img -= img.min()
#             if img.max()>0: img /= img.max()
#             img = cv2.resize(img, (target_size[1], target_size[0]))
#             lbl = parse_birads(xml_path, pid)
#             if lbl is None:
#                 continue

#             X_list.append(img[...,None])
#             y_list.append(lbl)

#     # 4) Final check
#     if not X_list:
#         raise ValueError("No INbreast images loaded in either ROI or DICOM branch.")

#     X = np.stack(X_list, axis=0)
#     y = np.array(y_list, dtype=np.int32)
#     y = label_encoder.fit_transform(y)
#     if label_encoder.classes_.size > 2:
#         y = to_categorical(y)
#     return X, y
# Trong src/data_operations/data_preprocessing.py
# Trong src/data_operations/data_preprocessing.py
# ở đầu file, thêm import:
# def import_inbreast_dataset(data_dir: str, label_encoder, target_size=None):
#     """
#     Load the INbreast dataset with support for three classes (Normal, Benign, Malignant).
#     - If config.is_roi is True and ROI files are available, crops lesions from DICOM images using .roi coordinates.
#     - Falls back to full DICOM images for cases with no ROI (e.g., normal cases or missing ROI data).
#     Returns:
#         X: NumPy array of shape (N, H, W, 1) containing image data (grayscale, normalized).
#         y: NumPy array of shape (N,) for binary labels or (N, C) for one-hot multi-class labels.
#     """
#     import os
#     import numpy as np
#     import pandas as pd
#     import pydicom
#     import cv2
#     import config
#     from tensorflow.keras.utils import to_categorical

#     # 1. Read the metadata CSV file
#     csv_path = os.path.join(data_dir, "INbreast.csv")
#     df = pd.read_csv(csv_path, sep=';')
    
#     # 2. Identify the relevant columns for file name and BI-RADS category
#     file_col  = next(c for c in df.columns if "file"    in c.lower())
#     birad_col = next(c for c in df.columns if "bi-rads" in c.lower() or "bi_rads" in c.lower())
#     df[file_col]  = df[file_col].astype(str).str.strip()
#     df[birad_col] = df[birad_col].astype(str).str.strip()
    
#     # Remove any "BI-RADS" prefix in the BI-RADS column for consistency (e.g., "BI-RADS 3" -> "3")
#     df[birad_col] = df[birad_col].str.replace("BI-RADS", "").str.strip()
    
#     # 3. Create a mapping from image ID (without extension) to BI-RADS value
#     df["pid"] = df[file_col].str.replace(r"(?i)\.dcm$", "", regex=True)
#     birad_map = dict(zip(df["pid"], df[birad_col]))
    
#     # 4. Determine target image size for resizing
#     if target_size is None:
#         ts = config.INBREAST_IMG_SIZE
#         target_size = (ts["HEIGHT"], ts["WIDTH"])
    
#     # Define directories for DICOM images and ROI files
#     dicom_dir = os.path.join(data_dir, "AllDICOMs")
#     roi_dir   = os.path.join(data_dir, "AllROI")
    
#     # 5. Decide whether to use ROI cropping (if ROI files exist and config.is_roi is True)
#     use_roi = config.is_roi and os.path.isdir(roi_dir) and any(f.lower().endswith(".roi") for f in os.listdir(roi_dir))
    
#     X_list = []
#     y_list = []
#     loaded_pids = set()  # keep track of images processed via ROI
    
#     # --- ROI branch: process images with ROI annotations ---
#     if use_roi:
#         print("Loading cropped ROI images...")
#         for fn in sorted(os.listdir(roi_dir)):
#             if not fn.lower().endswith(".roi"):
#                 continue
#             # Image ID (pid) from ROI filename (strip extension and any trailing index after '_')
#             pid_full = os.path.splitext(fn)[0]
#             pid_base = pid_full.split('_', 1)[0]    # e.g., "22678622_1" -> "22678622"
#             birad_val = birad_map.get(pid_base)
#             dicom_fp = os.path.join(dicom_dir, pid_base + ".dcm")
#             if birad_val is None or not os.path.exists(dicom_fp):
#                 continue  # skip ROI if corresponding image or BI-RADS not found
            
#             # Read ROI coordinates from the .roi file (assuming text with one X Y per line)
#             coords = []
#             with open(os.path.join(roi_dir, fn), 'r') as f_roi:
#                 for line in f_roi:
#                     parts = line.strip().split()
#                     if len(parts) >= 2:
#                         # Parse X and Y coordinates (float to int for pixel indices)
#                         x, y = map(float, parts[:2])
#                         coords.append((int(x), int(y)))
#             if not coords:
#                 # If no coordinates found, skip this ROI (could log a warning)
#                 continue
            
#             # Load the full DICOM image
#             ds = pydicom.dcmread(dicom_fp, force=True)
#             arr = ds.pixel_array.astype(np.float32)
#             # Normalize pixel intensities to [0,1]
#             arr -= arr.min()
#             if arr.max() > 0:
#                 arr /= arr.max()
            
#             # Compute bounding box of ROI coordinates
#             xs, ys = zip(*coords)
#             x_min, x_max = max(0, min(xs)), min(arr.shape[1], max(xs))
#             y_min, y_max = max(0, min(ys)), min(arr.shape[0], max(ys))
#             crop = arr[y_min:y_max, x_min:x_max]
            
#             # Resize crop to target size and normalize
#             roi_img = cv2.resize(crop, (target_size[1], target_size[0]))  # cv2 expects (width, height)
#             roi_img = roi_img.astype(np.float32)
#             # (arr was already 0-1, so roi_img is 0-1; no additional /255 scaling needed)
            
#             # Append image and label
#             X_list.append(roi_img[..., None])  # add channel dimension
#             # Determine class label from BI-RADS using config mapping
#             birad_str = str(birad_val).strip()  # e.g., "2", "4a", etc.
#             label_name = None
#             for cls, vals in config.INBREAST_BIRADS_MAPPING.items():
#                 if birad_str in [v.replace("BI-RADS ", "").strip() for v in vals]:
#                     label_name = cls
#                     break
#             if label_name is None:
#                 # If BI-RADS is not in mapping (e.g., '0'), skip this entry
#                 continue
#             y_list.append(label_name)
#             loaded_pids.add(pid_base)
#         # If no images were loaded via ROI (edge case), fallback to full images
#         if not X_list:
#             print("[WARN] No ROI crops loaded; falling back to full DICOM images.")
#             use_roi = False
    
#     # --- Full DICOM branch: process all images (or those not handled by ROI) ---
#     if not use_roi:
#         print("Loading full DICOM images...")
#     for fn in sorted(os.listdir(dicom_dir)):
#         if not fn.lower().endswith(".dcm"):
#             continue
#         pid_base = os.path.splitext(fn)[0].split('_', 1)[0]
#         # If we already added this image via ROI crop, skip to avoid duplicates
#         if use_roi and pid_base in loaded_pids:
#             continue
#         birad_val = birad_map.get(pid_base)
#         if birad_val is None:
#             continue  # skip if no BI-RADS info for this image
        
#         # Load DICOM image
#         ds = pydicom.dcmread(os.path.join(dicom_dir, fn))
#         arr = ds.pixel_array.astype(np.float32)
#         # Normalize to [0,1]
#         arr -= arr.min()
#         if arr.max() > 0:
#             arr /= arr.max()
#         # Resize image to target size
#         img_resized = cv2.resize(arr, (target_size[1], target_size[0]))
#         img_resized = img_resized.astype(np.float32)
#         # (arr normalized to 0-1, so img_resized is 0-1 range as well)
        
#         X_list.append(img_resized[..., None])
#         birad_str = str(birad_val).strip()
#         label_name = None
#         for cls, vals in config.INBREAST_BIRADS_MAPPING.items():
#             if birad_str in [v.replace("BI-RADS ", "").strip() for v in vals]:
#                 label_name = cls
#                 break
#         if label_name is None:
#             continue
#         y_list.append(label_name)
    
#     # 6. Finalize arrays and encode labels
#     if len(X_list) == 0:
#         raise ValueError("No INbreast images were loaded. Check data paths and ROI files.")
#     X = np.stack(X_list, axis=0).astype(np.float32)
#     y = np.array(y_list)  # labels as array of class names
#     # Fit label encoder and transform labels to numeric values
#     y_enc = label_encoder.fit_transform(y)
#     # If multi-class (more than 2 classes), convert to one-hot encoding
#     if label_encoder.classes_.size > 2:
#         y_enc = to_categorical(y_enc)
#     return X, y_enc
# def import_inbreast_roi_dataset(data_dir: str, label_encoder, target_size=None):
#     """
#     Chỉ load & crop ROI on-the-fly, trả về tf.data.Dataset (img, one-hot/label).
#     """
#     # 1) Build sample list
#     samples = []
#     dicom_dir = os.path.join(data_dir, "AllDICOMs")
#     roi_dir   = os.path.join(data_dir, "AllROI")
#     for roi_fn in sorted(os.listdir(roi_dir)):
#         if not roi_fn.lower().endswith(".roi"):
#             continue
#         pid = roi_fn.split("_roi")[0]
#         dicom_fp = os.path.join(dicom_dir, pid + ".dcm")
#         coords, label = load_roi_and_label(os.path.join(roi_dir, roi_fn))
#         if coords and label:
#             samples.append((dicom_fp, coords, label))

#     # 2) Fit LabelEncoder
#     labels = [lbl for _, _, lbl in samples]
#     label_encoder.fit(labels)
#     num_classes = label_encoder.classes_.size

#     # 3) Generator từng sample
#     def gen():
#         for dicom_fp, coords, label in samples:
#             try:
#                 arr = _load_and_crop(dicom_fp, coords, target_size)
#             except InvalidDicomError:
#                 continue
#             yield arr, label.encode()

#     # 4) Tạo Dataset & encode nhãn
#     H, W = target_size or (
#         config.INBREAST_IMG_SIZE["HEIGHT"], config.INBREAST_IMG_SIZE["WIDTH"]
#     )
#     output_signature = (
#         tf.TensorSpec(shape=(H, W, 1), dtype=tf.float32),
#         tf.TensorSpec(shape=(),       dtype=tf.string),
#     )
#     ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

#     def _encode(img, lbl):
#         i = tf.py_function(
#             lambda x: label_encoder.transform([x.decode()])[0],
#             [lbl], tf.int32
#         )
#         i.set_shape([])  # scalar
#         if num_classes > 2:
#             i = tf.one_hot(i, num_classes)
#         return img, i

#     ds = (
#         ds
#         .map(_encode, num_parallel_calls=tf.data.AUTOTUNE)
#         .shuffle(len(samples))
#         .batch(config.BATCH_SIZE)
#         .prefetch(tf.data.AUTOTUNE)
#     )
#     return ds


# def import_inbreast_full_dataset(data_dir: str, label_encoder, target_size=None):
#     """
#     Load toàn bộ DICOM + trích BI-RADS từ XML, trả về (X: np.ndarray, y: np.ndarray).
#     """
#     # 1) Build birad_map từ XML
#     xml_dir = os.path.join(data_dir, "AllXML")
#     birad_map = {}
#     for xml_fn in os.listdir(xml_dir):
#         if xml_fn.lower().endswith(".xml"):
#             pid = os.path.splitext(xml_fn)[0]
#             val = _parse_birads(os.path.join(xml_dir, xml_fn))
#             if val is not None:
#                 birad_map[pid] = val

#     # 2) Build sample list
#     samples = []
#     dicom_dir = os.path.join(data_dir, "AllDICOMs")
#     for fn in sorted(os.listdir(dicom_dir)):
#         if not fn.lower().endswith(".dcm"):
#             continue
#         pid = os.path.splitext(fn)[0].split("_", 1)[0]
#         birad = birad_map.get(pid)
#         if birad is None:
#             continue
#         # map birad -> class name
#         label = None
#         for cls, vals in config.INBREAST_BIRADS_MAPPING.items():
#             if str(birad) in [v.replace("BI-RADS", "").strip() for v in vals]:
#                 label = cls
#                 break
#         if label:
#             samples.append((os.path.join(dicom_dir, fn), label))
#     # ————————————————————————————————————————
#     # Guard: nếu không tìm thấy sample nào, báo lỗi rõ ràng
#     if not samples:
#         raise ValueError(
#             f"No valid InBreast DICOM samples found in {dicom_dir}. "
#             "Kiểm tra lại đường dẫn DATA_ROOT và thư mục AllXML/AllDICOMs."
#         )
#     # ————————————————————————————————————————
#     # 3) Load vào arrays
#     X_list, y_list = [], []
#     for dicom_fp, label in samples:
#         try:
#             ds = pydicom.dcmread(dicom_fp, force=True)
#         except InvalidDicomError:
#             continue
#         arr = ds.pixel_array.astype(np.float32)
#         arr -= arr.min()
#         if arr.max() > 0:
#             arr /= arr.max()
#         H, W = target_size or (
#             config.INBREAST_IMG_SIZE["HEIGHT"], config.INBREAST_IMG_SIZE["WIDTH"]
#         )
#         if (arr.shape[0], arr.shape[1]) != (H, W):
#             arr = cv2.resize(arr, (W, H))
#         X_list.append(arr[..., np.newaxis])
#         y_list.append(label)
#     # Guard thêm: sau khi lấy sample, vẫn rỗng => hỏng data
#     if not X_list:
#         raise ValueError(
#             "After filtering invalid DICOM, no images left to load. "
#             "Hãy kiểm tra file DICOM và XML annotation."
#         )

#     X = np.stack(X_list, axis=0)
#     # 4) Fit & transform labels
#     y_enc = label_encoder.fit_transform(y_list)
#     if label_encoder.classes_.size > 2:
#         from tensorflow.keras.utils import to_categorical
#         y_enc = to_categorical(y_enc)
#     return X.astype(np.float32), y_enc

# # --- Helpers ---

# def _load_and_crop(dicom_fp, coords, target_size):
#     """
#     Dùng trong ROI‐branch.
#     """
#     ds = pydicom.dcmread(dicom_fp, force=True)
#     arr = ds.pixel_array.astype(np.float32)
#     arr -= arr.min()
#     if arr.max()>0: arr /= arr.max()
#     xs, ys = zip(*coords)
#     x0, x1 = max(0,min(xs)), min(arr.shape[1],max(xs))
#     y0, y1 = max(0,min(ys)), min(arr.shape[0],max(ys))
#     roi = arr[y0:y1, x0:x1]
#     H, W = target_size or (arr.shape[0], arr.shape[1])
#     roi = cv2.resize(roi, (W, H))
#     return roi[..., np.newaxis]


# def _parse_birads(xml_fp):
#     """
#     Đọc BI-RADS từ XML và trả về giá trị (chuỗi đã strip "BI-RADS").
#     """
#     root = ET.parse(xml_fp).getroot()
#     bi = root.find(".//bi_rads")
#     vals = [bi.text.strip()] if bi is not None and bi.text else []
#     if not vals:
#         for e in root.iter():
#             if e.text and e.text.strip().startswith("BI-RADS"):
#                 vals = [e.text.strip()]; break
#     if not vals:
#         return None
#     return vals[0].replace("BI-RADS", "").strip()
from typing import List, Tuple, Optional, Dict

# def load_roi_and_label(
#     roi_path: str,
#     birad_map: Dict[str,str]
# ) -> Tuple[Optional[List[Tuple[int,int]]], Optional[str]]:
#     """
#     Đọc file .roi, trả về:
#       - coords: List[(x,y)] vùng ROI
#       - label_name: 'Benign' / 'Malignant' / 'Normal'
#     Nếu không tìm được coords hoặc nhãn, trả về (None, None).
#     """
#     # 1) load text, bỏ qua lỗi decode
#     raw = open(roi_path, 'rb').read().decode('utf-8', errors='ignore')

#     # 2) regex tìm tất cả {x, y}
#     pts = re.findall(r'\{\s*([\d\.]+)\s*,\s*([\d\.]+)\s*\}', raw)
#     if not pts:
#         return None, None

#     # 3) chuyển float → int, bỏ cặp mặc định {75,19}
#     coords: List[Tuple[int,int]] = []
#     for xs, ys in pts:
#         x, y = float(xs), float(ys)
#         if abs(x-75.0)<1e-6 and abs(y-19.0)<1e-6:
#             continue
#         coords.append((int(x),int(y)))
#     if not coords:
#         return None, None

#     # 4) tìm PID từ tên file .roi
#     fn       = os.path.basename(roi_path)
#     pid_base = os.path.splitext(fn)[0].split('_',1)[0]

#     # 5) lấy giá trị BI-RADS gốc từ birad_map
#     birad_val = birad_map.get(pid_base)
#     if birad_val is None or not birad_val.strip():
#         return None, None
#     birad_val = birad_val.strip()

#     # 6) tìm nhãn cuối cùng qua config.INBREAST_BIRADS_MAPPING
#     label_name: Optional[str] = None
#     for cls, raw_vals in config.INBREAST_BIRADS_MAPPING.items():
#         # chuẩn hoá: xóa "BI-RADS" và khoảng trắng
#         normalized = [v.replace("BI-RADS","").strip() for v in raw_vals]
#         if birad_val in normalized:
#             label_name = cls
#             break

#     # 7) nếu không map được hoặc là Normal, bỏ luôn
#     if label_name is None or label_name == "Normal":
#         return None, None

#     return coords, label_name
def load_roi_and_label(
    roi_path: str,
    birad_map: Dict[str,str]
) -> Tuple[Optional[List[Tuple[int,int]]], Optional[str]]:
    """
    Read a .roi file (lines of "x y"), return
       - coords: List[(x,y)]
       - label_name: one of 'Benign'/'Malignant' (we drop 'Normal')
    or (None,None) if we can’t extract a valid ROI+label.
    """
    coords: List[Tuple[int,int]] = []
    with open(roi_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                x, y = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            # drop the meaningless default mark that INbreast .roi files include
            if abs(x - 75.0) < 1e-6 and abs(y - 19.0) < 1e-6:
                continue
            coords.append((int(x), int(y)))
    if not coords:
        return None, None

    # map PID → BI-RADS from the CSV-derived dictionary
    pid = os.path.basename(roi_path).split('_',1)[0]
    birad_val = birad_map.get(pid)
    if not birad_val:
        return None, None

    # map BI-RADS number → 'Benign'/'Malignant' via your config
    label_name: Optional[str] = None
    for cls, raw_vals in config.INBREAST_BIRADS_MAPPING.items():
        normalized = [v.replace("BI-RADS","").strip() for v in raw_vals]
        if birad_val.strip() in normalized:
            label_name = cls
            break

    # skip unrecognized or "Normal"
    if label_name is None or label_name == "Normal":
        return None, None

    return coords, label_name

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

def import_inbreast_roi_dataset(
    data_dir: str,
    label_encoder,
    target_size=None,
    csv_path="/kaggle/input/breastdata/INbreast/INbreast/INbreast.csv"
):
    """
    Load & crop ROI on-the-fly từ INbreast:
     - Dùng load_roi_and_label() để parse .roi
     - Trả về tf.data.Dataset(img: H×W×1, lbl: int or one-hot)
     - Đã tự loại bỏ 'Normal' vì load_roi_and_label trả về None cho nó.
    """
    # 0) Đọc CSV BI-RADS để tạo birad_map
    df = pd.read_csv(csv_path, sep=';')
    df.columns = [c.strip() for c in df.columns]
    birad_map: Dict[str,str] = {
        str(fn).strip(): str(val).strip()
        for fn, val in zip(df['File Name'], df['Bi-Rads'])
    }

    # 1) Duyệt các file .roi trong AllROI
    # samples: List[Tuple[str, List[Tuple[int,int]], str]] = []
    samples = []
    dicom_dir = os.path.join(data_dir, "AllDICOMs")
    roi_dir   = os.path.join(data_dir, "AllROI")

    print(f"[DEBUG] roi_dir = {roi_dir}")
    print(f"[DEBUG] contents = {os.listdir(roi_dir)}")

    for roi_fn in sorted(os.listdir(roi_dir)):
        if not roi_fn.lower().endswith(".roi"):
            continue
        roi_path = os.path.join(roi_dir, roi_fn)

        # (a) parse coords + label từ .roi
        coords, label_name = load_roi_and_label(roi_path, birad_map)
        if coords is None or label_name is None:
            continue

        # (b) tìm file .dcm tương ứng
        pid = os.path.splitext(roi_fn)[0].split('_',1)[0]
        dcm_fp = os.path.join(dicom_dir, f"{pid}.dcm")
        if not os.path.exists(dcm_fp):
            continue

        samples.append((dcm_fp, coords, label_name))
        print(f"[DEBUG] thêm ROI sample #{len(samples)}: PID={pid}, label={label_name}")

    if not samples:
        raise ValueError(f"No ROI samples found in {roi_dir}")

    # 2) Fit LabelEncoder để có num_classes
    labels = [lbl for _,_,lbl in samples]
    label_encoder.fit(labels)
    num_classes = label_encoder.classes_.size

    # 3) Tạo generator đọc DICOM, crop, resize, normalize
    def gen():
        for dcm_fp, coords, label_name in samples:
            try:
                ds = pydicom.dcmread(dcm_fp, force=True)
            except InvalidDicomError:
                continue
            arr = ds.pixel_array.astype(np.float32)
            # normalize 0–1
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
            # crop bounding box từ coords
            xs, ys = zip(*coords)
            x0, x1 = max(0,min(xs)), min(arr.shape[1], max(xs))
            y0, y1 = max(0,min(ys)), min(arr.shape[0], max(ys))
            roi = arr[y0:y1, x0:x1]
            # resize về target_size
            H, W = target_size or (config.INBREAST_IMG_SIZE["HEIGHT"],
                                   config.INBREAST_IMG_SIZE["WIDTH"])
            roi = cv2.resize(roi, (W, H), interpolation=cv2.INTER_AREA)
            # thêm channel dim
            yield roi[..., np.newaxis], label_name.encode('utf-8')

    # 4) Xây Dataset và encode label → int (và one-hot nếu cần)
    H, W = target_size or (config.INBREAST_IMG_SIZE["HEIGHT"],
                           config.INBREAST_IMG_SIZE["WIDTH"])
    sig = (
        tf.TensorSpec((H, W, 1), tf.float32),
        tf.TensorSpec((),     tf.string),
    )
    ds = tf.data.Dataset.from_generator(gen, output_signature=sig)

    def _encode(img, lbl):
        idx = tf.py_function(
            lambda b: label_encoder.transform([b.decode('utf-8')])[0],
            [lbl], tf.int32
        )
        idx.set_shape([])
        if num_classes > 2:
            idx = tf.one_hot(idx, num_classes)
        return img, idx

    ds = (ds
          .map(_encode, num_parallel_calls=tf.data.AUTOTUNE)
          .shuffle(len(samples))
          .batch(config.batch_size)
          .prefetch(tf.data.AUTOTUNE))

    return ds
# def dataset_stratified_split(split, data, labels):
#     return train_test_split(data, labels,
#                             test_size=split,
#                             stratify=labels,
#                             random_state=config.RANDOM_SEED,
#                             shuffle=True)

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


# def calculate_class_weights(y_train, label_encoder):
#     """
#     Calculate class  weights for imbalanced datasets.
#     """
#     if label_encoder.classes_.size != 2:
#         y_train = label_encoder.inverse_transform(np.argmax(y_train, axis=1))

#     # Balanced class weights
#     weights = class_weight.compute_class_weight("balanced",
#                                                 np.unique(y_train),
#                                                 y_train)
#     class_weights = dict(enumerate(weights))

#     # Manual class weights for CBIS-DDSM
#     #class_weights = {0: 1.0, 1:1.5}

#     # No class weights
#     #class_weights = None

#     if config.verbose_mode:
#         print("Class weights: {}".format(str(class_weights)))

#     return class_weights
#     # return None
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
