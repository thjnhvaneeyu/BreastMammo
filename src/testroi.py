import os
import glob
import pandas as pd
import pydicom
from pydicom.errors import InvalidDicomError
import cv2

# nhập hàm của bạn
from data_operations.data_preprocessing import load_roi_and_label

# --- CONFIGURATION ---
DATA_ROOT = "/kaggle/input/breastdata/INbreast/INbreast"
ROI_DIR   = os.path.join(DATA_ROOT, "AllROI")
DCM_DIR   = os.path.join(DATA_ROOT, "AllDICOMs")
CSV_PATH  = os.path.join(DATA_ROOT, "INbreast.csv")
NUM_TEST  = 5    # số file ROI muốn test nhanh

# 1) Kiểm tra thư mục
print(f"ROI_DIR exists? {os.path.isdir(ROI_DIR)}, total ROI files = {len(os.listdir(ROI_DIR))}")
print(f"DCM_DIR exists? {os.path.isdir(DCM_DIR)}, total DCM files = {len(os.listdir(DCM_DIR))}")

# 2) Đọc bảng BI-RADS
df = pd.read_csv(CSV_PATH, sep=';')
df.columns = [c.strip() for c in df.columns]
birad_map = { str(fn).strip(): str(val).strip()
              for fn, val in zip(df["File Name"], df["Bi-Rads"]) }

print("Sample entries in birad_map:", list(birad_map.items())[:3])

# 3) Lấy list file ROI
roi_files = sorted([f for f in os.listdir(ROI_DIR) if f.lower().endswith(".roi")])
print(f"Testing first {NUM_TEST} ROI files:", roi_files[:NUM_TEST])

for roi_fn in roi_files[:NUM_TEST]:
    roi_path = os.path.join(ROI_DIR, roi_fn)
    roi_id   = os.path.splitext(roi_fn)[0].split("_",1)[0]
    print("\n" + "="*60)
    print(f"ROI file: {roi_fn} → id = {roi_id}")

    # 4) Tìm file DICOM tương ứng
    pattern = os.path.join(DCM_DIR, f"{roi_id}_*.dcm")
    dcm_matches = glob.glob(pattern)
    if not dcm_matches:
        print(f"  ❌ No DICOM found with pattern {pattern}")
        continue
    dcm_path = dcm_matches[0]
    print(f"  ✅ DICOM matched: {os.path.basename(dcm_path)}")

    # 5) Đọc coords + label
    coords, label = load_roi_and_label(roi_path, birad_map)
    if coords is None or label is None:
        print(f"  ❌ load_roi_and_label returned None → skip")
        continue
    print(f"  ✅ coords count = {len(coords)}, label = {label}")
    print(f"    first 5 coords: {coords[:5]}")

    # 6) Đọc ảnh DICOM
    try:
        ds  = pydicom.dcmread(dcm_path, force=True)
        img = ds.pixel_array
        print(f"  ✅ DICOM read: dtype={img.dtype}, shape={img.shape}")
    except InvalidDicomError as e:
        print(f"  ❌ InvalidDicomError: {e}")
        continue
    except Exception as e:
        print(f"  ❌ error reading DICOM: {e}")
        continue

    # 7) Tính bounding-box và crop
    xs = [x for x,y in coords]
    ys = [y for x,y in coords]
    x0, x1 = max(0,min(xs)), min(img.shape[1]-1,max(xs))
    y0, y1 = max(0,min(ys)), min(img.shape[0]-1,max(ys))
    print(f"  Bounding box: x0={x0},x1={x1}, y0={y0},y1={y1}")

    roi_img = img[y0:y1+1, x0:x1+1]
    print(f"  ROI crop shape: {roi_img.shape}, size={roi_img.size}")
    if roi_img.size == 0:
        print("  ❌ Empty ROI after crop → skip")
        continue

    # 8) Resize thử
    try:
        resized = cv2.resize(roi_img, (224,224), interpolation=cv2.INTER_AREA)
        print(f"  ✅ Resized to {resized.shape}")
    except Exception as e:
        print(f"  ❌ Error resizing ROI: {e}")
        continue

    print(f"  🎉 ROI {roi_id} processed successfully!")
