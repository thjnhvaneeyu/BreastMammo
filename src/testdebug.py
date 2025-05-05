import os
import glob
import pandas as pd
import pydicom
from pydicom.errors import InvalidDicomError
import cv2

from data_operations.data_preprocessing import load_roi_and_label

# --- CẤU HÌNH ---
DATA_ROOT = "/kaggle/input/breastdata/INbreast/INbreast"
ROI_DIR   = os.path.join(DATA_ROOT, "AllROI")
DCM_DIR   = os.path.join(DATA_ROOT, "AllDICOMs")
CSV_PATH  = os.path.join(DATA_ROOT, "INbreast.csv")
NUM_TEST  = 5

# 1) Đọc map BI-RADS
df = pd.read_csv(CSV_PATH, sep=';')
df.columns = [c.strip() for c in df.columns]
birad_map = { str(fn).strip(): str(val).strip()
              for fn,val in zip(df['File Name'], df['Bi-Rads']) }

roi_files = sorted([f for f in os.listdir(ROI_DIR) if f.lower().endswith(".roi")])[:NUM_TEST]
print(f"== Testing {NUM_TEST} ROI files: {roi_files}")

for fn in roi_files:
    roi_path = os.path.join(ROI_DIR, fn)
    roi_id   = os.path.splitext(fn)[0].split('_',1)[0]
    print("\n" + "="*40)
    print(f"ROI    : {fn} → id={roi_id}")

    # 2) match DICOM
    patt = os.path.join(DCM_DIR, f"{roi_id}_*.dcm")
    dcm_list = glob.glob(patt)
    if not dcm_list:
        print(f"  ❌ No DICOM matches '{patt}'")
        continue
    dcm_path = dcm_list[0]
    print(f"  ✅ DICOM: {os.path.basename(dcm_path)}")

    # 3) load coords & label
    coords, label = load_roi_and_label(roi_path, birad_map)
    if coords is None:
        print("  ❌ load_roi_and_label → coords=None")
        continue
    print(f"  ✅ coords: {len(coords)} pts, label={label}")

    # 4) đọc ảnh DICOM
    try:
        ds  = pydicom.dcmread(dcm_path, force=True)
        img = ds.pixel_array
        print(f"  ✅ DICOM read shape={img.shape}, dtype={img.dtype}")
    except (InvalidDicomError, Exception) as e:
        print(f"  ❌ Error reading DICOM: {e}")
        continue

    # 5) crop & resize
    xs = [x for x,y in coords]; ys = [y for x,y in coords]
    x0,x1 = max(0,min(xs)), min(img.shape[1]-1,max(xs))
    y0,y1 = max(0,min(ys)), min(img.shape[0]-1,max(ys))
    roi_img = img[y0:y1+1, x0:x1+1]
    print(f"  → crop bbox=({x0},{y0})–({x1},{y1}), shape={roi_img.shape}")
    if roi_img.size == 0:
        print("  ❌ Empty ROI after crop")
        continue

    try:
        out = cv2.resize(roi_img, (224,224), interpolation=cv2.INTER_AREA)
        print(f"  ✅ Resized to {out.shape}")
    except Exception as e:
        print(f"  ❌ Error resizing: {e}")
        continue

    print("  🎉 Sample processed OK")
