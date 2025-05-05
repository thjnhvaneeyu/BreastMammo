import os
import pandas as pd

from data_operations.data_preprocessing import load_roi_and_label

# --- CẤU HÌNH ---
DATA_ROOT = "/kaggle/input/breastdata/INbreast/INbreast"
ROI_DIR   = os.path.join(DATA_ROOT, "AllROI")
CSV_PATH  = os.path.join(DATA_ROOT, "INbreast.csv")
NUM_TEST  = 10   # số file thử

# # 1) Load bản đồ BI-RADS
# df = pd.read_csv(CSV_PATH, sep=';')
# df.columns = [c.strip() for c in df.columns]
# birad_map = {
#     str(fn).strip(): str(val).strip()
#     for fn, val in zip(df['File Name'], df['Bi-Rads'])
# }

# # 2) Lấy vài file .roi
# roi_files = sorted(f for f in os.listdir(ROI_DIR) if f.lower().endswith(".roi"))[:NUM_TEST]

# print(f"Testing {len(roi_files)} ROI files:")
# for fn in roi_files:
#     path = os.path.join(ROI_DIR, fn)
#     coords, label = load_roi_and_label(path, birad_map)
#     # print(f"- {fn:15s} → "
#     print(f"\nFile: {fn}"
#           f"coords: {len(coords) if coords else None:4s} pts, "
#           f"label: {label}")
#     # print(f"\nFile: {fn}")
#     # print(" → coords:", coords[:5], "... total", len(coords) if coords else 0)
#     # print(" → label:", label)

df = pd.read_csv(CSV_PATH, sep=';')
df.columns = [c.strip() for c in df.columns]
birad_map = {str(fn).strip(): str(val).strip() 
             for fn,val in zip(df['File Name'], df['Bi-Rads'])}

# Chọn một vài file .roi
roi_files = sorted(f for f in os.listdir(ROI_DIR) if f.lower().endswith(".roi"))[:NUM_TEST]
print(f"Testing {len(roi_files)} ROI files:")

for fn in roi_files:
    path = os.path.join(ROI_DIR, fn)
    coords, label = load_roi_and_label(path, birad_map)

    # chuẩn bị chuỗi hiển thị
    coords_count = len(coords) if coords is not None else None
    coords_str   = str(coords_count)          # 'None' hoặc số
    label_str    = label or 'None'

    # in với width chỉ định cho chuỗi
    print(f"- {fn:20} → coords: {coords_str:>4} pts, label: {label_str}")