# import os
# import pandas as pd
# from data_operations.data_preprocessing import load_roi_and_label

# # --- CẤU HÌNH ---
# DATA_ROOT = "/kaggle/input/breastdata/INbreast/INbreast"
# ROI_DIR   = os.path.join(DATA_ROOT, "AllROI")
# CSV_PATH  = os.path.join(DATA_ROOT, "INbreast.csv")

# # 1) Đọc CSV BI-RADS
# df = pd.read_csv(CSV_PATH, sep=';')
# df.columns = [c.strip() for c in df.columns]
# birad_map = {
#     str(fn).strip(): str(val).strip()
#     for fn, val in zip(df['File Name'], df['Bi-Rads'])
# }

# # 2) Lấy một vài file ROI để test
# roi_files = sorted([f for f in os.listdir(ROI_DIR) if f.lower().endswith(".roi")])[:10]
# print(f"Testing {len(roi_files)} ROI files:\n", roi_files)

# # 3) Debug load_roi_and_label trên từng file
# for roi_fn in roi_files:
#     roi_path = os.path.join(ROI_DIR, roi_fn)

#     # PID thô: split trước dấu "_" nhưng vẫn có đuôi ".roi"
#     pid_raw = os.path.basename(roi_fn).split('_', 1)[0]
#     # PID đúng: bỏ phần mở rộng trước và split
#     no_ext = os.path.splitext(roi_fn)[0]
#     pid    = no_ext.split('_', 1)[0]

#     print("\n" + "-"*60)
#     print(f"ROI file        : {roi_fn}")
#     print(f" - pid_raw      : {pid_raw!r}")
#     print(f" - pid (no_ext) : {pid!r}")

#     # lookup map với cả hai PID
#     val_raw = birad_map.get(pid_raw)
#     val     = birad_map.get(pid)
#     print(f" - birad_map.get(pid_raw) = {val_raw!r}")
#     print(f" - birad_map.get(pid)     = {val!r}")

#     # gọi chính hàm, in kết quả
#     coords, label = load_roi_and_label(roi_path, birad_map)
#     print(f" → load_roi_and_label returns coords={None if coords is None else len(coords)} points, label={label!r}")
import os
import pandas as pd
from data_operations.data_preprocessing import load_roi_and_label

# --- CẤU HÌNH ---
DATA_ROOT = "/kaggle/input/breastdata/INbreast/INbreast"
ROI_DIR   = os.path.join(DATA_ROOT, "AllROI")
CSV_PATH  = os.path.join(DATA_ROOT, "INbreast.csv")

# 1) Load bản đồ BI-RADS
df = pd.read_csv(CSV_PATH, sep=';')
df.columns = [c.strip() for c in df.columns]
birad_map = {
    str(fn).strip(): str(val).strip()
    for fn, val in zip(df['File Name'], df['Bi-Rads'])
}

# 2) Danh sách file .roi
roi_files = sorted(f for f in os.listdir(ROI_DIR) if f.lower().endswith(".roi"))

# 3) Thống kê
stats = {
    'total':       len(roi_files),
    'no_coords':   0,
    'no_label':    0,
    'success':     0,
}
examples = {
    'no_coords': [],
    'no_label':  [],
    'success':   [],
}

# 4) Chạy kiểm thử
for roi_fn in roi_files:
    path = os.path.join(ROI_DIR, roi_fn)
    coords, label = load_roi_and_label(path, birad_map)

    if coords is None:
        stats['no_coords'] += 1
        if len(examples['no_coords']) < 5:
            examples['no_coords'].append(roi_fn)
    elif label is None:
        stats['no_label'] += 1
        if len(examples['no_label']) < 5:
            examples['no_label'].append(roi_fn)
    else:
        stats['success'] += 1
        if len(examples['success']) < 5:
            examples['success'].append((roi_fn, label, len(coords)))

# 5) In kết quả
print("=== ROI TEST SUMMARY ===")
print(f"Total ROI files   : {stats['total']}")
print(f"No coords         : {stats['no_coords']}  (first examples: {examples['no_coords']})")
print(f"No label          : {stats['no_label']}  (first examples: {examples['no_label']})")
print(f"Successfully load : {stats['success']}  (first examples: {examples['success']})")
