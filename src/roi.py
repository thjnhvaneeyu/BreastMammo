# # test_roi.py
# import os
# import pandas as pd
# from data_operations.data_preprocessing import load_roi_and_label

# # 1) Thiết lập đúng data_dir và roi_dir
# DATA_ROOT_BREAST = "/kaggle/input/breastdata"
# data_dir = os.path.join(DATA_ROOT_BREAST, "INbreast", "INbreast")
# roi_dir  = os.path.join(data_dir, "AllROI")

# print("== ROI dir =", roi_dir)
# print("Exists?", os.path.isdir(roi_dir))
# print("Contents of AllROI:", os.listdir(roi_dir)[:10], "... total", len(os.listdir(roi_dir)))

# # 2) Load bản đồ BI-RADS
# csv_path = os.path.join(data_dir, "INbreast.csv")
# df = pd.read_csv(csv_path, sep=';')
# df.columns = [c.strip() for c in df.columns]
# birad_map = {
#     str(fn).strip(): str(val).strip()
#     for fn, val in zip(df['File Name'], df['Bi-Rads'])
# }
# print("Loaded birad_map entries:", list(birad_map.items())[:5])

# # 3) Thử parse 1–3 file .roi
# roi_files = [fn for fn in os.listdir(roi_dir) if fn.lower().endswith(".roi")]
# for fn in roi_files[:3]:
#     path = os.path.join(roi_dir, fn)
#     coords, label = load_roi_and_label(path, birad_map)
#     print(f"\nFile: {fn}")
#     print(" → coords:", coords[:5], "... total", len(coords) if coords else 0)
#     print(" → label:", label)

from data_operations.data_preprocessing import import_inbreast_roi_dataset
from sklearn.preprocessing import LabelEncoder

ds = import_inbreast_roi_dataset(
    "/kaggle/input/breastdata/INbreast/INbreast",
    LabelEncoder(),
    target_size=(224,224),
    csv_path="/kaggle/input/breastdata/INbreast/INbreast/INbreast.csv"
)
print("OK, found dataset:", ds)
# In vài batch đầu
for imgs, labels in ds.take(2):
    print(imgs.shape, labels.shape)
