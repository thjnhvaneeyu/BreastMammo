import os
import pandas as pd
import pydicom
from PIL import Image, ImageOps
import numpy as np
from scipy.ndimage import gaussian_filter
import cv2

def preprocess_image(image_array):
    """
    Full preprocessing pipeline:
    1. Normalization
    2. Gaussian filtering
    3. Histogram equalization
    4. Resizing and padding to 224x224
    """
    # 1. Normalization (scale pixel intensities to [0, 1])
    normalized_image = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array) + 1e-8)
    # 2. Gaussian filtering (reduce noise)
    smoothed_image = gaussian_filter(normalized_image, sigma=0.5)
    # 3. Histogram Equalization for contrast
    uint8_image = (smoothed_image * 255).astype(np.uint8)
    equalized_image = cv2.equalizeHist(uint8_image)
    # 4. Resize to 224x224 with padding to maintain aspect ratio
    desired_size = 224
    image_pil = Image.fromarray(equalized_image)
    old_size = image_pil.size  # (width, height)
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    image_pil = image_pil.resize(new_size, Image.LANCZOS)
    delta_w = desired_size - new_size[0]
    delta_h = desired_size - new_size[1]
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    padded_image = ImageOps.expand(image_pil, padding, fill=0)  # black padding
    final_image = np.array(padded_image).astype(np.uint8)
    print("Output image shape:", final_image.shape)  # e.g. (224, 224)
    # Đảm bảo kích thước chính xác 224x224 (nếu chưa đúng do làm tròn)
    return cv2.resize(final_image, (224, 224)) if final_image.shape[:2] != (224, 224) else final_image

def process_and_save_dicom_images(root_dir, output_dir, metadata_excel, output_csv):
    """
    Process DICOM images using metadata, save processed PNG images in class folders,
    and record filenames and labels in a CSV.
    :param root_dir: Root directory containing DICOM files (subfolders by ID).
    :param output_dir: Directory to save processed images (will contain 'benign' and 'malignant' subdirs).
    :param metadata_excel: Path to the Excel file containing metadata with IDs and classifications.
    :param output_csv: Path to save the output CSV file with filename and label.
    """
    # Đọc file metadata Excel
    metadata = pd.read_excel(metadata_excel, engine='openpyxl')
    # Tạo thư mục đầu ra và các thư mục con cho 2 lớp
    os.makedirs(output_dir, exist_ok=True)
    benign_dir = os.path.join(output_dir, 'benign')
    malignant_dir = os.path.join(output_dir, 'malignant')
    os.makedirs(benign_dir, exist_ok=True)
    os.makedirs(malignant_dir, exist_ok=True)
    # Danh sách ghi log thông tin file ảnh và nhãn
    records = []
    # Lặp qua từng dòng metadata
    for _, row in metadata.iterrows():
        dicom_id = str(row['ID1']).strip()
        label = str(row['classification']).strip().lower()
        # Xác định thư mục DICOM theo ID
        dicom_folder = os.path.join(root_dir, dicom_id)
        for root, _, files in os.walk(dicom_folder):
            for file in files:
                if file.endswith('.dcm'):
                    dicom_path = os.path.join(root, file)
                    try:
                        # Đọc DICOM
                        dicom_data = pydicom.dcmread(dicom_path)
                        image_array = dicom_data.pixel_array
                        # Xử lý ảnh
                        processed_image = preprocess_image(image_array)
                        # Xác định nhãn (benign/malignant)
                        if 'malignant' in label: 
                            class_dir = malignant_dir
                            label_name = 'malignant'
                        else:
                            class_dir = benign_dir
                            label_name = 'benign'
                        # Tạo tên file PNG và lưu ảnh vào thư mục tương ứng
                        output_filename = f"{dicom_id}_{file.replace('.dcm', '.png')}"
                        output_filepath = os.path.join(class_dir, output_filename)
                        Image.fromarray(processed_image).save(output_filepath)
                        # Ghi log filename (kèm thư mục) và label
                        records.append({"filename": f"{label_name}/{output_filename}", "label": label_name})
                        print(f"Processed and saved: {output_filepath}")
                    except Exception as e:
                        print(f"Error processing {dicom_path}: {e}")
    # Ghi metadata CSV
    pd.DataFrame(records).to_csv(output_csv, index=False, encoding='utf-8')
    print(f"Saved metadata to {output_csv}")

# Ví dụ sử dụng (có thể thay đổi đường dẫn phù hợp trước khi chạy):
if __name__ == "__main__":
    root_directory = "/home/neeyuhuynh/cggm-mammography-classification/dataset/TheChineseMammographyDatabase/CMMD/"  # Thư mục gốc chứa các file DICOM
    output_directory = "/home/neeyuhuynh/Desktop/Breast-Cancer-Detection-Mammogram-Deep-Learning-Publication/data/CMMD"  # Thư mục sẽ chứa ảnh PNG (có subfolders benign/malignant)
    metadata_excel = "/home/neeyuhuynh/Breast-Cancer-Detection-Mammogram-Deep-Learning-Publication/data/CMMD/CMMD_clinicaldata_revision.xlsx"  # File Excel chứa thông tin ID và nhãn
    output_csv = os.path.join(output_directory, "processed_metadata.csv")
    process_and_save_dicom_images(root_directory, output_directory, metadata_excel, output_csv)
