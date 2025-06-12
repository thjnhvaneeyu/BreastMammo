import os
import argparse
import random
import numpy as np
import tensorflow as tf
import pydicom
import pandas as pd
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0
from sklearn.model_selection import train_test_split
from collections import Counter
import skimage as sk
import skimage.transform
import skimage.exposure
from PIL import Image

# ===================================================================
# PHẦN 1: CÁC HÀM HELPER VÀ AUGMENTATION (TỰ CHỨA, KHÔNG IMPORT)
# ===================================================================
def import_cmmd_dataset_standalone(data_dir):
    print(f"--- [STANDALONE] Loading CMMD data from: {data_dir}")
    metadata_path = os.path.join(data_dir, "CMMD_clinicaldata_revision.csv")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    df = pd.read_csv(metadata_path)
    images_list = []
    labels_list = []
    
    for index, row in df.iterrows():
        patient_id = row['ID1']
        side = row['LeftRight']
        classification = row['classification']
        
        # Tạo đường dẫn tới file ảnh DICOM
        image_path = os.path.join(data_dir, patient_id, f"{patient_id}_{side}_1.dcm")
        
        if os.path.exists(image_path):
            try:
                dcm_data = pydicom.dcmread(image_path)
                image = dcm_data.pixel_array
                image = (image - np.min(image)) / (np.max(image) - np.min(image)) # Chuẩn hóa về [0,1]
                image = np.array(Image.fromarray(image).resize((224, 224))) # Resize
                
                images_list.append(image)
                # Chuyển đổi nhãn: Benign -> 0, Malignant -> 1
                labels_list.append(0 if classification == 'Benign' else 1)
            except Exception as e:
                print(f"Warning: Could not process file {image_path}. Error: {e}")
        else:
            print(f"Warning: File not found {image_path}")

    return np.array(images_list, dtype=np.float32), np.array(labels_list, dtype=np.int32)

def horizontal_flip(image_array):
    return image_array[:, ::-1]

def random_rotation(image_array):
    random_degree = random.uniform(-15, 15)
    return sk.transform.rotate(image_array, random_degree, mode='reflect', preserve_range=True)

def create_individual_transform(image, transforms):
    transformed_image = image.copy()
    if 'horizontal_flip' in transforms and random.random() < 0.5:
        transformed_image = transforms['horizontal_flip'](transformed_image)
    if 'rotate' in transforms and random.random() < 0.3:
        transformed_image = transforms['rotate'](transformed_image)
    return transformed_image

def generate_image_transforms(images, labels):
    print("--- [STANDALONE] Running controlled oversampling augmentation. ---")
    numeric_labels = np.argmax(labels, axis=1)
    counts = Counter(numeric_labels)
    
    if len(counts) < 2: return images, labels
    majority_class = max(counts, key=counts.get)
    minority_class = min(counts, key=counts.get)
    num_to_generate = counts[majority_class] - counts[minority_class]

    if num_to_generate <= 0: return images, labels
        
    print(f"[INFO] Generating {num_to_generate} new samples for class {minority_class}.")
    
    basic_transforms = {'horizontal_flip': horizontal_flip, 'rotate': random_rotation}
    minority_indices = np.where(numeric_labels == minority_class)[0]
    
    new_images = [create_individual_transform(images[minority_indices[i % len(minority_indices)]], basic_transforms) for i in range(num_to_generate)]
    
    augmented_images = np.concatenate((images, np.array(new_images)), axis=0)
    new_labels = tf.keras.utils.to_categorical(np.full(num_to_generate, minority_class), num_classes=labels.shape[1])
    augmented_labels = np.concatenate((labels, new_labels), axis=0)
    
    indices = np.random.permutation(len(augmented_images))
    return augmented_images[indices], augmented_labels[indices]

# ===================================================================
# PHẦN 3: LOGIC MAIN ĐỂ CHẠY TRỰC TIẾP
# ===================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--weights_path", required=True)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--augment", action="store_true")
    args = parser.parse_args()

    # Tải dữ liệu bằng hàm độc lập
    X_np, y_scalar_labels = import_cmmd_dataset_standalone(args.data_dir)
    X_np = np.expand_dims(X_np, axis=-1) # Thêm chiều kênh (H, W) -> (H, W, 1)
    y_np = tf.keras.utils.to_categorical(y_scalar_labels, num_classes=2)

    # Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size=0.2, stratify=y_scalar_labels, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, stratify=np.argmax(y_train, axis=1), random_state=42)

    class_weights = None
    if args.augment:
        X_train, y_train = generate_image_transforms(X_train, y_train)
        print("[INFO] Data is balanced. Class weights are not needed.")
    else:
        # Tính class weights nếu không augmentation
        from sklearn.utils.class_weight import compute_class_weight
        y_train_numeric = np.argmax(y_train, axis=1)
        weights = compute_class_weight('balanced', classes=np.unique(y_train_numeric), y=y_train_numeric)
        class_weights = dict(enumerate(weights))
        print(f"[INFO] Data is imbalanced. Using class weights: {class_weights}")

    print(f"[INFO] Final training samples: {len(X_train)}")
    
    # Tạo mô hình
    input_tensor = Input(shape=(224, 224, 1))
    x = Concatenate()([input_tensor, input_tensor, input_tensor])
    base_model = EfficientNetB0(include_top=False, weights=args.weights_path, input_tensor=x)
    base_model.trainable = True
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(2, activation='softmax')(x)
    model = Model(inputs=input_tensor, outputs=outputs)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("[INFO] Starting training...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100, # Tăng epochs vì có early stopping
        batch_size=args.batch_size,
        class_weight=class_weights,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)]
    )
    
    print("\n[INFO] Final Evaluation on Test Set")
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    from sklearn.metrics import classification_report
    print(classification_report(y_true, y_pred, target_names=['Benign', 'Malignant']))

if __name__ == '__main__':
    main()