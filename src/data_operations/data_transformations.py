import random
import numpy as np
import skimage as sk
import skimage.transform
import config
import os
import tensorflow as tf # Cần thiết cho MixUp nếu bạn muốn làm trên batch
import numpy as np
import skimage as sk
import skimage.filters # Cho GaussianBlur
import skimage.exposure # Cho điều chỉnh độ sáng (gamma) hoặc rescale_intensity
import random
# def generate_image_transforms(images, labels):
#     """
#     Oversample data by creating transformed copies of existing images.
#     Applies random rotations, flips, noise, shearing to balance classes.
#     """
#     augmentation_multiplier = 1
#     if config.dataset == "mini-MIAS-binary" or config.dataset == "CMMD-binary":
#         augmentation_multiplier = 3  # tăng cường nhiều hơn cho bộ dữ liệu binary nếu mất cân bằng
    
#     images_with_transforms = images
#     labels_with_transforms = labels
#     # Các phép biến đổi có thể áp dụng
#     available_transforms = {
#         'rotate': random_rotation,
#         'noise': random_noise,
#         'horizontal_flip': horizontal_flip,
#         'shear': random_shearing
#     }
#     # Tính số lượng mẫu mỗi lớp và xác định cần thêm bao nhiêu mẫu để cân bằng
#     class_balance = get_class_balances(labels)
#     max_count = max(class_balance) * augmentation_multiplier
#     to_add = [int(max_count - count) for count in class_balance]
#     for i in range(len(to_add)):
#         if to_add[i] <= 0:
#             continue
#         # Lấy các ảnh của lớp i
#         if label_is_binary(labels):
#             indices = [j for j, x in enumerate(labels) if x == i]
#             base_label = i  # nhãn số (0 hoặc 1)
#         else:
#             # Tạo vector one-hot cho lớp i để so sánh
#             label_vector = np.zeros(len(to_add)); label_vector[i] = 1
#             indices = [j for j, x in enumerate(labels) if np.array_equal(x, label_vector)]
#             base_label = label_vector
#         if len(indices) == 0:
#             continue
#         indiv_class_images = [images[j] for j in indices]
#         # Tạo các ảnh mới cho lớp i
#         for k in range(to_add[i]):
#             orig_img = indiv_class_images[k % len(indiv_class_images)]
#             transformed_image = create_individual_transform(orig_img, available_transforms)
#             # Đảm bảo shape của ảnh transform đúng (H, W, 1)
#             if config.is_roi or config.model == "CNN":
#                 transformed_image = transformed_image.reshape(1, config.ROI_IMG_SIZE['HEIGHT'], config.ROI_IMG_SIZE["WIDTH"], 1)
#             elif config.model == "VGG" or config.model == "Inception":
#                 transformed_image = transformed_image.reshape(1, config.MINI_MIAS_IMG_SIZE['HEIGHT'], config.MINI_MIAS_IMG_SIZE["WIDTH"], 1)
#             elif config.model == "VGG-common":
#                 transformed_image = transformed_image.reshape(1, config.VGG_IMG_SIZE['HEIGHT'], config.VGG_IMG_SIZE["WIDTH"], 1)
#             elif config.model == "ResNet":
#                 transformed_image = transformed_image.reshape(1, config.RESNET_IMG_SIZE['HEIGHT'], config.RESNET_IMG_SIZE["WIDTH"], 1)
#             elif config.model == "MobileNet":
#                 transformed_image = transformed_image.reshape(1, config.MOBILE_NET_IMG_SIZE['HEIGHT'], config.MOBILE_NET_IMG_SIZE["WIDTH"], 1)
#             elif config.model == "DenseNet" or config.model == "Inception":
#                 transformed_image = transformed_image.reshape(1, config.INCEPTION_IMG_SIZE['HEIGHT'], config.INCEPTION_IMG_SIZE["WIDTH"], 1)
#             # Thêm ảnh mới và nhãn mới vào tập
#             images_with_transforms = np.append(images_with_transforms, transformed_image, axis=0)
#             # Tạo label tương ứng cho ảnh mới
#             if label_is_binary(labels):
#                 new_label = np.array([base_label])  # giữ dạng số
#             else:
#                 new_label = base_label.reshape(1, len(base_label))
#             labels_with_transforms = np.append(labels_with_transforms, new_label, axis=0)
#     return images_with_transforms, labels_with_transforms

# def load_roi_and_label(roi_file_path: str):
#     """
#     Đọc file .roi, trả về:
#       - coords: List[(x:int,y:int)]  
#       - label_name: str (lấy từ BI-RADS mapping trong config)
#     """
#     # 1) Đọc tọa độ
#     coords = []
#     with open(roi_file_path, 'r', encoding='utf-8', errors='ignore') as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) >= 2:
#                 x, y = map(float, parts[:2])
#                 coords.append((int(x), int(y)))
#     # Nếu không có coords, trả về None để skip
#     if not coords:
#         return None, None

#     # 2) Lấy PID (image ID) từ tên file .roi
#     #    Ví dụ roi_file_path = ".../22678622_1.roi" -> pid_base = "22678622"
#     fn = os.path.basename(roi_file_path)
#     pid_base = os.path.splitext(fn)[0].split('_', 1)[0]

#     # 3) Xác định BI-RADS value (giả sử bạn đã load map birad_map ở đâu đó)
#     birad_val = config.INBREAST_BIRADS_MAPPING_RAW.get(pid_base)
#     # Nếu không tìm thấy, bạn có thể ném warning hoặc skip
#     if birad_val is None:
#         return coords, None

#     # 4) Map BI-RADS number sang class name theo config.INBREAST_BIRADS_MAPPING
#     label_name = None
#     for cls, vals in config.INBREAST_BIRADS_MAPPING.items():
#         # vals là danh sách các string như "BI-RADS 2", "BI-RADS 3",…
#         normalized = [v.replace("BI-RADS", "").strip() for v in vals]
#         if str(birad_val) in normalized:
#             label_name = cls
#             break

#     return coords, label_name
# --- HÀM MIXUP (TensorFlow) ---
def mixup_tf(images_batch, labels_batch, alpha=0.2):
    batch_size = tf.shape(images_batch)[0]
    indices = tf.random.shuffle(tf.range(batch_size))
    
    images_shuffled = tf.gather(images_batch, indices)
    labels_shuffled = tf.gather(labels_batch, indices)
    
    l = tf.compat.v1.distributions.Beta(alpha, alpha).sample(1)
    l = tf.cast(l, images_batch.dtype) # Đảm bảo lambda cùng kiểu dữ liệu
    # l = tf.maximum(l, 1 - l) # Đảm bảo lambda >= 0.5 không nhất thiết

    mixed_images = l * images_batch + (1 - l) * images_shuffled
    mixed_labels = l * labels_batch + (1 - l) * labels_shuffled
    return mixed_images, mixed_labels

# --- HÀM CUTMIX (TensorFlow) ---
def get_random_box_tf(img_height, img_width, lam):
    cut_rat = tf.sqrt(1. - lam)
    cut_h = tf.cast(tf.cast(img_height, tf.float32) * cut_rat, dtype=tf.int32)
    cut_w = tf.cast(tf.cast(img_width, tf.float32) * cut_rat, dtype=tf.int32)

    cx = tf.random.uniform([], 0, img_width, dtype=tf.int32)
    cy = tf.random.uniform([], 0, img_height, dtype=tf.int32)

    bbx1 = tf.clip_by_value(cx - cut_w // 2, 0, img_width)
    bby1 = tf.clip_by_value(cy - cut_h // 2, 0, img_height)
    bbx2 = tf.clip_by_value(cx + cut_w // 2, 0, img_width)
    bby2 = tf.clip_by_value(cy + cut_h // 2, 0, img_height)
    return bbx1, bby1, bbx2, bby2

def cutmix_tf(images_batch, labels_batch, alpha=1.0):
    batch_size = tf.shape(images_batch)[0]
    img_height = tf.shape(images_batch)[1]
    img_width = tf.shape(images_batch)[2]
    channels = tf.shape(images_batch)[3]

    indices = tf.random.shuffle(tf.range(batch_size))
    images_shuffled = tf.gather(images_batch, indices)
    labels_shuffled = tf.gather(labels_batch, indices)

    lam_value = tf.compat.v1.distributions.Beta(alpha, alpha).sample(1)
    bbx1, bby1, bbx2, bby2 = get_random_box_tf(img_height, img_width, lam_value)

    # Tạo mask
    mask_h = bby2 - bby1
    mask_w = bbx2 - bbx1

    # Đảm bảo mask_h và mask_w không âm và lớn hơn 0
    if mask_h == 0 or mask_w == 0:
        return images_batch, labels_batch

    # Phần được cắt từ ảnh shuffle
    patch = images_shuffled[:, bby1:bby2, bbx1:bbx2, :]

    # Tạo ảnh mới với patch
    # Cách 1: Dùng tf.tensor_scatter_nd_update (phức tạp hơn để tạo indices)
    # Cách 2: Dùng mask và tf.where (đơn giản hơn)
    y_coords_range = tf.range(img_height)
    x_coords_range = tf.range(img_width)
    Y_grid, X_grid = tf.meshgrid(y_coords_range, x_coords_range, indexing='ij') # W, H -> H, W

    # Expand X_grid, Y_grid for broadcasting with bbx1, etc.
    # bbx1, bby1, bbx2, bby2 là scalar tensors.
    # Y_grid, X_grid là [H, W]
    
    cut_mask_2d = (Y_grid >= bby1) & (Y_grid < bby2) & (X_grid >= bbx1) & (X_grid < bbx2)
    cut_mask_4d = tf.expand_dims(tf.expand_dims(cut_mask_2d, axis=0), axis=-1) # (1, H, W, 1)
    cut_mask_4d = tf.tile(cut_mask_4d, [batch_size, 1, 1, channels]) # (batch_size, H, W, channels)

    mixed_images = tf.where(cut_mask_4d, images_shuffled, images_batch)
    
    # Điều chỉnh lambda
    actual_lam = 1.0 - tf.cast((bbx2 - bbx1) * (bby2 - bby1), tf.float32) / tf.cast(img_height * img_width, tf.float32)
    actual_lam = tf.cast(actual_lam, labels_batch.dtype)

    mixed_labels = actual_lam * labels_batch + (1.0 - actual_lam) * labels_shuffled
    return mixed_images, mixed_labels

# --- HÀM AUGMENTATION CHÍNH ---
def generate_image_transforms(images: np.ndarray, labels: np.ndarray):
    """
    Oversample data. Áp dụng CutMix/MixUp chỉ cho INbreast.
    images: (N, H, W, C) - giả sử C=1 cho ảnh xám
    labels: (N,) hoặc (N, num_classes) - one-hot được ưu tiên cho MixUp/CutMix
    """
    print(f"Augmentation - Initial shapes: images={images.shape}, labels={labels.shape}")

    # Xác định số lớp và chuyển labels sang one-hot nếu cần (đặc biệt cho MixUp/CutMix)
    is_binary_scalar_labels = label_is_binary(labels)
    if is_binary_scalar_labels: # Binary 0/1
        num_classes = 2
        labels_one_hot = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
    elif labels.ndim == 2 and labels.shape[1] > 1: # Already one-hot
        num_classes = labels.shape[1]
        labels_one_hot = labels.astype(np.float32) # Đảm bảo float cho MixUp
    else: # Trường hợp khác (ví dụ: nhãn số đa lớp)
        unique_labels_count = len(np.unique(labels.ravel()))
        num_classes = unique_labels_count if unique_labels_count > 1 else np.max(labels.astype(int)) + 1
        labels_one_hot = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
        print(f"Converted scalar multi-class labels to one-hot. Num_classes: {num_classes}")

    # --- Các phép biến đổi hình học cơ bản ---
    basic_transforms = {
        'rotate': random_rotation,
        'noise': random_noise,
        'horizontal_flip': horizontal_flip,
        'shear': random_shearing,
        'gaussian_blur': random_gaussian_blur,         # <--- THÊM MỚI
        'brightness_adjust': random_brightness_adjustment # <--- THÊM MỚI
    }
    if config.dataset in ["CMMD_binary", "CMMD-binary", "INbreast"]: # Thêm zoom, contrast
         basic_transforms.update({'zoom': random_zoom, 'contrast': random_contrast})


    # --- Logic tăng cường ---
    augmented_images_list = list(images.astype(np.float32)) # Chuyển sang float sớm
    augmented_labels_list = list(labels_one_hot.astype(np.float32))

    # Hệ số nhân (chỉ INbreast được nhân 3 lần tổng cộng)
    # Các bộ khác sẽ chỉ cân bằng lớp mà không nhất thiết nhân 3
    target_multiplier = 1
    if config.dataset == "INbreast":
        target_multiplier = 3 
    elif config.dataset in ["mini-MIAS-binary", "CMMD-binary"]:
        target_multiplier = 2 # Hoặc 3 tùy bạn muốn tăng cường mạnh đến đâu cho các bộ này

    class_counts = get_class_balances(labels_one_hot, num_classes) # labels_one_hot là (N, num_classes)
    
    # Nếu target_multiplier > 1, mục tiêu là max_count * target_multiplier
    # Nếu target_multiplier = 1, mục tiêu chỉ là max_count (cân bằng)
    if not class_counts: # Xử lý trường hợp class_counts rỗng
        print("Warning: class_counts is empty. Skipping basic augmentation.")
        target_count_per_class = 0
    else:
        target_count_per_class = max(class_counts)
        if target_multiplier > 1:
             target_count_per_class *= target_multiplier
    
    print(f"Class counts: {class_counts}, Target count per class (approx): {target_count_per_class}")

    for class_idx in range(num_classes):
        current_class_count = class_counts[class_idx]
        num_to_generate = target_count_per_class - current_class_count
        
        if num_to_generate <= 0:
            continue

        # Tìm ảnh gốc thuộc lớp này
        # labels_one_hot là (N, num_classes)
        original_indices_for_class = [
            j for j, lab_vec in enumerate(labels_one_hot) if lab_vec[class_idx] == 1
        ]

        if not original_indices_for_class:
            print(f"Warning: No original images found for class {class_idx} to augment from.")
            continue
        
        print(f"Augmenting class {class_idx}: Need to generate {num_to_generate} images.")
        
        for k in range(num_to_generate):
            original_image_to_transform = images[original_indices_for_class[k % len(original_indices_for_class)]]
            
            # Áp dụng các phép biến đổi hình học cơ bản
            transformed_image = create_individual_transform(original_image_to_transform.astype(np.float32), basic_transforms)
            
            augmented_images_list.append(transformed_image)
            augmented_labels_list.append(labels_one_hot[original_indices_for_class[k % len(original_indices_for_class)]].copy())

    # --- Áp dụng MixUp và CutMix CHỈ CHO INBREAST ---
    final_images_np = np.array(augmented_images_list, dtype=np.float32)
    final_labels_np = np.array(augmented_labels_list, dtype=np.float32)

    if config.dataset == "INbreast":
        print(f"INbreast - Before MixUp/CutMix: images_shape={final_images_np.shape}, labels_shape={final_labels_np.shape}")
        
        # Chuyển sang tf.Tensor
        # Đảm bảo ảnh có channel dimension (H, W, 1)
        if final_images_np.ndim == 3: # (N, H, W)
            final_images_np = np.expand_dims(final_images_np, axis=-1)

        tf_images = tf.convert_to_tensor(final_images_np, dtype=tf.float32)
        tf_labels = tf.convert_to_tensor(final_labels_np, dtype=tf.float32) # labels đã là one-hot

        # Quyết định áp dụng (ví dụ: 50% cho mỗi loại)
        if random.random() < 0.5: # Xác suất áp dụng MixUp
            print("Applying MixUp for INbreast...")
            tf_images, tf_labels = mixup_tf(tf_images, tf_labels, alpha=0.2)
            print(f"After MixUp: images_shape={tf_images.shape}, labels_shape={tf_labels.shape}")

        if random.random() < 0.5: # Xác suất áp dụng CutMix
            print("Applying CutMix for INbreast...")
            # CutMix có thể làm thay đổi giá trị pixel, cần clip lại nếu model yêu cầu 0-1
            tf_images, tf_labels = cutmix_tf(tf_images, tf_labels, alpha=1.0)
            tf_images = tf.clip_by_value(tf_images, 0.0, 1.0) # Đảm bảo pixel trong khoảng [0,1]
            print(f"After CutMix: images_shape={tf_images.shape}, labels_shape={tf_labels.shape}")
        
        final_images_np = tf_images.numpy()
        final_labels_np = tf_labels.numpy()

    # Reshape cuối cùng (nếu cần) và trả về
    # Hàm này không còn reshape bên trong nữa, việc reshape sẽ do hàm gọi (ví dụ: model input)
    # Hoặc bạn có thể thêm logic reshape ở đây nếu tất cả output phải có cùng kích thước cố định
    # Nếu bạn cần đảm bảo tất cả ảnh có cùng một target_size từ config:
    
    target_h_final, target_w_final = config.INBREAST_IMG_SIZE['HEIGHT'], config.INBREAST_IMG_SIZE['WIDTH']
    if config.is_roi: # Nếu là ROI, có thể dùng ROI_IMG_SIZE
        target_h_final, target_w_final = config.ROI_IMG_SIZE['HEIGHT'], config.ROI_IMG_SIZE['WIDTH']
    
    # # Code cũ có logic reshape phức tạp dựa trên model, có thể giữ lại nếu cần
    # # nhưng thường augmentation sẽ giữ nguyên kích thước hoặc resize về một kích thước chung
    
    resized_final_images = []
    for img in final_images_np:
        if img.shape[0] != target_h_final or img.shape[1] != target_w_final:
            # Kiểm tra img có phải là tensor không, nếu có thì .numpy()
            img_to_resize = img.numpy() if hasattr(img, 'numpy') else img
            res_img = cv2.resize(img_to_resize, (target_w_final, target_h_final), interpolation=cv2.INTER_AREA)
            if res_img.ndim == 2: # Nếu cv2.resize trả về ảnh 2D (grayscale)
                res_img = np.expand_dims(res_img, axis=-1)
            resized_final_images.append(res_img)
        else:
            resized_final_images.append(img)
    
    final_images_output = np.array(resized_final_images, dtype=np.float32)

    # Nếu nhãn gốc là binary scalar, và sau MixUp/CutMix nó thành one-hot (hoặc float vector)
    # bạn có thể muốn chuyển nó về lại dạng scalar nếu model của bạn yêu cầu vậy
    if is_binary_scalar_labels and final_labels_np.ndim > 1:
        # Nếu final_labels_np là one-hot hoặc vector xác suất, lấy argmax
        final_labels_output = np.argmax(final_labels_np, axis=1)
    else:
        final_labels_output = final_labels_np # Giữ nguyên (có thể là one-hot hoặc đã mix)

    print(f"Augmentation - Final shapes: images={final_images_output.shape}, labels={final_labels_output.shape}")
    return final_images_output, final_labels_output

def sample_benign_patch(full_img: np.ndarray, x0: int, y0: int, w: int, h: int) -> np.ndarray:
    """Randomly crop a non-overlapping benign region of size (h,w)."""
    H, W = full_img.shape
    while True:
        tx = np.random.randint(0, W - w)
        ty = np.random.randint(0, H - h)
        # ensure no overlap
        if not (abs(tx - x0) < w and abs(ty - y0) < h):
            return full_img[ty:ty+h, tx:tx+w]

def fourier_domain_adaptation(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """Swap amplitude spectra from src into tgt domain (FDA)."""
    src_fft = np.fft.fft2(src)
    tgt_fft = np.fft.fft2(tgt)
    src_amp, src_phase = np.abs(src_fft), np.angle(src_fft)
    tgt_amp = np.abs(tgt_fft)
    merged_fft = tgt_amp * np.exp(1j * src_phase)
    adapted = np.fft.ifft2(merged_fft).real
    return np.clip(adapted, 0.0, 1.0)

def smooth_mix(adapted: np.ndarray, benign: np.ndarray, sigma: float) -> np.ndarray:
    """Blend two patches with a Gaussian radial mask."""
    H, W = adapted.shape
    cy, cx = H/2, W/2
    y, x = np.ogrid[:H, :W]
    d2 = (x-cx)**2 + (y-cy)**2
    mask = np.exp(-d2/(2*sigma**2))
    return mask * adapted + (1-mask) * benign

def augment_roi_patch(full_img: np.ndarray, coords: list, target_size: tuple=None) -> np.ndarray:
    """
    Given full image and ROI coords, crop lesion, sample benign,
    apply FDA + SmoothMix, resize to target_size, return patch [H,W,1].
    """
    # crop lesion
    xs, ys = zip(*coords)
    x0, x1 = max(0,min(xs)), min(full_img.shape[1], max(xs))
    y0, y1 = max(0,min(ys)), min(full_img.shape[0], max(ys))
    lesion = full_img[y0:y1, x0:x1]
    h, w = lesion.shape

    # sample benign region
    benign = sample_benign_patch(full_img, x0, y0, w, h)

    # FDA adapt + smooth mix
    adapted = fourier_domain_adaptation(lesion, benign)
    mixed   = smooth_mix(adapted, benign, sigma=min(h,w)/4.)

    # resize
    H, W = target_size or (config.INBREAST_IMG_SIZE["HEIGHT"],
                           config.INBREAST_IMG_SIZE["WIDTH"])
    patch = cv2.resize(mixed, (W, H), interpolation=cv2.INTER_AREA)
    return patch[..., np.newaxis].astype(np.float32)

def label_is_binary(labels):
    # Kiểm tra nếu labels là mảng 1D chứa toàn số (0/1) => binary
    return labels.ndim == 1 or (labels.ndim == 2 and labels.shape[1] == 1)

# ... (các hàm random_rotation, random_noise, horizontal_flip, random_shearing,
# random_zoom, random_contrast, create_individual_transform, get_class_balances
# generate_image_transforms đã có trong file data_transformations.py của bạn) ...

def random_gaussian_blur(image_array: np.ndarray, sigma_max: float = 1.0) -> np.ndarray:
    """
    Áp dụng Gaussian Blur với sigma ngẫu nhiên.
    Ảnh đầu vào nên có giá trị pixel dạng float.

    :param image_array: Mảng NumPy chứa ảnh (H, W) hoặc (H, W, 1).
    :param sigma_max: Giá trị sigma tối đa cho Gaussian Blur.
                      Sigma nhỏ -> ít mờ, sigma lớn -> mờ nhiều.
    :return: Ảnh đã được làm mờ.
    """
    if image_array.max() > 1.0: # Giả định ảnh chưa được chuẩn hóa về [0,1]
        image_array_float = sk.util.img_as_float(image_array)
    else:
        image_array_float = image_array.astype(np.float32)

    sigma = random.uniform(0, sigma_max)
    # preserve_range=True rất quan trọng để giữ dải giá trị pixel sau khi làm mờ,
    # đặc biệt nếu ảnh đầu vào không phải là [0,1]
    # multichannel=True nếu ảnh có chiều kênh (ví dụ H,W,1), False nếu là (H,W)
    # Tuy nhiên, gaussian thường hoạt động trên từng kênh nếu multichannel=True và ảnh có nhiều kênh.
    # Đối với ảnh xám (H,W) hoặc (H,W,1), bạn có thể bỏ qua multichannel hoặc đặt là False nếu ảnh là 2D.
    
    is_multichannel = image_array_float.ndim == 3 and image_array_float.shape[-1] > 1
    # Nếu ảnh là (H,W,1), thì nên coi nó là grayscale, không phải multichannel theo nghĩa màu sắc
    if image_array_float.ndim == 3 and image_array_float.shape[-1] == 1:
        # Bỏ chiều cuối để gaussian hoạt động trên ảnh 2D, sau đó thêm lại
        blurred_image = sk.filters.gaussian(image_array_float.squeeze(axis=-1),
                                            sigma=sigma,
                                            preserve_range=True,
                                            mode='reflect') # mode='reflect' giúp xử lý biên ảnh tốt hơn
        return np.expand_dims(blurred_image, axis=-1).astype(np.float32)
    else: # Ảnh 2D (H,W) hoặc ảnh màu thực sự (H,W,C với C>1)
        blurred_image = sk.filters.gaussian(image_array_float,
                                            sigma=sigma,
                                            preserve_range=True,
                                            multichannel=is_multichannel,
                                            mode='reflect')
        return blurred_image.astype(np.float32)


def random_brightness_adjustment(image_array: np.ndarray, factor_range: tuple = (0.7, 1.3)) -> np.ndarray:
    """
    Điều chỉnh độ sáng của ảnh bằng cách nhân với một hệ số ngẫu nhiên.
    Ảnh đầu vào nên có giá trị pixel dạng float trong khoảng [0, 1].

    :param image_array: Mảng NumPy chứa ảnh (H, W) hoặc (H, W, 1).
    :param factor_range: Khoảng (min_factor, max_factor) để chọn hệ số điều chỉnh độ sáng.
                         Factor < 1 làm ảnh tối hơn, factor > 1 làm ảnh sáng hơn.
    :return: Ảnh đã được điều chỉnh độ sáng.
    """
    if image_array.max() > 1.0: # Đảm bảo ảnh trong khoảng [0,1]
        image_array_float = sk.util.img_as_float(image_array)
    else:
        image_array_float = image_array.astype(np.float32)

    brightness_factor = random.uniform(factor_range[0], factor_range[1])
    
    # Nhân với hệ số và cắt giá trị về khoảng [0, 1]
    adjusted_image = np.clip(image_array_float * brightness_factor, 0.0, 1.0)
    
    return adjusted_image.astype(np.float32)


def random_rotation(image_array: np.ndarray):
    """
    Randomly rotate the image.
    
    :param image_array: input image
    :return: randomly rotated image
    """
    random_degree = random.uniform(-20, 20)
    return sk.transform.rotate(image_array, random_degree, mode='wrap')

def random_noise(image_array: np.ndarray):
    """
    Add a random amount of noise to the image.
    
    :param image_array: input image
    :return: image with added random noise
    """
    return sk.util.random_noise(image_array)

def horizontal_flip(image_array: np.ndarray):
    """
    Flip image horizontally.
    
    :param image_array: input image
    :return: horizontally flipped image
    """
    return image_array[:, ::-1]

def random_shearing(image_array: np.ndarray):
    """
    Add random amount of shearing to image.
    
    :param image_array: input image
    :return: sheared image
    """
    random_degree = random.uniform(-0.2, 0.2)
    tf = sk.transform.AffineTransform(shear=random_degree)
    return sk.transform.warp(image_array, tf, order=1, preserve_range=True, mode='wrap')

# def create_individual_transform(image: np.array, transforms: dict):
#     """
#     Create transformation of an individual image by applying multiple transforms.
    
#     :param image: input image
#     :param transforms: dictionary of possible transforms
#     :return: transformed image
#     """
#     num_transformations_to_apply = random.randint(1, len(transforms))
#     transformed_image = image.copy()  # Start with a copy of the original image
    
#     # Apply random transformations
#     for _ in range(num_transformations_to_apply):
#         key = random.choice(list(transforms))
#         transformed_image = sk.transform.resize(transformed_image, (config.CMMD_IMG_SIZE['HEIGHT'], config.CMMD_IMG_SIZE['WIDTH']))
#     if transformed_image.ndim == 3 and transformed_image.shape[2] == 3:
#         transformed_image = sk.color.rgb2gray(transformed_image)

    
#     return transformed_image
# def create_individual_transform(image: np.array, transforms: dict):
#     """
#     Tạo biến đổi cho một ảnh cụ thể.
#     """
#     num_transformations_to_apply = random.randint(1, len(transforms))
#     transformed_image = image.copy()  # Bắt đầu với bản sao của ảnh gốc
    
#     # Áp dụng các biến đổi ngẫu nhiên
#     for _ in range(num_transformations_to_apply):
#         key = random.choice(list(transforms))
#         transformed_image = transforms[key](transformed_image)
    
#     return transformed_image
def create_individual_transform(image: np.ndarray, transforms: dict):
    """
    Apply a random combination of transformations to an image.
    """
    num_transformations_to_apply = random.randint(1, len(transforms))
    transformed_image = image.copy()
    for _ in range(num_transformations_to_apply):
        transform_func = random.choice(list(transforms.values()))
        transformed_image = transform_func(transformed_image)
    return transformed_image

# def get_class_balances(y_vals):
#     """
#     Count occurrences of each class.
#     Supports CMMD, INbreast, and mini-MIAS datasets.
    
#     :param y_vals: labels
#     :return: array count of each class
#     """
#     # Initialize counts with default value
#     counts = np.zeros(2)  # Default to binary classification
    
#     if config.dataset in ["CMMD", "INbreast"]:
#         num_classes = 2  # benign and malignant
#         counts = np.zeros(num_classes)
#         for y_val in y_vals:
#             if np.array_equal(y_val, [1, 0]):  # benign
#                 counts[0] += 1
#             elif np.array_equal(y_val, [0, 1]):  # malignant
#                 counts[1] += 1
#     else:
#         # Check if one-hot encoded or scalar labels
#         if len(y_vals) > 0 and hasattr(y_vals[0], '__len__'):
#             # One-hot encoded labels
#             num_classes = len(y_vals[0])
#             counts = np.zeros(num_classes)
#             for y_val in y_vals:
#                 for i in range(num_classes):
#                     counts[i] += y_val[i]
#         else:
#             # Scalar labels
#             unique_values = np.unique(y_vals)
#             num_classes = len(unique_values)
#             counts = np.zeros(num_classes)
#             for i, cls in enumerate(unique_values):
#                 counts[i] = np.sum(np.array(y_vals) == cls)
    
# #     return counts.tolist()
# def get_class_balances(y_vals):
#     """Đếm số lượng mẫu trong mỗi lớp."""
#     # Khởi tạo biến counts với giá trị mặc định
#     counts = np.zeros(2)  # Mặc định 2 lớp
    
#     if config.dataset in ["CMMD", "INbreast"]:
#         num_classes = 2  # benign và malignant
#         counts = np.zeros(num_classes)
#         for y_val in y_vals:
#             if np.array_equal(y_val, [1, 0]):  # benign
#                 counts[0] += 1
#             elif np.array_equal(y_val, [0, 1]):  # malignant
#                 counts[1] += 1
#     else:
#         # Kiểm tra cấu trúc y_vals để xử lý phù hợp
#         if len(y_vals) > 0 and hasattr(y_vals[0], '__len__'):
#             # One-hot encoded labels
#             num_classes = len(y_vals[0])
#             counts = np.zeros(num_classes)
#             for y_val in y_vals:
#                 for i in range(num_classes):
#                     counts[i] += y_val[i]
#         else:
#             # Nhãn dạng scalar
#             unique_values = np.unique(y_vals)
#             num_classes = len(unique_values)
#             counts = np.zeros(num_classes)
#             for i, cls in enumerate(unique_values):
#                 counts[i] = np.sum(np.array(y_vals) == cls)
    
#     return counts.tolist()
def label_is_binary(labels: np.ndarray) -> bool:
    """
    Check if labels are binary (0/1) scalar array.
    """
    arr = np.array(labels)
    return arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] == 1)

def get_class_balances(y_vals):
    """
    Đếm số lượng mẫu cho mỗi lớp trong y_vals.
    Trả về list [count_class0, count_class1, ...].
    """
    if config.dataset == "mini-MIAS":
        # multi-class (nhãn one-hot)
        num_classes = y_vals.shape[1]  # số cột = số lớp
        counts = np.zeros(num_classes, dtype=int)
        for y in y_vals:
            # y là vector one-hot, cộng dồn cho class index tương ứng
            counts += y.astype(int)
    elif config.dataset == "mini-MIAS-binary" or config.dataset == "CMMD-binary":
        # binary (nhãn 0/1 dạng số)
        counts = np.zeros(2, dtype=int)
        for y in y_vals:
            if y == 0: counts[0] += 1
            elif y == 1: counts[1] += 1
    else:
        # Mặc định cho các trường hợp khác (nếu label one-hot)
        try:
            num_classes = y_vals.shape[1]
        except IndexError:
            num_classes = len(np.unique(y_vals))
        counts = np.zeros(num_classes, dtype=int)
        for y in y_vals:
            if isinstance(y, np.ndarray):
                # nếu one-hot vector
                counts += y.astype(int)
            else:
                # nếu label số
                counts[int(y)] += 1
    return counts.tolist()

def random_rotation(image_array: np.ndarray) -> np.ndarray:
    """
    Randomly rotate the image.
    """
    angle = random.uniform(-20, 20)
    return sk.transform.rotate(image_array, angle, mode='wrap')

def random_noise(image_array: np.ndarray) -> np.ndarray:
    """
    Add random noise to the image.
    """
    return sk.util.random_noise(image_array)

def horizontal_flip(image_array: np.ndarray) -> np.ndarray:
    """
    Flip the image horizontally.
    """
    return image_array[:, ::-1]

def random_shearing(image_array: np.ndarray) -> np.ndarray:
    """
    Apply random shearing to the image.
    """
    shear = random.uniform(-0.2, 0.2)
    tform = sk.transform.AffineTransform(shear=shear)
    return sk.transform.warp(image_array, tform, order=1, preserve_range=True, mode='wrap')

def random_zoom(image_array: np.ndarray) -> np.ndarray:
    """
    Zoom randomly into the central region of the image.
    """
    h, w = image_array.shape[:2]
    factor = random.uniform(0.8, 1.0)
    ch = int(h * factor / 2)
    cw = int(w * factor / 2)
    center_h, center_w = h // 2, w // 2
    top = max(0, center_h - ch)
    bottom = min(h, center_h + ch)
    left = max(0, center_w - cw)
    right = min(w, center_w + cw)
    zoomed = image_array[top:bottom, left:right]
    return sk.transform.resize(zoomed, (h, w), mode='reflect')

def random_contrast(image_array: np.ndarray) -> np.ndarray:
    """
    Adjust contrast randomly by stretching pixel intensities.
    """
    low_p = random.uniform(0.01, 0.1) * 100
    high_p = random.uniform(0.9, 0.99) * 100
    vmin, vmax = np.percentile(image_array, [low_p, high_p])
    return sk.exposure.rescale_intensity(image_array, in_range=(vmin, vmax), out_range=(0, 1))

def create_individual_transform(image: np.ndarray, transforms: dict) -> np.ndarray:
    """
    Apply a random combination of transforms to a single image.
    """
    num = random.randint(1, len(transforms))
    transformed = image.copy()
    for _ in range(num):
        func = random.choice(list(transforms.values()))
        transformed = func(transformed)
    return transformed

def label_is_binary(labels: np.ndarray) -> bool:
    """
    Check if labels are binary (0/1) scalar array.
    """
    arr = np.array(labels)
    return arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] == 1)

def get_class_balances(y_vals: np.ndarray) -> list:
    """
    Count samples per class in y_vals.
    Returns [count_class0, count_class1, ...].
    """
    arr = np.array(y_vals)
    if arr.ndim == 2 and arr.shape[1] > 1:
        return list(arr.sum(axis=0).astype(int))
    else:
        unique, counts = np.unique(arr, return_counts=True)
        return [int(counts[unique.tolist().index(i)]) if i in unique else 0
                for i in range(len(unique))]

# def generate_image_transforms(images: np.ndarray, labels: np.ndarray):
#     """
#     Oversample data by creating transformed copies to balance classes.
#     """
#     # choose multiplier
#     if config.dataset in ["mini-MIAS-binary", "CMMD_binary"]:
#         multiplier = 3
#     else:
#         multiplier = 1

#     imgs = images.copy()
#     labs = labels.copy()

#     # base transforms
#     transforms = {
#         'rotate': random_rotation,
#         'noise': random_noise,
#         'horizontal_flip': horizontal_flip,
#         'shear': random_shearing
#     }
#     # # dataset-specific
#     # if config.dataset in ["CMMD_binary", "CMMD"]:
#     #     transforms.update({'zoom': random_zoom, 'contrast': random_contrast})
#     # if config.dataset == "INbreast":
#     #     transforms.update({'zoom': random_zoom, 'contrast': random_contrast})
#     # dataset-specific: thêm zoom & contrast cho CMMD và INbreast
#     if config.dataset in ["CMMD_binary", "CMMD-binary", "INbreast"]:
#         transforms.update({
#             'zoom': random_zoom,
#             'contrast': random_contrast
#         })

#     balances = get_class_balances(labels)
#     target = max(balances) * multiplier
#     to_add = [target - b for b in balances]

#     for cls_idx, n in enumerate(to_add):
#         if n <= 0:
#             continue
#         # find indices
#         if label_is_binary(labels):
#             idxs = [i for i, l in enumerate(labels) if l == cls_idx]
#             base_label = cls_idx
#         else:
#             one_hot = np.zeros(len(balances))
#             one_hot[cls_idx] = 1
#             idxs = [i for i, l in enumerate(labels) if np.array_equal(l, one_hot)]
#             base_label = one_hot
#         if not idxs:
#             continue
#         class_imgs = [images[i] for i in idxs]
#         for i in range(n):
#             orig = class_imgs[i % len(class_imgs)]
#             transformed = create_individual_transform(orig, transforms)
#             h, w = transformed.shape[:2]
#             c = 1 if transformed.ndim == 2 else transformed.shape[2]
#             batch_img = transformed.reshape((1, h, w, c))
#             imgs = np.append(imgs, batch_img, axis=0)
#             if label_is_binary(labels):
#                 labs = np.append(labs, np.array([base_label]), axis=0)
#             else:
#                 labs = np.append(labs, base_label.reshape(1, -1), axis=0)

#     return imgs, labs

# import random

# import numpy as np
# import skimage as sk
# import skimage.transform

# import config

# def generate_image_transforms(images, labels):
#     """
#     Oversample data by transforming existing images.
#     Adjusted for CMMD dataset which has more malignant than benign samples.
    
#     :param images: input images
#     :param labels: input labels
#     :return: updated list of images and labels with extra transformed images and labels
#     """
#     # Thiết lập augmentation_multiplier dựa trên dataset
#     augmentation_multiplier = 1
#     if config.dataset == "mini-MIAS-binary":
#         augmentation_multiplier = 3
#     elif config.dataset == "CMMD":
#         # Tăng số lượng mẫu benign nhiều hơn vì CMMD có ít mẫu benign hơn
#         augmentation_multiplier = 3
#     elif config.dataset == "INbreast":
#         # Tăng số lượng mẫu malignant vì INbreast có ít mẫu malignant hơn
#         augmentation_multiplier = 3

#     images_with_transforms = images
#     labels_with_transforms = labels

#     available_transforms = {
#         'rotate': random_rotation,
#         'noise': random_noise,
#         'horizontal_flip': horizontal_flip,
#         'shear': random_shearing,
#         # Thêm các phép biến đổi đặc biệt cho CMMD nếu cần
#         'gamma_correction': gamma_correction,
#         'zoom': random_zoom,
#         'contrast': random_contrast,
#     }

#     # Thêm biến đổi đặc biệt cho INbreast
#     if config.dataset == "INbreast":
#         available_transforms['zoom'] = random_zoom
#         available_transforms['contrast'] = random_contrast

#     # Tính toán số lượng mẫu trong mỗi lớp
#     class_balance = get_class_balances(labels)

#     # Điều chỉnh cách cân bằng lớp dựa trên dataset
#     if config.dataset == "CMMD":
#         # Với CMMD, chỉ tăng cường dữ liệu cho lớp benign (lớp 0)
#         max_count = class_balance[1]  # Lấy số lượng mẫu malignant làm mục tiêu
#         to_add = [max_count - class_balance[0], 0]  # Chỉ tăng cường lớp benign
#     elif config.dataset == "INbreast":
#         # Với INbreast, chỉ tăng cường dữ liệu cho lớp malignant (lớp 1)
#         max_count = class_balance[0]  # Lấy số lượng mẫu benign làm mục tiêu
#         to_add = [0, max_count - class_balance[1]]  # Chỉ tăng cường lớp malignant
#     else:
#         # Cách cân bằng lớp ban đầu
#         max_count = max(class_balance) * augmentation_multiplier
#         to_add = [max_count - i for i in class_balance]

#     for i in range(len(to_add)):
#         if int(to_add[i]) == 0:
#             continue
        
        
#         # Tạo nhãn one-hot
#         label = np.zeros(len(to_add))
#         label[i] = 1
        
#         # Tìm các chỉ số của ảnh thuộc lớp hiện tại
#         indices = [j for j, x in enumerate(labels) if np.array_equal(x, label)]
#         indiv_class_images = [images[j] for j in indices]

#         for k in range(int(to_add[i])):
#             # Tạo ảnh đã biến đổi
#             # transformed_image = create_individual_transform(indiv_class_images[k % len(indiv_class_images)],
#             #                                                available_transforms)
#             # # In the generate_image_transforms function, ensure the reshape matches the actual image dimensions
#             # transformed_image = transformed_image.reshape(1, 224, 224, 1)  # If your images are 224×224
#             transformed_image = None
#             for k in range(int(to_add[i])):
#                 transformed_image = create_individual_transform(indiv_class_images[k % len(indiv_class_images)],
#                                                                 available_transforms)
#                 if transformed_image is None:
#                     continue  # Skip this iteration if no transformation was applied
#                 if transformed_image is not None:
#                     transformed_image = transformed_image.reshape(1, 224, 224, 1)
#                 else:
#                     continue  # Skip this iteration if transformed_image is None


#             # Reshape ảnh dựa trên model được sử dụng
#             # if config.dataset == "CMMD":
#             #     transformed_image = transformed_image.reshape(1, config.CMMD_IMG_SIZE['HEIGHT'], config.CMMD_IMG_SIZE['WIDTH'], 1)
#             # elif config.dataset == "INbreast":
#             #     transformed_image = transformed_image.reshape(1, config.INBREAST_IMG_SIZE['HEIGHT'], config.INBREAST_IMG_SIZE['WIDTH'], 1)
#             #     if hasattr(config, 'INBREAST_IMG_SIZE'):
#             #         transformed_image = transformed_image.reshape(1, config.INBREAST_IMG_SIZE['HEIGHT'], config.INBREAST_IMG_SIZE["WIDTH"], 1)
#             #     else:
#             #         # Mặc định 224x224 nếu không có cấu hình riêng
#             #         transformed_image = transformed_image.reshape(1, 224, 224, 1)
#             # if config.dataset == "CMMD":
#             #     transformed_image = transformed_image.reshape(1, config.CMMD_IMG_SIZE['HEIGHT'], config.CMMD_IMG_SIZE['WIDTH'], 1)
#             # elif config.dataset == "INbreast":
#             #     transformed_image = transformed_image.reshape(1, config.INBREAST_IMG_SIZE['HEIGHT'], config.INBREAST_IMG_SIZE['WIDTH'], 1)
#             if config.is_roi or config.model == "CNN":
#                 transformed_image = transformed_image.reshape(1, config.ROI_IMG_SIZE['HEIGHT'],
#                                                              config.ROI_IMG_SIZE["WIDTH"], 1)
#             elif config.model == "VGG" or config.model == "Inception":
#                 transformed_image = transformed_image.reshape(1, config.MINI_MIAS_IMG_SIZE['HEIGHT'],
#                                                              config.MINI_MIAS_IMG_SIZE["WIDTH"], 1)
#             elif config.model == "VGG-common":
#                 transformed_image = transformed_image.reshape(1, config.VGG_IMG_SIZE['HEIGHT'],
#                                                              config.VGG_IMG_SIZE["WIDTH"], 1)
#             elif config.model == "ResNet":
#                 transformed_image = transformed_image.reshape(1, config.RESNET_IMG_SIZE['HEIGHT'],
#                                                              config.RESNET_IMG_SIZE["WIDTH"], 1)
#             elif config.model == "MobileNet":
#                 transformed_image = transformed_image.reshape(1, config.MOBILE_NET_IMG_SIZE['HEIGHT'],
#                                                              config.MOBILE_NET_IMG_SIZE["WIDTH"], 1)
#             elif config.model == "Inception":
#                 transformed_image = transformed_image.reshape(1, config.INCEPTION_IMG_SIZE['HEIGHT'],
#                                                              config.INCEPTION_IMG_SIZE["WIDTH"], 1)
#             elif config.model == "DenseNet":
#                 transformed_image = transformed_image.reshape(1, config.DENSE_NET_IMG_SIZE['HEIGHT'],
#                                                              config.DENSE_NET_IMG_SIZE["WIDTH"], 1)

#             # Thêm ảnh và nhãn đã biến đổi vào tập dữ liệu
#             images_with_transforms = np.append(images_with_transforms, transformed_image, axis=0)
#             transformed_label = label.reshape(1, len(label))
#             labels_with_transforms = np.append(labels_with_transforms, transformed_label, axis=0)

#     return images_with_transforms, labels_with_transforms
# # from skimage.transform import resize

# # def generate_image_transforms(images, labels):
# #     """
# #     Oversample data by transforming existing images.
# #     Adjusted for CMMD and INbreast datasets.
    
# #     :param images: input images
# #     :param labels: input labels
# #     :return: updated list of images and labels with extra transformed images and labels
# #     """
# #     augmentation_multiplier = 3 if config.dataset in ["CMMD", "INbreast"] else 1

# #     images_with_transforms = images.copy()
# #     labels_with_transforms = labels.copy()

# #     available_transforms = {
# #         'rotate': random_rotation,
# #         'noise': random_noise,
# #         'horizontal_flip': horizontal_flip,
# #         'shear': random_shearing,
# #         'gamma_correction': gamma_correction,
# #         'zoom': random_zoom,
# #         'contrast': random_contrast,
# #     }

# #     class_balance = get_class_balances(labels)
    
# #     max_count = max(class_balance) * augmentation_multiplier
# #     to_add = [max_count - count for count in class_balance]

# #     for i in range(len(to_add)):
# #         if int(to_add[i]) == 0:
# #             continue
        
# #         label_one_hot = np.zeros(len(to_add))
# #         label_one_hot[i] = 1

# #         indices = [j for j, x in enumerate(labels) if np.array_equal(x, label_one_hot)]
# #         indiv_class_images = [images[j] for j in indices]

# #         for k in range(int(to_add[i])):
# #             transformed_image = create_individual_transform(indiv_class_images[k % len(indiv_class_images)], available_transforms)
            
# #             # Resize lại ảnh sau mỗi phép biến đổi
# #             transformed_image = resize(transformed_image, (config.INBREAST_IMG_SIZE['HEIGHT'], config.INBREAST_IMG_SIZE['WIDTH']), mode='reflect')
# #             transformed_image = transformed_image.reshape(1, config.INBREAST_IMG_SIZE['HEIGHT'], config.INBREAST_IMG_SIZE['WIDTH'], 1)

# #             images_with_transforms = np.append(images_with_transforms, transformed_image, axis=0)
# #             transformed_label = label_one_hot.reshape(1, len(label_one_hot))
# #             labels_with_transforms = np.append(labels_with_transforms, transformed_label, axis=0)

# #     return images_with_transforms, labels_with_transforms


# def random_rotation(image_array: np.ndarray):
#     """
#     Randomly rotate the image
#     Originally written as a group for the common pipeline.
#     :param image_array: input image
#     :return: randomly rotated image
#     """
#     random_degree = random.uniform(-20, 20)
#     return sk.transform.rotate(image_array, random_degree)


# def random_noise(image_array: np.ndarray):
#     """
#     Add a random amount of noise to the image.
#     Originally written as a group for the common pipeline.
#     :param image_array: input image.
#     :return: image with added random noise.
#     """
#     return sk.util.random_noise(image_array)


# def horizontal_flip(image_array: np.ndarray):
#     """
#     Flip image horizontally.
#     Originally written as a group for the common pipeline.
#     :param image_array: input image.
#     :return: horizantally flipped image.
#     """
#     return image_array[:, ::-1]


# def random_shearing(image_array: np.ndarray):
#     """
#     Add random amount of shearing to image.
#     :param image_array: input image.
#     :return: sheared image.
#     """
#     random_degree = random.uniform(-0.2, 0.2)
#     tf = sk.transform.AffineTransform(shear=random_degree)
#     return sk.transform.warp(image_array, tf, order=1, preserve_range=True, mode='wrap')


# def create_individual_transform(image: np.array, transforms: dict):
#     """
#     Create transformation of an individual image.
#     Originally written as a group for the common pipeline.
#     :param image: input image
#     :param transforms: the possible transforms to do on the image
#     :return: transformed image
#     """
#     num_transformations_to_apply = random.randint(1, len(transforms))
#     num_transforms = 0
#     transformed_image = None
#     while num_transforms <= num_transformations_to_apply:
#         key = random.choice(list(transforms))
#         transformed_image = transforms[key](image)
#         num_transforms += 1

#     return transformed_image

# # Thêm hàm biến đổi mới cho CMMD
# def random_zoom(image_array):
#     """Zoom ngẫu nhiên vào vùng trung tâm của ảnh."""
#     h, w = image_array.shape[:2]
#     zoom_factor = random.uniform(0.8, 1.0)
#     zoomed_image = sk.transform.rescale(image_array[int(h*(1-zoom_factor))//2:int(h*(1+zoom_factor))//2,
#                                                     int(w*(1-zoom_factor))//2:int(w*(1+zoom_factor))//2],
#                                         scale=(h/w), mode='reflect')
#     return sk.transform.resize(zoomed_image, (h, w))

# def random_contrast(image_array):
#     """Thay đổi độ tương phản ngẫu nhiên."""
#     factor = random.uniform(0.8, 1.2)
#     return sk.exposure.adjust_gamma(image_array, gamma=factor)

# def gamma_correction(image_array):
#     """Áp dụng gamma correction để tăng cường độ tương phản."""
#     gamma = random.uniform(0.8, 1.5)
#     return sk.exposure.adjust_gamma(image_array, gamma)

# # Thêm các hàm biến đổi mới cho INbreast
# def random_zoom(image_array: np.ndarray):
#     """
#     Áp dụng zoom ngẫu nhiên vào vùng trung tâm của ảnh.
#     Đặc biệt hữu ích cho ảnh mammogram để tập trung vào vùng tổn thương.
    
#     :param image_array: input image
#     :return: zoomed image
#     """
#     h, w = image_array.shape[:2]
#     center = (w // 2, h // 2)
#     zoom_factor = random.uniform(0.8, 0.95)
#     zoom_size = int(min(h, w) * zoom_factor)
    
#     # Đảm bảo vùng zoom không vượt quá kích thước ảnh
#     top = max(0, center[1] - zoom_size//2)
#     bottom = min(h, center[1] + zoom_size//2)
#     left = max(0, center[0] - zoom_size//2)
#     right = min(w, center[0] + zoom_size//2)
    
#     zoomed = image_array[top:bottom, left:right]
#     return sk.transform.resize(zoomed, (h, w))

# # def random_contrast(image_array: np.ndarray):
# #     """
# #     Điều chỉnh độ tương phản của ảnh một cách ngẫu nhiên.
# #     Hữu ích cho việc làm nổi bật các đặc trưng trong ảnh mammogram.
    
# #     :param image_array: input image
# #     :return: contrast adjusted image
# #     """
# #     # Điều chỉnh độ tương phản bằng cách kéo giãn histogram
# #     p_low = random.uniform(0.01, 0.1)
# #     p_high = random.uniform(0.9, 0.99)
    
# #     # Đảm bảo giá trị pixel nằm trong khoảng [0, 1]
# #     image_array = np.clip(image_array, 0, 1)
    
# #     # Áp dụng contrast stretching
# #     v_min, v_max = np.percentile(image_array, [p_low*100, p_high*100])
# #     return sk.exposure.rescale_intensity(image_array, in_range=(v_min, v_max), out_range=(0, 1))


# # # Cập nhật hàm get_class_balances để hỗ trợ cả CMMD và INbreast
# # def get_class_balances(y_vals):
# #     """
# #     Count occurrences of each class.
# #     Updated to support CMMD and INbreast datasets.
    
# #     :param y_vals: labels
# #     :return: array count of each class
# #     """
# #     if config.dataset == "CMMD" or config.dataset == "INbreast":
# #         num_classes = 2  # benign và malignant
# #         counts = np.zeros(num_classes)
# #         for y_val in y_vals:
# #             if np.array_equal(y_val, [1, 0]):  # benign
# #                 counts[0] += 1
# #             elif np.array_equal(y_val, [0, 1]):  # malignant
# #                 counts[1] += 1
# #     elif config.dataset == "mini-MIAS":
# #         num_classes = len(y_vals[0])
# #         counts = np.zeros(num_classes)
# #         for y_val in y_vals:
# #             for i in range(num_classes):
# #                 counts[i] += y_val[i]
# #     elif config.dataset == "mini-MIAS-binary":
# #         num_classes = 2
# #         counts = np.zeros(num_classes)
# #         for y_val in y_vals:
# #             if y_val == 0:
# #                 counts[0] += 1
# #             elif y_val == 1:
# #                 counts[1] += 1
# #     return counts.tolist()

# def get_class_balances(y_vals):
#     """
#     Count occurrences of each class.
    
#     :param y_vals: labels
#     :return: array count of each class
#     """
#     if config.dataset in ["CMMD", "INbreast"]:
#         num_classes = 2  # benign và malignant
#         counts = np.zeros(num_classes)
#         for y_val in y_vals:
#             if np.array_equal(y_val, [1, 0]):  # benign
#                 counts[0] += 1
#             elif np.array_equal(y_val, [0, 1]):  # malignant
#                 counts[1] += 1
#     else:
#         num_classes = len(y_vals[0])
#         counts = np.zeros(num_classes)
#         for y_val in y_vals:
#             for i in range(num_classes):
#                 counts[i] += y_val[i]
    
#     return counts.tolist()

from typing import Tuple
# import numpy as np
# import random
# import config

# def generate_image_transforms(images: np.ndarray,
#                               labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     - images: np.ndarray, shape (N, H, W, C) or (N, H, W)
#     - labels: np.ndarray, shape (N,) for binary or (N, num_classes) for one-hot
#     Returns augmented images & labels với logic mix ngẫu nhiên 1–N phép.
#     """
#     available_transforms = {
#         'rotate': random_rotation,
#         'noise': random_noise,
#         'horizontal_flip': horizontal_flip,
#         'shear': random_shearing
#     }
#     # 1. Khởi tạo lists
#     imgs = list(images)
#     labs = list(labels)

#     # 2. Oversampling multiplier
#     # choose multiplier
#     if config.dataset == "INbreast": # THÊM HOẶC CHỈNH SỬA ĐIỀU KIỆN NÀY
#         multiplier = 3
#     elif config.dataset in ["mini-MIAS-binary", "CMMD-binary"]: # Giữ nguyên logic cũ nếu có
#         multiplier = 3
#     else:
#         multiplier = 1

#     # 3. Tính to_add cho mỗi lớp
#     class_counts = get_class_balances(labels)  # e.g. [n0, n1, ...]
#     max_count = max(class_counts) * multiplier
#     to_add = [int(max_count - cnt) for cnt in class_counts]

#     # 4. Với từng lớp i, tạo thêm đúng to_add[i] ảnh
#     for i, add_count in enumerate(to_add):
#         if add_count <= 0:
#             continue

#         # 4.1 Lấy indices của lớp i
#         if label_is_binary(labels):
#             indices = [j for j, v in enumerate(labels) if v == i]
#             base_label = i
#         else:
#             vec = np.zeros(len(class_counts), dtype=labels.dtype)
#             vec[i] = 1
#             indices = [j for j, v in enumerate(labels) if np.array_equal(v, vec)]
#             base_label = vec

#         if not indices:
#             continue

#         # 4.2 Sinh từng ảnh mới bằng mix ngẫu nhiên 1–len(available_transforms) phép
#         for k in range(add_count):
#             orig = images[indices[k % len(indices)]]
#             ops = random.randint(1, len(available_transforms))
#             aug = orig.copy()
#             for _ in range(ops):
#                 func = random.choice(list(available_transforms.values()))
#                 aug = func(aug)
#             imgs.append(aug)
#             labs.append(base_label.copy() if not label_is_binary(labels) else base_label)

#     # 5. Trả về mảng
#     return np.stack(imgs), np.array(labs)
