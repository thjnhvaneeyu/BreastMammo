import random
import numpy as np
import skimage as sk
import skimage.transform
import skimage.color      # <<< THÊM IMPORT NÀY
import skimage.exposure   # <<< THÊM IMPORT NÀY
import skimage.util       # <<< THÊM IMPORT NÀY
import config             # Đảm bảo file config.py tồn tại và đúng cấu trúc

# ==============================================================
# 1. Hàm Resize và đảm bảo Grayscale (Lấy từ paste-3.txt)
# ==============================================================
def resize_before_reshape(image, target_height, target_width):
    """Resize ảnh về kích thước mục tiêu và đảm bảo output là grayscale (H, W, 1)."""
    current_height, current_width = image.shape[0], image.shape[1]

    # Đảm bảo ảnh đầu vào là float để resize hoạt động tốt
    img_float = sk.util.img_as_float(image)

    # Resize nếu kích thước khác biệt
    if current_height != target_height or current_width != target_width:
        # Preserve range giữ giá trị trong [0,1] hoặc gốc nếu không phải float
        # Anti-aliasing nên được dùng khi giảm kích thước
        anti_aliasing = (current_height > target_height) or (current_width > target_width)
        resized_image = sk.transform.resize(img_float, (target_height, target_width),
                                            mode='reflect',
                                            anti_aliasing=anti_aliasing,
                                            preserve_range=True) # Giữ range giá trị gốc
    else:
        resized_image = img_float # Không cần resize

    # Đảm bảo ảnh là grayscale và có đúng shape (H, W, 1)
    if resized_image.ndim == 3:
        # Nếu resize trả về RGB hoặc RGBA
        if resized_image.shape[2] == 3:
            resized_image = sk.color.rgb2gray(resized_image)
        elif resized_image.shape[2] == 4: # RGBA
            resized_image = sk.color.rgba2rgb(resized_image) # Chuyển sang RGB trước
            resized_image = sk.color.rgb2gray(resized_image)
        # Loại bỏ kênh thừa nếu kênh cuối cùng bằng 1 nhưng ndim=3 (hiếm gặp)
        elif resized_image.shape[2] == 1:
             resized_image = resized_image.squeeze(axis=-1) # Bỏ chiều kênh = 1

    # Luôn đảm bảo có chiều kênh cuối cùng là 1 nếu ảnh đang là 2D
    if resized_image.ndim == 2:
        resized_image = np.expand_dims(resized_image, axis=-1)

    # Kiểm tra shape cuối cùng một lần nữa và ép shape nếu cần (phòng ngừa lỗi lạ)
    if resized_image.shape != (target_height, target_width, 1):
         print(f"Warning: resize_before_reshape correcting shape from {resized_image.shape} to {(target_height, target_width, 1)}")
         # Cố gắng reshape, nhưng cần cẩn thận nếu số phần tử không khớp
         try:
             resized_image = resized_image.reshape(target_height, target_width, 1)
         except ValueError as e:
             print(f"Error reshaping in resize_before_reshape: {e}. Returning original shape.")
             # Trong trường hợp này, có thể trả về ảnh gốc hoặc raise lỗi tùy logic
             # Ở đây ta trả về ảnh đã resize nhưng chưa reshape lại lần cuối
             if resized_image.ndim == 2:
                 resized_image = np.expand_dims(resized_image, axis=-1)


    # Trả về float32 là chuẩn cho các framework DL
    return resized_image.astype(np.float32)

# ==============================================================
# 2. Các hàm transformation cá nhân (Lấy từ paste-3.txt)
# ==============================================================
def random_rotation(image_array: np.ndarray):
    """Xoay ảnh ngẫu nhiên."""
    random_degree = random.uniform(-20, 20)
    # resize=False để giữ nguyên kích thước sau khi xoay
    # preserve_range=True để giữ nguyên dải giá trị pixel
    return sk.transform.rotate(image_array, random_degree, resize=False, mode='reflect', preserve_range=True)

def random_noise(image_array: np.ndarray):
    """Thêm nhiễu ngẫu nhiên."""
    # Đảm bảo ảnh là float trong [0, 1] trước khi thêm nhiễu
    img_float = sk.util.img_as_float(np.clip(image_array, 0, 1 if image_array.max() <=1 else 255))
    noisy_image = sk.util.random_noise(img_float, mode='gaussian', var=random.uniform(0.001, 0.01))
    # Trả về cùng kiểu dữ liệu với đầu vào nếu cần, hoặc luôn là float32
    return noisy_image.astype(np.float32)

def horizontal_flip(image_array: np.ndarray):
    """Lật ảnh ngang."""
    return image_array[:, ::-1]

def random_shearing(image_array: np.ndarray):
    """Biến dạng cắt (shear) ngẫu nhiên."""
    random_degree = random.uniform(-0.2, 0.2)
    tf = sk.transform.AffineTransform(shear=random_degree)
    # preserve_range=True để giữ dải giá trị
    return sk.transform.warp(image_array, tf, order=1, preserve_range=True, mode='wrap')

def gamma_correction(image_array: np.ndarray):
    """Hiệu chỉnh Gamma ngẫu nhiên."""
    gamma = random.uniform(0.8, 1.5)
    # Đảm bảo giá trị trong khoảng hợp lệ cho gamma correction
    img_clipped = np.clip(image_array, 0, 1 if image_array.max() <= 1 else image_array.max())
    return sk.exposure.adjust_gamma(img_clipped, gamma)

def random_zoom(image_array: np.ndarray):
    """Zoom ngẫu nhiên (zoom out rồi resize lại)."""
    h, w = image_array.shape[:2]
    zoom_factor = random.uniform(0.8, 0.99) # Zoom out nhẹ
    crop_h = int(h * zoom_factor)
    crop_w = int(w * zoom_factor)

    # Tính toán điểm bắt đầu crop ngẫu nhiên hoặc từ tâm
    # start_h = random.randint(0, h - crop_h)
    # start_w = random.randint(0, w - crop_w)
    start_h = (h - crop_h) // 2 # Crop từ tâm
    start_w = (w - crop_w) // 2 # Crop từ tâm

    zoomed_out = image_array[start_h : start_h + crop_h, start_w : start_w + crop_w]

    # Resize back to original size, đảm bảo anti-aliasing khi phóng to
    return sk.transform.resize(zoomed_out, (h, w), mode='reflect', anti_aliasing=True, preserve_range=True)

def random_contrast(image_array: np.ndarray):
    """Thay đổi độ tương phản ngẫu nhiên bằng giãn lược đồ (histogram stretching)."""
    p_low = random.uniform(1, 10)   # Percentile thấp
    p_high = random.uniform(90, 99) # Percentile cao
    # Đảm bảo ảnh là float và trong khoảng [0, 1]
    img_float = sk.util.img_as_float(np.clip(image_array, 0, 1 if image_array.max() <=1 else 255))
    v_min, v_max = np.percentile(img_float, [p_low, p_high])
    # Giãn cường độ về khoảng [0, 1]
    return sk.exposure.rescale_intensity(img_float, in_range=(v_min, v_max), out_range=(0.0, 1.0)).astype(np.float32)

# ==============================================================
# 3. Hàm tạo biến đổi kết hợp (Lấy từ paste-3.txt)
# ==============================================================
def create_individual_transform(image: np.array, transforms: dict):
    """
    Tạo biến đổi cho một ảnh cụ thể bằng cách áp dụng ngẫu nhiên một số phép biến đổi.
    Đảm bảo output là ảnh grayscale (H, W, 1).

    :param image: ảnh đầu vào, kỳ vọng là (H, W, 1)
    :param transforms: dict các hàm biến đổi có sẵn
    :return: ảnh đã biến đổi (H, W, 1), float32
    """
    if image.ndim == 2: # Nếu đầu vào là 2D, thêm kênh
        image = np.expand_dims(image, axis=-1)
    elif image.ndim == 3 and image.shape[-1] != 1:
        print(f"Warning: Input to create_individual_transform has shape {image.shape}. Forcing grayscale.")
        image = image[..., :1] # Chỉ lấy kênh đầu tiên nếu không phải grayscale

    if image.shape[-1] != 1:
         print(f"Error: Cannot proceed with create_individual_transform, input shape not (H, W, 1): {image.shape}")
         return image.astype(np.float32) # Trả lại ảnh gốc

    num_transformations_to_apply = random.randint(1, len(transforms))
    transformed_image = image.copy() # Bắt đầu với bản sao

    applied_keys = random.sample(list(transforms.keys()), num_transformations_to_apply)

    for key in applied_keys:
        # Áp dụng transform
        transformed_image = transforms[key](transformed_image)

        # --- KIỂM TRA và FIX Shape/Kênh SAU MỖI TRANSFORM ---
        if transformed_image is None: # Nếu transform bị lỗi
            print(f"Error: Transform '{key}' returned None. Reverting to previous state.")
            transformed_image = image.copy() # Lấy lại ảnh gốc hoặc ảnh trước đó
            continue

        # 1. Đảm bảo ảnh vẫn là grayscale
        if transformed_image.ndim == 3 and transformed_image.shape[-1] != 1:
            print(f"Warning: Transform '{key}' changed channels to {transformed_image.shape[-1]}. Converting back to grayscale.")
            if transformed_image.shape[-1] == 3:
                 gray_img = sk.color.rgb2gray(transformed_image)
                 transformed_image = np.expand_dims(gray_img, axis=-1) # rgb2gray trả về 2D
            elif transformed_image.shape[-1] == 4:
                 gray_img = sk.color.rgb2gray(sk.color.rgba2rgb(transformed_image))
                 transformed_image = np.expand_dims(gray_img, axis=-1)
            else: # Trường hợp khác, lấy kênh đầu tiên
                 print(f"Warning: Taking first channel of shape {transformed_image.shape}")
                 transformed_image = transformed_image[..., 0:1] # Giữ lại ndim=3

        # 2. Đảm bảo có chiều kênh nếu transform trả về (H, W)
        elif transformed_image.ndim == 2:
             transformed_image = np.expand_dims(transformed_image, axis=-1)

        # 3. Kiểm tra ndim hợp lệ
        if transformed_image.ndim != 3 or transformed_image.shape[-1] != 1:
             print(f"Error: Transform '{key}' resulted in invalid shape {transformed_image.shape}. Attempting to restore.")
             # Cố gắng khôi phục về shape ảnh gốc nếu lỗi nghiêm trọng
             transformed_image = image.copy()
             break # Dừng áp dụng transform khác cho ảnh này

    # Kiểm tra lần cuối trước khi trả về
    if transformed_image.ndim != 3 or transformed_image.shape[-1] != 1:
         print(f"Error: create_individual_transform final shape is {transformed_image.shape}. Returning original.")
         return image.copy().astype(np.float32) # Trả về bản gốc nếu không fix được

    return transformed_image.astype(np.float32)

# ==============================================================
# 4. Hàm đếm số lượng lớp (Lấy từ paste-3.txt, có cải tiến)
# ==============================================================
def get_class_balances(y_vals):
    """Đếm số lượng mẫu trong mỗi lớp. Xử lý one-hot và integer labels."""
    counts = []
    if y_vals is None or len(y_vals) == 0:
        print("Warning: y_vals is empty in get_class_balances.")
        return counts

    y_array = np.array(y_vals) # Chuyển sang numpy array để dễ thao tác

    # Kiểm tra cấu trúc y_vals để xử lý phù hợp
    if y_array.ndim > 1 and y_array.shape[1] > 1: # Có vẻ là One-hot encoded (N, num_classes)
        num_classes = y_array.shape[1]
        counts = np.sum(y_array, axis=0).astype(int).tolist()
    elif y_array.ndim == 1 or y_array.shape[1] == 1: # Có vẻ là nhãn dạng scalar (integer)
        if y_array.ndim == 2 and y_array.shape[1] == 1: # Ép về 1D nếu shape là (N, 1)
            y_array = y_array.flatten()

        unique_values, unique_counts = np.unique(y_array, return_counts=True)
        # Giả sử các lớp là 0, 1, ..., max_val
        if len(unique_values) > 0:
            try:
                 max_val = int(np.max(unique_values))
                 num_classes = max_val + 1
                 counts = np.zeros(num_classes, dtype=int)
                 for val, count in zip(unique_values.astype(int), unique_counts):
                      if 0 <= val < num_classes: # Đảm bảo index hợp lệ
                          counts[val] = count
                 counts = counts.tolist()
            except ValueError:
                 print(f"Warning: Could not determine classes from labels like: {y_array[:5]}")
                 counts = [] # Không xác định được lớp
        else:
             counts = [] # Không có nhãn nào
    else:
        print(f"Warning: Unexpected label format in get_class_balances. Shape: {y_array.shape}")
        counts = []

    return counts

# ==============================================================
# 5. Hàm helper lấy kích thước reshape (Lấy từ paste-3.txt)
# ==============================================================
def get_reshape_size_from_config():
    """ Lấy target (height, width) từ config cho việc reshape cuối cùng sau augmentation. """
    default_h, default_w = 224, 224 # Kích thước mặc định phổ biến
    try:
        dataset = getattr(config, 'dataset', None)
        model_name = getattr(config, 'model', None)
        is_roi = getattr(config, 'is_roi', False)

        if dataset == "CMMD":
            size = getattr(config, 'CMMD_IMG_SIZE', {'HEIGHT': default_h, 'WIDTH': default_w})
            return size['HEIGHT'], size['WIDTH']
        elif dataset == "INbreast":
             size = getattr(config, 'INBREAST_IMG_SIZE', {'HEIGHT': default_h, 'WIDTH': default_w})
             return size['HEIGHT'], size['WIDTH']
        elif is_roi or model_name == "CNN":
             size = getattr(config, 'ROI_IMG_SIZE', {'HEIGHT': default_h, 'WIDTH': default_w})
             return size['HEIGHT'], size['WIDTH']
        # Thêm các model khác nếu cần, sử dụng getattr để an toàn
        elif model_name in ["VGG", "Inception", "VGG-common", "ResNet", "MobileNet", "DenseNet"]:
             # Tìm size tương ứng hoặc dùng default
             size_attr_name = f"{model_name.upper().replace('-', '_')}_IMG_SIZE"
             # Xử lý trường hợp MINI_MIAS riêng
             if model_name in ["VGG", "Inception"] and hasattr(config, 'MINI_MIAS_IMG_SIZE'):
                 size_attr_name = 'MINI_MIAS_IMG_SIZE'
             size = getattr(config, size_attr_name, {'HEIGHT': default_h, 'WIDTH': default_w})
             return size.get('HEIGHT', default_h), size.get('WIDTH', default_w)

        else:
            print(f"Warning: No specific reshape size found for model '{model_name}' / dataset '{dataset}'. Using default {default_h}x{default_w}.")
            return default_h, default_w

    except AttributeError as e:
        print(f"Warning: Config attribute missing for image size ({e}). Using default {default_h}x{default_w}.")
        return default_h, default_w
    except Exception as e:
        print(f"Error getting reshape size from config: {e}. Using default {default_h}x{default_w}.")
        return default_h, default_w

# ==============================================================
# 6. Hàm Augmentation chính (Lấy từ paste-3.txt, có cải tiến)
# ==============================================================
def generate_image_transforms(images, labels):
    """
    Oversample data by transforming existing images. Ensures output is (N, H, W, 1).
    Handles different datasets and label types.

    :param images: input images numpy array, kỳ vọng là (N, H, W, 1) or (N, H, W)
    :param labels: input labels numpy array (one-hot hoặc integer)
    :return: updated numpy arrays of images (N', H', W', 1) and labels (N', num_classes) or (N',)
    """
    print(f"Starting augmentation. Initial shapes: images={images.shape}, labels={labels.shape}")

    # --- Setup Multiplier ---
    augmentation_multiplier = 1 # Mặc định không tăng cường quá nhiều nếu không phải dataset cụ thể
    dataset_name = getattr(config, 'dataset', '')
    if dataset_name == "mini-MIAS-binary": augmentation_multiplier = 3
    elif dataset_name == "CMMD": augmentation_multiplier = 3
    elif dataset_name == "INbreast": augmentation_multiplier = 3
    print(f"Augmentation multiplier set to: {augmentation_multiplier} for dataset: '{dataset_name}'")

    # --- Sử dụng list để append hiệu quả ---
    # Chuyển ảnh đầu vào thành list các ảnh (H, W, 1)
    initial_images_list = []
    for img in images:
        if img.ndim == 2:
            initial_images_list.append(np.expand_dims(img, axis=-1))
        elif img.ndim == 3 and img.shape[-1] == 1:
            initial_images_list.append(img)
        elif img.ndim == 3 and img.shape[-1] > 1: # Nếu ảnh gốc là màu, chuyển sang xám
             print("Warning: Initial image seems to be color, converting to grayscale for augmentation.")
             gray_img = sk.color.rgb2gray(img) if img.shape[-1]==3 else sk.color.rgb2gray(sk.color.rgba2rgb(img))
             initial_images_list.append(np.expand_dims(gray_img, axis=-1))
        else:
             print(f"Warning: Skipping initial image with unexpected shape: {img.shape}")
             continue

    # Labels cũng chuyển thành list
    initial_labels_list = list(labels)

    if not initial_images_list or len(initial_images_list) != len(initial_labels_list):
        print("Error: Mismatch between processed images and labels or no images to process. Returning original data.")
        return images, labels

    # --- Define Transforms ---
    available_transforms = {
        'rotate': random_rotation, 'noise': random_noise, 'horizontal_flip': horizontal_flip,
        'shear': random_shearing, 'gamma_correction': gamma_correction,
        'zoom': random_zoom, 'contrast': random_contrast,
    }

    # --- Calculate Class Balance and Augmentation Needs ---
    class_balance = get_class_balances(labels)
    if not class_balance:
         print("Warning: Could not determine class balance. Skipping augmentation.")
         return images, labels

    num_classes = len(class_balance)
    to_add = np.zeros(num_classes, dtype=int)
    target_count = 0

    # Logic cân bằng lớp (tính toán số lượng cần thêm cho mỗi lớp)
    if dataset_name == "CMMD" and num_classes == 2:
        # Tăng lớp 0 (benign) lên bằng lớp 1 (malignant)
        target_count = class_balance[1]
        to_add[0] = max(0, target_count - class_balance[0]) * (augmentation_multiplier -1) # Chỉ thêm phần chênh lệch nhân multi
    elif dataset_name == "INbreast" and num_classes == 2:
        # Tăng lớp 1 (malignant) lên bằng lớp 0 (benign)
        target_count = class_balance[0]
        to_add[1] = max(0, target_count - class_balance[1]) * (augmentation_multiplier -1)
    else: # Chiến lược mặc định: tăng tất cả các lớp thiểu số lên bằng lớp đa số
        if class_balance:
             max_count = max(class_balance)
             target_count = max_count # Mục tiêu là số lượng của lớp đa số
             for i in range(num_classes):
                  to_add[i] = max(0, target_count - class_balance[i]) * (augmentation_multiplier -1) # Thêm phần chênh lệch
        else:
             target_count = 0

    print(f"Class counts: {class_balance}")
    print(f"Target count per class (approx): {target_count * augmentation_multiplier if target_count > 0 else 'N/A'}")
    print(f"Images to generate per class: {to_add.tolist()}")

    # --- Lấy kích thước ảnh mục tiêu ---
    target_height, target_width = get_reshape_size_from_config()
    print(f"Target image size after augmentation: ({target_height}, {target_width})")

    augmented_images_list = []
    augmented_labels_list = []

    # --- Vòng lặp Augmentation ---
    label_is_one_hot = labels.ndim > 1 and labels.shape[1] > 1

    for class_idx in range(num_classes):
        num_to_add_for_class = to_add[class_idx]
        if num_to_add_for_class <= 0:
            continue

        # Tạo nhãn cho lớp này (one-hot nếu cần)
        current_label_repr = np.zeros(num_classes, dtype=np.float32)
        current_label_repr[class_idx] = 1.0
        label_to_append = current_label_repr if label_is_one_hot else class_idx

        # Tìm ảnh gốc thuộc lớp này
        if label_is_one_hot:
             indices = [j for j, lab in enumerate(initial_labels_list) if np.array_equal(lab, current_label_repr)]
        else: # Integer labels
             indices = [j for j, lab in enumerate(initial_labels_list) if lab == class_idx]

        if not indices:
            print(f"Warning: No original images found for class index {class_idx} to augment from.")
            continue

        # Lấy list các ảnh gốc của lớp này (đã đảm bảo là H, W, 1)
        original_class_images = [initial_images_list[j] for j in indices]
        num_available_originals = len(original_class_images)
        print(f"Augmenting class {class_idx}: Found {num_available_originals} originals. Generating {num_to_add_for_class} new images.")

        for k in range(num_to_add_for_class):
            # Chọn ảnh gốc để transform (xoay vòng)
            original_image = original_class_images[k % num_available_originals] # Shape (H, W, 1)

            # 1. Tạo ảnh biến đổi (đảm bảo trả về H, W, 1)
            transformed_image = create_individual_transform(original_image, available_transforms)

            # 2. Resize ảnh *sau khi* transform về kích thước mục tiêu
            # Hàm này đảm bảo trả về (target_height, target_width, 1)
            resized_transformed_image = resize_before_reshape(transformed_image, target_height, target_width)

            # 3. Thêm ảnh và nhãn vào list kết quả
            augmented_images_list.append(resized_transformed_image) # Đã có shape (H, W, 1)
            augmented_labels_list.append(label_to_append)

    # --- Kết hợp dữ liệu gốc và dữ liệu đã tăng cường ---
    final_images_list = initial_images_list + augmented_images_list
    final_labels_list = initial_labels_list + augmented_labels_list

    # Chuyển list thành numpy array
    # Reshape ảnh để có chiều batch (N, H, W, 1)
    final_images = np.array(final_images_list) # Sẽ có shape (N', H', W', 1)
    final_labels = np.array(final_labels_list) # Sẽ có shape (N', num_classes) hoặc (N',)

    print("-" * 30)
    print(f"Augmentation complete.")
    print(f"Initial image count: {len(initial_images_list)}")
    print(f"Generated image count: {len(augmented_images_list)}")
    print(f"Final image count: {len(final_images)}")
    print(f"Final image shape: {final_images.shape}")
    print(f"Final label shape: {final_labels.shape}")

    # Kiểm tra shape cuối cùng
    if final_images.ndim != 4 or final_images.shape[-1] != 1:
        # Cố gắng sửa lỗi shape cuối cùng nếu có thể
        if final_images.ndim == 3: # Thiếu kênh màu
            final_images = np.expand_dims(final_images, axis=-1)
            print(f"Corrected final image shape to: {final_images.shape}")
        else:
             # Raise lỗi nếu không sửa được, vì nó sẽ gây lỗi cho model
            raise ValueError(f"Data augmentation resulted in incorrect final image shape: {final_images.shape}. Expected (N, H, W, 1)")

    if len(final_images) != len(final_labels):
        raise ValueError(f"Mismatch between final images ({len(final_images)}) and labels ({len(final_labels)}) after augmentation.")

    return final_images, final_labels

