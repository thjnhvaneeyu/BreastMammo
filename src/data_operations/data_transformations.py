import random
import numpy as np
import skimage as sk
import skimage.transform
import config
import os

def generate_image_transforms(images, labels):
    """
    Oversample data by creating transformed copies of existing images.
    Applies random rotations, flips, noise, shearing to balance classes.
    """
    augmentation_multiplier = 1
    if config.dataset == "mini-MIAS-binary" or config.dataset == "CMMD-binary":
        augmentation_multiplier = 3  # tăng cường nhiều hơn cho bộ dữ liệu binary nếu mất cân bằng
    
    images_with_transforms = images
    labels_with_transforms = labels
    # Các phép biến đổi có thể áp dụng
    available_transforms = {
        'rotate': random_rotation,
        'noise': random_noise,
        'horizontal_flip': horizontal_flip,
        'shear': random_shearing
    }
    # Tính số lượng mẫu mỗi lớp và xác định cần thêm bao nhiêu mẫu để cân bằng
    class_balance = get_class_balances(labels)
    max_count = max(class_balance) * augmentation_multiplier
    to_add = [int(max_count - count) for count in class_balance]
    for i in range(len(to_add)):
        if to_add[i] <= 0:
            continue
        # Lấy các ảnh của lớp i
        if label_is_binary(labels):
            indices = [j for j, x in enumerate(labels) if x == i]
            base_label = i  # nhãn số (0 hoặc 1)
        else:
            # Tạo vector one-hot cho lớp i để so sánh
            label_vector = np.zeros(len(to_add)); label_vector[i] = 1
            indices = [j for j, x in enumerate(labels) if np.array_equal(x, label_vector)]
            base_label = label_vector
        if len(indices) == 0:
            continue
        indiv_class_images = [images[j] for j in indices]
        # Tạo các ảnh mới cho lớp i
        for k in range(to_add[i]):
            orig_img = indiv_class_images[k % len(indiv_class_images)]
            transformed_image = create_individual_transform(orig_img, available_transforms)
            # Đảm bảo shape của ảnh transform đúng (H, W, 1)
            if config.is_roi or config.model == "CNN":
                transformed_image = transformed_image.reshape(1, config.ROI_IMG_SIZE['HEIGHT'], config.ROI_IMG_SIZE["WIDTH"], 1)
            elif config.model == "VGG" or config.model == "Inception":
                transformed_image = transformed_image.reshape(1, config.MINI_MIAS_IMG_SIZE['HEIGHT'], config.MINI_MIAS_IMG_SIZE["WIDTH"], 1)
            elif config.model == "VGG-common":
                transformed_image = transformed_image.reshape(1, config.VGG_IMG_SIZE['HEIGHT'], config.VGG_IMG_SIZE["WIDTH"], 1)
            elif config.model == "ResNet":
                transformed_image = transformed_image.reshape(1, config.RESNET_IMG_SIZE['HEIGHT'], config.RESNET_IMG_SIZE["WIDTH"], 1)
            elif config.model == "MobileNet":
                transformed_image = transformed_image.reshape(1, config.MOBILE_NET_IMG_SIZE['HEIGHT'], config.MOBILE_NET_IMG_SIZE["WIDTH"], 1)
            elif config.model == "DenseNet" or config.model == "Inception":
                transformed_image = transformed_image.reshape(1, config.INCEPTION_IMG_SIZE['HEIGHT'], config.INCEPTION_IMG_SIZE["WIDTH"], 1)
            # Thêm ảnh mới và nhãn mới vào tập
            images_with_transforms = np.append(images_with_transforms, transformed_image, axis=0)
            # Tạo label tương ứng cho ảnh mới
            if label_is_binary(labels):
                new_label = np.array([base_label])  # giữ dạng số
            else:
                new_label = base_label.reshape(1, len(base_label))
            labels_with_transforms = np.append(labels_with_transforms, new_label, axis=0)
    return images_with_transforms, labels_with_transforms

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

def generate_image_transforms(images: np.ndarray, labels: np.ndarray):
    """
    Oversample data by creating transformed copies to balance classes.
    """
    # choose multiplier
    if config.dataset in ["mini-MIAS-binary", "CMMD_binary"]:
        multiplier = 3
    else:
        multiplier = 1

    imgs = images.copy()
    labs = labels.copy()

    # base transforms
    transforms = {
        'rotate': random_rotation,
        'noise': random_noise,
        'horizontal_flip': horizontal_flip,
        'shear': random_shearing
    }
    # # dataset-specific
    # if config.dataset in ["CMMD_binary", "CMMD"]:
    #     transforms.update({'zoom': random_zoom, 'contrast': random_contrast})
    # if config.dataset == "INbreast":
    #     transforms.update({'zoom': random_zoom, 'contrast': random_contrast})
    # dataset-specific: thêm zoom & contrast cho CMMD và INbreast
    if config.dataset in ["CMMD_binary", "CMMD-binary", "INbreast"]:
        transforms.update({
            'zoom': random_zoom,
            'contrast': random_contrast
        })

    balances = get_class_balances(labels)
    target = max(balances) * multiplier
    to_add = [target - b for b in balances]

    for cls_idx, n in enumerate(to_add):
        if n <= 0:
            continue
        # find indices
        if label_is_binary(labels):
            idxs = [i for i, l in enumerate(labels) if l == cls_idx]
            base_label = cls_idx
        else:
            one_hot = np.zeros(len(balances))
            one_hot[cls_idx] = 1
            idxs = [i for i, l in enumerate(labels) if np.array_equal(l, one_hot)]
            base_label = one_hot
        if not idxs:
            continue
        class_imgs = [images[i] for i in idxs]
        for i in range(n):
            orig = class_imgs[i % len(class_imgs)]
            transformed = create_individual_transform(orig, transforms)
            h, w = transformed.shape[:2]
            c = 1 if transformed.ndim == 2 else transformed.shape[2]
            batch_img = transformed.reshape((1, h, w, c))
            imgs = np.append(imgs, batch_img, axis=0)
            if label_is_binary(labels):
                labs = np.append(labs, np.array([base_label]), axis=0)
            else:
                labs = np.append(labs, base_label.reshape(1, -1), axis=0)

    return imgs, labs

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

def generate_image_transforms(images: np.ndarray,
                              labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    - images: np.ndarray, shape (N, H, W, C) or (N, H, W)
    - labels: np.ndarray, shape (N,) for binary or (N, num_classes) for one-hot
    Returns augmented images & labels với logic mix ngẫu nhiên 1–N phép.
    """
    available_transforms = {
        'rotate': random_rotation,
        'noise': random_noise,
        'horizontal_flip': horizontal_flip,
        'shear': random_shearing
    }
    # 1. Khởi tạo lists
    imgs = list(images)
    labs = list(labels)

    # 2. Oversampling multiplier
    multiplier = 3 if config.dataset in ["mini-MIAS-binary", "CMMD-binary"] else 1

    # 3. Tính to_add cho mỗi lớp
    class_counts = get_class_balances(labels)  # e.g. [n0, n1, ...]
    max_count = max(class_counts) * multiplier
    to_add = [int(max_count - cnt) for cnt in class_counts]

    # 4. Với từng lớp i, tạo thêm đúng to_add[i] ảnh
    for i, add_count in enumerate(to_add):
        if add_count <= 0:
            continue

        # 4.1 Lấy indices của lớp i
        if label_is_binary(labels):
            indices = [j for j, v in enumerate(labels) if v == i]
            base_label = i
        else:
            vec = np.zeros(len(class_counts), dtype=labels.dtype)
            vec[i] = 1
            indices = [j for j, v in enumerate(labels) if np.array_equal(v, vec)]
            base_label = vec

        if not indices:
            continue

        # 4.2 Sinh từng ảnh mới bằng mix ngẫu nhiên 1–len(available_transforms) phép
        for k in range(add_count):
            orig = images[indices[k % len(indices)]]
            ops = random.randint(1, len(available_transforms))
            aug = orig.copy()
            for _ in range(ops):
                func = random.choice(list(available_transforms.values()))
                aug = func(aug)

            # 4.3 Nếu cần reshape theo model/ROI
            if getattr(config, "is_roi", False) or config.model == "CNN":
                aug = aug.reshape(1,
                                  config.ROI_IMG_SIZE['HEIGHT'],
                                  config.ROI_IMG_SIZE['WIDTH'],
                                  1)
            elif config.model in ("VGG", "Inception"):
                aug = aug.reshape(1,
                                  config.MINI_MIAS_IMG_SIZE['HEIGHT'],
                                  config.MINI_MIAS_IMG_SIZE['WIDTH'],
                                  1)
            elif config.model == "VGG-common":
                aug = aug.reshape(1,
                                  config.VGG_IMG_SIZE['HEIGHT'],
                                  config.VGG_IMG_SIZE['WIDTH'],
                                  1)
            elif config.model == "ResNet":
                aug = aug.reshape(1,
                                  config.RESNET_IMG_SIZE['HEIGHT'],
                                  config.RESNET_IMG_SIZE['WIDTH'],
                                  1)
            elif config.model == "MobileNet":
                aug = aug.reshape(1,
                                  config.MOBILE_NET_IMG_SIZE['HEIGHT'],
                                  config.MOBILE_NET_IMG_SIZE['WIDTH'],
                                  1)
            elif config.model in ("DenseNet", "Inception"):
                aug = aug.reshape(1,
                                  config.INCEPTION_IMG_SIZE['HEIGHT'],
                                  config.INCEPTION_IMG_SIZE['WIDTH'],
                                  1)

            imgs.append(aug)
            labs.append(base_label.copy() if not label_is_binary(labels) else base_label)

    # 5. Trả về mảng
    return np.vstack(imgs), np.array(labs)
