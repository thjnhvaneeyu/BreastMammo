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
import tensorflow.keras.backend as K
from typing import Tuple
# src/data_operations/data_transformations.py
import albumentations as A
import random
import numpy as np
import skimage as sk
import skimage.transform
import skimage.color
import skimage.exposure
import skimage.util
import skimage.filters # Cho GaussianBlur và map_coordinates
from scipy.ndimage import map_coordinates
import tensorflow as tf
import cv2 # cv2.resize có thể được thay thế bằng skimage.transform.resize nếu muốn đồng nhất

import config # Đảm bảo file config.py tồn tại và đúng cấu trúc
def focal_loss_factory(alpha=0.25, gamma=2.0):
    """
    Factory for creating a focal loss function.
    This version is designed to work with one-hot encoded y_true
    and softmax predictions y_pred for binary or multi-class cases.
    """
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate cross_entropy for each class
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate p_t (probability of the true class)
        p_t = K.sum(y_true * y_pred, axis=-1) # Sum over class dimension

        # Calculate modulating factor (1 - p_t)^gamma
        modulating_factor = K.pow(1.0 - p_t, gamma)

        # Calculate alpha factor
        # For binary (num_classes=2), if y_true is [0,1] for Malignant (class 1)
        # alpha_t for Malignant = alpha, for Benign = 1-alpha
        # A more general way for multi-class is to have an alpha per class or apply alpha to positive samples.
        # Here, we apply alpha to the "true" class contributions.
        # The `alpha` parameter of the factory is typically the weight for the positive/minority class.
        # If using with num_classes=2, and y_true=[?, alpha_class_target_prob],
        # this alpha is applied when y_true_for_alpha_class = 1.
        
        # For this implementation, alpha will weight the loss of each sample.
        # If y_true is one-hot, let's assume alpha applies to the samples of the class
        # that is considered "positive". For binary, if malignant is class 1:
        # alpha_weights = tf.where(K.equal(K.argmax(y_true, axis=-1), 1), alpha, 1.0 - alpha) # if argmax=1 is positive
        # A simpler approach (as in many papers) is to scale the contribution of each class by its alpha_t
        # For focal loss, alpha usually balances importance of positive/negative examples.
        # FL(pt) = - alpha_t * (1 - pt)^gamma * log(pt)
        # Here, pt is the prob of the true class. The ce term above is -log(pt) for the true class.
        # So, we need to multiply ce by alpha_t * (1-pt)^gamma.
        # If alpha is for the "positive" class examples (e.g. malignant):
        
        # Assuming alpha is the weight for the class indicated by the second column in one-hot y_true
        # This is a common setup for binary classification where class 1 is the positive class.
        alpha_per_sample = tf.where(tf.equal(y_true[:, 1], 1.0), alpha, 1.0 - alpha)
        alpha_per_sample = K.expand_dims(alpha_per_sample, axis=-1) # Make it broadcastable with cross_entropy if summing later
        
        # If using multi-class, alpha might be a list/tensor of weights per class.
        # For now, this alpha logic is more suited for binary (as num_classes=2).
        # If num_classes > 2, alpha typically becomes a vector of class weights.
        # For binary as CCE with 2 classes, this application of alpha is one way.

        f_loss = modulating_factor * K.sum(cross_entropy, axis=-1) # Summing CE over classes first
        
        # If alpha is meant to balance positive/negative examples as a whole
        # weighted_loss = alpha_per_sample * f_loss # This would apply alpha to the sample's total loss
        
        # A more direct application for focal loss is:
        # loss = alpha_t * (1-p_t)^gamma * CE_t
        # where CE_t is the cross_entropy of the true class.
        # cross_entropy sum already handles the CE_t part for one-hot.
        # So, the loss for a sample is alpha_for_that_sample_type * (1-p_t)^gamma * (sum over class (y_true_c * -log(y_pred_c)))
        # If alpha is a scalar, it is often applied to the positive class instances.
        # Let's assume the alpha parameter is for the "positive" class contribution to the loss.
        # The most common formulation:
        focal_loss_unweighted = modulating_factor * K.sum(cross_entropy, axis=-1) # (1-pt)^gamma * CE

        # If y_true has shape (batch, 2), where y_true[:,1] is for malignant class
        # We want to apply `alpha` if it's a malignant sample, `1-alpha` if benign
        alpha_weight_per_sample = tf.where(tf.equal(y_true[:, 1], 1.0), alpha, 1.0 - alpha)
        
        weighted_focal_loss = alpha_weight_per_sample * focal_loss_unweighted

        return K.mean(weighted_focal_loss) # Mean over batch
        
    return focal_loss_fixed
# --- HÀM ELASTIC TRANSFORM (Đã implement ở lượt trước) ---
def elastic_transform(image_array_input: np.ndarray,
                      alpha: float,
                      sigma: float,
                      alpha_affine: float = 0.0,
                      random_state_seed=None) -> np.ndarray:
    if image_array_input.ndim not in [2, 3]:
        raise ValueError("Image_array_input phải là ảnh 2D (grayscale) hoặc 3D (grayscale/RGB).")
    image_to_transform = sk.util.img_as_float64(image_array_input)
    is_multichannel = image_to_transform.ndim == 3 and image_to_transform.shape[-1] > 1

    if random_state_seed is None: random_state = np.random.RandomState(None)
    else: random_state = np.random.RandomState(random_state_seed)

    shape = image_to_transform.shape[:2]

    if alpha_affine > 0:
        angle = random_state.uniform(-alpha_affine, alpha_affine) * 15
        log_scale = random_state.uniform(-alpha_affine, alpha_affine) * 0.1
        scale = np.exp(log_scale)
        shear_val = random_state.uniform(-alpha_affine, alpha_affine) * 0.1
        translation_x = random_state.uniform(-alpha_affine, alpha_affine) * shape[1] * 0.05
        translation_y = random_state.uniform(-alpha_affine, alpha_affine) * shape[0] * 0.05
        tform_center = sk.transform.SimilarityTransform(translation=(-shape[1]/2, -shape[0]/2))
        tform_affine = sk.transform.AffineTransform(scale=(scale, scale), rotation=np.deg2rad(angle), shear=np.deg2rad(shear_val), translation=(translation_x, translation_y))
        tform_uncenter = sk.transform.SimilarityTransform(translation=(shape[1]/2, shape[0]/2))
        tform = tform_center + tform_affine + tform_uncenter
        if is_multichannel:
            for c in range(image_to_transform.shape[-1]):
                image_to_transform[..., c] = sk.transform.warp(image_to_transform[..., c], tform, mode='reflect', preserve_range=True)
        else:
            if image_to_transform.ndim == 3 and image_to_transform.shape[-1] == 1:
                squeezed_img = image_to_transform.squeeze(axis=-1)
                warped_squeezed = sk.transform.warp(squeezed_img, tform, mode='reflect', preserve_range=True)
                image_to_transform = np.expand_dims(warped_squeezed, axis=-1)
            else:
                image_to_transform = sk.transform.warp(image_to_transform, tform, mode='reflect', preserve_range=True)

    dx_field = random_state.rand(*shape) * 2 - 1
    dy_field = random_state.rand(*shape) * 2 - 1
    dx_smoothed = sk.filters.gaussian(dx_field, sigma=sigma, mode="reflect", preserve_range=False) * alpha
    dy_smoothed = sk.filters.gaussian(dy_field, sigma=sigma, mode="reflect", preserve_range=False) * alpha
    
    y_coords, x_coords = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices_y = y_coords + dy_smoothed
    indices_x = x_coords + dx_smoothed
    
    distorted_image = np.empty_like(image_to_transform)
    if is_multichannel:
        for c in range(image_to_transform.shape[-1]):
            distorted_image[..., c] = map_coordinates(image_to_transform[..., c],
                                                      [indices_y, indices_x],
                                                      order=1, mode='reflect', cval=0.0).reshape(shape)
    else:
        img_to_map = image_to_transform.squeeze(axis=-1) if (image_to_transform.ndim == 3 and image_to_transform.shape[-1] == 1) else image_to_transform
        mapped_img = map_coordinates(img_to_map,
                                     [indices_y, indices_x],
                                     order=1, mode='reflect', cval=0.0).reshape(shape)
        if image_array_input.ndim == 3 and image_array_input.shape[-1] == 1:
            distorted_image = np.expand_dims(mapped_img, axis=-1)
        else:
            distorted_image = mapped_img
            
    distorted_image = np.clip(distorted_image, 0.0, 1.0)
    return distorted_image.astype(np.float32)

# # --- HÀM MIXUP (TensorFlow) ---
# def mixup_tf(images_batch, labels_batch, alpha=0.2):
#     batch_size = tf.shape(images_batch)[0]
#     if batch_size <= 1: # Cần ít nhất 2 sample để mixup
#         return images_batch, labels_batch
#     indices = tf.random.shuffle(tf.range(batch_size))
    
#     images_shuffled = tf.gather(images_batch, indices)
#     labels_shuffled = tf.gather(labels_batch, indices)
    
#     l = tf.compat.v1.distributions.Beta(alpha, alpha).sample([]) # Lấy một scalar sample
#     l = tf.cast(l, images_batch.dtype)

#     mixed_images = l * images_batch + (1.0 - l) * images_shuffled
#     mixed_labels = l * labels_batch + (1.0 - l) * labels_shuffled
#     return mixed_images, mixed_labels

# --- HÀM CUTMIX (TensorFlow) ---
# def get_random_box_tf(img_height, img_width, lam):
#     cut_rat = tf.sqrt(1. - lam) # Tỷ lệ của vùng cắt
#     cut_h = tf.cast(tf.cast(img_height, tf.float32) * cut_rat, dtype=tf.int32)
#     cut_w = tf.cast(tf.cast(img_width, tf.float32) * cut_rat, dtype=tf.int32)

#     # Đảm bảo kích thước vùng cắt ít nhất là 1 pixel
#     cut_h = tf.maximum(cut_h, 1)
#     cut_w = tf.maximum(cut_w, 1)

#     # Chọn tâm ngẫu nhiên cho vùng cắt
#     cx = tf.random.uniform([], 0, img_width, dtype=tf.int32)
#     cy = tf.random.uniform([], 0, img_height, dtype=tf.int32)

#     # Tính tọa độ bounding box, đảm bảo nằm trong ảnh
#     bbx1 = tf.clip_by_value(cx - cut_w // 2, 0, img_width - 1)
#     bby1 = tf.clip_by_value(cy - cut_h // 2, 0, img_height - 1)
#     bbx2 = tf.clip_by_value(cx + cut_w // 2, 0, img_width -1) # Sửa: phải là cx + cut_w // 2
#     bby2 = tf.clip_by_value(cy + cut_h // 2, 0, img_height -1) # Sửa: phải là cy + cut_h // 2
    
#     # Điều chỉnh lại width và height nếu clipping xảy ra
#     actual_cut_w = bbx2 - bbx1
#     actual_cut_h = bby2 - bby1

#     # Đảm bảo width và height của box > 0
#     if actual_cut_w <= 0 or actual_cut_h <=0:
#         # Fallback: trả về một box nhỏ ở góc hoặc không làm gì cả
#         return 0,0,1,1 # Box 1x1 pixel để tránh lỗi chia cho 0

#     return bbx1, bby1, bbx2, bby2

# def cutmix_tf(images_batch, labels_batch, alpha=1.0):
#     batch_size = tf.shape(images_batch)[0]
#     if batch_size <= 1: # Cần ít nhất 2 sample
#         return images_batch, labels_batch

#     img_height = tf.shape(images_batch)[1]
#     img_width = tf.shape(images_batch)[2]
#     channels = tf.shape(images_batch)[3]

#     indices = tf.random.shuffle(tf.range(batch_size))
#     images_shuffled = tf.gather(images_batch, indices)
#     labels_shuffled = tf.gather(labels_batch, indices)

#     lam_value = tf.compat.v1.distributions.Beta(alpha, alpha).sample([])

#     bbx1, bby1, bbx2, bby2 = get_random_box_tf(img_height, img_width, lam_value)
    
#     # Tạo mask cho vùng cần thay thế
#     mask_y = tf.sequence_mask(tf.fill([batch_size, img_width], bby2 - bby1), img_height, dtype=tf.bool)[..., bby1:bby2, :]
#     mask_x = tf.sequence_mask(tf.fill([batch_size, img_height], bbx2 - bbx1), img_width, dtype=tf.bool)[..., :, bbx1:bbx2]
    
#     # Điều chỉnh mask cho đúng shape
#     # mask_y shape (batch_size, H_box, W_img)
#     # mask_x shape (batch_size, H_img, W_box)
#     # Ta cần mask (batch_size, H_img, W_img)
    
#     y_indices = tf.range(img_height)
#     x_indices = tf.range(img_width)
#     grid_y, grid_x = tf.meshgrid(y_indices, x_indices, indexing='ij') # (H, W)

#     cut_mask_2d = (grid_y >= bby1) & (grid_y < bby2) & (grid_x >= bbx1) & (grid_x < bbx2)
#     cut_mask_4d = tf.cast(tf.expand_dims(tf.expand_dims(cut_mask_2d, axis=0), axis=-1), images_batch.dtype)
#     cut_mask_4d = tf.tile(cut_mask_4d, [batch_size, 1, 1, channels])

#     mixed_images = images_batch * (1.0 - cut_mask_4d) + images_shuffled * cut_mask_4d
    
#     # Điều chỉnh lambda dựa trên diện tích thực tế của bounding box
#     actual_lam = 1.0 - tf.cast((bbx2 - bbx1) * (bby2 - bby1), tf.float32) / tf.cast(img_height * img_width, tf.float32)
#     actual_lam = tf.cast(actual_lam, labels_batch.dtype)

#     mixed_labels = actual_lam * labels_batch + (1.0 - actual_lam) * labels_shuffled
#     return mixed_images, mixed_labels

# # --- HÀM CUTMIX (TensorFlow) ---
# def get_random_box_tf(img_height, img_width, lam):
#     cut_rat = tf.sqrt(1. - lam)
#     cut_h = tf.cast(tf.cast(img_height, tf.float32) * cut_rat, dtype=tf.int32)
#     cut_w = tf.cast(tf.cast(img_width, tf.float32) * cut_rat, dtype=tf.int32)

#     cx = tf.random.uniform([], 0, img_width, dtype=tf.int32)
#     cy = tf.random.uniform([], 0, img_height, dtype=tf.int32)

#     bbx1 = tf.clip_by_value(cx - cut_w // 2, 0, img_width)
#     bby1 = tf.clip_by_value(cy - cut_h // 2, 0, img_height)
#     bbx2 = tf.clip_by_value(cx + cut_w // 2, 0, img_width)
#     bby2 = tf.clip_by_value(cy + cut_h // 2, 0, img_height)
#     return bbx1, bby1, bbx2, bby2

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

# def label_is_binary(labels):
#     # Kiểm tra nếu labels là mảng 1D chứa toàn số (0/1) => binary
#     return labels.ndim == 1 or (labels.ndim == 2 and labels.shape[1] == 1)

# ... (các hàm random_rotation, random_noise, horizontal_flip, random_shearing,
# random_zoom, random_contrast, create_individual_transform, get_class_balances
# generate_image_transforms đã có trong file data_transformations.py của bạn) ...

# def random_gaussian_blur(image_array: np.ndarray, sigma_max: float = 1.0) -> np.ndarray:
#     """
#     Áp dụng Gaussian Blur với sigma ngẫu nhiên.
#     Ảnh đầu vào nên có giá trị pixel dạng float.

#     :param image_array: Mảng NumPy chứa ảnh (H, W) hoặc (H, W, 1).
#     :param sigma_max: Giá trị sigma tối đa cho Gaussian Blur.
#                       Sigma nhỏ -> ít mờ, sigma lớn -> mờ nhiều.
#     :return: Ảnh đã được làm mờ.
#     """
#     if image_array.max() > 1.0: # Giả định ảnh chưa được chuẩn hóa về [0,1]
#         image_array_float = sk.util.img_as_float(image_array)
#     else:
#         image_array_float = image_array.astype(np.float32)

#     sigma = random.uniform(0, sigma_max)
#     # preserve_range=True rất quan trọng để giữ dải giá trị pixel sau khi làm mờ,
#     # đặc biệt nếu ảnh đầu vào không phải là [0,1]
#     # multichannel=True nếu ảnh có chiều kênh (ví dụ H,W,1), False nếu là (H,W)
#     # Tuy nhiên, gaussian thường hoạt động trên từng kênh nếu multichannel=True và ảnh có nhiều kênh.
#     # Đối với ảnh xám (H,W) hoặc (H,W,1), bạn có thể bỏ qua multichannel hoặc đặt là False nếu ảnh là 2D.
    
#     is_multichannel = image_array_float.ndim == 3 and image_array_float.shape[-1] > 1
#     # Nếu ảnh là (H,W,1), thì nên coi nó là grayscale, không phải multichannel theo nghĩa màu sắc
#     if image_array_float.ndim == 3 and image_array_float.shape[-1] == 1:
#         # Bỏ chiều cuối để gaussian hoạt động trên ảnh 2D, sau đó thêm lại
#         blurred_image = sk.filters.gaussian(image_array_float.squeeze(axis=-1),
#                                             sigma=sigma,
#                                             preserve_range=True,
#                                             mode='reflect') # mode='reflect' giúp xử lý biên ảnh tốt hơn
#         return np.expand_dims(blurred_image, axis=-1).astype(np.float32)
#     else: # Ảnh 2D (H,W) hoặc ảnh màu thực sự (H,W,C với C>1)
#         blurred_image = sk.filters.gaussian(image_array_float,
#                                             sigma=sigma,
#                                             preserve_range=True,
#                                             multichannel=is_multichannel,
#                                             mode='reflect')
#         return blurred_image.astype(np.float32)


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

# def get_class_balances(y_vals):
#     """
#     Đếm số lượng mẫu cho mỗi lớp trong y_vals.
#     Trả về list [count_class0, count_class1, ...].
#     """
#     if config.dataset == "mini-MIAS":
#         # multi-class (nhãn one-hot)
#         num_classes = y_vals.shape[1]  # số cột = số lớp
#         counts = np.zeros(num_classes, dtype=int)
#         for y in y_vals:
#             # y là vector one-hot, cộng dồn cho class index tương ứng
#             counts += y.astype(int)
#     elif config.dataset == "mini-MIAS-binary" or config.dataset == "CMMD-binary":
#         # binary (nhãn 0/1 dạng số)
#         counts = np.zeros(2, dtype=int)
#         for y in y_vals:
#             if y == 0: counts[0] += 1
#             elif y == 1: counts[1] += 1
#     else:
#         # Mặc định cho các trường hợp khác (nếu label one-hot)
#         try:
#             num_classes = y_vals.shape[1]
#         except IndexError:
#             num_classes = len(np.unique(y_vals))
#         counts = np.zeros(num_classes, dtype=int)
#         for y in y_vals:
#             if isinstance(y, np.ndarray):
#                 # nếu one-hot vector
#                 counts += y.astype(int)
#             else:
#                 # nếu label số
#                 counts[int(y)] += 1
#     return counts.tolist()

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

# def create_individual_transform(image: np.ndarray, transforms: dict) -> np.ndarray:
#     """
#     Apply a random combination of transforms to a single image.
#     """
#     num = random.randint(1, len(transforms))
#     transformed = image.copy()
#     for _ in range(num):
#         func = random.choice(list(transforms.values()))
#         transformed = func(transformed)
#     return transformed

# def label_is_binary(labels: np.ndarray) -> bool:
#     """
#     Check if labels are binary (0/1) scalar array.
#     """
#     arr = np.array(labels)
#     return arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] == 1)

# def get_class_balances(y_vals: np.ndarray) -> list:
#     """
#     Count samples per class in y_vals.
#     Returns [count_class0, count_class1, ...].
#     """
#     arr = np.array(y_vals)
#     if arr.ndim == 2 and arr.shape[1] > 1:
#         return list(arr.sum(axis=0).astype(int))
#     else:
#         unique, counts = np.unique(arr, return_counts=True)
#         return [int(counts[unique.tolist().index(i)]) if i in unique else 0
#                 for i in range(len(unique))]

import random
import numpy as np
import skimage as sk
import skimage.transform
import skimage.color
import skimage.exposure
import skimage.util
import skimage.filters # Cho GaussianBlur
import tensorflow as tf # Cần thiết cho MixUp/CutMix
# Để resize (nếu cần)

# ==============================================================
# 0. Các hàm MixUp và CutMix (giữ nguyên từ phiên bản trước của bạn)
# ==============================================================
def mixup_tf(images_batch, labels_batch, alpha=0.2):
    batch_size = tf.shape(images_batch)[0]
    # Đảm bảo batch_size > 0 trước khi shuffle
    if batch_size == 0:
        return images_batch, labels_batch
    indices = tf.random.shuffle(tf.range(batch_size))

    images_shuffled = tf.gather(images_batch, indices)
    labels_shuffled = tf.gather(labels_batch, indices)

    l = tf.compat.v1.distributions.Beta(alpha, alpha).sample([]) # sample() trả về scalar
    l = tf.cast(l, images_batch.dtype)

    mixed_images = l * images_batch + (1.0 - l) * images_shuffled
    mixed_labels = l * labels_batch + (1.0 - l) * labels_shuffled
    return mixed_images, mixed_labels

def get_random_box_tf(img_height, img_width, lam):
    cut_rat = tf.sqrt(1. - lam)
    cut_h = tf.cast(tf.cast(img_height, tf.float32) * cut_rat, dtype=tf.int32)
    cut_w = tf.cast(tf.cast(img_width, tf.float32) * cut_rat, dtype=tf.int32)

    # Đảm bảo cut_h và cut_w không lớn hơn kích thước ảnh
    cut_h = tf.clip_by_value(cut_h, 1, img_height)
    cut_w = tf.clip_by_value(cut_w, 1, img_width)


    cx = tf.random.uniform([], 0, img_width, dtype=tf.int32)
    cy = tf.random.uniform([], 0, img_height, dtype=tf.int32)

    bbx1 = tf.clip_by_value(cx - cut_w // 2, 0, img_width - cut_w) # Đảm bảo bbx1 + cut_w <= img_width
    bby1 = tf.clip_by_value(cy - cut_h // 2, 0, img_height - cut_h) # Đảm bảo bby1 + cut_h <= img_height
    bbx2 = bbx1 + cut_w
    bby2 = bby1 + cut_h
    return bbx1, bby1, bbx2, bby2


def cutmix_tf(images_batch, labels_batch, alpha=1.0):
    batch_size = tf.shape(images_batch)[0]
    if batch_size == 0:
        return images_batch, labels_batch

    img_height = tf.shape(images_batch)[1]
    img_width = tf.shape(images_batch)[2]
    channels = tf.shape(images_batch)[3]

    indices = tf.random.shuffle(tf.range(batch_size))
    images_shuffled = tf.gather(images_batch, indices)
    labels_shuffled = tf.gather(labels_batch, indices)

    lam_value = tf.compat.v1.distributions.Beta(alpha, alpha).sample([]) # Sample a scalar

    bbx1, bby1, bbx2, bby2 = get_random_box_tf(img_height, img_width, lam_value)

    # Tạo mask
    # Phần được cắt từ ảnh shuffle
    # Tạo một tensor chứa patch từ images_shuffled
    patch = images_shuffled[:, bby1:bby2, bbx1:bbx2, :]

    # Tạo ảnh mới bằng cách ghép phần không bị che của ảnh gốc và patch
    # Cách 1: Tạo mask rồi dùng tf.where
    mask_area = tf.ones_like(images_batch[:, bby1:bby2, bbx1:bbx2, :], dtype=images_batch.dtype)
    padding = [
        [0, 0],  # batch
        [bby1, img_height - bby2],  # height
        [bbx1, img_width - bbx2],  # width
        [0, 0]  # channels
    ]
    mask = tf.pad(mask_area, padding, "CONSTANT", constant_values=0)

    mixed_images = tf.where(tf.cast(mask, tf.bool), images_shuffled, images_batch)


    # Điều chỉnh lambda dựa trên diện tích thực tế của bounding box
    actual_lam = 1.0 - tf.cast((bbx2 - bbx1) * (bby2 - bby1), tf.float32) / tf.cast(img_height * img_width, tf.float32)
    actual_lam = tf.cast(actual_lam, labels_batch.dtype) # Đảm bảo cùng kiểu dữ liệu

    mixed_labels = actual_lam * labels_batch + (1.0 - actual_lam) * labels_shuffled
    return mixed_images, mixed_labels

# ==============================================================
# 1. Các hàm transformation cá nhân (giữ nguyên)
# ==============================================================
def random_rotation(image_array: np.ndarray):
    random_degree = random.uniform(-20, 20)
    return sk.transform.rotate(image_array, random_degree, resize=False, mode='reflect', preserve_range=True)

def random_noise(image_array: np.ndarray):
    img_float = sk.util.img_as_float(np.clip(image_array, 0, 1 if image_array.max() <=1 else 255))
    noisy_image = sk.util.random_noise(img_float, mode='gaussian', var=random.uniform(0.001, 0.01))
    return noisy_image.astype(np.float32)

def horizontal_flip(image_array: np.ndarray):
    return image_array[:, ::-1]

def random_shearing(image_array: np.ndarray):
    random_degree = random.uniform(-0.2, 0.2)
    tf = sk.transform.AffineTransform(shear=random_degree)
    return sk.transform.warp(image_array, tf, order=1, preserve_range=True, mode='wrap')

def gamma_correction(image_array: np.ndarray):
    gamma = random.uniform(0.8, 1.5)
    img_clipped = np.clip(image_array, 0, 1 if image_array.max() <= 1 else image_array.max())
    return sk.exposure.adjust_gamma(img_clipped, gamma)

def random_zoom(image_array: np.ndarray):
    h, w = image_array.shape[:2]
    zoom_factor = random.uniform(0.8, 0.99)
    crop_h = int(h * zoom_factor)
    crop_w = int(w * zoom_factor)
    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2
    zoomed_out = image_array[start_h : start_h + crop_h, start_w : start_w + crop_w]
    return sk.transform.resize(zoomed_out, (h, w), mode='reflect', anti_aliasing=True, preserve_range=True)

def random_contrast(image_array: np.ndarray):
    p_low = random.uniform(1, 10)
    p_high = random.uniform(90, 99)
    img_float = sk.util.img_as_float(np.clip(image_array, 0, 1 if image_array.max() <=1 else 255))
    v_min, v_max = np.percentile(img_float, [p_low, p_high])
    return sk.exposure.rescale_intensity(img_float, in_range=(v_min, v_max), out_range=(0.0, 1.0)).astype(np.float32)

def random_gaussian_blur(image_array: np.ndarray, sigma_max: float = 1.0) -> np.ndarray:
    if image_array.max() > 1.0:
        image_array_float = sk.util.img_as_float(image_array)
    else:
        image_array_float = image_array.astype(np.float32)
    sigma = random.uniform(0, sigma_max)
    if image_array_float.ndim == 3 and image_array_float.shape[-1] == 1:
        blurred_image = sk.filters.gaussian(image_array_float.squeeze(axis=-1), sigma=sigma, preserve_range=True, mode='reflect')
        return np.expand_dims(blurred_image, axis=-1).astype(np.float32)
    else:
        is_multichannel = image_array_float.ndim == 3 and image_array_float.shape[-1] > 1
        blurred_image = sk.filters.gaussian(image_array_float, sigma=sigma, preserve_range=True, multichannel=is_multichannel, mode='reflect')
        return blurred_image.astype(np.float32)

def random_brightness_adjustment(image_array: np.ndarray, factor_range: tuple = (0.7, 1.3)) -> np.ndarray:
    if image_array.max() > 1.0:
        image_array_float = sk.util.img_as_float(image_array)
    else:
        image_array_float = image_array.astype(np.float32)
    brightness_factor = random.uniform(factor_range[0], factor_range[1])
    adjusted_image = np.clip(image_array_float * brightness_factor, 0.0, 1.0)
    return adjusted_image.astype(np.float32)

# ==============================================================
# 2. Hàm tạo biến đổi kết hợp (giữ nguyên)
# ==============================================================
# def create_individual_transform(image: np.ndarray, transforms: dict):
#     if image.ndim == 2:
#         image = np.expand_dims(image, axis=-1)
#     elif image.ndim == 3 and image.shape[-1] != 1:
#         image = image[..., :1]
#     if image.shape[-1] != 1:
#          return image.astype(np.float32)

#     num_transformations_to_apply = random.randint(1, len(transforms))
#     transformed_image = image.copy()
#     applied_keys = random.sample(list(transforms.keys()), num_transformations_to_apply)
#     for key in applied_keys:
#         transformed_image = transforms[key](transformed_image)
#         if transformed_image is None:
#             transformed_image = image.copy()
#             continue
#         if transformed_image.ndim == 3 and transformed_image.shape[-1] != 1:
#             if transformed_image.shape[-1] == 3:
#                  gray_img = sk.color.rgb2gray(transformed_image)
#                  transformed_image = np.expand_dims(gray_img, axis=-1)
#             elif transformed_image.shape[-1] == 4:
#                  gray_img = sk.color.rgb2gray(sk.color.rgba2rgb(transformed_image))
#                  transformed_image = np.expand_dims(gray_img, axis=-1)
#             else:
#                  transformed_image = transformed_image[..., 0:1]
#         elif transformed_image.ndim == 2:
#              transformed_image = np.expand_dims(transformed_image, axis=-1)
#         if transformed_image.ndim != 3 or transformed_image.shape[-1] != 1:
#              transformed_image = image.copy()
#              break
#     if transformed_image.ndim != 3 or transformed_image.shape[-1] != 1:
#          return image.copy().astype(np.float32)
#     return transformed_image.astype(np.float32)
# def create_individual_transform(image: np.ndarray, transforms: dict) -> np.ndarray:
#     """
#     Apply a random combination of transformations to a single image.
#     This corrected version handles both grayscale and RGB images without unwanted channel reduction.
#     """
#     # Tạo bản sao để không thay đổi ảnh gốc
#     transformed_image = image.copy()

#     # Đảm bảo ảnh đầu vào có ít nhất 3 chiều (H, W, C)
#     if transformed_image.ndim == 2:
#         # Nếu ảnh là 2D (H, W), thêm chiều kênh vào cuối
#         transformed_image = np.expand_dims(transformed_image, axis=-1)

#     # Logic cũ có vấn đề đã được gỡ bỏ. Các hàm transform từ skimage
#     # thường có thể tự xử lý ảnh đa kênh.

#     # Chọn ngẫu nhiên số lượng phép biến đổi để áp dụng
#     num_transformations_to_apply = random.randint(1, len(transforms))
#     # Chọn ngẫu nhiên các phép biến đổi từ danh sách
#     applied_keys = random.sample(list(transforms.keys()), num_transformations_to_apply)

#     for key in applied_keys:
#         transform_func = transforms[key]
#         transformed_image = transform_func(transformed_image)

#         # Kiểm tra sau mỗi phép biến đổi để đảm bảo shape không bị thay đổi không mong muốn
#         if transformed_image.ndim == 2:
#              transformed_image = np.expand_dims(transformed_image, axis=-1)

#         if transformed_image.shape[-1] != image.shape[-1]:
#             print(f"Warning: Transform '{key}' changed channel count from {image.shape[-1]} to {transformed_image.shape[-1]}. This may cause issues.")
#             # Trong trường hợp phức tạp, bạn có thể muốn xử lý thêm ở đây
#             # nhưng với các transform hiện tại, việc này ít khả năng xảy ra
#             # nếu input đã được chuẩn hóa.

#     return transformed_image.astype(np.float32)
# def create_individual_transform(image: np.ndarray, transforms: dict) -> np.ndarray:
#     """
#     PHIÊN BẢN TINH CHỈNH VÀ ỔN ĐỊNH
#     Áp dụng một pipeline augmentation cơ bản có kiểm soát thay vì logic hỗn loạn.
#     """
#     transformed_image = image.copy()

#     # == LOGIC MỚI BẮT ĐẦU TỪ ĐÂY ==
#     # Thay vì chọn ngẫu nhiên, chúng ta đi qua một chuỗi các phép biến đổi
#     # và mỗi phép có một xác suất được áp dụng riêng.
    
#     # 1. Lật ngang (Horizontal Flip): Xác suất 50%
#     if 'horizontal_flip' in transforms and random.random() < 0.5:
#         transformed_image = transforms['horizontal_flip'](transformed_image)

#     # 2. Xoay nhẹ (Random Rotation): Xác suất 30%
#     if 'rotate' in transforms and random.random() < 0.3:
#         transformed_image = transforms['rotate'](transformed_image)

#     # 3. Trượt ảnh (Shear): Xác suất 20%
#     if 'shear' in transforms and random.random() < 0.2:
#         transformed_image = transforms['shear'](transformed_image)

#     # 4. Điều chỉnh Gamma (Tương phản): Xác suất 20%
#     if 'gamma_correction' in transforms and random.random() < 0.2:
#         transformed_image = transforms['gamma_correction'](transformed_image)
        
#     # Bạn có thể thêm các phép biến đổi khác vào đây theo cùng một logic
#     # ví dụ: zoom, noise... với một xác suất nhất định.
#     # == KẾT THÚC LOGIC MỚI ==
    
#     # Đảm bảo định dạng đầu ra đúng
#     if transformed_image.ndim == 2:
#         transformed_image = np.expand_dims(transformed_image, axis=-1)

#     return transformed_image.astype(np.float32)
# # ==============================================================
# # 3. Hàm đếm số lượng lớp (giữ nguyên)
# # ==============================================================
# def get_class_balances(y_vals):
#     counts = []
#     if y_vals is None or len(y_vals) == 0:
#         return counts
#     y_array = np.array(y_vals)
#     if y_array.ndim > 1 and y_array.shape[1] > 1:
#         num_classes = y_array.shape[1]
#         counts = np.sum(y_array, axis=0).astype(int).tolist()
#     elif y_array.ndim == 1 or y_array.shape[1] == 1:
#         if y_array.ndim == 2 and y_array.shape[1] == 1:
#             y_array = y_array.flatten()
#         unique_values, unique_counts = np.unique(y_array, return_counts=True)
#         if len(unique_values) > 0:
#             try:
#                  max_val = int(np.max(unique_values))
#                  num_classes_derived = max_val + 1
#                  counts_arr = np.zeros(num_classes_derived, dtype=int)
#                  for val, count in zip(unique_values.astype(int), unique_counts):
#                       if 0 <= val < num_classes_derived:
#                           counts_arr[val] = count
#                  counts = counts_arr.tolist()
#             except ValueError:
#                  counts = []
#         else:
#              counts = []
#     else:
#         counts = []
#     return counts

# # ==============================================================
# # 4. Hàm helper lấy kích thước resize (giữ nguyên)
# # ==============================================================
# def get_reshape_size_from_config():
#     default_h, default_w = 224, 224
#     try:
#         dataset_name = getattr(config, 'dataset', None)
#         model_name = getattr(config, 'model', None)
#         is_roi_flag = getattr(config, 'is_roi', False)

#         if dataset_name == "CMMD":
#             size = getattr(config, 'CMMD_IMG_SIZE', {'HEIGHT': default_h, 'WIDTH': default_w})
#         elif dataset_name == "INbreast":
#              size = getattr(config, 'INBREAST_IMG_SIZE', {'HEIGHT': default_h, 'WIDTH': default_w})
#         elif is_roi_flag or model_name == "CNN":
#              size = getattr(config, 'ROI_IMG_SIZE', {'HEIGHT': default_h, 'WIDTH': default_w})
#         elif model_name in ["VGG", "Inception", "VGG-common", "ResNet", "MobileNet", "DenseNet"]:
#              size_attr_name = f"{model_name.upper().replace('-', '_')}_IMG_SIZE"
#              if model_name in ["VGG", "Inception"] and hasattr(config, 'MINI_MIAS_IMG_SIZE'):
#                  size_attr_name = 'MINI_MIAS_IMG_SIZE'
#              size = getattr(config, size_attr_name, {'HEIGHT': default_h, 'WIDTH': default_w})
#         else:
#             return default_h, default_w
#         return size.get('HEIGHT', default_h), size.get('WIDTH', default_w)
#     except AttributeError: return default_h, default_w
#     except Exception: return default_h, default_w

# def label_is_binary(labels: np.ndarray) -> bool:
#     arr = np.array(labels)
#     return arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] == 1)
# # ==============================================================
# # 5. Hàm Augmentation chính (CẬP NHẬT)
# # ==============================================================
# # --- HÀM AUGMENTATION CHÍNH ---
# def generate_image_transforms(images: np.ndarray, labels: np.ndarray,
#                               apply_elastic: bool = False, elastic_alpha: float = 34.0, elastic_sigma: float = 4.0,
#                               apply_mixup: bool = False, mixup_alpha: float = 0.2,
#                               apply_cutmix: bool = False, cutmix_alpha: float = 1.0
#                               ):
#     # print(f"[generate_image_transforms] Initial shapes: images={images.shape}, labels={labels.shape}")
#     # print(f"  Flags: Elastic={apply_elastic} (a:{elastic_alpha},s:{elastic_sigma}), MixUp={apply_mixup} (a:{mixup_alpha}), CutMix={apply_cutmix} (a:{cutmix_alpha})")

#     is_binary_scalar_original = label_is_binary(labels) # Lưu lại dạng nhãn gốc
    
#     # 1. Chuẩn bị labels_one_hot (luôn là float32)
#     if is_binary_scalar_original:
#         num_classes = 2
#         labels_one_hot = tf.keras.utils.to_categorical(labels, num_classes=num_classes).astype(np.float32)
#     elif labels.ndim == 2 and labels.shape[1] > 1: # Đã là one-hot
#         num_classes = labels.shape[1]
#         labels_one_hot = labels.astype(np.float32)
#     else: # Nhãn số đa lớp
#         unique_labels_count = len(np.unique(labels.ravel()))
#         num_classes = unique_labels_count if unique_labels_count > 0 else (int(np.max(labels)) + 1 if labels.size > 0 else 2)
#         if num_classes == 0 : num_classes = 2 # Fallback
#         labels_one_hot = tf.keras.utils.to_categorical(labels, num_classes=num_classes).astype(np.float32)
#     # print(f"    [INFO Aug] num_classes determined: {num_classes}, labels_one_hot shape: {labels_one_hot.shape}")

#     # 2. Chuẩn bị initial_images_processed_for_aug
#     initial_images_processed_for_aug = []
#     for img_idx, img_orig in enumerate(images):
#         img_f = img_orig.astype(np.float32)
#         if np.max(img_f) > 1.0: # Chuẩn hóa nếu chưa
#             min_v, max_v = np.min(img_f), np.max(img_f)
#             if max_v - min_v > 1e-8: img_f = (img_f - min_v) / (max_v - min_v)
#             else: img_f = np.zeros_like(img_f)
#         img_f = np.clip(img_f, 0.0, 1.0)

#         if img_f.ndim == 2: img_f = np.expand_dims(img_f, axis=-1)
#         elif img_f.ndim == 3:
#             if img_f.shape[-1] > 3 : img_f = sk.color.rgba2rgb(img_f) # RGBA -> RGB
#             if img_f.shape[-1] == 2: img_f = img_f[..., :1] # Lấy kênh đầu nếu 2 kênh
#         # img_f giờ là (H,W,1) hoặc (H,W,3)

#         # if apply_elastic:
#         #     img_f = elastic_transform(img_f, alpha=elastic_alpha, sigma=elastic_sigma, alpha_affine=0.05) # Thêm alpha_affine nhỏ
#         # initial_images_processed_for_aug.append(img_f)
#         if apply_elastic:
#             # print(f"    [DEBUG Elastic] Processing image {img_idx} with elastic. Original img_f shape: {img_f.shape}, dtype: {img_f.dtype}, min: {np.min(img_f):.2f}, max: {np.max(img_f):.2f}")
#             img_f_before_elastic = img_f.copy() # Giữ bản sao để so sánh
#             try:
#                 img_f = elastic_transform(img_f, alpha=elastic_alpha, sigma=elastic_sigma, alpha_affine=0.05)
#                 if img_f is None:
#                     print(f"    [CRITICAL Elastic] Elastic transform returned None for image {img_idx}! Using original.")
#                     img_f = img_f_before_elastic # Sử dụng lại ảnh gốc nếu elastic trả về None
#                 # else:
#                     # print(f"    [DEBUG Elastic] Image {img_idx} after elastic. Shape: {img_f.shape}, dtype: {img_f.dtype}, min: {np.min(img_f):.2f}, max: {np.max(img_f):.2f}, NaNs: {np.isnan(img_f).sum()}")
#             except Exception as e_elastic:
#                 print(f"    [CRITICAL Elastic] Exception during elastic_transform for image {img_idx}: {e_elastic}. Using original.")
#                 img_f = img_f_before_elastic # Sử dụng ảnh gốc nếu có lỗi

#         # Kiểm tra lại shape và dtype trước khi append
#         if not isinstance(img_f, np.ndarray) or img_f.ndim not in [2, 3] or (img_f.ndim == 3 and img_f.shape[-1] not in [1,3]):
#             print(f"    [CRITICAL Elastic] img_f for image {img_idx} has problematic shape/type after elastic processing: {img_f.shape if hasattr(img_f, 'shape') else type(img_f)}. Reverting to original for this image.")
#             # Lấy lại ảnh gốc img_orig, chuẩn hóa và đảm bảo đúng số kênh
#             temp_img_orig = img_orig.astype(np.float32)
#             if np.max(temp_img_orig) > 1.0:
#                 min_v, max_v = np.min(temp_img_orig), np.max(temp_img_orig)
#                 if max_v - min_v > 1e-8: temp_img_orig = (temp_img_orig - min_v) / (max_v - min_v)
#                 else: temp_img_orig = np.zeros_like(temp_img_orig)
#             temp_img_orig = np.clip(temp_img_orig, 0.0, 1.0)
#             if temp_img_orig.ndim == 2: temp_img_orig = np.expand_dims(temp_img_orig, axis=-1)
#             img_f = temp_img_orig

#         initial_images_processed_for_aug.append(img_f)
#     if not initial_images_processed_for_aug:
#         return images, labels

#     augmented_images_list = list(initial_images_processed_for_aug)
#     augmented_labels_list = list(labels_one_hot)

#     # 3. Logic cân bằng lớp và áp dụng basic_transforms
#     # basic_transforms = {
#     #     'rotate': random_rotation, 'noise': random_noise, 'horizontal_flip': horizontal_flip,
#     #     'shear': random_shearing, 'gamma_correction': gamma_correction,
#     #     'zoom': random_zoom, 'contrast': random_contrast,
#     #     'gaussian_blur': random_gaussian_blur, 'brightness_adjust': random_brightness_adjustment
#     # }
#     basic_transforms = {
#         'rotate': random_rotation, 'horizontal_flip': horizontal_flip,
#         'shear': random_shearing, 'gamma_correction': gamma_correction
#     }
#     target_multiplier = 1
#     dataset_name = getattr(config, 'dataset', '')
#     # Đọc các hằng số multiplier từ config nếu có, nếu không dùng default
#     if dataset_name == "INbreast": target_multiplier = int(getattr(config, 'INBREAST_AUG_MULTIPLIER', 3))
#     elif dataset_name in ["mini-MIAS-binary", "CMMD-binary", "CMMD"]: target_multiplier = int(getattr(config, 'BINARY_AUG_MULTIPLIER', 1))

#     # print(f"    [INFO Aug] Augmentation target_multiplier for {dataset_name}: {target_multiplier}")

#     class_counts = get_class_balances(labels_one_hot) # labels_one_hot là (N, num_classes)
#     # print(f"    [INFO Aug] Initial class counts (from one-hot labels): {class_counts}")

#     # if not class_counts or num_classes == 0:
#     #     print("    [WARN] Aug: Could not determine class balance or num_classes is 0. Skipping basic augmentation.")
#     # else:
#     #     target_count_per_class = max(class_counts) * target_multiplier if class_counts and len(class_counts) > 0 else 0
#     if not class_counts or num_classes == 0 or len(class_counts) != num_classes or not any(c > 0 for c in class_counts):
#         print("    [WARN Aug] Invalid class_counts or num_classes. Skipping basic augmentation/balancing.")
#     else:
#         # Mục tiêu là làm cho tất cả các lớp có ít nhất số lượng mẫu bằng lớp đa số hiện tại,
#         # sau đó nhân thêm nếu target_multiplier > 1.
#         max_present_class_count = 0
#         for count in class_counts:
#             if count > 0: max_present_class_count = max(max_present_class_count, count)
        
#         if max_present_class_count == 0: # Không có mẫu nào trong bất kỳ lớp nào
#             print("    [WARN Aug] All class counts are zero. Skipping augmentation.")
#         else:
#             # Mục tiêu cuối cùng cho mỗi lớp
#             target_count_per_class = max_present_class_count * target_multiplier
#             # print(f"    [INFO Aug] Max present class count: {max_present_class_count}. Final target count per class: {final_target_count_per_class}")

#         for class_idx in range(num_classes):
#             current_class_count_for_idx = class_counts[class_idx] if class_idx < len(class_counts) else 0
#             num_to_generate = target_count_per_class - current_class_count_for_idx
            
#             if num_to_generate <= 0: continue

#             original_indices_for_class = [j for j, lab_vec in enumerate(labels_one_hot) if lab_vec[class_idx] == 1.0]
#             if not original_indices_for_class: continue
            
#             # print(f"    [INFO] Aug: Basic augmenting class {class_idx}: Need {num_to_generate} more from {len(original_indices_for_class)} originals.")

#             for k in range(int(num_to_generate)): # Chuyển sang int
#                 original_image_idx = original_indices_for_class[k % len(original_indices_for_class)]
#                 # Lấy ảnh đã qua elastic (nếu có) để làm cơ sở cho transform cơ bản
#                 original_image_for_transform = initial_images_processed_for_aug[original_image_idx]
                
#                 transformed_image = create_individual_transform(original_image_for_transform, basic_transforms)
#                 augmented_images_list.append(transformed_image)
#                 augmented_labels_list.append(labels_one_hot[original_image_idx].copy())

#     final_images_np = np.array(augmented_images_list, dtype=np.float32)
#     final_labels_np = np.array(augmented_labels_list, dtype=np.float32)

#     # 4. Áp dụng MixUp
#     if apply_mixup and final_images_np.shape[0] > 1:
#         if final_images_np.ndim == 3: final_images_np = np.expand_dims(final_images_np, axis=-1)
#         # Đảm bảo số kênh hợp lệ cho MixUp (thường là 1 hoặc 3)
#         if final_images_np.ndim == 4 and final_images_np.shape[-1] not in [1, 3]:
#              final_images_np = final_images_np[..., :1] # Lấy kênh đầu
        
#         tf_images = tf.convert_to_tensor(final_images_np, dtype=tf.float32)
#         tf_labels = tf.convert_to_tensor(final_labels_np, dtype=tf.float32)
#         if tf.shape(tf_images)[0] > 0: # Kiểm tra batch không rỗng
#             tf_images, tf_labels = mixup_tf(tf_images, tf_labels, alpha=mixup_alpha)
#             final_images_np = tf_images.numpy()
#             final_labels_np = tf_labels.numpy()

#     # 5. Áp dụng CutMix
#     if apply_cutmix and final_images_np.shape[0] > 1:
#         if final_images_np.ndim == 3: final_images_np = np.expand_dims(final_images_np, axis=-1)
#         if final_images_np.ndim == 4 and final_images_np.shape[-1] not in [1, 3]:
#              final_images_np = final_images_np[..., :1]

#         tf_images = tf.convert_to_tensor(final_images_np, dtype=tf.float32)
#         tf_labels = tf.convert_to_tensor(final_labels_np, dtype=tf.float32)
#         if tf.shape(tf_images)[0] > 0:
#             tf_images, tf_labels = cutmix_tf(tf_images, tf_labels, alpha=cutmix_alpha)
#             final_images_np = tf.clip_by_value(tf_images, 0.0, 1.0).numpy()
#             final_labels_np = tf_labels.numpy()
    
#     final_images_output = final_images_np
#     final_labels_output = final_labels_np

#     # 6. Xử lý nhãn cuối cùng: Nếu nhãn gốc là binary scalar, chuyển lại từ one-hot/mixed
#     if is_binary_scalar_original and final_labels_output.ndim > 1 and final_labels_output.shape[1] == 2 :
#         final_labels_output = np.argmax(final_labels_output, axis=1)
#     elif is_binary_scalar_original and final_labels_output.ndim == 1 and num_classes == 2:
#         # Đã là scalar, nhưng có thể là float do mixup, làm tròn
#         final_labels_output = np.round(final_labels_output).astype(int)


#     # print(f"[generate_image_transforms] DEBUG Final output shapes before return: images={final_images_output.shape}, labels={final_labels_output.shape}")
#     if final_images_output.shape[0] != final_labels_output.shape[0]:
#         print(f"[FATAL ERROR generate_image_transforms] Mismatch in samples between augmented images and labels BEFORE returning!")
#         # Bạn có thể raise lỗi ở đây để dừng sớm nếu muốn:
#         # raise ValueError("Mismatch in samples from generate_image_transforms after augmentation")
#     return final_images_output, final_labels_output

# ==============================================================================
# CÁC HÀM PHỤ TRỢ ĐÃ ĐƯỢC KIỂM TRA
# ==============================================================================

def _prepare_images(images: np.ndarray) -> np.ndarray:
    """Hàm phụ trợ: Chuẩn hóa và đảm bảo định dạng ảnh đúng."""
    if images.dtype != np.float32:
        processed = images.astype(np.float32)
    else:
        processed = images

    if np.max(processed) > 1.0:
        min_val, max_val = np.min(processed), np.max(processed)
        if (max_val - min_val) > 1e-6:
            processed = (processed - min_val) / (max_val - min_val)
        else:
            processed = np.zeros_like(processed)
    
    if processed.ndim == 3:
        processed = np.expand_dims(processed, axis=-1)
        
    return np.clip(processed, 0.0, 1.0)

def _prepare_labels(labels: np.ndarray) -> np.ndarray:
    """Hàm phụ trợ: Chuyển đổi các định dạng nhãn về one-hot float32."""
    try:
        if labels.ndim == 1:
            unique_labels = np.unique(labels)
            num_classes = len(unique_labels)
            if num_classes < 2:
                num_classes = int(np.max(labels)) + 1 if labels.size > 0 else 2
            return tf.keras.utils.to_categorical(labels, num_classes=num_classes).astype(np.float32)
        elif labels.ndim == 2 and labels.shape[1] > 1:
            return labels.astype(np.float32)
        else:
            return None
    except Exception as e:
        print(f"[ERROR _prepare_labels] Lỗi khi xử lý nhãn: {e}")
        return None

# ==============================================================================
# HÀM CHÍNH ĐÃ ĐƯỢC TÁI CẤU TRÚC VÀ KIỂM TRA
# ==============================================================================

def generate_image_transforms(images: np.ndarray, labels: np.ndarray,
                              apply_elastic: bool = False, elastic_alpha: float = 34.0, elastic_sigma: float = 4.0,
                              apply_mixup: bool = False, mixup_alpha: float = 0.2,
                              apply_cutmix: bool = False, cutmix_alpha: float = 1.0
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Áp dụng augmentation cho một batch ảnh và nhãn.
    Hàm này chỉ làm nhiệm vụ augmentation, không cân bằng lớp.
    """
    if images.shape[0] == 0:
        return images, labels

    processed_images = _prepare_images(images)
    labels_one_hot = _prepare_labels(labels)
    
    if labels_one_hot is None:
        print("[ERROR generate_image_transforms] Định dạng nhãn không hợp lệ, bỏ qua augmentation.")
        return images, labels

    if apply_elastic:
        elastic_images = []
        for img in processed_images:
            try:
                transformed_img = elastic_transform(img, alpha=elastic_alpha, sigma=elastic_sigma, alpha_affine=0.05)
                elastic_images.append(transformed_img)
            except Exception as e:
                print(f"[WARNING generate_image_transforms] Lỗi Elastic Transform, sử dụng ảnh gốc. Lỗi: {e}")
                elastic_images.append(img)
        processed_images = np.array(elastic_images, dtype=np.float32)

    basic_augmentations = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.7, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        A.GaussNoise(var_limit=(10.0 / 255.0, 50.0 / 255.0), p=0.5)
    ])

    augmented_batch = [basic_augmentations(image=img)['image'] for img in processed_images]
    final_images_np = np.array(augmented_batch, dtype=np.float32)
    final_labels_np = labels_one_hot

    if apply_mixup and final_images_np.shape[0] > 1:
        tf_images, tf_labels = mixup_tf(tf.convert_to_tensor(final_images_np), tf.convert_to_tensor(final_labels_np), alpha=mixup_alpha)
        final_images_np, final_labels_np = tf_images.numpy(), tf_labels.numpy()

    if apply_cutmix and final_images_np.shape[0] > 1:
        tf_images, tf_labels = cutmix_tf(tf.convert_to_tensor(final_images_np), tf.convert_to_tensor(final_labels_np), alpha=cutmix_alpha)
        final_images_np = tf.clip_by_value(tf_images, 0.0, 1.0).numpy()
        final_labels_np = tf_labels.numpy()
        
    if final_images_np.shape[0] != final_labels_np.shape[0]:
        raise ValueError("[FATAL] Lỗi logic: Số lượng ảnh và nhãn không khớp sau augmentation.")

    return final_images_np, final_labels_np