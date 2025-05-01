import random
import numpy as np
import skimage as sk
import skimage.transform
import config

def resize_before_reshape(transformed_image, target_height, target_width):
    # """Resize ảnh về kích thước mục tiêu trước khi reshape"""
    # # Kiểm tra xem ảnh có cần resize không
    # if image.shape[:2] != (target_height, target_width):
    #     image = sk.transform.resize(image, (target_height, target_width), mode='reflect', anti_aliasing=True)
    
    # # Kiểm tra số kênh màu và chuyển đổi nếu cần
    # if len(image.shape) == 3 and image.shape[2] == 3:
    #     image = sk.color.rgb2gray(image)
    
    # return image
    # Xác định kích thước mục tiêu dựa vào dataset
    resized_transformed_image = resize_before_reshape(transformed_image, target_height, target_width)
    # resize_before_reshape PHẢI trả về (target_H, target_W, 1)

    # --- Kiểm tra lại shape sau resize ---
    if resized_transformed_image.shape != (target_height, target_width, 1):
            print(f"Error after resize: Expected {(target_height, target_width, 1)}, got {resized_transformed_image.shape}. Reshaping forcefully.")
            resized_transformed_image = resized_transformed_image.reshape(target_height, target_width, 1)

def generate_image_transforms(images, labels):
    """
    Oversample data by transforming existing images.
    Adjusted for both CMMD and INbreast datasets.
    
    :param images: input images
    :param labels: input labels
    :return: updated list of images and labels with extra transformed images and labels
    """
    # Set augmentation multiplier based on dataset
    augmentation_multiplier = 1
    if config.dataset == "mini-MIAS-binary":
        augmentation_multiplier = 3
    elif config.dataset == "CMMD":
        augmentation_multiplier = 3  # Increase benign samples for CMMD
    elif config.dataset == "INbreast":
        augmentation_multiplier = 3  # Increase malignant samples for INbreast

    images_with_transforms = images.copy()
    labels_with_transforms = labels.copy()

    # Define available transformations
    available_transforms = {
        'rotate': random_rotation,
        'noise': random_noise,
        'horizontal_flip': horizontal_flip,
        'shear': random_shearing,
        'gamma_correction': gamma_correction,
        'zoom': random_zoom,
        'contrast': random_contrast,
    }

    # Calculate class balance
    class_balance = get_class_balances(labels)
    
    # Adjust class balancing strategy based on dataset
    if config.dataset == "CMMD":
        # For CMMD, only augment benign class (class 0)
        max_count = class_balance[1]  # Target is number of malignant samples
        to_add = [max_count - class_balance[0], 0]
    elif config.dataset == "INbreast":
        # For INbreast, only augment malignant class (class 1)
        max_count = class_balance[0]  # Target is number of benign samples
        to_add = [0, max_count - class_balance[1]]
    else:
        # Default balancing strategy
        max_count = max(class_balance) * augmentation_multiplier
        to_add = [max_count - i for i in class_balance]

    # Xác định kích thước mục tiêu dựa vào dataset
    if config.dataset == "CMMD":
        target_height = config.CMMD_IMG_SIZE['HEIGHT']
        target_width = config.CMMD_IMG_SIZE['WIDTH']
    elif config.dataset == "INbreast":
        target_height = config.INBREAST_IMG_SIZE['HEIGHT']
        target_width = config.INBREAST_IMG_SIZE['WIDTH']
    else:
        target_height = 224
        target_width = 224

    # # Resize và reshape ảnh
    # transformed_image = resize_before_reshape(transformed_image, target_height, target_width)
    # transformed_image = transformed_image.reshape(1, target_height, target_width, 1)

    # Generate transformations for each class that needs augmentation
    for i in range(len(to_add)):
        num_to_add = int(to_add[i])
        # if int(to_add[i]) == 0:
        if num_to_add <= 0:
            continue
        
        # Create one-hot encoded label
        label_one_hot = np.zeros(len(to_add))
        label_one_hot[i] = 1
        
        # Find images of current class
        indices = [j for j, x in enumerate(labels) if np.array_equal(x, label_one_hot)]
        indiv_class_images = [images[j] for j in indices]
        num_available_images = len(indiv_class_images)
        print(f"Augmenting class {i}: Found {num_available_images} images. Need to add {num_to_add}.")
        # for k in range(int(to_add[i])):
        #     # Apply transformations to create new image

        for k in range(num_to_add):
            original_image = indiv_class_images[k % num_available_images]

            # --- Đảm bảo original_image là (H, W, 1) ---
            if original_image.ndim == 4 and original_image.shape[0] == 1:
                 original_image = original_image.squeeze(axis=0)
            if original_image.ndim == 2:
                 original_image = np.expand_dims(original_image, axis=-1)
            if original_image.shape[-1] != 1:
                 print(f"Error: Original image for augmentation (class {i}) is not grayscale (shape: {original_image.shape}). Skipping.")
                 continue

        for k in range(int(to_add[i])):
            transformed_image = create_individual_transform(original_image, available_transforms)
            
            # Resize và reshape về 224x224x1
            # transformed_image = sk.transform.resize(transformed_image, (224, 224))
            # transformed_image = np.concatenate([transformed_image]*3, axis=-1)  # Tạo 3 kênh
            # transformed_image = transformed_image.reshape(1, 224, 224, 1)
            # transformed_image = transformed_image.reshape(1, 224, 224, 3)
            if transformed_image.ndim == 2:
                 transformed_image = np.expand_dims(transformed_image, axis=-1)
            if transformed_image.shape[-1] != 1:
                 print(f"Warning: Transform resulted in shape {transformed_image.shape}. Forcing grayscale.")
                 # Cố gắng chuyển về grayscale nếu có nhiều kênh
                 if transformed_image.shape[-1] == 3:
                     transformed_image = sk.color.rgb2gray(transformed_image)
                     transformed_image = np.expand_dims(transformed_image, axis=-1)
                 else: # Lấy kênh đầu tiên nếu không phải 3
                     transformed_image = transformed_image[..., :1]
            transformed_image = transformed_image.astype(np.float32)
            resized_transformed_image = resize_before_reshape(transformed_image, target_height, target_width)
            # resize_before_reshape PHẢI trả về (target_H, target_W, 1)

            # --- Kiểm tra lại shape sau resize ---
            if resized_transformed_image.shape != (target_height, target_width, 1):
                 print(f"Error after resize: Expected {(target_height, target_width, 1)}, got {resized_transformed_image.shape}. Reshaping forcefully.")
                 resized_transformed_image = resized_transformed_image.reshape(target_height, target_width, 1)            
            # # Resize image to ensure consistent dimensions
            # if transformed_image.shape[:2] != original_image.shape[:2]:
            #     transformed_image = sk.transform.resize(
            #         transformed_image, 
            #         original_image.shape[:2], 
            #         mode='reflect', 
            #         anti_aliasing=True
            #     )
            
            # Reshape based on model and dataset requirements
            if config.dataset == "CMMD":
                transformed_image = transformed_image.reshape(1, config.CMMD_IMG_SIZE['HEIGHT'], 
                                                             config.CMMD_IMG_SIZE['WIDTH'], 1)
            elif config.dataset == "INbreast":
                if hasattr(config, 'INBREAST_IMG_SIZE'):
                    transformed_image = transformed_image.reshape(1, config.INBREAST_IMG_SIZE['HEIGHT'],
                                                                config.INBREAST_IMG_SIZE['WIDTH'], 1)
                else:
                    transformed_image = transformed_image.reshape(1, 224, 224, 1)
            elif config.is_roi or config.model == "CNN":
                transformed_image = transformed_image.reshape(1, config.ROI_IMG_SIZE['HEIGHT'],
                                                             config.ROI_IMG_SIZE['WIDTH'], 1)
            elif config.model == "VGG" or config.model == "Inception":
                transformed_image = transformed_image.reshape(1, config.MINI_MIAS_IMG_SIZE['HEIGHT'],
                                                             config.MINI_MIAS_IMG_SIZE['WIDTH'], 1)
            elif config.model == "VGG-common":
                transformed_image = transformed_image.reshape(1, config.VGG_IMG_SIZE['HEIGHT'],
                                                             config.VGG_IMG_SIZE['WIDTH'], 1)
            elif config.model == "ResNet":
                transformed_image = transformed_image.reshape(1, config.RESNET_IMG_SIZE['HEIGHT'],
                                                             config.RESNET_IMG_SIZE['WIDTH'], 1)
            elif config.model == "MobileNet":
                transformed_image = transformed_image.reshape(1, config.MOBILE_NET_IMG_SIZE['HEIGHT'],
                                                             config.MOBILE_NET_IMG_SIZE['WIDTH'], 1)
            elif config.model == "DenseNet":
                transformed_image = transformed_image.reshape(1, config.DENSE_NET_IMG_SIZE['HEIGHT'],
                                                             config.DENSE_NET_IMG_SIZE['WIDTH'], 1)

            # Add transformed image and label to dataset
            images_with_transforms = np.append(images_with_transforms, transformed_image, axis=0)
            transformed_label = label_one_hot.reshape(1, len(label_one_hot))
            labels_with_transforms = np.append(labels_with_transforms, transformed_label, axis=0)

    return images_with_transforms, labels_with_transforms

def random_rotation(image_array: np.ndarray):
    """
    Randomly rotate the image.
    
    :param image_array: input image
    :return: randomly rotated image
    """
    random_degree = random.uniform(-20, 20)
    return sk.transform.rotate(image_array, random_degree)

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

def gamma_correction(image_array: np.ndarray):
    """
    Apply gamma correction to enhance contrast.
    Especially useful for mammogram images.
    
    :param image_array: input image
    :return: gamma corrected image
    """
    gamma = random.uniform(0.8, 1.5)
    return sk.exposure.adjust_gamma(image_array, gamma)

def random_zoom(image_array: np.ndarray):
    """
    Apply random zoom to focus on center region.
    Especially useful for mammogram images to focus on lesion areas.
    
    :param image_array: input image
    :return: zoomed image
    """
    h, w = image_array.shape[:2]
    center = (w // 2, h // 2)
    zoom_factor = random.uniform(0.8, 0.95)
    zoom_size = int(min(h, w) * zoom_factor)
    
    # Ensure zoom region doesn't exceed image boundaries
    top = max(0, center[1] - zoom_size//2)
    bottom = min(h, center[1] + zoom_size//2)
    left = max(0, center[0] - zoom_size//2)
    right = min(w, center[0] + zoom_size//2)
    
    zoomed = image_array[top:bottom, left:right]
    return sk.transform.resize(zoomed, (h, w))

def random_contrast(image_array: np.ndarray):
    """
    Adjust image contrast randomly.
    Useful for highlighting features in mammogram images.
    
    :param image_array: input image
    :return: contrast adjusted image
    """
    # Adjust contrast using histogram stretching
    p_low = random.uniform(0.01, 0.1)
    p_high = random.uniform(0.9, 0.99)
    
    # Ensure pixel values are in [0, 1] range
    image_array = np.clip(image_array, 0, 1)
    
    # Apply contrast stretching
    v_min, v_max = np.percentile(image_array, [p_low*100, p_high*100])
    return sk.exposure.rescale_intensity(image_array, in_range=(v_min, v_max), out_range=(0, 1))

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
def create_individual_transform(image: np.array, transforms: dict):
    """
    Tạo biến đổi cho một ảnh cụ thể.
    """
    num_transformations_to_apply = random.randint(1, len(transforms))
    transformed_image = image.copy()  # Bắt đầu với bản sao của ảnh gốc
    
    # Áp dụng các biến đổi ngẫu nhiên
    for _ in range(num_transformations_to_apply):
        key = random.choice(list(transforms))
        transformed_image = transforms[key](transformed_image)
    
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
    
#     return counts.tolist()
def get_class_balances(y_vals):
    """Đếm số lượng mẫu trong mỗi lớp."""
    # Khởi tạo biến counts với giá trị mặc định
    counts = np.zeros(2)  # Mặc định 2 lớp
    
    if config.dataset in ["CMMD", "INbreast"]:
        num_classes = 2  # benign và malignant
        counts = np.zeros(num_classes)
        for y_val in y_vals:
            if np.array_equal(y_val, [1, 0]):  # benign
                counts[0] += 1
            elif np.array_equal(y_val, [0, 1]):  # malignant
                counts[1] += 1
    else:
        # Kiểm tra cấu trúc y_vals để xử lý phù hợp
        if len(y_vals) > 0 and hasattr(y_vals[0], '__len__'):
            # One-hot encoded labels
            num_classes = len(y_vals[0])
            counts = np.zeros(num_classes)
            for y_val in y_vals:
                for i in range(num_classes):
                    counts[i] += y_val[i]
        else:
            # Nhãn dạng scalar
            unique_values = np.unique(y_vals)
            num_classes = len(unique_values)
            counts = np.zeros(num_classes)
            for i, cls in enumerate(unique_values):
                counts[i] = np.sum(np.array(y_vals) == cls)
    
    return counts.tolist()


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