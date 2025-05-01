import tensorflow as tf
import tensorflow_io as tfio

import config


def create_dataset(x, y):
    """
    Generates a TF dataset for feeding in the data.
    Originally written as a group for the common pipeline.
    :param x: X inputs - paths to images
    :param y: y values - labels for images
    :return: the dataset
    """
    dataset = tf.data.Dataset.from_tensor_slices((x, y))

    # Map values from dicom image path to array
    dataset = dataset.map(parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Dataset to cache data and repeat until all samples have been run once in each epoch
    dataset = dataset.cache().repeat(1)
    dataset = dataset.batch(config.batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


# def parse_function(filename, label):
#     """
#     Mapping function to convert filename to array of pixel values.
#     Supports resizing based on the model and dataset being used.
#     """
#     image_bytes = tf.io.read_file(filename)
    
#     # Decode DICOM or PNG/JPG based on file format
#     if filename.endswith('.dcm'):
#         image = tfio.image.decode_dicom_image(image_bytes, color_dim=True, dtype=tf.uint16)
#         as_png = tf.image.encode_png(image[0])
#         decoded_png = tf.io.decode_png(as_png, channels=1)
#         image = decoded_png
#     else:
#         image = tf.io.decode_image(image_bytes, channels=1)

#     # Resize based on the model and dataset configuration
#     if config.dataset == "CMMD":
#         height = config.CMMD_IMG_SIZE["HEIGHT"]
#         width = config.CMMD_IMG_SIZE["WIDTH"]
#         image = tf.image.resize_with_pad(image, height, width)
#     elif config.dataset == "INbreast":
#         height = config.INBREAST_IMG_SIZE["HEIGHT"]
#         width = config.INBREAST_IMG_SIZE["WIDTH"]
#         image = tf.image.resize_with_pad(image, height, width)
    
#     # Normalize pixel values to [0, 1]
#     image /= 255.0

#     return image, label

def parse_function(filename, label):
    """
    Hàm map để đọc file ảnh (DICOM hoặc PNG) thành tensor và tiền xử lý (đổi kích thước, số kênh).
    """
    image_bytes = tf.io.read_file(filename)
    # Kiểm tra phần mở rộng để quyết định decode
    # Nếu là DICOM (*.dcm) dùng tfio, nếu không thì decode hình ảnh (PNG/JPG)
    def decode_dicom():
        image = tfio.image.decode_dicom_image(image_bytes, color_dim=True, dtype=tf.uint16)  # shape [1, H, W, 1]
        image_png = tf.image.encode_png(image[0])    # chuyển sang định dạng PNG trong bộ nhớ
        img = tf.io.decode_png(image_png, channels=1)  # decode thành tensor 1 kênh
        return img
    def decode_image():
        # Giải mã ảnh thường (PNG/JPG) - giả sử ảnh grayscale 1 kênh
        img = tf.io.decode_image(image_bytes, channels=1, dtype=tf.uint8)
        return img
    # Chọn decode phù hợp
    image = tf.cond(tf.strings.regex_full_match(filename, ".*\\.dcm$"), decode_dicom, decode_image)
    # Nếu model cần 3 kênh, chuyển ảnh xám 1 kênh -> 3 kênh
    # if config.dataset.upper().startswith("CMMD"):
    #     target_height = config.CMMD_IMG_SIZE["HEIGHT"]
    #     target_width  = config.CMMD_IMG_SIZE["WIDTH"]
    # # else để nguyên các branch model cũ
    # elif config.model in ["VGG", "VGG-common", "ResNet", "MobileNet", "DenseNet", "Inception"]:
    #     image = tf.image.grayscale_to_rgb(image)  # từ (H,W,1) -> (H,W,3)

    # --- Chuyển grayscale->RGB cho CMMD và INbreast (dùng pretrained ImageNet) ---
    if config.dataset in ["CMMD_binary", "INbreast"]:
        image = tf.image.grayscale_to_rgb(image)
    # Ngoài ra với các backbone ImageNet khác (mini-MIAS, CBIS-DDSM…)
    elif config.model in ["VGG", "VGG-common", "ResNet", "MobileNet", "DenseNet", "Inception"]:
        image = tf.image.grayscale_to_rgb(image)

    # Xác định kích thước đích
    if config.model == "CNN" or config.is_roi:
        target_height = config.ROI_IMG_SIZE["HEIGHT"]; target_width = config.ROI_IMG_SIZE["WIDTH"]
    elif config.model == "VGG" or config.model == "Inception":
        target_height = config.MINI_MIAS_IMG_SIZE["HEIGHT"]; target_width = config.MINI_MIAS_IMG_SIZE["WIDTH"]
    elif config.model == "VGG-common":
        target_height = config.VGG_IMG_SIZE["HEIGHT"]; target_width = config.VGG_IMG_SIZE["WIDTH"]
    elif config.model == "ResNet":
        target_height = config.RESNET_IMG_SIZE["HEIGHT"]; target_width = config.RESNET_IMG_SIZE["WIDTH"]
    elif config.model == "MobileNet":
        target_height = config.MOBILE_NET_IMG_SIZE["HEIGHT"]; target_width = config.MOBILE_NET_IMG_SIZE["WIDTH"]
    elif config.model == "DenseNet" or config.model == "Inception":
        target_height = config.INCEPTION_IMG_SIZE["HEIGHT"]; target_width = config.INCEPTION_IMG_SIZE["WIDTH"]
    else:
        target_height = 224; target_width = 224
    # Resize (và pad nếu cần) về kích thước đích
    image = tf.image.resize_with_pad(image, target_height, target_width)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label