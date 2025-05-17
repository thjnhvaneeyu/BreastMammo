# import ssl

# from tensorflow.keras.applications import VGG19
# from tensorflow.keras.layers import Concatenate, Dense, Dropout, Flatten, Input
# from tensorflow.python.keras.models import Model
# from tensorflow.python.keras import Sequential
# from tensorflow.python.keras.layers import Conv2D, MaxPooling2D

# import config

# # Required to download pre-trained weights for ImageNet (stored in ~/.keras/models/)
# ssl._create_default_https_context = ssl._create_unverified_context


# def create_vgg19_model(num_classes: int):
#     """
#     Creates a CNN from an existing architecture with pre-trained weights on ImageNet.
#     :return: The VGG19 model.
#     """
#     base_model = Sequential(name="Base_Model")

#     # Reconfigure a single channel image input (greyscale) into a 3-channel greyscale input (tensor).
#     single_channel_input = Input(shape=(config.MINI_MIAS_IMG_SIZE['HEIGHT'], config.MINI_MIAS_IMG_SIZE['WIDTH'], 1))
#     triple_channel_input = Concatenate()([single_channel_input, single_channel_input, single_channel_input])
#     input_model = Model(inputs=single_channel_input, outputs=triple_channel_input)
#     base_model.add(input_model)

#     # Generate extra convolutional layers for model to put at the beginning
#     base_model.add(Conv2D(64, (5, 5),
#                           activation='relu',
#                           padding='same'))
#     base_model.add(Conv2D(32, (5, 5),
#                           activation='relu',
#                           padding='same'))
#     base_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

#     # Generate a VGG19 model with pre-trained ImageNet weights, input as given above, excluding the fully
#     # connected layers.
#     base_model.add(Conv2D(64, (3, 3),
#                           activation='relu',
#                           padding='same'))
#     pre_trained_model = VGG19(include_top=False, weights="imagenet",
#                               input_shape=[config.VGG_IMG_SIZE['HEIGHT'], config.VGG_IMG_SIZE['WIDTH'], 3])

#     # Exclude input layer and first convolutional layer of VGG model.
#     pre_trained_model_trimmed = Sequential(name="Pre-trained_Model")
#     for layer in pre_trained_model.layers[2:]:
#         pre_trained_model_trimmed.add(layer)

#     # Add fully connected layers
#     model = Sequential(name="Breast_Cancer_Model")

#     # Start with base model consisting of convolutional layers
#     model.add(base_model)
#     model.add(pre_trained_model_trimmed)

#     # Flatten layer to convert each input into a 1D array (no parameters in this layer, just simple pre-processing).
#     model.add(Flatten())

#     # Add fully connected hidden layers and dropout layers between each for regularisation.
#     model.add(Dropout(0.2))
#     model.add(Dense(units=512, activation='relu', kernel_initializer="random_uniform", name='Dense_1'))
#     # model.add(Dropout(0.2))
#     model.add(Dense(units=32, activation='relu', kernel_initializer="random_uniform", name='Dense_2'))

#     # Final output layer that uses softmax activation function (because the classes are exclusive).
#     # if config.dataset == "CBIS-DDSM" or config.dataset == "mini-MIAS-binary":
#     if config.dataset in ["CBIS-DDSM", "mini-MIAS-binary", "CMMD", "CMMD_binary"]:
#         model.add(Dense(1, activation='sigmoid', kernel_initializer="random_uniform", name='Output'))
#     elif config.dataset == "mini-MIAS":
#         model.add(Dense(num_classes, activation='softmax', kernel_initializer="random_uniform", ame='Output'))

#     # Print model details if running in debug mode.
#     if config.verbose_mode:
#         print(base_model.summary())
#         print(pre_trained_model_trimmed.summary())
#         print(model.summary())

#     return model

import ssl
import tensorflow as tf # Thêm import
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D # Thêm Conv2D, MaxPooling2D từ keras.layers
from tensorflow.keras.models import Model, Sequential # Sửa từ tensorflow.python.keras

import config

# Required to download pre-trained weights for ImageNet (stored in ~/.keras/models/)
ssl._create_default_https_context = ssl._create_unverified_context


def create_vgg19_model(num_classes: int): # Thêm num_classes vào tham số để nhất quán
    """
    Creates a CNN from an existing architecture with pre-trained weights on ImageNet.
    :param num_classes: The number of classes for the output layer.
    :return: The VGG19 model.
    """
    # Sử dụng giá trị từ config một cách an toàn
    input_height = getattr(config, 'MINI_MIAS_IMG_SIZE', {}).get('HEIGHT', 224)
    input_width = getattr(config, 'MINI_MIAS_IMG_SIZE', {}).get('WIDTH', 224)
    vgg_target_height = getattr(config, 'VGG_IMG_SIZE', {}).get('HEIGHT', 224)
    vgg_target_width = getattr(config, 'VGG_IMG_SIZE', {}).get('WIDTH', 224)


    # Input Layer
    single_channel_input = Input(shape=(input_height, input_width, 1), name="Input_Grayscale")
    triple_channel_input = Concatenate(name="Input_RGB_Grayscale")([single_channel_input, single_channel_input, single_channel_input])
    
    # Custom Convolutional Layers at the beginning
    # x = Conv2D(64, (5, 5), activation='relu', padding='same', name="CustomConv1")(triple_channel_input)
    # x = Conv2D(32, (5, 5), activation='relu', padding='same', name="CustomConv2")(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name="CustomPool1")(x)
    # x = Conv2D(64, (3, 3), activation='relu', padding='same', name="CustomConv3_BridgeToVGG")(x) # Lớp này để khớp input shape của VGG19 nếu cần điều chỉnh
    
    # Pre-trained VGG19 model
    # input_shape của VGG19 cần khớp với output của lớp CustomConv3_BridgeToVGG
    # Giả sử output của CustomConv3_BridgeToVGG là (None, H_vgg, W_vgg, 64)
    # và VGG19 yêu cầu (H_vgg, W_vgg, 3). Điều này không khớp.
    # Cách tiếp cận phổ biến là dùng VGG19 trực tiếp với input_tensor=triple_channel_input (đã resize phù hợp)
    # Hoặc là chỉ dùng các khối của VGG19.

    # -------- CÁCH TIẾP CẬN ĐƠN GIẢN HƠN VÀ PHỔ BIẾN HƠN: --------
    # Resize input cho VGG19
    # (Lưu ý: input_shape của VGG19 gốc là 224x224. Nếu config.VGG_IMG_SIZE khác, cần xem xét)
    # resized_for_vgg_input = tf.keras.layers.Resizing(vgg_img_height, vgg_img_width, name="ResizeForVGG")(triple_channel_input)
    # Thay vào đó, hãy Resize input cho VGG19 nếu kích thước input ban đầu khác với kích thước VGG mong đợi
    if input_height != vgg_target_height or input_width != vgg_target_width:
        processed_input_for_vgg = tf.keras.layers.Resizing(vgg_target_height, vgg_target_width, name="ResizeForVGG")(triple_channel_input)
    else:
        processed_input_for_vgg = triple_channel_input
    pre_trained_model_base = VGG19(include_top=False, weights="imagenet", input_tensor=processed_input_for_vgg)
    
    # Lấy output của base model
    x = pre_trained_model_base.output

    # Flatten layer
    # x = Flatten(name="Flatten")(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="GlobalAvgPool")(x)

    # Fully connected hidden layers and dropout layers
    # Sử dụng giá trị từ config một cách an toàn
    random_seed_val = getattr(config, 'RANDOM_SEED', None)
    x = Dropout(0.2, seed=random_seed_val, name="Dropout_FC1")(x) # Thêm seed nếu cần
    x = Dense(units=512, activation='relu', kernel_initializer="random_uniform", name='Dense_1')(x)
    x = Dense(units=32, activation='relu', kernel_initializer="random_uniform", name='Dense_2')(x)

    # Final output layer - Sửa đổi để dùng num_classes và softmax cho trường hợp 2 lớp
    # Logic `config.dataset` ở đây có thể không còn phù hợp nếu `main.py` đã chuẩn hóa `num_classes`.
    if num_classes == 2:
        outputs = Dense(num_classes, activation='softmax', kernel_initializer="random_uniform", name='Output')(x)
    elif num_classes > 2: # Các trường hợp đa lớp khác
        outputs = Dense(num_classes, activation='softmax', kernel_initializer="random_uniform", name='Output')(x)
    else: # num_classes = 1 hoặc < 1
        print(f"[WARNING] vgg19: num_classes is {num_classes}. Defaulting output to 1 neuron with sigmoid for safety, but review CnnModel's compile logic.")
        outputs = Dense(1, activation='sigmoid', kernel_initializer="random_uniform", name='Output')(x)

    model = Model(inputs=single_channel_input, outputs=outputs, name="VGG19_Custom")

    verbose_mode_val = getattr(config, 'verbose_mode', False)
    if verbose_mode_val:
        model.summary()

    return model