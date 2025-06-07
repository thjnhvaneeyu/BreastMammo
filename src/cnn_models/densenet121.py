import ssl

from tensorflow.keras.applications import DenseNet121
from tensorflow.python.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Flatten, Input, GlobalAveragePooling2D # Thêm GlobalAveragePooling2D

import config

# Needed to download pre-trained weights for ImageNet
ssl._create_default_https_context = ssl._create_unverified_context


# def create_densenet121_model(num_classes: int):
#     """
#     Function to create a DenseNet121 model pre-trained with custom FC Layers.
#     If the "advanced" command line argument is selected, adds an extra convolutional layer with extra filters to support
#     larger images.
#     :param num_classes: The number of classes (labels).
#     :return: The DenseNet121 model.
#     """
#     # Reconfigure single channel input into a greyscale 3 channel input
#     img_input = Input(shape=(config.DENSE_NET_IMG_SIZE['HEIGHT'], config.DENSE_NET_IMG_SIZE['WIDTH'], 1))
#     img_conc = Concatenate()([img_input, img_input, img_input])

#     # Generate a DenseNet121 model with pre-trained ImageNet weights, input as given above, excluded fully connected layers.
#     model_base = DenseNet121(include_top=False, weights="imagenet", input_tensor=img_conc)

#     # Add fully connected layers
#     model = Sequential()
#     # Start with base model consisting of convolutional layers
#     model.add(model_base)

#     # Flatten layer to convert each input into a 1D array (no parameters in this layer, just simple pre-processing).
#     model.add(Flatten())

#     fully_connected = Sequential(name="Fully_Connected")
#     # Fully connected layers.
#     fully_connected.add(Dropout(0.2, seed=config.RANDOM_SEED, name="Dropout_1"))
#     fully_connected.add(Dense(units=512, activation='relu', name='Dense_1'))
#     # fully_connected.add(Dropout(0.2, name="Dropout_2"))
#     fully_connected.add(Dense(units=32, activation='relu', name='Dense_2'))

#     # Final output layer that uses softmax activation function (because the classes are exclusive).
#     if num_classes == 2:
#         fully_connected.add(Dense(1, activation='sigmoid', kernel_initializer="random_uniform", name='Output'))
#     else:
#         fully_connected.add(
#             Dense(num_classes, activation='softmax', kernel_initializer="random_uniform", name='Output'))

#     model.add(fully_connected)

#     # Print model details if running in debug mode.
#     if config.verbose_mode:
#         print("CNN Model used:")
#         print(model.summary())
#         print("Fully connected layers:")
#         print(fully_connected.summary())

#     return model

# def create_densenet121_model(num_classes: int):
#     """
#     Function to create a DenseNet121 model pre-trained with custom FC Layers.
#     :param num_classes: The number of classes (labels).
#     :return: The DenseNet121 model.
#     """
#     # Sử dụng giá trị từ config một cách an toàn
#     img_height = getattr(config, 'DENSE_NET_IMG_SIZE', {}).get('HEIGHT', 224)
#     img_width = getattr(config, 'DENSE_NET_IMG_SIZE', {}).get('WIDTH', 224)

#     # Reconfigure single channel input into a greyscale 3 channel input
#     img_input = Input(shape=(img_height, img_width, 1), name="Input_Grayscale")
#     img_conc = Concatenate(name="Input_RGB_Grayscale")([img_input, img_input, img_input])

#     # Generate a DenseNet121 model with pre-trained ImageNet weights
#     model_base = DenseNet121(include_top=False, weights="imagenet", input_tensor=img_conc)

#     # Get the output of the base model
#     x = model_base.output
    
#     # Add custom top layers
#     x = GlobalAveragePooling2D(name="GlobalAvgPool")(x) # Thường dùng GAP sau backbone
#     # Hoặc Flatten nếu muốn: x = Flatten(name="Flatten")(x)

#     # Fully connected layers.
#     # Sử dụng giá trị từ config một cách an toàn
#     random_seed_val = getattr(config, 'RANDOM_SEED', None)
#     x = Dropout(0.2, seed=random_seed_val, name="Dropout_1")(x)
#     x = Dense(units=512, activation='relu', name='Dense_1')(x)
#     x = Dense(units=32, activation='relu', name='Dense_2')(x)

#     # Final output layer - Đã sửa đổi
#     if num_classes == 2:
#         # Nhị phân, nhưng target là one-hot (2 classes) và loss là CategoricalCrossentropy
#         outputs = Dense(num_classes, activation='softmax', name='Output')(x)
#     elif num_classes > 2:
#         outputs = Dense(num_classes, activation='softmax', name='Output')(x)
#     else: # num_classes = 1 hoặc < 1
#         print(f"[WARNING] densenet121: num_classes is {num_classes}. Defaulting output to 1 neuron with sigmoid for safety, but review CnnModel's compile logic.")
#         outputs = Dense(1, activation='sigmoid', name='Output')(x)
        
#     # Create the final model
#     model = Model(inputs=img_input, outputs=outputs, name="DenseNet121_Custom")

#     # Print model details if running in debug mode.
#     verbose_mode_val = getattr(config, 'verbose_mode', False)
#     if verbose_mode_val:
#         print("CNN Model used (DenseNet121_Custom):")
#         model.summary()

#     return model


def create_densenet121_model(num_classes: int, input_shape: tuple):
    """
    Function to create a flexible DenseNet121 model that accepts 1-channel or 3-channel input.
    :param num_classes: The number of classes (labels).
    :param input_shape: The shape of the input images, e.g., (224, 224, 1) or (224, 224, 3).
    """
    # Lấy số kênh từ input_shape
    num_channels = input_shape[2]

    # Định nghĩa lớp Input linh hoạt dựa trên shape được cung cấp
    img_input = Input(shape=input_shape, name="flexible_input")

    # --- LOGIC XỬ LÝ KÊNH ĐẦU VÀO ---
    # Kiểm tra số kênh để quyết định có cần concatenate hay không
    if num_channels == 1:
        # Nếu đầu vào là ảnh xám (1 kênh), nhân 3 để tạo thành tensor 3 kênh
        print("[INFO] Input is 1-channel. Concatenating to 3 channels.")
        processed_input = Concatenate(name="concat_to_3_channels")([img_input, img_input, img_input])
    elif num_channels == 3:
        # Nếu đầu vào đã là 3 kênh, sử dụng trực tiếp
        print("[INFO] Input is 3-channel. Using it directly.")
        processed_input = img_input
    else:
        # Ném ra lỗi nếu số kênh không được hỗ trợ
        raise ValueError(f"Unsupported number of channels: {num_channels}. Expected 1 or 3.")

    # Generate a DenseNet121 model with pre-trained ImageNet weights
    # 'processed_input' giờ đây chắc chắn là 3 kênh
    model_base = DenseNet121(include_top=False, weights="imagenet", input_tensor=processed_input)

    # Lấy output của model cơ sở
    x = model_base.output
    
    # Thêm các lớp tùy chỉnh (giữ nguyên như code của bạn)
    x = GlobalAveragePooling2D(name="GlobalAvgPool")(x)

    random_seed_val = getattr(config, 'RANDOM_SEED', None)
    x = Dropout(0.2, seed=random_seed_val, name="Dropout_1")(x)
    x = Dense(units=512, activation='relu', name='Dense_1')(x)
    x = Dense(units=32, activation='relu', name='Dense_2')(x)

    # Lớp output cuối cùng (giữ nguyên logic của bạn)
    if num_classes == 2:
        outputs = Dense(num_classes, activation='softmax', name='Output')(x)
    elif num_classes > 2:
        outputs = Dense(num_classes, activation='softmax', name='Output')(x)
    else:
        print(f"[WARNING] densenet121: num_classes is {num_classes}. Defaulting to 1 neuron with sigmoid.")
        outputs = Dense(1, activation='sigmoid', name='Output')(x)
        
    # Tạo model cuối cùng
    model = Model(inputs=img_input, outputs=outputs, name="DenseNet121_Flexible")

    verbose_mode_val = getattr(config, 'verbose_mode', False)
    if verbose_mode_val:
        print("CNN Model used (DenseNet121_Flexible):")
        model.summary()

    return model