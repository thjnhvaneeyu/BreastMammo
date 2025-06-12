# import ssl

# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras.layers import Concatenate, Dense, Dropout, Flatten, Input
# from tensorflow.python.keras import Sequential

# import config

# # Needed to download pre-trained weights for ImageNet
# ssl._create_default_https_context = ssl._create_unverified_context


# def create_resnet50_model(num_classes: int):
#     """
#     Function to create a ResNet50 model pre-trained with custom FC Layers.
#     If the "advanced" command line argument is selected, adds an extra convolutional layer with extra filters to support
#     larger images.
#     :param num_classes: The number of classes (labels).
#     :return: The ResNet50 model.
#     """
#     # Reconfigure single channel input into a greyscale 3 channel input
#     img_input = Input(shape=(config.RESNET_IMG_SIZE['HEIGHT'], config.RESNET_IMG_SIZE['WIDTH'], 1))
#     img_conc = Concatenate()([img_input, img_input, img_input])

#     # Generate a ResNet50 model with pre-trained ImageNet weights, input as given above, excluding fully connected
#     # layers.
#     model_base = ResNet50(include_top=False, weights="imagenet", input_tensor=img_conc)

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

import ssl
import tensorflow as tf # Thêm import này
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Flatten, Input, GlobalAveragePooling2D # Thêm GlobalAveragePooling2D
from tensorflow.keras.models import Model # Sửa từ tensorflow.python.keras sang tensorflow.keras.models

import config

# Needed to download pre-trained weights for ImageNet
ssl._create_default_https_context = ssl._create_unverified_context


def create_resnet50_model(num_classes: int):
    """
    Function to create a ResNet50 model pre-trained with custom FC Layers.
    :param num_classes: The number of classes (labels).
    :return: The ResNet50 model.
    """
    # Sử dụng giá trị từ config một cách an toàn
    img_height = getattr(config, 'RESNET_IMG_SIZE', {}).get('HEIGHT', 224)
    img_width = getattr(config, 'RESNET_IMG_SIZE', {}).get('WIDTH', 224)

    # Reconfigure single channel input into a greyscale 3 channel input
    img_input = Input(shape=(img_height, img_width, 1), name="Input_Grayscale")
    img_conc = Concatenate(name="Input_RGB_Grayscale")([img_input, img_input, img_input])

    # Generate a ResNet50 model with pre-trained ImageNet weights
    model_base = ResNet50(include_top=False, weights="imagenet", input_tensor=img_conc)

    x = model_base.output
    x = GlobalAveragePooling2D(name="GlobalAvgPool")(x) # Hoặc Flatten()

    # Fully connected layers.
    # Sử dụng giá trị từ config một cách an toàn
    random_seed_val = getattr(config, 'RANDOM_SEED', None)
    x = Dropout(0.2, seed=random_seed_val, name="Dropout_1")(x)
    x = Dense(units=512, activation='relu', name='Dense_1')(x)
    x = Dense(units=32, activation='relu', name='Dense_2')(x)

    # Final output layer - Đã sửa đổi
    if num_classes == 2:
        outputs = Dense(num_classes, activation='softmax', name='Output')(x)
    elif num_classes > 2:
        outputs = Dense(num_classes, activation='softmax', name='Output')(x)
    else: # num_classes = 1 hoặc < 1
        print(f"[WARNING] resnet50: num_classes is {num_classes}. Defaulting output to 1 neuron with sigmoid for safety, but review CnnModel's compile logic.")
        outputs = Dense(1, activation='sigmoid', name='Output')(x)
        
    model = Model(inputs=img_input, outputs=outputs, name="ResNet50_Custom")

    verbose_mode_val = getattr(config, 'verbose_mode', False)
    # if verbose_mode_val:
    #     print("CNN Model used (ResNet50_Custom):")
    #     model.summary()

    return model