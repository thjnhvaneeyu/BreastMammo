import ssl
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Flatten, Input, GlobalAveragePooling2D
from tensorflow.python.keras import Sequential

import config

# Needed to download pre-trained weights for ImageNet
ssl._create_default_https_context = ssl._create_unverified_context


# def create_mobilenet_model(num_classes: int):
#     """
#     Function to create a MobileNetV2 model pre-trained with custom FC Layers.
#     If the "advanced" command line argument is selected, adds an extra convolutional layer with extra filters to support
#     larger images.
#     :param num_classes: The number of classes (labels).
#     :return: The MobileNetV2 model.
#     """
#     # Reconfigure single channel input into a greyscale 3 channel input
#     img_input = Input(shape=(config.DENSE_NET_IMG_SIZE['HEIGHT'], config.DENSE_NET_IMG_SIZE['WIDTH'], 1))
#     img_conc = Concatenate()([img_input, img_input, img_input])

#     # Generate a MobileNetV2 model with pre-trained ImageNet weights, input as given above, excluded fully connected layers.
#     model_base = MobileNetV2(include_top=False, weights="imagenet", input_tensor=img_conc)

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

# def create_mobilenet_model(num_classes: int):
#     # 1) Input grayscale → 3 channels
#     inp = Input(shape=(config.MOBILE_NET_IMG_SIZE['HEIGHT'],
#                        config.MOBILE_NET_IMG_SIZE['WIDTH'], 1))
#     x = Concatenate()([inp, inp, inp])

#     # 2) Base MobileNetV2
#     base = MobileNetV2(include_top=False,
#                        weights='imagenet',
#                        input_tensor=x)
#     x = base.output

#     # 3) Head
#     x = Flatten()(x)
#     x = Dropout(0.2, seed=config.RANDOM_SEED)(x)
#     x = Dense(512, activation='relu')(x)
#     x = Dense(32, activation='relu')(x)
#     if num_classes == 2:
#         out = Dense(1, activation='sigmoid', name='Output')(x)
#     else:
#         out = Dense(num_classes, activation='softmax', name='Output')(x)

#     return Model(inputs=inp, outputs=out, name='MobileNetV2_Custom')

def create_mobilenet_model(num_classes: int):
    # Sử dụng giá trị từ config một cách an toàn
    img_height = getattr(config, 'MOBILE_NET_IMG_SIZE', {}).get('HEIGHT', 224)
    img_width = getattr(config, 'MOBILE_NET_IMG_SIZE', {}).get('WIDTH', 224)
    
    # 1) Input grayscale → 3 channels
    # inp = Input(shape=(img_height, img_width, 1), name="Input_Grayscale")
    # x_conc = Concatenate(name="Input_RGB_Grayscale")([inp, inp, inp]) # Đổi tên biến để không trùng inp

    # # 2) Base MobileNetV2
    # base = MobileNetV2(include_top=False,
    #                    weights='imagenet',
    #                    input_tensor=x_conc) # Sử dụng x_conc
    # x = base.output
    inp = Input(shape=(img_height, img_width, 3), name="Input_RGB") # THAY ĐỔI Ở ĐÂY: từ 1 thành 3 kênh

    base = MobileNetV2(include_top=False,
                       weights='imagenet',
                       input_tensor=inp) # input_tensor bây giờ là inp (3 kênh)
    x = base.output
    # 3) Head
    # Có thể chọn GlobalAveragePooling2D thay vì Flatten tùy theo hiệu năng mong muốn
    x = GlobalAveragePooling2D(name="GlobalAvgPool")(x) 
    # x = Flatten()(x) # Nếu giữ Flatten

    # Sử dụng giá trị từ config một cách an toàn
    random_seed_val = getattr(config, 'RANDOM_SEED', None)
    x = Dropout(0.2, seed=random_seed_val, name="Dropout_1")(x) # Đặt tên để dễ debug
    x = Dense(512, activation='relu', name="Dense_1")(x)
    x = Dense(32, activation='relu', name="Dense_2")(x)

    # Lớp output - Đã sửa đổi
    if num_classes == 2:
        out = Dense(num_classes, activation='softmax', name='Output')(x) # 2 units, softmax
    elif num_classes > 2:
        out = Dense(num_classes, activation='softmax', name='Output')(x)
    else: # num_classes = 1 hoặc < 1
        print(f"[WARNING] mobilenet_v2: num_classes is {num_classes}. Defaulting output to 1 neuron with sigmoid for safety, but review CnnModel's compile logic.")
        out = Dense(1, activation='sigmoid', name='Output')(x)

    final_model = Model(inputs=inp, outputs=out, name='MobileNetV2_Custom') # Đổi tên biến model

    verbose_mode_val = getattr(config, 'verbose_mode', False)
    if verbose_mode_val:
        print("CNN Model used (MobileNetV2_Custom):")
        final_model.summary()
        
    return final_model