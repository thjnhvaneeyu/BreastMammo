# from tensorflow.keras.layers import Dense, Dropout, Flatten
# from tensorflow.python.keras import Sequential
# from tensorflow.python.keras.layers import Conv2D, MaxPooling2D

import config


# def create_basic_cnn_model(num_classes: int):
#     """
#     Function to create a basic CNN.
#     :param num_classes: The number of classes (labels).
#     :return: A basic CNN model.
#     """
#     model = Sequential()

#     # Convolutional + spooling layers
#     model.add(Conv2D(64, (5, 5), input_shape=(config.ROI_IMG_SIZE['HEIGHT'], config.ROI_IMG_SIZE['WIDTH'], 1)))
#     model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#     model.add(Conv2D(32, (5, 5), padding='same'))
#     model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#     model.add(Flatten())

#     # Dropout
#     model.add(Dropout(0.5, seed=config.RANDOM_SEED, name="Dropout_1"))

#     # FC
#     model.add(Dense(1024, activation='relu', name='Dense_2'))

#     # Output
#     if num_classes == 2:
#         model.add(Dense(1, activation='sigmoid', kernel_initializer="random_uniform", name='Output'))
#     else:
#         model.add(Dense(num_classes, activation='softmax', kernel_initializer="random_uniform", name='Output'))

#     # Print model details if running in debug mode.
#     if config.verbose_mode:
#         print(model.summary())

#     return model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    InputLayer,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dropout,
    Dense
)

def create_basic_cnn_model(num_classes: int):
    """
    Function to create a basic CNN.
    :param num_classes: The number of classes (labels).
    :return: A basic CNN model.
    """
    model = Sequential()
    
    # 1) Định nghĩa InputLayer ngay từ đầu,
    #    để mọi layer sau này đều biết được input_shape
    model.add(InputLayer(
        input_shape=(
            config.ROI_IMG_SIZE['HEIGHT'],
            config.ROI_IMG_SIZE['WIDTH'],
            1
        ),
        name="Input"
    ))
    
    # 2) Convolutional + pooling layers
    model.add(Conv2D(64, (5, 5), activation='relu', name="Conv1"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name="Pool1"))
    model.add(Conv2D(32, (5, 5), padding='same', activation='relu', name="Conv2"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name="Pool2"))
    
    # 3) Flatten giờ không còn lỗi về layer thiếu input_shape
    model.add(Flatten(name="Flatten"))
    
    # 4) Dropout
    model.add(Dropout(0.5, seed=config.RANDOM_SEED, name="Dropout_1"))
    
    # 5) Fully Connected
    model.add(Dense(1024, activation='relu', name='Dense_2'))
    
    # # 6) Output layer
    # if num_classes == 2:
    #     # Nhị phân
    #     model.add(Dense(
    #         2,
    #         activation='sigmoid',
    #         kernel_initializer="random_uniform",
    #         name='Output'
    #     ))
    # else:
    #     # Đa lớp
    #     model.add(Dense(
    #         num_classes,
    #         activation='softmax',
    #         kernel_initializer="random_uniform",
    #         name='Output'
    #     ))
    
    # # In summary nếu đang ở chế độ debug
    # if config.verbose_mode:
    #     model.summary()
    
    # return model
    if num_classes == 2:
        # Nhị phân, nhưng target là one-hot (2 classes) và loss là CategoricalCrossentropy
        model.add(Dense(
            num_classes, # Sử dụng num_classes (sẽ là 2)
            activation='softmax', # Dùng softmax cho CategoricalCrossentropy
            # kernel_initializer="random_uniform", # Có thể bỏ để dùng default của Keras
            name='Output'
        ))
    elif num_classes > 2: # Trường hợp đa lớp rõ ràng
        model.add(Dense(
            num_classes,
            activation='softmax',
            # kernel_initializer="random_uniform",
            name='Output'
        ))
    else: # Trường hợp num_classes = 1 (ít khả thi với CategoricalCrossentropy, nhưng để phòng)
          # Hoặc nếu có lỗi logic num_classes < 1
        print(f"[WARNING] basic_cnn: num_classes is {num_classes}. Defaulting output to 1 neuron with sigmoid for safety, but review CnnModel's compile logic.")
        model.add(Dense(
            1,
            activation='sigmoid',
            # kernel_initializer="random_uniform",
            name='Output'
        ))
    
    # In summary nếu đang ở chế độ debug
    verbose_mode_val = getattr(config, 'verbose_mode', False)
    if verbose_mode_val:
        model.summary()
    
    return model