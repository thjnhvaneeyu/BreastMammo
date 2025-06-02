# from tensorflow.keras.layers import Dense, Dropout, Flatten
# from tensorflow.python.keras import Sequential
# from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf
import config
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras import regularizers
lambda_val = 0.0001 # Start with a small value like 0.001 or 0.0001
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
    Dense,
    Activation,
    BatchNormalization
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
    
    # # 2) Convolutional + pooling layers
    # # model.add(Conv2D(32, (5, 5), activation='relu', name="Conv1"))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2), name="Pool1"))
    # model.add(Conv2D(16, (5, 5), padding='same', activation='relu', name="Conv2"))
    # model.add(tf.keras.layers.Conv2D(
    #     64, (5, 5),
    #     activation='relu',
    #     kernel_regularizer=regularizers.l2(lambda_val), # Added L2 regularizer
    #     name="Conv1"
    # ))
    # model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name="Pool1"))

    # model.add(tf.keras.layers.Conv2D(
    #     32, (5, 5),
    #     padding='same',
    #     activation='relu',
    #     kernel_regularizer=regularizers.l2(lambda_val), # Added L2 regularizer
    #     name="Conv2"
    # ))
    # model.add(BatchNormalization(name="BN1")) # Ngay sau Conv2D
    # model.add(Activation('relu', name="Relu1")) # Hoặc gộp activation='relu' vào Conv2D
    # model.add(MaxPooling2D((2, 2), strides=(2, 2), name="Pool2"))
    
    # # 3) Flatten giờ không còn lỗi về layer thiếu input_shape
    # model.add(Flatten(name="Flatten"))
    
    # # 4) Dropout
    # model.add(Dropout(0.6, seed=config.RANDOM_SEED, name="Dropout_1"))
    
    # # 5) Fully Connected
    # model.add(Dense(256, activation='relu', name='Dense_2'))
    # # Block 1
    # model.add(Conv2D(
    #     64, (5, 5), # Kernel 5x5, 64 filters
    #     activation='relu', # Sử dụng 'relu' trực tiếp thay vì BN + Activation riêng ở lớp đầu
    #     padding='same', # Thêm padding để giữ kích thước không gian
    #     kernel_regularizer=regularizers.l2(lambda_val),
    #     name="Conv1_5x5_64"
    # ))
    # model.add(BatchNormalization(name="BN1"))
    # model.add(Activation('relu', name="Relu1_after_BN"))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2), name="Pool1"))
    
    # # Block 2
    # model.add(Conv2D(
    #     128, (3, 3), # Kernel 3x3, 128 filters
    #     padding='same',
    #     kernel_regularizer=regularizers.l2(lambda_val),
    #     name="Conv2_3x3_128"
    # ))
    # model.add(BatchNormalization(name="BN2"))
    # model.add(Activation('relu', name="Relu2"))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2), name="Pool2"))

    # # Block 3 (New)
    # model.add(Conv2D(
    #     64, (3, 3), # Kernel 3x3, 64 filters
    #     padding='same',
    #     # kernel_regularizer=regularizers.l2(lambda_val),
    #     name="Conv3_3x3_64"
    # ))
    # model.add(BatchNormalization(name="BN3"))
    # model.add(Activation('relu', name="Relu3"))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2), name="Pool3"))
    
    # model.add(Flatten(name="Flatten"))
    # model.add(Dropout(0.3, seed=config.RANDOM_SEED if hasattr(config, 'RANDOM_SEED') else None, name="Dropout_FC")) # Tăng Dropout
    
    # model.add(Dense(512, activation='relu', name='Dense_FC_512'))
    # # Trong basic_cnn.py
    model.add(Conv2D(32, (3, 3), padding='same',kernel_initializer='he_normal', name="Conv1_32"))
    model.add(BatchNormalization(name="BN1"))
    # model.add(Activation('relu', name="Relu1"))
    model.add(LeakyReLU(alpha=0.01, name="LeakyRelu1")) # Hoặc alpha=0.2
    model.add(MaxPooling2D((2, 2), name="Pool1"))

    model.add(Conv2D(64, (3, 3), padding='same', name="Conv2_64"))
    model.add(BatchNormalization(name="BN2"))
    model.add(Activation('relu', name="Relu2"))
    model.add(MaxPooling2D((2, 2), name="Pool2"))

    model.add(Conv2D(128, (3, 3), padding='same', name="Conv3_128"))
    model.add(BatchNormalization(name="BN3"))
    model.add(Activation('relu', name="Relu3"))
    model.add(MaxPooling2D((2, 2), name="Pool3"))

    model.add(Conv2D(256, (3, 3), padding='same', name="Conv4_256")) # Thêm lớp
    model.add(BatchNormalization(name="BN4"))
    model.add(Activation('relu', name="Relu4"))
    model.add(MaxPooling2D((2, 2), name="Pool4"))

    model.add(Flatten(name="Flatten"))
    model.add(Dropout(0.2, name="Dropout_FC")) # Giảm dropout hoặc bỏ hẳn ban đầu
    model.add(Dense(512, activation='relu', name='Dense_FC1'))
    model.add(Dense(256, activation='relu', name='Dense_FC2')) # Có thể thêm 1 lớp Dense nữa
    # Lớp output giữ nguyên (softmax với 2 units)
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
            kernel_initializer="random_uniform", # Có thể bỏ để dùng default của Keras
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
    # verbose_mode_val = getattr(config, 'verbose_mode', False)
    # if verbose_mode_val:
    #     model.summary()
    
    return model