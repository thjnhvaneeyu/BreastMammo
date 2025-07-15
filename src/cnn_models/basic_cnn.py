# from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
# from tensorflow.keras import Sequential
# import config # File cấu hình của bạn

# def create_basic_cnn_model(num_classes: int):
#     """
#     Hàm để tạo một mô hình CNN cơ bản, đã được điều chỉnh lớp output
#     cho phù hợp với CategoricalCrossentropy khi num_classes = 2.

#     :param num_classes: Số lượng lớp (nhãn) cho bài toán phân loại.
#     :return: Một mô hình CNN cơ bản đã được tạo.
#     """
#     model = Sequential(name="Basic_CNN_Adjusted")

#     # Các lớp Convolutional (tích chập) và Pooling (gộp)
#     model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(config.ROI_IMG_SIZE['HEIGHT'], config.ROI_IMG_SIZE['WIDTH'], 1), name="Conv1"))
#     model.add(MaxPooling2D((2, 2), strides=(2, 2), name="Pool1"))
#     model.add(Conv2D(32, (5, 5), activation='relu', padding='same', name="Conv2"))
#     model.add(MaxPooling2D((2, 2), strides=(2, 2), name="Pool2"))
#     model.add(Flatten(name="Flatten"))

#     # Lớp Dropout
#     model.add(Dropout(0.5, seed=getattr(config, 'RANDOM_SEED', None), name="Dropout_FC"))

#     # Lớp Fully Connected (Dense)
#     model.add(Dense(1024, activation='relu', name='Dense_FC'))

#     # Lớp Output (đầu ra)
#     if num_classes == 2:
#         # Phân loại nhị phân: SỬ DỤNG 2 nơ-ron và hàm softmax
#         # để tương thích với CategoricalCrossentropy trong CnnModel.
#         model.add(Dense(num_classes, activation='softmax', kernel_initializer="random_uniform", name='Output_Softmax_Binary'))
#     elif num_classes > 2:
#         # Phân loại đa lớp (>2 lớp): num_classes nơ-ron, hàm softmax
#         model.add(Dense(num_classes, activation='softmax', kernel_initializer="random_uniform", name='Output_Softmax_Multiclass'))
#     else: # Trường hợp num_classes = 1 (hoặc lỗi)
#         # Vẫn giữ 1 nơ-ron sigmoid cho trường hợp này, mặc dù ít khả năng xảy ra
#         # nếu num_classes được xác định đúng từ đầu.
#         # CnnModel.compile_model cũng có nhánh xử lý cho num_classes = 1.
#         model.add(Dense(1, activation='sigmoid', kernel_initializer="random_uniform", name='Output_Sigmoid_SingleClass'))


#     if getattr(config, 'verbose_mode', False):
#         print("\n--- Basic CNN (Adjusted Output) Model Summary ---")
#         model.summary()
#         print("-------------------------------------------------\n")

#     return model
import config
import tensorflow as tf
from tensorflow.keras.initializers import HeNormal

def create_efficient_medical_cnn(num_classes: int):
    """
    Tạo một kiến trúc CNN hiệu quả với SE-Attention block.
    
    :param num_classes: Số lượng lớp output.
    :return: Một model Keras đã được tạo.
    """
    input_layer = tf.keras.layers.Input(shape=(224, 224, 1), name="Input_Grayscale")
    
    # --- Block 1: Conv + SE-Attention ---
    # Lớp tích chập đầu tiên để trích xuất đặc trưng cơ bản
    x = tf.keras.layers.Conv2D(64, 3, padding='same',
                              kernel_initializer=HeNormal(),
                              bias_initializer='zeros',
                              name="Conv1")(input_layer)
    x = tf.keras.layers.BatchNormalization(name="BN1")(x)
    x = tf.keras.layers.ReLU(name="ReLU1")(x)
    
    # SE-Attention Block: Giúp model học cách "chú ý" vào các kênh quan trọng
    # Squeeze: Nén thông tin không gian của mỗi kênh thành một giá trị duy nhất
    se = tf.keras.layers.GlobalAveragePooling2D(name="SE_Squeeze")(x)
    # Excitation: Học mối quan hệ phi tuyến giữa các kênh
    se = tf.keras.layers.Dense(64 // 16, activation='relu', kernel_initializer=HeNormal(), name="SE_Excite_Dense1")(se)
    se = tf.keras.layers.Dense(64, activation='sigmoid', kernel_initializer=HeNormal(), name="SE_Excite_Dense2")(se)
    se = tf.keras.layers.Reshape((1, 1, 64), name="SE_Reshape")(se)
    # Recalibrate: Nhân lại các feature map ban đầu với trọng số chú ý đã học
    x = tf.keras.layers.Multiply(name="SE_Recalibrate")([x, se])
    
    # --- Block 2: Feature Extraction & Down-sampling ---
    # Dùng strides=2 để giảm kích thước ảnh và tăng độ sâu đặc trưng
    x = tf.keras.layers.Conv2D(128, 3, strides=2, padding='same', name="Conv2_Downsample")(x)
    x = tf.keras.layers.BatchNormalization(name="BN2")(x)
    x = tf.keras.layers.ReLU(name="ReLU2")(x)
    x = tf.keras.layers.Dropout(0.3, name="Dropout_Conv")(x)
    
    # --- Classifier Head ---
    x = tf.keras.layers.GlobalAveragePooling2D(name="GlobalPool")(x)
    x = tf.keras.layers.Dense(128, activation='relu', kernel_initializer=HeNormal(), name="Dense1")(x)
    x = tf.keras.layers.Dropout(0.5, name="Dropout_Final")(x)
    
    # Lớp Output
    if num_classes >= 2:
        output_layer = tf.keras.layers.Dense(num_classes, activation='softmax', name="Output_Softmax")(x)
    else:
        output_layer = tf.keras.layers.Dense(1, activation='sigmoid', name="Output_Sigmoid")(x)
    
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer, name="Efficient_Medical_CNN")
    if getattr(config, 'verbose_mode', False):
        print("\n--- Basic CNN (Adjusted Output) Model Summary ---")
        model.summary()
        print("-------------------------------------------------\n")    
    return model

def create_basic_cnn_model(num_classes: int):
    """
    Hàm này được giữ lại để tương thích với CnnModel,
    nó sẽ gọi đến kiến trúc mới và hiệu quả hơn của bạn.
    """
    print("[INFO] Creating model using new 'Efficient_Medical_CNN' architecture.")
    return create_efficient_medical_cnn(num_classes)