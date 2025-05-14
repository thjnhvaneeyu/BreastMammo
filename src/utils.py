import matplotlib.pyplot as plt
from numpy.random import seed
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import config


def set_random_seeds() -> None:
    """
    Set random seeds for reproducible results.
    :return: None.
    """
    seed(config.RANDOM_SEED)  # NumPy
    tf.random.set_seed(config.RANDOM_SEED)  # Tensorflow


def print_runtime(text: str, runtime: float) -> None:
    """
    Print runtime in seconds.
    :param text: Message to print to the console indicating what was measured.
    :param runtime: The runtime in seconds.
    :return: None.
    """
    print("\n--- {} runtime: {} seconds ---".format(text, runtime))


def show_raw_image(img) -> None:
    """
    Displays a PIL image.
    :param img: the image in PIL format (before being converted to an array).
    :return: None.
    """
    img.show()


def print_num_gpus_available() -> None:
    """
    Prints the number of GPUs available on the current machine.
    :return: None
    """
    print("Number of GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


def print_error_message() -> None:
    """
    Print error message and exit code when a CLI-related error occurs.
    :return:
    """
    print("Wrong command line arguments passed, please use 'python main.py --help' for instructions on which arguments"
          "to pass to the program.")
    exit(1)


def create_label_encoder():
    """
    Creates the label encoder.
    :return: The instantiated label encoder.
    """
    return LabelEncoder()


def print_cli_arguments() -> None:
    """
    Print command line arguments and all code configurations to the terminal.
    :return: None
    """
    print("\nSettings used:")
    print("Dataset: {}".format(config.dataset))
    print("Mammogram type: {}".format(config.mammogram_type))
    print("CNN Model: {}".format(config.model))
    print("Run mode: {}".format(config.run_mode))
    print("Learning rate: {}".format(config.learning_rate))
    print("Batch size: {}".format(config.batch_size))
    print("Max number of epochs when original CNN layers are frozen: {}".format(config.max_epoch_frozen))
    print("Max number of epochs when original CNN layers are unfrozen: {}".format(config.max_epoch_unfrozen))
    print("Verbose mode: {}".format(config.verbose_mode))
    print("Experiment name: {}\n".format(config.name))


# def save_output_figure(title: str) -> None:
#     """
#     Save a figure on the output directory.
#     :param title: The title of the figure.
#     :return: None
#     """
#     plt.savefig(
#         "../output/{}_dataset-{}_mammogramtype-{}_model-{}_lr-{}_b-{}_e1-{}_e2-{}_roi-{}_{}_{}.png".format(
#             config.run_mode,
#             config.dataset,
#             config.mammogram_type,
#             config.model,
#             config.learning_rate,
#             config.batch_size,
#             config.max_epoch_frozen,
#             config.max_epoch_unfrozen,
#             config.is_roi,
#             config.name,
#             title))  # bbox_inches='tight'
def save_output_figure(title: str) -> None:
    """
    Save a figure into the output directory (auto-create it if missing).
    :param title: The title suffix of the file.
    """
    # 1) Xác định project root (thư mục trên src/)
    util_dir    = os.path.dirname(os.path.abspath(__file__))  # .../BreastMammo/src
    project_dir = os.path.dirname(util_dir)                   # .../BreastMammo
    output_dir  = os.path.join(project_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    # 2) Build filename theo config và title
    fname = (
        f"{config.run_mode}_dataset-{config.dataset}"
        f"_mammogramtype-{config.mammogram_type}"
        f"_model-{config.model}"
        f"_lr-{config.learning_rate}"
        f"_b-{config.batch_size}"
        f"_e1-{config.max_epoch_frozen}"
        f"_e2-{config.max_epoch_unfrozen}"
        f"_roi-{config.is_roi}"
        f"_{config.name}_{title}.png"
    )
    save_path = os.path.join(output_dir, fname)

    # 3) Lưu ảnh
    plt.savefig(save_path)  # bạn có thể thêm bbox_inches='tight' nếu cần

# def load_trained_model() -> None:
#     """
#     Load the model previously trained for the final evaluation using the test set.
#     :return: None
#     """
#     print("Loading trained model")
#     return load_model(
#         "../saved_models/dataset-{}_mammogramtype-{}_model-{}_lr-{}_b-{}_e1-{}_e2-{}_roi-{}_{}_saved-model.h5".format(
#             config.dataset,
#             config.mammogram_type,
#             config.model[0:],
#             config.learning_rate,
#             config.batch_size,
#             config.max_epoch_frozen,
#             config.max_epoch_unfrozen,
#             config.is_roi,
#             config.name)
#     )

def load_trained_model() -> tf.keras.Model: # Sửa lại kiểu trả về
    """
    Load the model previously trained for the final evaluation using the test set.
    SỬA ĐỔI ĐỂ PHÙ HỢP VÀ THỐNG NHẤT.
    """
    print("Loading trained model...") # Giữ lại print statement này

    # --- Bắt đầu phần code mới cho load_trained_model ---
    # Xác định PROJECT_ROOT một cách đáng tin cậy từ vị trí của utils.py
    # utils.py nằm trong PROJECT_ROOT/src/
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_actual = os.path.dirname(current_file_dir) # Đi lên 1 cấp từ src -> PROJECT_ROOT

    save_dir = os.path.join(project_root_actual, "saved_models")

    # Tạo tên file cơ sở từ config (hàm này có thể được tách ra nếu muốn)
    base_name = (
        f"dataset-{config.dataset}_type-{config.mammogram_type}_"
        f"model-{config.model}_lr-{config.learning_rate}_"
        f"b-{config.batch_size}_e1-{config.max_epoch_frozen}_"
        f"e2-{config.max_epoch_unfrozen}_roi-{config.is_roi}_{config.name}"
    )

    # Các đường dẫn tiềm năng
    tf_saved_model_path = os.path.join(save_dir, base_name)  # Thư mục cho định dạng TF
    full_model_h5_path = os.path.join(save_dir, f"{base_name}.h5")
    arch_json_path = os.path.join(save_dir, f"{base_name}_arch.json")
    weights_h5_path = os.path.join(save_dir, f"{base_name}.weights.h5")

    model = None
    compile_after_load = True # Mặc định là True, sẽ compile lại sau khi tải

    # 1. Ưu tiên định dạng TensorFlow SavedModel (thư mục)
    if os.path.exists(tf_saved_model_path) and os.path.isdir(tf_saved_model_path):
        print(f"Attempting to load model from TF SavedModel directory: {tf_saved_model_path}")
        try:
            model = tf.keras.models.load_model(tf_saved_model_path, compile=False)
            print(f"Successfully loaded model from TF SavedModel directory: {model_summary_short(model)}")
        except Exception as e:
            print(f"Failed to load TF SavedModel: {e}. Trying other formats.")
            model = None

    # 2. Nếu không có định dạng TF, thử HDF5 (.h5)
    if model is None and os.path.exists(full_model_h5_path) and os.path.isfile(full_model_h5_path):
        print(f"Attempting to load full model from HDF5: {full_model_h5_path}")
        try:
            model = tf.keras.models.load_model(full_model_h5_path, compile=False)
            print(f"Successfully loaded model from HDF5: {model_summary_short(model)}")
        except Exception as e:
            print(f"Failed to load full HDF5 model: {e}. Trying architecture + weights.")
            model = None

    # 3. Nếu cả hai trên thất bại, thử tải từ kiến trúc JSON và trọng số .weights.h5
    if model is None:
        if os.path.exists(arch_json_path) and os.path.exists(weights_h5_path):
            print(f"Attempting to load model from architecture ({arch_json_path}) and weights ({weights_h5_path}).")
            try:
                with open(arch_json_path, "r") as json_file:
                    loaded_model_json = json_file.read()
                model = tf.keras.models.model_from_json(loaded_model_json) # type: tf.keras.Model
                model.load_weights(weights_h5_path)
                print(f"Successfully loaded model from architecture and weights: {model_summary_short(model)}")
            except Exception as e:
                print(f"Failed to load model from architecture and weights: {e}")
                model = None
        else:
            print(f"Architecture or weights file missing for separate loading.")
            if not os.path.exists(arch_json_path): print(f"  Missing JSON: {arch_json_path}")
            if not os.path.exists(weights_h5_path): print(f"  Missing Weights: {weights_h5_path}")


    if model is None:
        print(f"CRITICAL: Model loading failed. No valid model found for base name '{base_name}' in {save_dir}")
        # Có thể raise lỗi ở đây nếu muốn chương trình dừng lại khi không tải được model
        # raise FileNotFoundError(f"Model loading failed for base name '{base_name}' in {save_dir}")
        return None # Hoặc trả về None để main.py xử lý

    if compile_after_load:
        output_neurons = model.output_shape[-1]
        num_classes_for_compile = output_neurons if output_neurons > 1 else 2

        print(f"Recompiling loaded model with {num_classes_for_compile} classes (output neurons: {output_neurons})...")
        
        # Sử dụng optimizer và learning rate từ config
        # Cần import Adam từ tensorflow.keras.optimizers
        # Cần import BinaryCrossentropy, CategoricalCrossentropy, BinaryAccuracy, CategoricalAccuracy
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
        from tensorflow.keras.metrics import BinaryAccuracy, CategoricalAccuracy

        optimizer_instance = Adam(learning_rate=config.learning_rate)
        setattr(optimizer_instance, 'lr', optimizer_instance.learning_rate) # For callbacks

        if num_classes_for_compile == 2:
            model.compile(
                optimizer=optimizer_instance,
                loss=BinaryCrossentropy(),
                metrics=[BinaryAccuracy(name='accuracy')] # Đặt tên 'accuracy' để nhất quán
            )
        else:
            model.compile(
                optimizer=optimizer_instance,
                loss=CategoricalCrossentropy(),
                metrics=[CategoricalAccuracy(name='accuracy')] # Đặt tên 'accuracy'
            )
        print("Model recompiled.")

    return model

def get_model_base_name_from_config(cfg_module):
    """Tạo tên file cơ sở cho các thành phần của model dựa trên module config."""
    return (
        f"dataset-{cfg_module.dataset}_type-{cfg_module.mammogram_type}_"
        f"model-{cfg_module.model}_lr-{cfg_module.learning_rate}_"
        f"b-{cfg_module.batch_size}_e1-{cfg_module.max_epoch_frozen}_"
        f"e2-{cfg_module.max_epoch_unfrozen}_roi-{cfg_module.is_roi}_{cfg_module.name}"
    )

def model_summary_short(model: tf.keras.Model) -> str:
    """Trả về một chuỗi tóm tắt ngắn về model."""
    if model is None:
        return "None"
    return f"Model(name='{model.name}', inputs={model.input_shape}, outputs={model.output_shape})"
