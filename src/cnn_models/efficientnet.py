import ssl
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Concatenate, Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras import regularizers

import config

# Needed to download pre-trained weights for ImageNet
ssl._create_default_https_context = ssl._create_unverified_context


def create_efficientnet_model(num_classes: int, weights_path: str = None):
    """
    Function to create an EfficientNetB0 model that flexibly handles input channels
    based on the dataset specified in the config, similar to the MobileNetV2 implementation.

    - For INbreast, it expects a 3-channel (RGB) input.
    - For CMMD, it expects a 1-channel (grayscale) input and concatenates it to 3 channels.
    - For other datasets, it defaults to the CMMD behavior.

    :param num_classes: The number of classes for the output layer.
    :param weights_path: Optional path to a local .h5 weights file. If provided,
                         it loads weights from this file instead of downloading.
    :return: The custom EfficientNetB0 model.
    """
    # Get image dimensions from config, defaulting to 224x224 for EfficientNetB0
    img_config = getattr(config, 'EFFICIENTNET_IMG_SIZE', {})
    img_height = img_config.get('HEIGHT', 224)
    img_width = img_config.get('WIDTH', 224)

    final_model_input_layer = None
    tensor_fed_to_efficientnet_base = None

    dataset_name_upper = getattr(config, 'dataset', '').upper()

    if config.verbose_mode:
        print(f"    [EfficientNet Create] Initializing for Dataset: {config.dataset}, Model: EfficientNet")

    # Logic to handle input channels based on the dataset
    if dataset_name_upper == "INBREAST":
        print("[INFO EfficientNet] Dataset is INBREAST. Expecting 3-channel input.")
        inp_rgb = Input(shape=(img_height, img_width, 3), name="Input_RGB_INbreast_EfficientNet")
        tensor_fed_to_efficientnet_base = inp_rgb
        final_model_input_layer = inp_rgb
    else:
        print(f"[INFO EfficientNet] Dataset is '{config.dataset}'. Assuming 1-channel input, will concatenate.")
        inp_gray = Input(shape=(img_height, img_width, 1), name="Input_Grayscale_Default_EfficientNet")
        concatenated_rgb = Concatenate(name="EfficientNet_Grayscale_to_RGB")([inp_gray, inp_gray, inp_gray])
        tensor_fed_to_efficientnet_base = concatenated_rgb
        final_model_input_layer = inp_gray

    if tensor_fed_to_efficientnet_base is None or final_model_input_layer is None:
        raise ValueError(f"Critical Error: Input tensors for EfficientNet could not be constructed for dataset '{config.dataset}'.")

    # Determine the weights to load: local path or download from 'imagenet'
    weights_to_load = 'imagenet'
    if weights_path:
        print(f"[INFO EfficientNet] Loading weights from local path: {weights_path}")
        weights_to_load = weights_path
    else:
        print("[WARNING EfficientNet] No local weights_path provided. Attempting to download 'imagenet' weights. This may fail.")

    # Instantiate EfficientNetB0 base model
    base_efficientnet_app = EfficientNetB0(
        input_tensor=tensor_fed_to_efficientnet_base,
        include_top=False,
        weights=weights_to_load,
        name="EfficientNetB0_Base"
    )

    # Add custom classification head
    x = base_efficientnet_app.output
    x = GlobalAveragePooling2D(name="EfficientNet_GlobalAvgPool")(x)
    x = Dropout(0.4, name="EfficientNet_Dropout_1")(x)
    x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0005), name="EfficientNet_Dense_1")(x)
    x = Dense(32, activation='relu', name="EfficientNet_Dense_2")(x)
    x = Dropout(0.3, name="EfficientNet_Dropout_2")(x)

    # Final output layer
    if num_classes >= 2:
        outputs = Dense(num_classes, activation='softmax', name='EfficientNet_Output')(x)
    else:
        outputs = Dense(1, activation='sigmoid', name='EfficientNet_Output')(x)

    final_model = Model(inputs=final_model_input_layer, outputs=outputs, name=f'EfficientNetB0_Custom_{config.dataset}')

    if getattr(config, 'verbose_mode', False):
        print(f"--- EfficientNetB0_Custom ({config.dataset}) Summary ---")
        final_model.summary(line_length=120)

    return final_model