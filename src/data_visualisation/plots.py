import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
# tạo thư mục output tương đối với working directory hiện tại
os.makedirs(os.path.join(os.getcwd(), "output"), exist_ok=True)
import config
from utils import save_output_figure


# def plot_confusion_matrix(cm: np.ndarray, fmt: str, label_encoder, is_normalised: bool) -> None:
#     """
#     Plot confusion matrix.
#     Originally written as a group for the common pipeline. Later amended by Adam Jaamour.
#     :param cm: Confusion matrix array.
#     :param fmt: The formatter for numbers in confusion matrix.
#     :param label_encoder: The label encoder used to get the number of classes.
#     :param is_normalised: Boolean specifying whether the confusion matrix is normalised or not.
#     :return: None.
#     """
#     title = str()
#     if is_normalised:
#         title = "Normalised Confusion Matrix"
#         vmax = 1  # Y scale.
#     elif not is_normalised:
#         title = "Confusion Matrix"
#         vmax = np.max(cm.sum(axis=1))  # Y scale.

#     # Plot.
#     fig, ax = plt.subplots(figsize=(6, 4))
#     sns.heatmap(cm, annot=True, ax=ax, fmt=fmt, cmap=plt.cm.Blues, vmin=0, vmax=vmax)  # annot=True to annotate cells

#     # Set labels, title, ticks and axis range.
#     ax.set_xlabel('Predicted classes')
#     ax.set_ylabel('True classes')
#     ax.set_title(title)
#     ax.xaxis.set_ticklabels(label_encoder.classes_)
#     ax.yaxis.set_ticklabels(label_encoder.classes_)
#     plt.setp(ax.yaxis.get_majorticklabels(), rotation=0, ha='right', rotation_mode='anchor')
#     plt.tight_layout()
#     bottom, top = ax.get_ylim()
#     if is_normalised:
#         save_output_figure("CM-normalised")
#     elif not is_normalised:
#         save_output_figure("CM")
# #     plt.show()
# def plot_confusion_matrix(cm: np.ndarray,
#                           fmt: str,
#                           label_encoder,
#                           is_normalised: bool) -> None:
#     """
#     Vẽ confusion matrix cm (K×K) với K = cm.shape[0].
#     Tự động chỉ lấy K nhãn đầu từ label_encoder.classes_.
#     """
#     n_classes = cm.shape[0]
#     class_names = list(label_encoder.classes_)[:n_classes]

#     title = "Normalized Confusion Matrix" if is_normalised else "Confusion Matrix"
#     vmax = 1.0 if is_normalised else None

#     fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
#     sns.heatmap(
#         cm,
#         annot=True,
#         fmt=fmt,
#         cmap="Blues",
#         vmin=0,
#         vmax=vmax,
#         ax=ax,
#         cbar_kws={"shrink": 0.8}
#     )

#     ax.set_title(title)
#     ax.set_xlabel("Predicted classes")
#     ax.set_ylabel("True classes")

#     ax.set_xticks(np.arange(n_classes) + 0.5)  # heatmap centers ticks on cell centers
#     ax.set_yticks(np.arange(n_classes) + 0.5)

#     ax.set_xticklabels(class_names, rotation=45, ha="right")
#     ax.set_yticklabels(class_names, rotation=0, ha="right")

#     # Lưu hình nếu bạn có hàm này
#     fname = "CM-normalised" if is_normalised else "CM"
#     save_output_figure(fname)
#     # plt.show()

def plot_confusion_matrix(cm: np.ndarray,
                          fmt: str,
                          label_encoder, # Đây là sklearn.preprocessing.LabelEncoder
                          is_normalised: bool) -> None:
    """
    Vẽ confusion matrix cm (K×K) với K = cm.shape[0].
    Tự động chỉ lấy K nhãn đầu từ label_encoder.classes_.
    """
    print(f"[DEBUG plot_confusion_matrix] Called. is_normalised: {is_normalised}") # Thêm debug print
    if cm is None or cm.size == 0:
        print(f"[ERROR plot_confusion_matrix] Confusion matrix 'cm' is None or empty. Skipping plot.")
        return
    if label_encoder is None or not hasattr(label_encoder, 'classes_') or label_encoder.classes_.size == 0:
        print(f"[ERROR plot_confusion_matrix] LabelEncoder is invalid. Skipping plot.")
        return

    n_classes_from_cm = cm.shape[0]
    
    # Cẩn thận khi lấy class_names, đảm bảo không vượt quá số lớp có trong label_encoder
    if n_classes_from_cm > len(label_encoder.classes_):
        print(f"[WARNING plot_confusion_matrix] Confusion matrix has {n_classes_from_cm} classes, but LabelEncoder only knows {len(label_encoder.classes_)}. Using generic names for extra classes.")
        class_names_le = list(label_encoder.classes_)
        class_names = class_names_le + [f"UnknownClass{i+1}" for i in range(n_classes_from_cm - len(class_names_le))]
    else:
        class_names = list(label_encoder.classes_)[:n_classes_from_cm]


    title = "Normalized Confusion Matrix" if is_normalised else "Confusion Matrix"
    # vmax nên được tính toán nếu không chuẩn hóa để heatmap đẹp hơn
    # vmax_val = 1.0 if is_normalised else np.max(cm) if cm.size > 0 else None
    # Tuy nhiên, seaborn thường tự xử lý vmax khá tốt nếu để None cho ma trận không chuẩn hóa
    vmax_val = 1.0 if is_normalised else None


    # Tăng figsize để có không gian cho các nhãn và tiêu đề
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True) # Ví dụ: (10,8)
    
    sns.heatmap(
        cm,
        annot=True, # Hiển thị số liệu trên ô
        fmt=fmt,    # Định dạng số liệu (ví dụ: '.2f' cho số thực, 'd' cho số nguyên)
        cmap="Blues", # Bảng màu
        vmin=0,       # Giá trị nhỏ nhất cho thang màu
        vmax=vmax_val,  # Giá trị lớn nhất cho thang màu (1.0 cho ma trận chuẩn hóa)
        ax=ax,
        cbar_kws={"shrink": 0.8} # Thu nhỏ color bar một chút
    )

    ax.set_title(title, fontsize=14) # Tăng fontsize tiêu đề
    ax.set_xlabel("Predicted classes", fontsize=12)
    ax.set_ylabel("True classes", fontsize=12)

    # Đảm bảo số lượng ticks và nhãn khớp nhau
    if class_names: # Chỉ đặt tick labels nếu có class_names
        ax.set_xticks(np.arange(len(class_names)) + 0.5)
        ax.set_yticks(np.arange(len(class_names)) + 0.5)
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=10)
        ax.set_yticklabels(class_names, rotation=0, ha="right", fontsize=10)

    # Sử dụng hàm save_output_figure từ utils.py
    plot_file_suffix = "CM-normalised" if is_normalised else "CM"
    save_output_figure(plot_file_suffix) # save_output_figure sẽ tự thêm .png và các thông tin khác

# def plot_comparison_chart(df: pd.DataFrame) -> None:
#     """
#     Plot comparison bar chart.
#     Originally written as a group for the common pipeline. Later amended by Adam Jaamour.
#     :param df: Compare data from json file.
#     :return: None.
#     """
#     title = "Accuracy Comparison"

#     # Plot.
#     fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
#     sns.barplot(x='paper', y='accuracy', data=df)

#     # Add number at the top of the bar.
#     for p in ax.patches:
#         height = p.get_height()
#         ax.text(p.get_x() + p.get_width() / 2., height + 0.01, height, ha='center')

#     # Set title.
#     plt.title(title)
#     plt.setp(ax.xaxis.get_majorticklabels(), rotation=60, ha='right', rotation_mode='anchor')
#     # plt.tight_layout()
#     plt.show()
#     save_output_figure(title)
#     # plt.show()
def plot_comparison_chart(df: pd.DataFrame) -> None:
    """
    Vẽ bar chart so sánh accuracy.
    """
    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    sns.barplot(x="paper", y="accuracy", data=df, ax=ax)

    # Ghi số lên đầu mỗi bar
    for p in ax.patches:
        ax.text(
            p.get_x() + p.get_width() / 2,
            p.get_height() + 0.005,
            f"{p.get_height():.2f}",
            ha="center"
        )

    ax.set_title("Accuracy Comparison")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha="right")
    save_output_figure("Accuracy Comparison")
    # plt.show()


# def plot_training_results(hist_input, plot_name: str, is_frozen_layers) -> None:
#     """
#     Function to plot loss and accuracy over epoch count for training.
#     Originally written as a group for the common pipeline.
#     :param is_frozen_layers: Boolean controlling whether some layers are frozen (for the plot title).
#     :param hist_input: The training history.
#     :param plot_name: The plot name.
#     """
#     title = "Training Loss on {}".format(config.dataset)
#     if not is_frozen_layers:
#         title += " (unfrozen layers)"

#     # fig = plt.figure()
#     # n = len(hist_input.history["loss"])
#     # 1) Lấy history dict và xác định key loss / val_loss
#     hist = hist_input.history
#     if "loss" in hist:
#         loss_key = "loss"
#     else:
#         # fallback: tìm key nào chứa 'loss' (tránh 'val_')
#         loss_cands = [k for k in hist.keys() if "loss" in k and not k.startswith("val_")]
#         if not loss_cands:
#             raise KeyError(f"No loss in history: {hist.keys()}")
#         loss_key = loss_cands[0]
#     val_loss_key = "val_loss" if "val_loss" in hist else f"val_{loss_key}"
#     n = len(hist[loss_key])
#     plt.style.use("ggplot")
#     # plt.figure()
#     # plt.plot(np.arange(0, n), hist_input.history["loss"], label="train set")
#     # plt.plot(np.arange(0, n), hist_input.history["val_loss"], label="validation set")
#     plt.title(title)
#     plt.xlabel("Epochs")
#     plt.ylabel("Cross entropy loss")
#     # plt.ylim(0, 1.5)
#     plt.legend(loc="upper right")
#     plt.savefig("../output/dataset-{}_model-{}_{}-Loss.png".format(config.dataset, config.model, plot_name))
#     # plt.show()

#     title = "Training Accuracy on {}".format(config.dataset)
#     if not is_frozen_layers:
#         title += " (unfrozen layers)"

#     fig = plt.figure()
#     n = len(hist_input.history["loss"])
#     plt.style.use("ggplot")
#     plt.figure()

#     # if config.dataset == "mini-MIAS":
#     #     plt.plot(np.arange(0, n), hist_input.history["categorical_accuracy"], label="train set")
#     #     plt.plot(np.arange(0, n), hist_input.history["val_categorical_accuracy"], label="validation set")
#     # elif config.dataset == "CBIS-DDSM" or config.dataset == "mini-MIAS-binary":
#     #     plt.plot(np.arange(0, n), hist_input.history["binary_accuracy"], label="train set")
#     #     plt.plot(np.arange(0, n), hist_input.history["val_binary_accuracy"], label="validation set")
#     # 2) Vẽ accuracy — gồm CMMD như binary
#     if config.dataset == "mini-MIAS":
#         acc_key = "categorical_accuracy"; val_acc = "val_categorical_accuracy"
#     elif config.dataset in ["CBIS-DDSM", "mini-MIAS-binary", "CMMD", "CMMD_binary"]:
#         # assume compile_model đã dùng BinaryAccuracy(name="binary_accuracy")
#         acc_key = "binary_accuracy"
#         val_acc  = "val_binary_accuracy"
#     else:
#         # fallback: tìm bất kỳ key nào chứa 'accuracy'
#         acc_key = next((k for k in hist if k.endswith("accuracy")), None)
#         val_acc  = f"val_{acc_key}" if acc_key and f"val_{acc_key}" in hist else None

#     if acc_key:
#         plt.figure()
#         plt.plot(np.arange(0, n), hist[acc_key], label="train acc")
#         if val_acc and val_acc in hist:
#             plt.plot(np.arange(0, n), hist[val_acc], label="val acc")
#     plt.title(title)
#     plt.xlabel("Epochs")
#     plt.ylabel("Accuracy")
#     plt.ylim(0, 1.1)
#     plt.legend(loc="upper right")
#     plt.savefig("../output/dataset-{}_model-{}_{}-Accuracy.png".format(config.dataset, config.model, plot_name))
#     # plt.show()
# def plot_training_results(hist_input, plot_name, is_frozen_layers=True):
#     # Thiết lập thư mục output trên cùng project
#     script_dir  = os.path.dirname(os.path.abspath(__file__))    # .../BreastMammo/src
#     project_dir = os.path.dirname(script_dir)                   # .../BreastMammo
#     out_dir     = os.path.join(project_dir, 'output')
#     os.makedirs(out_dir, exist_ok=True)

#     hist = hist_input.history
#     n    = len(hist.get('loss', []))

#     # -------- Loss --------
#     plt.figure()
#     plt.plot(np.arange(n), hist.get('loss', []),      label='train loss')
#     if 'val_loss' in hist:
#         plt.plot(np.arange(n), hist['val_loss'],       label='val loss')
#     title = f"Training Loss on {config.dataset}"
#     if not is_frozen_layers:
#         title += " (unfrozen layers)"
#     plt.title(title)
#     plt.xlabel("Epochs")
#     plt.ylabel("Cross entropy loss")
#     plt.legend(loc="upper right")
#     loss_fn = f"dataset-{config.dataset}_model-{config.model}_{plot_name}-Loss.png"
#     plt.savefig(os.path.join(out_dir, loss_fn))
#     plt.close()

#     # -------- Accuracy --------
#     plt.figure()
#     # Chọn key accuracy phù hợp
#     if config.dataset == "mini-MIAS":
#         acc_key, val_acc_key = "categorical_accuracy", "val_categorical_accuracy"
#     elif config.dataset in ["CBIS-DDSM","mini-MIAS-binary","CMMD","CMMD_binary"]:
#         acc_key, val_acc_key = "binary_accuracy",      "val_binary_accuracy"
#     else:
#         acc_key     = next((k for k in hist if k.endswith("accuracy")), None)
#         val_acc_key = f"val_{acc_key}" if acc_key and f"val_{acc_key}" in hist else None

#     if acc_key:
#         plt.plot(np.arange(n), hist.get(acc_key,[]),      label='train acc')
#         if val_acc_key:
#             plt.plot(np.arange(n), hist[val_acc_key],    label='val acc')

#     title = f"Training Accuracy on {config.dataset}"
#     if not is_frozen_layers:
#         title += " (unfrozen layers)"
#     plt.title(title)
#     plt.xlabel("Epochs")
#     plt.ylabel("Accuracy")
#     plt.ylim(0, 1.1)
#     plt.legend(loc="upper right")
#     acc_fn = f"dataset-{config.dataset}_model-{config.model}_{plot_name}-Accuracy.png"
#     plt.savefig(os.path.join(out_dir, acc_fn))
#     plt.close()

# def plot_training_results(hist_input, plot_name, is_frozen_layers=True):
#     """
#     Vẽ và lưu:
#       - Loss (train & val)
#       - Accuracy (train & val)
#     hist_input: History object trả về bởi model.fit()
#     plot_name: một chuỗi để phân biệt các giai đoạn (e.g. 'frozen' / 'unfrozen')
#     is_frozen_layers: nếu False, sẽ thêm note "(unfrozen layers)" lên title
#     """
#     # Thiết lập thư mục output
#     script_dir  = os.path.dirname(os.path.abspath(__file__))
#     project_dir = os.path.dirname(script_dir)
#     out_dir     = os.path.join(project_dir, 'output')
#     os.makedirs(out_dir, exist_ok=True)

#     hist = hist_input.history
#     epochs = np.arange(len(hist.get('loss', [])))

#     # -------- Loss --------
#     plt.figure()
#     plt.plot(epochs, hist.get('loss', []), label='train loss')
#     if 'val_loss' in hist:
#         plt.plot(epochs, hist['val_loss'], label='val loss')
#     title = f"Training Loss on {config.dataset}"
#     if not is_frozen_layers:
#         title += " (unfrozen layers)"
#     plt.title(title)
#     plt.xlabel("Epochs")
#     plt.ylabel("Loss")
#     plt.legend(loc="upper right")
#     loss_fn = f"dataset-{config.dataset}_model-{config.model}_{plot_name}-Loss.png"
#     plt.savefig(os.path.join(out_dir, loss_fn))
#     plt.close()

#     # -------- Accuracy --------
#     # Tự động tìm train-accuracy key
#     acc_key = None
#     for k in hist:
#         lk = k.lower()
#         if lk.endswith('accuracy') and not lk.startswith('val_'):
#             acc_key = k
#             break
#     # Tương ứng validation-accuracy
#     val_acc_key = None
#     if acc_key:
#         candidate = f"val_{acc_key}"
#         if candidate in hist:
#             val_acc_key = candidate
#         elif 'val_accuracy' in hist:
#             val_acc_key = 'val_accuracy'

#     plt.figure()
#     if acc_key:
#         plt.plot(epochs, hist.get(acc_key, []), label='train acc')
#         if val_acc_key:
#             plt.plot(epochs, hist[val_acc_key], label='val acc')

#     title = f"Training Accuracy on {config.dataset}"
#     if not is_frozen_layers:
#         title += " (unfrozen layers)"
#     plt.title(title)
#     plt.xlabel("Epochs")
#     plt.ylabel("Accuracy")
#     plt.ylim(0, 1.05)
#     plt.legend(loc="upper right")
#     acc_fn = f"dataset-{config.dataset}_model-{config.model}_{plot_name}-Accuracy"
#     plt.savefig(os.path.join(out_dir, acc_fn))
#     # plt.close()

def plot_training_results(hist_input, plot_name: str, is_frozen_layers=True):
    """
    Vẽ và lưu:
      - Loss (train & val)
      - Accuracy (train & val)
    Sử dụng hàm tiện ích save_output_figure.
    """
    hist = hist_input.history
    if not hist: # Kiểm tra nếu history rỗng
        print(f"[WARNING plot_training_results] History object is empty for plot_name: {plot_name}. Skipping plotting.")
        return

    # Lấy số epochs một cách an toàn
    epochs_list = hist.get('loss')
    primary_acc_key = next((k for k in hist if k.endswith('accuracy') and not k.startswith('val_')), None)
    
    if not epochs_list and primary_acc_key and hist.get(primary_acc_key):
        epochs_list = hist.get(primary_acc_key)
    
    if not epochs_list:
        print(f"[WARNING plot_training_results] No 'loss' or primary 'accuracy' data in history for plot_name: {plot_name}. Cannot determine epochs. Skipping plotting.")
        return
        
    epochs = np.arange(len(epochs_list))

    # -------- Đồ thị Loss --------
    if hist.get('loss'):
        plt.figure(figsize=(10, 7)) # Có thể tăng figsize nếu cần
        plt.plot(epochs, hist['loss'], label='train loss')
        if hist.get('val_loss'):
            plt.plot(epochs, hist['val_loss'], label='val loss')
        
        title_str = f"Training Loss on {getattr(config, 'dataset', 'UnknownDataset')}"
        if not is_frozen_layers:
            title_str += " (unfrozen layers)"
        plt.title(title_str)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        
        # Sử dụng hàm save_output_figure
        loss_fn_suffix = f"{plot_name}-Loss" 
        save_output_figure(loss_fn_suffix) # save_output_figure sẽ tự thêm .png và xử lý đường dẫn
                                           # và cũng sẽ gọi plt.close() nếu bạn đã sửa nó
    else:
        print(f"[INFO plot_training_results] 'loss' key not found or empty in history for {plot_name}. Skipping loss plot.")

    # -------- Đồ thị Accuracy --------
    acc_key = primary_acc_key # Sử dụng key đã tìm ở trên
            
    val_acc_key = None
    if acc_key and hist.get(acc_key):
        candidate_val_key = f"val_{acc_key}"
        if candidate_val_key in hist and hist.get(candidate_val_key):
            val_acc_key = candidate_val_key
        # Các fallback khác nếu tên key validation không theo chuẩn val_<train_key>
        elif 'val_accuracy' in hist and hist.get('val_accuracy'): 
            val_acc_key = 'val_accuracy'
        elif 'val_categorical_accuracy' in hist and hist.get('val_categorical_accuracy'):
            val_acc_key = 'val_categorical_accuracy'
        elif 'val_binary_accuracy' in hist and hist.get('val_binary_accuracy'): # Thêm kiểm tra này
             val_acc_key = 'val_binary_accuracy'

    if acc_key and hist.get(acc_key):
        plt.figure(figsize=(10, 7)) # Có thể tăng figsize
        plt.plot(epochs, hist[acc_key], label='train acc')
        if val_acc_key and hist.get(val_acc_key):
            plt.plot(epochs, hist[val_acc_key], label='val acc')

        title_str_acc = f"Training Accuracy on {getattr(config, 'dataset', 'UnknownDataset')}"
        if not is_frozen_layers:
            title_str_acc += " (unfrozen layers)"
        plt.title(title_str_acc)
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1.05)
        plt.legend(loc="lower right")
        
        # Sử dụng hàm save_output_figure
        acc_fn_suffix = f"{plot_name}-Accuracy" # Chỉ tên gốc, không có .png
        save_output_figure(acc_fn_suffix) # save_output_figure sẽ tự thêm .png và xử lý đường dẫn
                                          # và cũng sẽ gọi plt.close()
    else:
        print(f"[INFO plot_training_results] Suitable 'accuracy' key (e.g., 'accuracy', 'categorical_accuracy') not found or empty in history for {plot_name}. Skipping accuracy plot.")
