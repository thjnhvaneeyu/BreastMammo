import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
# tạo thư mục output tương đối với working directory hiện tại
os.makedirs(os.path.join(os.getcwd(), "output"), exist_ok=True)
import config
from utils import save_output_figure


def plot_confusion_matrix(cm: np.ndarray, fmt: str, label_encoder, is_normalised: bool) -> None:
    """
    Plot confusion matrix.
    Originally written as a group for the common pipeline. Later amended by Adam Jaamour.
    :param cm: Confusion matrix array.
    :param fmt: The formatter for numbers in confusion matrix.
    :param label_encoder: The label encoder used to get the number of classes.
    :param is_normalised: Boolean specifying whether the confusion matrix is normalised or not.
    :return: None.
    """
    title = str()
    if is_normalised:
        title = "Normalised Confusion Matrix"
        vmax = 1  # Y scale.
    elif not is_normalised:
        title = "Confusion Matrix"
        vmax = np.max(cm.sum(axis=1))  # Y scale.

    # Plot.
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, ax=ax, fmt=fmt, cmap=plt.cm.Blues, vmin=0, vmax=vmax)  # annot=True to annotate cells

    # Set labels, title, ticks and axis range.
    ax.set_xlabel('Predicted classes')
    ax.set_ylabel('True classes')
    ax.set_title(title)
    ax.xaxis.set_ticklabels(label_encoder.classes_)
    ax.yaxis.set_ticklabels(label_encoder.classes_)
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=0, ha='right', rotation_mode='anchor')
    plt.tight_layout()
    bottom, top = ax.get_ylim()
    if is_normalised:
        save_output_figure("CM-normalised")
    elif not is_normalised:
        save_output_figure("CM")
    # plt.show()


def plot_comparison_chart(df: pd.DataFrame) -> None:
    """
    Plot comparison bar chart.
    Originally written as a group for the common pipeline. Later amended by Adam Jaamour.
    :param df: Compare data from json file.
    :return: None.
    """
    title = "Accuracy Comparison"

    # Plot.
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(x='paper', y='accuracy', data=df)

    # Add number at the top of the bar.
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2., height + 0.01, height, ha='center')

    # Set title.
    plt.title(title)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=60, ha='right', rotation_mode='anchor')
    plt.tight_layout()
    save_output_figure(title)
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
def plot_training_results(hist_input, plot_name, is_frozen_layers=True):
    # Thiết lập thư mục output trên cùng project
    script_dir  = os.path.dirname(os.path.abspath(__file__))    # .../BreastMammo/src
    project_dir = os.path.dirname(script_dir)                   # .../BreastMammo
    out_dir     = os.path.join(project_dir, 'output')
    os.makedirs(out_dir, exist_ok=True)

    hist = hist_input.history
    n    = len(hist.get('loss', []))

    # -------- Loss --------
    plt.figure()
    plt.plot(np.arange(n), hist.get('loss', []),      label='train loss')
    if 'val_loss' in hist:
        plt.plot(np.arange(n), hist['val_loss'],       label='val loss')
    title = f"Training Loss on {config.dataset}"
    if not is_frozen_layers:
        title += " (unfrozen layers)"
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Cross entropy loss")
    plt.legend(loc="upper right")
    loss_fn = f"dataset-{config.dataset}_model-{config.model}_{plot_name}-Loss.png"
    plt.savefig(os.path.join(out_dir, loss_fn))
    plt.close()

    # -------- Accuracy --------
    plt.figure()
    # Chọn key accuracy phù hợp
    if config.dataset == "mini-MIAS":
        acc_key, val_acc_key = "categorical_accuracy", "val_categorical_accuracy"
    elif config.dataset in ["CBIS-DDSM","mini-MIAS-binary","CMMD","CMMD_binary"]:
        acc_key, val_acc_key = "binary_accuracy",      "val_binary_accuracy"
    else:
        acc_key     = next((k for k in hist if k.endswith("accuracy")), None)
        val_acc_key = f"val_{acc_key}" if acc_key and f"val_{acc_key}" in hist else None

    if acc_key:
        plt.plot(np.arange(n), hist.get(acc_key,[]),      label='train acc')
        if val_acc_key:
            plt.plot(np.arange(n), hist[val_acc_key],    label='val acc')

    title = f"Training Accuracy on {config.dataset}"
    if not is_frozen_layers:
        title += " (unfrozen layers)"
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.1)
    plt.legend(loc="upper right")
    acc_fn = f"dataset-{config.dataset}_model-{config.model}_{plot_name}-Accuracy.png"
    plt.savefig(os.path.join(out_dir, acc_fn))
    plt.close()