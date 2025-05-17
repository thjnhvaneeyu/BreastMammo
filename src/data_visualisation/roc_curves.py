import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve

import config
from utils import save_output_figure


# def plot_roc_curve_binary(y_true: list, y_pred: list) -> None:
#     """
#     Plot ROC curve for binary classification.
#     Originally written as a group for the common pipeline.
#     :param y_true: Ground truth of the data in one-hot-encoding type.
#     :param y_pred: Prediction result of the data in one-hot-encoding type.
#     :return: None.
#     """
#     # Calculate fpr, tpr, and area under the curve(auc)
#     # Transform y_true and y_pred from one-hot-encoding to the label-encoding.
#     fpr, tpr, _ = roc_curve(y_true, y_pred)
#     roc_auc = auc(fpr, tpr)

#     # Plot.
#     plt.figure(figsize=(8, 5))
#     plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)  # plot roc curve
#     plt.plot([0, 1], [0, 1], 'k--', color='navy', lw=2)  # plot random guess line

#     # Set labels, title, ticks, legend, axis range and annotation.
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.annotate('Random Guess', (.53, .48), color='navy')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic')
#     plt.legend(loc='lower right')
#     save_output_figure("ROC-binary")
#     # plt.show()
def plot_roc_curve_binary(y_true_one_hot: np.ndarray, y_pred_scores: np.ndarray) -> None:
    """
    Vẽ đường cong ROC cho phân loại nhị phân.
    :param y_true_one_hot: Nhãn thực tế ở dạng one-hot encoding (N, 2).
    :param y_pred_scores: Điểm số/xác suất dự đoán từ mô hình, thường là (N, 2) từ softmax
                           hoặc (N, 1) từ sigmoid.
    :return: None.
    """
    print(f"[DEBUG plot_roc_curve_binary] y_true_one_hot shape: {y_true_one_hot.shape}, y_pred_scores shape: {y_pred_scores.shape}")

    if y_true_one_hot.ndim != 2 or y_true_one_hot.shape[1] != 2:
        print(f"[ERROR plot_roc_curve_binary] y_true_one_hot phải có shape (N, 2). Shape hiện tại: {y_true_one_hot.shape}")
        return
    
    # Chuyển y_true từ one-hot sang nhãn 1D (0 hoặc 1).
    # Giả sử cột 1 (index 1) là lớp dương tính (ví dụ: Malignant).
    y_true_scalar = y_true_one_hot[:, 1] 

    # Lấy điểm số/xác suất cho lớp dương tính từ y_pred_scores.
    # Nếu y_pred_scores là (N, 2) (từ softmax), lấy cột của lớp dương tính.
    # Nếu y_pred_scores đã là (N, 1) (từ sigmoid), có thể dùng trực tiếp (cần đảm bảo nó là xác suất của lớp dương).
    if y_pred_scores.ndim == 2 and y_pred_scores.shape[1] == 2:
        y_scores_positive_class = y_pred_scores[:, 1]
    elif y_pred_scores.ndim == 1: # Trường hợp output model là (N,1) với sigmoid
        y_scores_positive_class = y_pred_scores
    else:
        print(f"[ERROR plot_roc_curve_binary] y_pred_scores có shape không hợp lệ: {y_pred_scores.shape}. Mong đợi (N, 2) hoặc (N, 1).")
        return

    print(f"[DEBUG plot_roc_curve_binary] y_true_scalar shape: {y_true_scalar.shape}, y_scores_positive_class shape: {y_scores_positive_class.shape}")


    # Calculate fpr, tpr, and area under the curve(auc)
    try:
        fpr, tpr, _ = roc_curve(y_true_scalar, y_scores_positive_class)
    except ValueError as e:
        print(f"[ERROR plot_roc_curve_binary] Lỗi khi gọi roc_curve: {e}")
        print(f"  Unique values in y_true_scalar: {np.unique(y_true_scalar, return_counts=True)}")
        # Nếu y_true_scalar chỉ chứa 1 giá trị (ví dụ toàn 0 hoặc toàn 1), roc_curve sẽ lỗi.
        # Điều này có thể xảy ra với tập test rất nhỏ hoặc mất cân bằng hoàn toàn.
        return # Không vẽ ROC nếu có lỗi

    roc_auc = auc(fpr, tpr)

    # Plot.
    plt.figure(figsize=(10, 8)) # Tăng figsize
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--', color='navy', lw=2) 

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.annotate('Random Guess', (.53, .48), color='navy')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (Binary)', fontsize=14)
    plt.legend(loc='lower right')
    
    save_output_figure("ROC-binary") # Gọi hàm save_output_figure đã sửa
    # plt.show() # Bỏ comment nếu muốn hiển thị khi chạy local

def plot_roc_curve_multiclass(y_true: list, y_pred: list, label_encoder) -> None:
    """
    Plot ROC curve for multi classification.

    Code reference: https://github.com/DeepmindHub/python-/blob/master/ROC%20Curve%20Multiclass.py
    Originally written as a group for the common pipeline.
    
    :param y_true: Ground truth of the data in one-hot-encoding type.
    :param y_pred: Prediction result of the data in one-hot-encoding type.
    :return: None.
    """
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Calculate fpr, tpr, area under the curve(auc) of each class.
    for i in range(label_encoder.classes_.size):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Calculate macro fpr, tpr and area under the curve (AUC).
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(label_encoder.classes_))]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(label_encoder.classes_.size):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= label_encoder.classes_.size

    fpr['macro'] = all_fpr
    tpr['macro'] = mean_tpr
    roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])

    # Calculate micro fpr, tpr and area under the curve (AUC).
    fpr['micro'], tpr['micro'], _ = roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

    # Plot.
    plt.figure(figsize=(10, 8))

    # Plot micro roc curve.
    plt.plot(fpr['micro'], tpr['micro'],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', lw=4)

    # Plot macro roc curve.
    plt.plot(fpr['macro'], tpr['macro'],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc['macro']),
             color='black', linestyle=':', lw=4)

    # Plot roc curve of each class.
    colors = ['#3175a1', '#e1812b', '#39923a', '#c03d3e', '#9372b2']
    for i, color in zip(range(len(label_encoder.classes_)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(label_encoder.classes_[i], roc_auc[i]))

    # Plot random guess line
    plt.plot([0, 1], [0, 1], 'k--', color='red', lw=2)

    # Set labels, title, ticks, legend, axis range and annotation.
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.annotate('Random Guess', (.53, .48), color='red')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    save_output_figure("ROC-multi")
    # plt.show()
