"""
Variables set by the command line arguments dictating which parts of the program to execute.
Originally written as a group for the common pipeline. Later amended by Adam Jaamour.
"""

# Constants
RANDOM_SEED = 1  # Seed for reproducibility

# Kích thước ảnh cho từng bộ dữ liệu
MINI_MIAS_IMG_SIZE = {
    "HEIGHT": 1024,
    "WIDTH": 1024
}
CMMD_IMG_SIZE = {
    "HEIGHT": 224,  # Sử dụng kích thước nhỏ hơn để phù hợp với pipeline
    "WIDTH": 224
}
INBREAST_IMG_SIZE = {
    "HEIGHT": 224,  # Đồng bộ với CMMD để sử dụng chung mô hình
    "WIDTH": 224
}
# In config.py, change MOBILE_NET_IMG_SIZE to match your actual image dimensions
MOBILE_NET_IMG_SIZE = {
    "HEIGHT": 224,
    "WIDTH": 224
}
# Kích thước ảnh cho từng mô hình
VGG_IMG_SIZE = {
    "HEIGHT": 512,
    "WIDTH": 512
}
RESNET_IMG_SIZE = VGG_IMG_SIZE
INCEPTION_IMG_SIZE = VGG_IMG_SIZE
DENSE_NET_IMG_SIZE = MOBILE_NET_IMG_SIZE
XCEPTION_IMG_SIZE = INCEPTION_IMG_SIZE
# Cho MobileNet tương thích với tên trong main.py
MOBILENET_IMG_SIZE = MOBILE_NET_IMG_SIZE
# Số epoch tối đa không cải thiện sẽ dừng training
early_stopping_patience = 20  
# Số epoch không cải thiện sẽ giảm learning rate
reduce_lr_patience      = 3  
# Hệ số giảm learning rate mỗi lần giảm
reduce_lr_factor        = 0.5
# Learning rate nhỏ nhất còn cho phép
min_learning_rate       = 1e-6
# Batch size mặc định
batch_size              = 8
random_state            = 42
# Kích thước ROI (Region of Interest)
ROI_IMG_SIZE = {
    "HEIGHT": 224,
    "WIDTH": 224
}

# Variables set by command line arguments/flags
dataset = "CMMD_binary"       # Bộ dữ liệu mặc định là CMMD (có thể thay đổi thành INbreast hoặc khác)
mammogram_type = "all"      # Loại mammogram (Calc hoặc Mass)
model = "VGG"               # Mô hình mặc định là VGG (có thể thay đổi thành ResNet, DenseNet,...)
run_mode = "training"       # Chế độ chạy: training hoặc testing
learning_rate = 1e-3        # Learning rate cho các lớp đã được pre-trained trên ImageNet
batch_size = 8              # Batch size (tăng lên để phù hợp với GPU mạnh hơn)
max_epoch_frozen = 50       # Số epoch khi các lớp CNN ban đầu bị đóng băng (frozen)
max_epoch_unfrozen = 50     # Số epoch khi các lớp CNN ban đầu được mở khóa (unfrozen)
is_roi = False              # Có sử dụng ROI hay không (False: toàn bộ ảnh)
verbose_mode = True         # In thêm thông tin log để debug nếu cần thiết
name = ""                   # Tên của thí nghiệm hiện tại

# Thêm thông số cho augmentation (tăng cường dữ liệu)
augment_data = True          # Có áp dụng augmentation hay không

# BI-RADS mapping cho INbreast dataset
# BI-RADS mapping cho INbreast dataset (thêm Normal)
BI_RADS_MAPPING = {
    "Normal":   ["BI-RADS 1"],                    # Mặc định BI-RADS 1 = không bất thường
    "Benign":   ["BI-RADS 2", "BI-RADS 3"],       # Lành tính
    "Malignant":["BI-RADS 4a", "BI-RADS 4b", "BI-RADS 4c", "BI-RADS 5", "BI-RADS 6"]
}

INBREAST_BIRADS_MAPPING = {
    "Normal":   ["BI-RADS 1"],                    # Mặc định BI-RADS 1 = không bất thường
    "Benign":   ["BI-RADS 2", "BI-RADS 3"],       # Lành tính
    "Malignant":["BI-RADS 4a", "BI-RADS 4b", "BI-RADS 4c", "BI-RADS 5", "BI-RADS 6"]
}

