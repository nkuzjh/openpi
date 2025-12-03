import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F
import random

class CoarseDropout:
    """
    Coarse Dropout (类似 CutOut，但多个方块)
    """
    def __init__(self, max_holes=8, max_height=16, max_width=16, fill_value=0, p=0.5):
        self.max_holes = max_holes
        self.max_height = max_height
        self.max_width = max_width
        self.fill_value = fill_value
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img

        h, w = img.shape[-2], img.shape[-1]
        mask = torch.ones_like(img)

        for _ in range(random.randint(1, self.max_holes)):
            hole_h = random.randint(1, self.max_height)
            hole_w = random.randint(1, self.max_width)
            y = random.randint(0, h - hole_h)
            x = random.randint(0, w - hole_w)
            mask[:, y:y+hole_h, x:x+hole_w] = self.fill_value

        return img * mask

class GridDropout:
    """
    Grid Dropout: 按网格规律性丢弃区域
    """
    def __init__(self, grid_size=4, fill_value=0, p=0.5):
        self.grid_size = grid_size
        self.fill_value = fill_value
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img

        c, h, w = img.shape
        mask = torch.ones_like(img)

        step_h = h // self.grid_size
        step_w = w // self.grid_size

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if random.random() < 0.5:  # 50% 概率丢弃每个网格
                    y1 = i * step_h
                    y2 = min((i + 1) * step_h, h)
                    x1 = j * step_w
                    x2 = min((j + 1) * step_w, w)
                    mask[:, y1:y2, x1:x2] = self.fill_value

        return img * mask

# 构建完整 transform
def get_train_transform():
    return T.Compose([
        # 颜色增强
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),

        # 转为 Tensor (如果输入是 PIL Image)
        # T.ToTensor(),  # 如果输入是 numpy/PIL，取消注释

        # 自定义 Dropout
        CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.5),
        GridDropout(grid_size=4, p=0.3),

        # ImageNet 标准化
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# 使用示例（假设输入是 [C, H, W] Tensor，值域 [0,1]）
transform = get_train_transform()
# augmented_img = transform(img_tensor)