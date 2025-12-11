import torch
import numpy as np
import os
from PIL import Image, ImageDraw
from torchvision import transforms
from typing import Dict, Any

# --- 1. [关键修改] 更新反归一化参数 ---
# 必须与 CsgoEvalDataset_IT.get_transform 中的参数严格对应
# Mean: (0.48145466, 0.4578275, 0.40821073)
# Std:  (0.26862954, 0.26130258, 0.27577711)
UNNORMALIZE_TRANSFORM = transforms.Compose([
    transforms.Normalize(
        mean=[-0.48145466/0.26862954, -0.4578275/0.26130258, -0.40821073/0.27577711],
        std=[1/0.26862954, 1/0.26130258, 1/0.27577711]
    ),
    transforms.ToPILImage(),
])

TOPIL_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
])

def visualize_batch_from_dataloader(
    batch: Dict[str, Any],
    id_to_map_dict: Dict[int, str],
    save_dir: str = "visualizations",
    prefix: str = "eval_vis"
) -> None:
    """
    可视化 DataLoader 输出的一个批次 (适配 CsgoEvalDataset_IT)。
    不再需要 loc_tokenizer，因为 Dataset 直接返回了归一化坐标。
    """

    os.makedirs(save_dir, exist_ok=True)

    # --- 2. [关键修改] 使用新的键名解包数据 ---
    # Dataset 返回: "wrist_image" (Map), "image" (FPS), "state" (Coords), "map_id"

    # 注意: 数据集返回的可能是 [B, 1, C, H, W] 或 [B, C, H, W]
    # 如果使用了 unsqueeze(0) 在 action 上，图片通常不需要，但要检查维度
    map_images_tensor = batch["wrist_image"]
    fps_images_tensor = batch["image"]
    state_tensor = batch["state"]           # Shape: [B, 5] -> [x, y, z, v, h]
    map_ids_tensor = batch["map_id"]        # Shape: [B]

    batch_size = map_images_tensor.shape[0]

    print(f"🖼️ Visualizing batch of size {batch_size}...")

    for i in range(batch_size):
    # try:
        # --- A. 反-标准化图像 ---
        # Map Image (wrist_image)
        map_tensor = map_images_tensor[i]
        if map_tensor.shape[-1] == 3 and map_tensor.dtype == torch.uint8:
            map_tensor = (map_tensor.permute(2,0,1).float() / 255)
            pil_map_img = TOPIL_TRANSFORM(map_tensor.cpu())
        else:
            pil_map_img = UNNORMALIZE_TRANSFORM(map_tensor.cpu())

        # FPS Image (image)
        fps_tensor = fps_images_tensor[i]#.permute(2,0,1)
        if fps_tensor.shape[-1] == 3 and fps_tensor.dtype == torch.uint8:
            fps_tensor = (fps_tensor.permute(2,0,1).float() / 255)
            pil_fps_img = TOPIL_TRANSFORM(fps_tensor.cpu())
        else:
            pil_fps_img = UNNORMALIZE_TRANSFORM(fps_tensor.cpu())

        # --- B. [关键修改] 直接获取坐标 (无需 Tokenizer 解码) ---
        # state_tensor[i] 是 [x_norm, y_norm, z_norm, v_norm, h_norm]
        # 它们已经是 0.0 - 1.0 之间的归一化浮点数了
        coords = state_tensor[i].cpu().numpy()
        x_norm, y_norm = coords[0], coords[1]

        # --- C. 在 Map 上绘制坐标 ---
        img_w, img_h = pil_map_img.size

        # 确保坐标在 0-1 范围内
        x_norm = np.clip(x_norm, 0.0, 1.0)
        y_norm = np.clip(y_norm, 0.0, 1.0)

        pixel_x = int(x_norm * img_w)
        pixel_y = int(y_norm * img_h)

        draw = ImageDraw.Draw(pil_map_img)
        radius = 5
        # 绘制红色圆点代表 Ground Truth 位置
        bbox = [pixel_x - radius, pixel_y - radius, pixel_x + radius, pixel_y + radius]
        draw.ellipse(bbox, fill="red", outline="white", width=2)

        # (可选) 可以在图上写上 map name
        # draw.text((10, 10), f"GT: ({x_norm:.2f}, {y_norm:.2f})", fill="white")

        # --- D. 拼接 FPS 和 Map 图像 ---
        # 调整高度一致 (以 Map 高度为准，通常 map_size 比较大)
        target_height = pil_map_img.height
        if pil_fps_img.height != target_height:
                aspect_ratio = pil_fps_img.width / pil_fps_img.height
                new_width = int(target_height * aspect_ratio)
                pil_fps_img = pil_fps_img.resize((new_width, target_height), Image.BICUBIC)

        total_width = pil_fps_img.width + pil_map_img.width
        combined_img = Image.new('RGB', (total_width, target_height))

        # 粘贴: 左边 FPS，右边 Map
        combined_img.paste(pil_fps_img, (0, 0))
        combined_img.paste(pil_map_img, (pil_fps_img.width, 0))

        # 分割线
        draw_combined = ImageDraw.Draw(combined_img)
        draw_combined.line([(pil_fps_img.width, 0), (pil_fps_img.width, target_height)], fill="yellow", width=3)

        # --- E. 保存 ---
        map_id = map_ids_tensor[i].item()
        map_name = id_to_map_dict.get(map_id, f"id_{map_id}")

        save_filename = f"{prefix}_idx{i}_map_{map_name}.png"
        save_path = os.path.join(save_dir, save_filename)
        combined_img.save(save_path)

        # except Exception as e:
        #     print(f"❌ Error visualizing sample {i}: {e}")
        #     continue

    print(f"✅ Saved visualizations to '{save_dir}'")