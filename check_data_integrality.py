from pathlib import Path
import json
import os
import sys

# ==============================================================================
# 1. 配置参数 (请根据您的实际路径进行修改)
# ==============================================================================

# [!!! 必须修改 !!!] 您的数据根目录 (包含 'de_dust2', 'de_inferno' 等文件夹的父目录)
DATA_ROOT = "data/processed_data"

# [!!! 必须修改 !!!] 您的元数据 JSON 文件路径
# 假设您的完整 JSON 文件路径如下：
METADATA_FILE = "data/processed_data/de_dust2/positions.json"

# 文件名常量
IMAGE_SUBDIR = "imgs"
IMAGE_EXTENSION = ".png" # 根据您的文件类型，通常是 .png 或 .jpg

# ==============================================================================

def check_image_existence(metadata_path: str, data_root: str):
    """
    遍历元数据 JSON 文件，检查每个 file_frame 对应的图像文件是否在磁盘上存在。
    """
    if not os.path.isfile(metadata_path):
        print(f"❌ 错误: 找不到元数据文件: {metadata_path}")
        return

    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
             metadata = json.load(f)
    except Exception as e:
        print(f"❌ 错误: 无法加载或解析 JSON 文件 {metadata_path}. 错误: {e}")
        return

    missing_files = []
    total_checked = 0

    data_root_path = Path(data_root)

    print(f"--- 开始检查 {len(metadata)} 个样本 ---")

    for entry in metadata:
        total_checked += 1

        try:
            map_name = entry['map']
            file_frame = entry['file_frame']

            # 构建完整的图像路径: data_root / map / imgs / file_frame.png
            expected_path = data_root_path / map_name / IMAGE_SUBDIR / f"{file_frame}{IMAGE_EXTENSION}"

            if not expected_path.is_file():
                # 记录缺失的文件的预期路径
                missing_files.append(str(expected_path))

        except KeyError as e:
            print(f"⚠️ 警告: JSON 结构错误，缺少键 {e}。跳过此条目。")
            continue

    # --- 打印总结 ---
    print("=" * 40)
    if not missing_files:
        print(f"✅ 检查完成。在 {total_checked} 个样本中，所有图像文件都存在。")
    else:
        print(f"❌ 严重警告: 在 {total_checked} 个样本中，发现 {len(missing_files)} 个缺失文件。")
        print("--- 缺失文件路径示例 ---")
        for path in missing_files:
            print(f"  {path}")
        print("---------------------------------------")

    return missing_files

if __name__ == "__main__":
    if DATA_ROOT == "/path/to/your/data":
        print("🛑 请先修改脚本顶部的 `DATA_ROOT` 和 `METADATA_FILE` 变量为您的实际路径。")
        sys.exit(1)

    check_image_existence(METADATA_FILE, DATA_ROOT)