import os
import json
from pathlib import Path
from typing import List
import shutil

# ==============================================================================
# 1. 配置参数 (请根据您的实际路径进行修改)
# ==============================================================================

# [!!! 必须修改 !!!] 您的数据根目录 (包含所有地图文件夹的父目录)
DATA_ROOT = "data/processed_data"

# [!!! 必须修改 !!!] 需要处理的地图列表 (例如，您的训练地图)
MAP_IDENTIFIERS = ["de_dust2", "de_mirage", "de_inferno", "de_nuke"]

# 图像文件的扩展名
IMAGE_EXTENSION = ".png"
# JSON 文件名
METADATA_FILENAME = "positions.json"

METADATA_FILENAME_BACKUP = "positions_all.json"

# ==============================================================================

def filter_metadata_by_image_existence(data_root: str, map_names: List[str], extension: str, metadata_filename: str, metadata_filename_backup: str):
    """
    遍历指定地图的元数据，检查对应的图像文件是否在原始 'imgs' 文件夹中存在。
    如果文件不存在 (已被移动)，则删除对应的 JSON 条目，并覆盖原 JSON 文件。
    """
    data_root_path = Path(data_root)
    total_removed = 0
    total_checked = 0

    print("--- 开始清理 JSON 元数据 ---")

    for map_name in map_names:
        map_path = data_root_path / map_name
        json_path = map_path / metadata_filename
        json_path_backup = map_path / metadata_filename_backup
        imgs_path = map_path / "imgs" # 图像的原始存放目录

        if not json_path.is_file():
            print(f"⚠️ 跳过 {map_name}：元数据文件 {metadata_filename} 不存在。")
            continue

        shutil.copy(json_path, json_path_backup)

        # 1. 加载原始元数据
        with open(json_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        filtered_metadata = []
        removed_count = 0

        # 2. 遍历并检查图像文件是否存在
        for entry in metadata:
            total_checked += 1
            file_frame = entry.get('file_frame', None)

            if not file_frame:
                continue # 跳过结构异常的条目

            # 构造原始图像路径: data_root / map / imgs / file_frame.png
            image_path = imgs_path / f"{file_frame}{extension}"

            if image_path.is_file():
                # 文件存在 (未被隔离) -> 保留
                filtered_metadata.append(entry)
            else:
                # 文件缺失 (已被隔离) -> 删除条目
                removed_count += 1

        # 3. 如果有条目被移除，则覆盖原始 JSON 文件
        if removed_count > 0:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(filtered_metadata, f, indent=4, ensure_ascii=False)

            print(f"✅ Map {map_name}: 成功移除 {removed_count} 个缺失的条目。新样本总数: {len(filtered_metadata)}.")
            total_removed += removed_count
        else:
            print(f"🟢 Map {map_name}: 没有发现缺失的图像条目。JSON 文件未修改。")

    print("\n" + "=" * 50)
    print(f"✨ JSON 清理完成。总共移除了 {total_removed} 个元数据条目。")
    print("=" * 50)


if __name__ == "__main__":
    # 请确保您的 MAP_IDENTIFIERS 包含了所有需要处理的地图
    filter_metadata_by_image_existence(
        DATA_ROOT,
        MAP_IDENTIFIERS,
        IMAGE_EXTENSION,
        METADATA_FILENAME,
        METADATA_FILENAME_BACKUP
    )