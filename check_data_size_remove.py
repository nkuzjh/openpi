import os
import shutil
from pathlib import Path
from typing import List

# ==============================================================================
# 1. 配置参数
# ==============================================================================

# [!!! 必须修改 !!!] 您的数据根目录 (包含 'de_dust2', 'de_inferno' 等地图文件夹的父目录)
DATA_ROOT = "data/processed_data"

# 最小文件大小阈值 (单位：字节)。小于此值的文件将被移动。
SIZE_THRESHOLD_BYTES = 10000

# [!!! 新增 !!!] 隔离目录路径 (将创建在脚本运行目录下)
QUARANTINE_DIR = Path("data/processed_data/quarantine_small_imgs")

# ==============================================================================

def filter_and_move_small_images(data_root: str, threshold: int, quarantine_dir: Path):
    """
    遍历目录，检查文件大小，并将小于阈值的图片移动到隔离目录，同时保留 map/imgs 结构。
    """
    data_root_path = Path(data_root)
    total_checked = 0
    total_moved = 0

    # [新增] 创建隔离目录的根目录
    quarantine_dir.mkdir(exist_ok=True)

    print(f"--- 任务配置 ---")
    print(f"根目录: {data_root_path}")
    print(f"大小阈值: {threshold / 1024:.2f} KB")
    print(f"隔离目录: {quarantine_dir.resolve()}")
    print("----------------")

    # 查找所有地图目录
    map_dirs = [d for d in data_root_path.iterdir() if d.is_dir()]

    if not map_dirs:
        print(f"❌ 未找到任何子目录 (地图文件夹) 在 {data_root_path} 下。")
        return

    for map_dir in map_dirs:
        imgs_path = map_dir / "imgs"

        if not imgs_path.is_dir():
            print(f"跳过 {map_dir.name}：未找到 imgs 目录。")
            continue

        print(f"\n正在检查地图: {map_dir.name}")

        # 定义该地图的隔离子目录 (例如 quarantine_small_files/de_dust2/imgs)
        dest_map_sub_path = quarantine_dir / map_dir.name / "imgs"
        dest_map_sub_path.mkdir(parents=True, exist_ok=True)


        for file_path in list(imgs_path.iterdir()): # 使用 list 复制，以便在迭代时移动文件
            if file_path.is_file() and file_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                total_checked += 1

                file_size = os.path.getsize(file_path)
                file_size_kb = file_size / 1024

                if file_size < threshold:
                    total_moved += 1

                    try:
                        # [!!! 核心移动操作 !!!]
                        shutil.move(str(file_path), str(dest_map_sub_path / file_path.name))

                        # [!! 打印文件大小和名称 !!]
                        print(f"  [移动] ⬇️ {file_path.name:<30} ({file_size_kb:.2f} KB) -> 成功隔离")

                    except OSError as e:
                        print(f"  [错误] ❌ 无法移动文件 {file_path.name}: {e}")
                else:
                    # 打印正常文件的大小 (可选，但有助于验证)
                    print(f"  [保留] ✅ {file_path.name:<30} ({file_size_kb:.2f} KB)")


    # --- 打印总结 ---
    print("\n" + "=" * 50)
    print(f"✅ 检查完成。总共检查了 {total_checked} 个文件。")
    print(f"总共移动到隔离区的文件数量: {total_moved} 个。")
    print(f"隔离区路径: {quarantine_dir.resolve()}")
    print("=" * 50)


if __name__ == "__main__":
    if not Path(DATA_ROOT).exists():
         print("🛑 错误: 请先修改脚本顶部的 `DATA_ROOT` 变量指向您正确的目录。")
    else:
        filter_and_move_small_images(DATA_ROOT, SIZE_THRESHOLD_BYTES, QUARANTINE_DIR)