
import os
import json
import numpy as np

import cv2
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from transformers import AutoTokenizer

from datasets.utils import CoarseDropout, GridDropout
from datasets.random_erasing import RandomErasing
from loc_tokenizer import LocTokenizer


cross_view_matching_prompts = {

    'de_dust2': '''You are an expert in Counter-Strike gameplay and map geometry.
Your task is to localize the player's position in the top-down radar map based on the first-person view screenshot.

**Context about the radar map**:
- This is the official overview radar map of "de_dust2" from Counter-Strike 2 (CS2).
- The map is set in a Middle Eastern desert town, inspired by real-world locations in Morocco or Egypt, featuring sandy textures, low-rise buildings, stone walls, and palm trees.
- The radar uses a fixed orientation: Terrorist (T) spawn is in the bottom-left, Counter-Terrorist (CT) spawn is in the top-right.
- Key areas include:
• **A Site**: Upper-right bombsite with long corridors and boxes
• **B Site**: Lower-left bombsite near the tunnels and car
• **Mid**: Central open area with crates and a raised platform
• **Tunnels**: Narrow passage connecting T spawn to B site
• **Long A / Short A**: Routes from mid to A site
- The radar image is a simplified 2D schematic with consistent scale and no dynamic elements (e.g., no player icons).

**Your input**:
1. A first-person perspective screenshot from a player currently in-game on de_dust2.
2. The full de_dust2 overview radar map (top-down, 1024×1024 PNG with transparent background).

**Task**:
Analyze visual cues in the first-person image (e.g., wall textures, staircase layouts, rooftop views, alley geometry, lighting, field of view, and viewing direction) to infer the player’s **full 5D pose**.
Output **five normalized values** as a list `[x, y, z, angle_h, angle_v]`, where:

- **`x`**: Normalized horizontal position from **left (0.0) to right (1.0)** of the radar map.
- **`y`**: Normalized vertical position from **top (0.0) to bottom (1.0)** of the radar map.
- **`z`**: Normalized eye-level height relative to the **lowest walkable ground** on the map (0.0 = ground level, 1.0 = maximum height).
- **`angle_h`**: Normalized horizontal (yaw) viewing angle in **radians**, mapped to `[0.0, 1.0]` (0.0 = facing east, increasing counter-clockwise).
- **`angle_v`**: Normalized vertical (pitch) viewing angle in **radians**, mapped to `[0.0, 1.0]` (0.0 = looking straight ahead, 1.0 = looking straight down).

Focus on geometric alignment, not player state or UI elements.''',

    'de_mirage': '''You are an expert in Counter-Strike gameplay and map geometry.
Your task is to localize the player's full pose in the top-down radar map based on the first-person view screenshot.

**Context about the radar map**:
- This is the official overview radar map of "de_mirage" from Counter-Strike 2 (CS2).
- The map is set in a North African city inspired by Chefchaouen, Morocco, featuring vivid blue-painted buildings, narrow alleyways, mosaic tiles, and rooftop terraces.
- The radar uses a fixed orientation: Terrorist (T) spawn is in the bottom-left, Counter-Terrorist (CT) spawn is in the top-right.
- Key areas include:
• **A Site (Palace)**: Upper-right bombsite with multi-level courtyards, a central fountain, and upper/lower apartments
• **B Site (Apartments)**: Lower-left bombsite with tight indoor corridors and a back alley
• **Mid**: Central open street with a raised platform, market stalls, and direct sightlines to both sites
• **CT Spawn**: Elevated position with quick access to Mid and A Site
• **T Spawn**: Ground-level area with routes through Mid, Short A, or Long A
- The radar image is a simplified 2D schematic with consistent scale and no dynamic elements (e.g., no player icons).

**Your input**:
1. A first-person perspective screenshot from a player currently in-game on de_mirage.
2. The full de_mirage overview radar map (top-down, 1024×1024 PNG with transparent background).

**Task**:
Analyze visual cues in the first-person image (e.g., wall textures, staircase layouts, rooftop views, alley geometry, lighting, field of view, and viewing direction) to infer the player’s **full 5D pose**.
Output **five normalized values** as a list `[x, y, z, angle_h, angle_v]`, where:

- **`x`**: Normalized horizontal position from **left (0.0) to right (1.0)** of the radar map.
- **`y`**: Normalized vertical position from **top (0.0) to bottom (1.0)** of the radar map.
- **`z`**: Normalized eye-level height relative to the **lowest walkable ground** on the map (0.0 = ground level, 1.0 = maximum height).
- **`angle_h`**: Normalized horizontal (yaw) viewing angle in **radians**, mapped to `[0.0, 1.0]` (0.0 = facing east, increasing counter-clockwise).
- **`angle_v`**: Normalized vertical (pitch) viewing angle in **radians**, mapped to `[0.0, 1.0]` (0.0 = looking straight ahead, 1.0 = looking straight down).

Focus on geometric alignment and architectural landmarks. Do not consider UI elements, crosshairs, or player state indicators.''',

    'de_inferno': '''You are an expert in Counter-Strike gameplay and map geometry.
Your task is to localize the player's full pose in the top-down radar map based on the first-person view screenshot.

**Context about the radar map**:
- This is the official overview radar map of "de_inferno" from Counter-Strike 2 (CS2).
- The map is set in a rustic Italian village inspired by Tuscany, featuring terracotta rooftops, narrow cobblestone alleys, vineyard walls, and central courtyards with fountains.
- The radar uses a fixed orientation: Terrorist (T) spawn is in the bottom-left, Counter-Terrorist (CT) spawn is in the top-right.
- Key areas include:
  • **A Site (Banana)**: Upper-left bombsite accessible via a long curved alley ("Banana"), with multi-level buildings and a raised balcony
  • **B Site**: Lower-right bombsite near the garage and back alley, featuring tight indoor corridors
  • **Mid**: Central open area with a fountain, direct sightlines to both sites, and elevated CT-side rooftops
  • **Heaven / CT Rooftop**: Elevated CT position overlooking Mid and Banana
  • **T Spawn**: Ground-level area with routes through Banana (to A) or Mid/Long B (to B)
- The radar image is a simplified 2D schematic with consistent scale and no dynamic elements (e.g., no player icons).

**Your input**:
1. A first-person perspective screenshot from a player currently in-game on de_inferno.
2. The full de_inferno overview radar map (top-down, 1024×1024 PNG with transparent background).

**Task**:
Analyze visual cues in the first-person image (e.g., terracotta walls, vineyard textures, fountain geometry, rooftop views, alley curvature, lighting direction, and field of view) to infer the player’s **full 5D pose**.
Output **five normalized values** as a list `[x, y, z, angle_h, angle_v]`, where:

- **`x`**: Normalized horizontal position from **left (0.0) to right (1.0)** of the radar map.
- **`y`**: Normalized vertical position from **top (0.0) to bottom (1.0)** of the radar map.
- **`z`**: Normalized eye-level height relative to the **lowest walkable ground** on the map (0.0 = ground level, 1.0 = maximum height).
- **`angle_h`**: Normalized horizontal (yaw) viewing angle in **radians**, mapped to `[0.0, 1.0]` (0.0 = facing east, increasing counter-clockwise).
- **`angle_v`**: Normalized vertical (pitch) viewing angle in **radians**, mapped to `[0.0, 1.0]` (0.0 = looking straight ahead, 1.0 = looking straight down).

Focus on geometric alignment and architectural landmarks. Do not consider UI elements, crosshairs, or player state indicators.''',

    'de_nuke': '''You are an expert in Counter-Strike gameplay and map geometry.
Your task is to localize the player's full pose in the top-down radar map based on the first-person view screenshot.

**Context about the radar map**:
- This is the official overview radar map of "de_nuke" from Counter-Strike 2 (CS2).
- The map is set in a secret underground nuclear facility with a two-level layout: an outdoor surface area and a subterranean basement.
- The radar uses a fixed orientation: Terrorist (T) spawn is in the bottom-left, Counter-Terrorist (CT) spawn is in the top-right.
- Key areas include:
  • **Surface (Upper Level)**: Open outdoor area with silos, trucks, and CT spawn
  • **Basement (Lower Level)**: Indoor corridors beneath the surface, accessible via ramps or ladders
  • **A Site**: Located in the basement, near the yellow container and vents
  • **B Site**: On the surface, near the red warehouse and truck
  • **Ramps / Ladders**: Vertical connections between surface and basement (critical for Z-height changes)
- The radar image is a simplified 2D schematic with consistent scale and no dynamic elements (e.g., no player icons). Note that both levels are projected onto the same 2D plane, so Z-height is essential to disambiguate positions.

**Your input**:
1. A first-person perspective screenshot from a player currently in-game on de_nuke.
2. The full de_nuke overview radar map (top-down, 1024×1024 PNG with transparent background).

**Task**:
Analyze visual cues in the first-person image (e.g., indoor vs. outdoor lighting, wall textures, presence of ladders/ramps, ceiling visibility, and field of view) to infer the player’s **full 5D pose**.
Output **five normalized values** as a list `[x, y, z, angle_h, angle_v]`, where:

- **`x`**: Normalized horizontal position from **left (0.0) to right (1.0)** of the radar map.
- **`y`**: Normalized vertical position from **top (0.0) to bottom (1.0)** of the radar map.
- **`z`**: Normalized eye-level height relative to the **lowest walkable ground** on the map (0.0 = basement floor, 1.0 = surface level or rooftop).
- **`angle_h`**: Normalized horizontal (yaw) viewing angle in **radians**, mapped to `[0.0, 1.0]` (0.0 = facing east, increasing counter-clockwise).
- **`angle_v`**: Normalized vertical (pitch) viewing angle in **radians**, mapped to `[0.0, 1.0]` (0.0 = looking straight ahead, 1.0 = looking straight down).

Focus on geometric alignment, vertical context (surface vs. basement), and architectural landmarks. Do not consider UI elements, crosshairs, or player state indicators.''',

}

cross_view_matching_prompts_64token = {
    'de_mirage': '''You’re a CS2 expert. Given an FPV screenshot and de_mirage radar map, output normalized [x,y,z,angle_h,angle_v].
x: left(0)→right(1)
y: top(0)→bottom(1)
z: height from min ground (0) to max (1)
angle_h: yaw ∈ [0,1] (east=0, CCW)
angle_v: pitch ∈ [0,1] (straight=0, down=1)
Use blue walls, alleys, Palace/Apartments, Mid platform. Ignore UI.''',

}




map_to_id_dict = {
    'de_dust2': 0,
    'de_inferno': 1,
    'de_mirage': 2,
    'de_nuke': 3
}

id_to_map_dict = {
    0: 'de_dust2',
    1: 'de_inferno',
    2: 'de_mirage',
    3: 'de_nuke'
}


class LocalizationTrainDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        self.config = config
        # self.map_names = []
        # self.fps_img_paths = []
        self.positions = []
        self.map_z_range = {}
        for map_name in config["data"]["train_maps"]:
            # self.map_names.append(f"{config['data']['data_dir']}/{map_name}/{map_name}_radar_psd.png")
            # self.fps_img_paths.extend(os.listdir(f"{config['data']['data_dir']}/{map_name}/imgs/*"))
            with open(f"{config['data']['data_dir']}/{map_name}/positions.json", "r", encoding="utf-8") as f:
                datas = json.load(f)
            self.positions.extend(datas)
            max_z = -9999
            min_z = 9999
            for data in datas:
                if data['z'] > max_z:
                    max_z = data['z']
                if data['z'] < min_z:
                    min_z = data['z']
            self.map_z_range[map_name] = {
                'max_z':max_z,
                'min_z':min_z
            }
        self.fps_transform, self.map_transform = self.get_transform(config)

        if config['debug']:
            indices = [3574, 180, 2456, 2540, 2606, 4466, 892, 31, 4779, 1279, 4014, 1939, 2121, 2897, 3275, 806, 1350, 2474, 1724, 3549, 2798, 4542, 1392, 3934, 672, 2886, 1174, 3905, 4848, 556, 1443, 2800]
            self.positions = [self.positions[i] for i in indices]
        else:
            self.positions = self.positions[:config['data'].get('debug_num_train_data', len(self.positions))]

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        map = self.positions[idx]['map']
        file_frame = self.positions[idx]['file_frame']
        x = self.positions[idx]['x'] / 1024
        y = self.positions[idx]['y'] / 1024
        z = ( self.positions[idx]['z'] - self.map_z_range[map]['min_z'] ) / (self.map_z_range[map]['max_z'] - self.map_z_range[map]['min_z'])
        angle_v = self.positions[idx]['angle_v'] / (2*np.pi)
        angle_h = self.positions[idx]['angle_h'] / (2*np.pi)

        # assert self.map_names.contains(map), f"Map {map} not in dataset.map_names !"
        # assert any(fps_img_path.endswith(f"{map}/imgs/{file_frame}.png") for fps_img_path in self.fps_img_paths), f"Map {map} and screenshot {file_frame} not in dataset.fps_img_paths !"

        map_img_path = f"{self.config['data']['data_dir']}/{map}/{map}_radar_psd.png"
        map_img = Image.open(map_img_path).convert('RGB')
        map_img = self.map_transform(map_img)

        fps_img_path = f"{self.config['data']['data_dir']}/{map}/imgs/{file_frame}.png"
        fps_img = Image.open(fps_img_path).convert('RGB')
        fps_img = self.fps_transform(fps_img)

        return fps_img, map_img, torch.Tensor((x,y,z,angle_v,angle_h)), map_to_id_dict[map]

    def get_transform(self, config):

        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

        if config["data"]["is_fps_aug"]:
            fps_transform = transforms.Compose([
                transforms.Resize((config['data']['fps_size'][0], config['data']['fps_size'][1]), interpolation=InterpolationMode.BICUBIC),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                normalize,
                CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.5),
                GridDropout(grid_size=4, p=0.3),
                RandomErasing(probability=config['data']['erasing_p'], mean=[0.0, 0.0, 0.0])
            ])
        else:
            fps_transform = transforms.Compose([
                transforms.Resize((config['data']['fps_size'][0], config['data']['fps_size'][1]), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])

        map_transform = transforms.Compose([
            transforms.Resize((config['data']['map_size'][0], config['data']['map_size'][1]), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])

        return fps_transform, map_transform


class LocalizationEvalDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        self.config = config
        # self.map_names = []
        # self.fps_img_paths = []
        self.positions = []
        self.map_z_range = {}
        for map_name in config["data"]["val_maps"]:
            # self.map_names.append(f"{config['data']['data_dir']}/{map_name}/{map_name}_radar_psd.png")
            # self.fps_img_paths.extend(os.listdir(f"{config['data']['data_dir']}/{map_name}/imgs/*"))
            with open(f"{config['data']['data_dir']}/{map_name}/positions.json", "r", encoding="utf-8") as f:
                datas = json.load(f)
            self.positions.extend(datas)
            max_z = -9999
            min_z = 9999
            for data in datas:
                if data['z'] > max_z:
                    max_z = data['z']
                if data['z'] < min_z:
                    min_z = data['z']
            self.map_z_range[map_name] = {
                'max_z':max_z,
                'min_z':min_z
            }
        self.fps_transform, self.map_transform = self.get_transform(config)

        if config['debug']:
            indices = [3574, 180, 2456, 2540, 2606, 4466, 892, 31, 4779, 1279, 4014, 1939, 2121, 2897, 3275, 806, 1350, 2474, 1724, 3549, 2798, 4542, 1392, 3934, 672, 2886, 1174, 3905, 4848, 556, 1443, 2800]
            self.positions = [self.positions[i] for i in indices]
        else:
            self.positions = self.positions[:config['data'].get('debug_num_val_data', len(self.positions))]

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        map = self.positions[idx]['map']
        file_frame = self.positions[idx]['file_frame']
        x = self.positions[idx]['x'] / 1024
        y = self.positions[idx]['y'] / 1024
        z = ( self.positions[idx]['z'] - self.map_z_range[map]['min_z'] ) / (self.map_z_range[map]['max_z'] - self.map_z_range[map]['min_z'])
        angle_v = self.positions[idx]['angle_v'] / (2*np.pi)
        angle_h = self.positions[idx]['angle_h'] / (2*np.pi)

        # assert self.map_names.contains(map), f"Map {map} not in dataset.map_names !"
        # assert any(fps_img_path.endswith(f"{map}/imgs/{file_frame}.png") for fps_img_path in self.fps_img_paths), f"Map {map} and screenshot {file_frame} not in dataset.fps_img_paths !"

        map_img_path = f"{self.config['data']['data_dir']}/{map}/{map}_radar_psd.png"
        map_img = Image.open(map_img_path).convert('RGB')
        map_img = self.map_transform(map_img)

        fps_img_path = f"{self.config['data']['data_dir']}/{map}/imgs/{file_frame}.png"
        fps_img = Image.open(fps_img_path).convert('RGB')
        fps_img = self.fps_transform(fps_img)

        return fps_img, map_img, torch.Tensor((x,y,z,angle_v,angle_h)), map_to_id_dict[map]

    def get_transform(self, config):

        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

        fps_transform = transforms.Compose([
                transforms.Resize((config['data']['fps_size'][0], config['data']['fps_size'][1]), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])

        map_transform = transforms.Compose([
            transforms.Resize((config['data']['map_size'][0], config['data']['map_size'][1]), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])

        return fps_transform, map_transform


class LocClassTrainDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        self.config = config
        # self.map_names = []
        # self.fps_img_paths = []
        self.positions = []
        self.map_z_range = {}
        for map_name in config["data"]["train_maps"]:
            # self.map_names.append(f"{config['data']['data_dir']}/{map_name}/{map_name}_radar_psd.png")
            # self.fps_img_paths.extend(os.listdir(f"{config['data']['data_dir']}/{map_name}/imgs/*"))
            with open(f"{config['data']['data_dir']}/{map_name}/positions.json", "r", encoding="utf-8") as f:
                datas = json.load(f)
            self.positions.extend(datas)
            max_z = -9999
            min_z = 9999
            for data in datas:
                if data['z'] > max_z:
                    max_z = data['z']
                if data['z'] < min_z:
                    min_z = data['z']
            self.map_z_range[map_name] = {
                'max_z':max_z,
                'min_z':min_z
            }
        self.fps_transform, self.map_transform = self.get_transform(config)

        if config['debug']:
            indices = [3574, 180, 2456, 2540, 2606, 4466, 892, 31, 4779, 1279, 4014, 1939, 2121, 2897, 3275, 806, 1350, 2474, 1724, 3549, 2798, 4542, 1392, 3934, 672, 2886, 1174, 3905, 4848, 556, 1443, 2800]
            self.positions = [self.positions[i] for i in indices]
        else:
            self.positions = self.positions[:config['data'].get('debug_num_train_data', len(self.positions))]

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        map = self.positions[idx]['map']
        file_frame = self.positions[idx]['file_frame']
        if self.config['model'].get('task_head', None) == 'class_hw':
            cell_id = self.get_wh_cell_id_class(self.positions[idx]['x'], self.positions[idx]['y'])
        else:
            cell_id = self.get_cell_id_class(self.positions[idx]['x'], self.positions[idx]['y'])
        x = self.positions[idx]['x'] / 1024
        y = self.positions[idx]['y'] / 1024
        z = ( self.positions[idx]['z'] - self.map_z_range[map]['min_z'] ) / (self.map_z_range[map]['max_z'] - self.map_z_range[map]['min_z'])
        angle_v = self.positions[idx]['angle_v'] / (2*np.pi)
        angle_h = self.positions[idx]['angle_h'] / (2*np.pi)

        # assert self.map_names.contains(map), f"Map {map} not in dataset.map_names !"
        # assert any(fps_img_path.endswith(f"{map}/imgs/{file_frame}.png") for fps_img_path in self.fps_img_paths), f"Map {map} and screenshot {file_frame} not in dataset.fps_img_paths !"

        map_img_path = f"{self.config['data']['data_dir']}/{map}/{map}_radar_psd.png"
        map_img = Image.open(map_img_path).convert('RGB')
        map_img = self.map_transform(map_img)

        fps_img_path = f"{self.config['data']['data_dir']}/{map}/imgs/{file_frame}.png"
        fps_img = Image.open(fps_img_path).convert('RGB')
        fps_img = self.fps_transform(fps_img)

        if self.config['model'].get('task_head', None) == 'class_hw':
            return fps_img, map_img, torch.LongTensor(cell_id), torch.Tensor((x,y,z,angle_v,angle_h)), map_to_id_dict[map]
        else:
            return fps_img, map_img, torch.LongTensor([cell_id]), torch.Tensor((x,y,z,angle_v,angle_h)), map_to_id_dict[map]

    def get_transform(self, config):

        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

        if config["data"].get("is_fps_aug", False):
            fps_transform = transforms.Compose([
                transforms.Resize((config['data']['fps_size'][0], config['data']['fps_size'][1]), interpolation=InterpolationMode.BICUBIC),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                normalize,
                CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.5),
                GridDropout(grid_size=4, p=0.3),
                RandomErasing(probability=config['data']['erasing_p'], mean=[0.0, 0.0, 0.0])
            ])
        else:
            fps_transform = transforms.Compose([
                transforms.Resize((config['data']['fps_size'][0], config['data']['fps_size'][1]), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])

        map_transform = transforms.Compose([
            transforms.Resize((config['data']['map_size'][0], config['data']['map_size'][1]), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])

        return fps_transform, map_transform

    def get_cell_id_class(self, x, y, cell_dim=64, map_size=1024):
        # 假设 x, y 是 1024x1024 地图上的像素坐标
        cell_size = map_size / cell_dim
        # col 对应 x (宽度), row 对应 y (高度)
        col_index = x // cell_size
        row_index = y // cell_size
        # 确保索引不会越界 (例如当 x=1024 时)
        col_index = min(col_index, 63)
        row_index = min(row_index, 63)

        # 展平得到cell index作为分类标签
        cell_id = row_index * cell_dim + col_index

        return cell_id

    def get_wh_cell_id_class(self, x, y, cell_dim=64, map_size=1024):
        # 假设 x, y 是 1024x1024 地图上的像素坐标
        cell_size = map_size / cell_dim
        # col 对应 x (宽度), row 对应 y (高度)
        col_index = x // cell_size
        row_index = y // cell_size
        # 确保索引不会越界 (例如当 x=1024 时)
        col_index = min(col_index, 63)
        row_index = min(row_index, 63)

        return [col_index, row_index]


class LocClassEvalDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        self.config = config
        # self.map_names = []
        # self.fps_img_paths = []
        self.positions = []
        self.map_z_range = {}
        for map_name in config["data"]["val_maps"]:
            # self.map_names.append(f"{config['data']['data_dir']}/{map_name}/{map_name}_radar_psd.png")
            # self.fps_img_paths.extend(os.listdir(f"{config['data']['data_dir']}/{map_name}/imgs/*"))
            with open(f"{config['data']['data_dir']}/{map_name}/positions.json", "r", encoding="utf-8") as f:
                datas = json.load(f)
            self.positions.extend(datas)
            max_z = -9999
            min_z = 9999
            for data in datas:
                if data['z'] > max_z:
                    max_z = data['z']
                if data['z'] < min_z:
                    min_z = data['z']
            self.map_z_range[map_name] = {
                'max_z':max_z,
                'min_z':min_z
            }
        self.fps_transform, self.map_transform = self.get_transform(config)

        if config['debug']:
            indices = [3574, 180, 2456, 2540, 2606, 4466, 892, 31, 4779, 1279, 4014, 1939, 2121, 2897, 3275, 806, 1350, 2474, 1724, 3549, 2798, 4542, 1392, 3934, 672, 2886, 1174, 3905, 4848, 556, 1443, 2800]
            self.positions = [self.positions[i] for i in indices]
        else:
            self.positions = self.positions[:config['data'].get('debug_num_val_data', len(self.positions))]

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        map = self.positions[idx]['map']
        file_frame = self.positions[idx]['file_frame']
        if self.config['model'].get('task_head', None) == 'class_hw':
            cell_id = self.get_wh_cell_id_class(self.positions[idx]['x'], self.positions[idx]['y'])
        else:
            cell_id = self.get_cell_id_class(self.positions[idx]['x'], self.positions[idx]['y'])
        x = self.positions[idx]['x'] / 1024
        y = self.positions[idx]['y'] / 1024
        z = ( self.positions[idx]['z'] - self.map_z_range[map]['min_z'] ) / (self.map_z_range[map]['max_z'] - self.map_z_range[map]['min_z'])
        angle_v = self.positions[idx]['angle_v'] / (2*np.pi)
        angle_h = self.positions[idx]['angle_h'] / (2*np.pi)

        # assert self.map_names.contains(map), f"Map {map} not in dataset.map_names !"
        # assert any(fps_img_path.endswith(f"{map}/imgs/{file_frame}.png") for fps_img_path in self.fps_img_paths), f"Map {map} and screenshot {file_frame} not in dataset.fps_img_paths !"

        map_img_path = f"{self.config['data']['data_dir']}/{map}/{map}_radar_psd.png"
        map_img = Image.open(map_img_path).convert('RGB')
        map_img = self.map_transform(map_img)

        fps_img_path = f"{self.config['data']['data_dir']}/{map}/imgs/{file_frame}.png"
        fps_img = Image.open(fps_img_path).convert('RGB')
        fps_img = self.fps_transform(fps_img)

        if self.config['model'].get('task_head', None) == 'class_hw':
            return fps_img, map_img, torch.LongTensor(cell_id), torch.Tensor((x,y,z,angle_v,angle_h)), map_to_id_dict[map]
        else:
            return fps_img, map_img, torch.LongTensor([cell_id]), torch.Tensor((x,y,z,angle_v,angle_h)), map_to_id_dict[map]

    def get_transform(self, config):

        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

        fps_transform = transforms.Compose([
                transforms.Resize((config['data']['fps_size'][0], config['data']['fps_size'][1]), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])

        map_transform = transforms.Compose([
            transforms.Resize((config['data']['map_size'][0], config['data']['map_size'][1]), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])

        return fps_transform, map_transform

    def get_cell_id_class(self, x, y, cell_dim=64, map_size=1024):
        # 假设 x, y 是 1024x1024 地图上的像素坐标
        cell_size = map_size / cell_dim
        # col 对应 x (宽度), row 对应 y (高度)
        col_index = x // cell_size
        row_index = y // cell_size
        # 确保索引不会越界 (例如当 x=1024 时)
        col_index = min(col_index, 63)
        row_index = min(row_index, 63)

        # 展平得到cell index作为分类标签
        cell_dim = 64
        cell_id = row_index * cell_dim + col_index

        return cell_id

    def get_wh_cell_id_class(self, x, y, cell_dim=64, map_size=1024):
        # 假设 x, y 是 1024x1024 地图上的像素坐标
        cell_size = map_size / cell_dim
        # col 对应 x (宽度), row 对应 y (高度)
        col_index = x // cell_size
        row_index = y // cell_size
        # 确保索引不会越界 (例如当 x=1024 时)
        col_index = min(col_index, 63)
        row_index = min(row_index, 63)

        return [col_index, row_index]

class LocalizationVLMTrainDataset(torch.utils.data.Dataset):
    def __init__(self, config, tokenizer=None):
        self.config = config
        # self.map_names = []
        # self.fps_img_paths = []
        self.positions = []
        self.map_z_range = {}
        for map_name in config["data"]["train_maps"]:
            # self.map_names.append(f"{config['data']['data_dir']}/{map_name}/{map_name}_radar_psd.png")
            # self.fps_img_paths.extend(os.listdir(f"{config['data']['data_dir']}/{map_name}/imgs/*"))
            with open(f"{config['data']['data_dir']}/{map_name}/positions.json", "r", encoding="utf-8") as f:
                datas = json.load(f)
            self.positions.extend(datas)
            max_z = -9999
            min_z = 9999
            for data in datas:
                if data['z'] > max_z:
                    max_z = data['z']
                if data['z'] < min_z:
                    min_z = data['z']
            self.map_z_range[map_name] = {
                'max_z':max_z,
                'min_z':min_z
            }
        self.fps_transform, self.map_transform = self.get_transform(config)

        if tokenizer == None:
            self.tokenizer = AutoTokenizer.from_pretrained(config['model']['vlm_model_name'])
        else:
            self.tokenizer = tokenizer

        if config['debug']:
            indices = [3574, 180, 2456, 2540, 2606, 4466, 892, 31, 4779, 1279, 4014, 1939, 2121, 2897, 3275, 806, 1350, 2474, 1724, 3549, 2798, 4542, 1392, 3934, 672, 2886, 1174, 3905, 4848, 556, 1443, 2800]
            self.positions = [self.positions[i] for i in indices]
        else:
            self.positions = self.positions[:config['data'].get('debug_num_train_data', len(self.positions))]

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        map = self.positions[idx]['map']
        file_frame = self.positions[idx]['file_frame']
        x = self.positions[idx]['x'] / 1024
        y = self.positions[idx]['y'] / 1024
        z = ( self.positions[idx]['z'] - self.map_z_range[map]['min_z'] ) / (self.map_z_range[map]['max_z'] - self.map_z_range[map]['min_z'])
        angle_v = self.positions[idx]['angle_v'] / (2*np.pi)
        angle_h = self.positions[idx]['angle_h'] / (2*np.pi)

        # assert self.map_names.contains(map), f"Map {map} not in dataset.map_names !"
        # assert any(fps_img_path.endswith(f"{map}/imgs/{file_frame}.png") for fps_img_path in self.fps_img_paths), f"Map {map} and screenshot {file_frame} not in dataset.fps_img_paths !"

        map_img_path = f"{self.config['data']['data_dir']}/{map}/{map}_radar_psd.png"
        map_img = Image.open(map_img_path).convert('RGB')
        map_img = self.map_transform(map_img)

        fps_img_path = f"{self.config['data']['data_dir']}/{map}/imgs/{file_frame}.png"
        fps_img = Image.open(fps_img_path).convert('RGB')
        fps_img = self.fps_transform(fps_img)

        if self.config['model'].get('prompt_template', False) == '64token':
            map_prompt_template = cross_view_matching_prompts_64token[map]
        else:
            map_prompt_template = cross_view_matching_prompts[map]
        text_inputs = self.tokenizer(map_prompt_template, max_length=64, return_tensors="pt", padding="max_length", truncation=True)

        return fps_img, map_img, text_inputs, torch.Tensor((x,y,z,angle_v,angle_h)), map_to_id_dict[map]

    def get_transform(self, config):

        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

        if config["data"]["is_fps_aug"]:
            fps_transform = transforms.Compose([
                transforms.Resize((config['data']['fps_size'][0], config['data']['fps_size'][1]), interpolation=InterpolationMode.BICUBIC),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                normalize,
                CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.5),
                GridDropout(grid_size=4, p=0.3),
                RandomErasing(probability=config['data']['erasing_p'], mean=[0.0, 0.0, 0.0])
            ])
        else:
            fps_transform = transforms.Compose([
                transforms.Resize((config['data']['fps_size'][0], config['data']['fps_size'][1]), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])

        map_transform = transforms.Compose([
            transforms.Resize((config['data']['map_size'][0], config['data']['map_size'][1]), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])

        return fps_transform, map_transform


class LocalizationVLMEvalDataset(torch.utils.data.Dataset):
    def __init__(self, config, tokenizer=None):
        self.config = config
        # self.map_names = []
        # self.fps_img_paths = []
        self.positions = []
        self.map_z_range = {}
        for map_name in config["data"]["val_maps"]:
            # self.map_names.append(f"{config['data']['data_dir']}/{map_name}/{map_name}_radar_psd.png")
            # self.fps_img_paths.extend(os.listdir(f"{config['data']['data_dir']}/{map_name}/imgs/*"))
            with open(f"{config['data']['data_dir']}/{map_name}/positions.json", "r", encoding="utf-8") as f:
                datas = json.load(f)
            self.positions.extend(datas)
            max_z = -9999
            min_z = 9999
            for data in datas:
                if data['z'] > max_z:
                    max_z = data['z']
                if data['z'] < min_z:
                    min_z = data['z']
            self.map_z_range[map_name] = {
                'max_z':max_z,
                'min_z':min_z
            }
        self.fps_transform, self.map_transform = self.get_transform(config)

        if tokenizer == None:
            self.tokenizer = AutoTokenizer.from_pretrained(config['model']['vlm_model_name'])
        else:
            self.tokenizer = tokenizer

        if config['debug']:
            indices = [3574, 180, 2456, 2540, 2606, 4466, 892, 31, 4779, 1279, 4014, 1939, 2121, 2897, 3275, 806, 1350, 2474, 1724, 3549, 2798, 4542, 1392, 3934, 672, 2886, 1174, 3905, 4848, 556, 1443, 2800]
            self.positions = [self.positions[i] for i in indices]
        else:
            self.positions = self.positions[:config['data'].get('debug_num_val_data', len(self.positions))]

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        map = self.positions[idx]['map']
        file_frame = self.positions[idx]['file_frame']
        x = self.positions[idx]['x'] / 1024
        y = self.positions[idx]['y'] / 1024
        z = ( self.positions[idx]['z'] - self.map_z_range[map]['min_z'] ) / (self.map_z_range[map]['max_z'] - self.map_z_range[map]['min_z'])
        angle_v = self.positions[idx]['angle_v'] / (2*np.pi)
        angle_h = self.positions[idx]['angle_h'] / (2*np.pi)

        # assert self.map_names.contains(map), f"Map {map} not in dataset.map_names !"
        # assert any(fps_img_path.endswith(f"{map}/imgs/{file_frame}.png") for fps_img_path in self.fps_img_paths), f"Map {map} and screenshot {file_frame} not in dataset.fps_img_paths !"

        map_img_path = f"{self.config['data']['data_dir']}/{map}/{map}_radar_psd.png"
        map_img = Image.open(map_img_path).convert('RGB')
        map_img = self.map_transform(map_img)

        fps_img_path = f"{self.config['data']['data_dir']}/{map}/imgs/{file_frame}.png"
        fps_img = Image.open(fps_img_path).convert('RGB')
        fps_img = self.fps_transform(fps_img)

        if self.config['model'].get('prompt_template', False) == '64token':
            map_prompt_template = cross_view_matching_prompts_64token[map]
        else:
            map_prompt_template = cross_view_matching_prompts[map]
        text_inputs = self.tokenizer(map_prompt_template, max_length=64, return_tensors="pt", padding="max_length", truncation=True)

        return fps_img, map_img, text_inputs, torch.Tensor((x,y,z,angle_v,angle_h)), map_to_id_dict[map]

    def get_transform(self, config):

        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

        fps_transform = transforms.Compose([
                transforms.Resize((config['data']['fps_size'][0], config['data']['fps_size'][1]), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])

        map_transform = transforms.Compose([
            transforms.Resize((config['data']['map_size'][0], config['data']['map_size'][1]), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])

        return fps_transform, map_transform


class LocalizationVLATrainDataset(torch.utils.data.Dataset):
    def __init__(self, config, tokenizer=None):
        self.config = config
        # self.map_names = []
        # self.fps_img_paths = []
        self.positions = []
        self.map_z_range = {}
        for map_name in config["data"]["train_maps"]:
            # self.map_names.append(f"{config['data']['data_dir']}/{map_name}/{map_name}_radar_psd.png")
            # self.fps_img_paths.extend(os.listdir(f"{config['data']['data_dir']}/{map_name}/imgs/*"))
            with open(f"{config['data']['data_dir']}/{map_name}/positions.json", "r", encoding="utf-8") as f:
                datas = json.load(f)
            self.positions.extend(datas)
            max_z = -9999
            min_z = 9999
            for data in datas:
                if data['z'] > max_z:
                    max_z = data['z']
                if data['z'] < min_z:
                    min_z = data['z']
            self.map_z_range[map_name] = {
                'max_z':max_z,
                'min_z':min_z
            }
        self.fps_transform, self.map_transform = self.get_transform(config)

        self.tokenizer = tokenizer

        if config['debug']:
            indices = [3574, 180, 2456, 2540, 2606, 4466, 892, 31, 4779, 1279, 4014, 1939, 2121, 2897, 3275, 806, 1350, 2474, 1724, 3549, 2798, 4542, 1392, 3934, 672, 2886, 1174, 3905, 4848, 556, 1443, 2800]
            self.positions = [self.positions[i] for i in indices]
        else:
            self.positions = self.positions[:config['data'].get('debug_num_train_data', len(self.positions))]

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        map = self.positions[idx]['map']
        file_frame = self.positions[idx]['file_frame']
        x = self.positions[idx]['x'] / 1024
        y = self.positions[idx]['y'] / 1024
        z = ( self.positions[idx]['z'] - self.map_z_range[map]['min_z'] ) / (self.map_z_range[map]['max_z'] - self.map_z_range[map]['min_z'])
        angle_v = self.positions[idx]['angle_v'] / (2*np.pi)
        angle_h = self.positions[idx]['angle_h'] / (2*np.pi)

        # assert self.map_names.contains(map), f"Map {map} not in dataset.map_names !"
        # assert any(fps_img_path.endswith(f"{map}/imgs/{file_frame}.png") for fps_img_path in self.fps_img_paths), f"Map {map} and screenshot {file_frame} not in dataset.fps_img_paths !"

        map_img_path = f"{self.config['data']['data_dir']}/{map}/{map}_radar_psd.png"
        map_img = Image.open(map_img_path).convert('RGB')
        map_img = self.map_transform(map_img)

        fps_img_path = f"{self.config['data']['data_dir']}/{map}/imgs/{file_frame}.png"
        fps_img = Image.open(fps_img_path).convert('RGB')
        fps_img = self.fps_transform(fps_img)

        map_prompt_template = cross_view_matching_prompts[map]
        input_ids = self.tokenizer(map_prompt_template, truncation=True, return_tensors="pt").input_ids[0]
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        return fps_img, map_img, input_ids, attention_mask, torch.Tensor((x,y,z,angle_v,angle_h)), map_to_id_dict[map]

    def get_transform(self, config):

        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

        if config["data"]["is_fps_aug"]:
            fps_transform = transforms.Compose([
                transforms.Resize((config['data']['fps_size'][0], config['data']['fps_size'][1]), interpolation=InterpolationMode.BICUBIC),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                normalize,
                CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.5),
                GridDropout(grid_size=4, p=0.3),
                RandomErasing(probability=config['data']['erasing_p'], mean=[0.0, 0.0, 0.0])
            ])
        else:
            fps_transform = transforms.Compose([
                transforms.Resize((config['data']['fps_size'][0], config['data']['fps_size'][1]), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])

        map_transform = transforms.Compose([
            transforms.Resize((config['data']['map_size'][0], config['data']['map_size'][1]), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])

        return fps_transform, map_transform


class LocalizationVLAEvalDataset(torch.utils.data.Dataset):
    def __init__(self, config, tokenizer=None):
        self.config = config
        # self.map_names = []
        # self.fps_img_paths = []
        self.positions = []
        self.map_z_range = {}
        for map_name in config["data"]["val_maps"]:
            # self.map_names.append(f"{config['data']['data_dir']}/{map_name}/{map_name}_radar_psd.png")
            # self.fps_img_paths.extend(os.listdir(f"{config['data']['data_dir']}/{map_name}/imgs/*"))
            with open(f"{config['data']['data_dir']}/{map_name}/positions.json", "r", encoding="utf-8") as f:
                datas = json.load(f)
            self.positions.extend(datas)
            max_z = -9999
            min_z = 9999
            for data in datas:
                if data['z'] > max_z:
                    max_z = data['z']
                if data['z'] < min_z:
                    min_z = data['z']
            self.map_z_range[map_name] = {
                'max_z':max_z,
                'min_z':min_z
            }
        self.fps_transform, self.map_transform = self.get_transform(config)

        self.tokenizer = tokenizer

        if config['debug']:
            indices = [3574, 180, 2456, 2540, 2606, 4466, 892, 31, 4779, 1279, 4014, 1939, 2121, 2897, 3275, 806, 1350, 2474, 1724, 3549, 2798, 4542, 1392, 3934, 672, 2886, 1174, 3905, 4848, 556, 1443, 2800]
            self.positions = [self.positions[i] for i in indices]
        else:
            self.positions = self.positions[:config['data'].get('debug_num_val_data', len(self.positions))]

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        map = self.positions[idx]['map']
        file_frame = self.positions[idx]['file_frame']
        x = self.positions[idx]['x'] / 1024
        y = self.positions[idx]['y'] / 1024
        z = ( self.positions[idx]['z'] - self.map_z_range[map]['min_z'] ) / (self.map_z_range[map]['max_z'] - self.map_z_range[map]['min_z'])
        angle_v = self.positions[idx]['angle_v'] / (2*np.pi)
        angle_h = self.positions[idx]['angle_h'] / (2*np.pi)

        # assert self.map_names.contains(map), f"Map {map} not in dataset.map_names !"
        # assert any(fps_img_path.endswith(f"{map}/imgs/{file_frame}.png") for fps_img_path in self.fps_img_paths), f"Map {map} and screenshot {file_frame} not in dataset.fps_img_paths !"

        map_img_path = f"{self.config['data']['data_dir']}/{map}/{map}_radar_psd.png"
        map_img = Image.open(map_img_path).convert('RGB')
        map_img = self.map_transform(map_img)

        fps_img_path = f"{self.config['data']['data_dir']}/{map}/imgs/{file_frame}.png"
        fps_img = Image.open(fps_img_path).convert('RGB')
        fps_img = self.fps_transform(fps_img)

        map_prompt_template = cross_view_matching_prompts[map]
        input_ids = self.tokenizer(map_prompt_template, truncation=True, return_tensors="pt").input_ids[0]
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        return fps_img, map_img, input_ids, attention_mask, torch.Tensor((x,y,z,angle_v,angle_h)), map_to_id_dict[map]

    def get_transform(self, config):

        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

        fps_transform = transforms.Compose([
                transforms.Resize((config['data']['fps_size'][0], config['data']['fps_size'][1]), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])

        map_transform = transforms.Compose([
            transforms.Resize((config['data']['map_size'][0], config['data']['map_size'][1]), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])

        return fps_transform, map_transform



### Qwen3-Max
# --- 选项 A：直接指令式 ---
# (假设 fps_text_description 是包含 Gemini 生成描述的变量)
# PROMPT_TEMPLATE = (
#     "<s><image>\n" # BOS Token 和 图像占位符
#     "First-person view description: \"{fps_text_description}\". " # 插入描述
#     "Based on the visual context from the first-person view and the overhead radar map, "
#     "predict the precise camera coordinates (x, y, z) and view angles (pitch, yaw). "
#     "Output:" # 引导模型输出
# )

# --- 选项 B：角色扮演式 (可能有助于格式化输出) ---
# PROMPT_TEMPLATE = (
#     "<s><fps>\n"
#     "USER: The First-person view environment shows: \"{fps_text_description}\". "
#     "<map>\n"
#     "Given this first-person view and the associated radar map, what are the precise normalized camera coordinates (x, y, z) and view angles (pitch, yaw)?\n"
#     "ASSISTANT: The predicted coordinates and angles are:"
# )

# --- 如何在代码中使用 (以选项 A 为例) ---
# fps_text_description = get_caption_for_image(...) # 获取当前 FPS 图的描述
# full_text_prompt = PROMPT_TEMPLATE.format(fps_text_description=fps_text_description)


### Gemini 2.5 pro
# (假设 fps_text_description 是包含 Gemini 生成描述的变量)
PROMPT_TEMPLATE_SINGLE_PLACEHOLDER = (
    "<image>\n" # BOS Token 和 统一的图像占位符
    "The visual context contains two parts: a first-person view and an overhead radar map. " # --- 明确说明 <image> 包含的内容 ---
    "The first-person view is described as: \"{fps_text_description}\". " # --- 关联 FPS 描述 ---
    "Based on relating the first-person view to the radar map, predict the precise camera coordinates (x, y, z) and view angles (pitch, yaw). " # --- 任务指令 ---
    "Output:"
)

# --- 如何在代码中使用 ---
# fps_text_description = get_caption_for_image(...)
# full_text_prompt = PROMPT_TEMPLATE_SINGLE_PLACEHOLDER.format(fps_text_description=fps_text_description)

# (假设 fps_text_description 是包含 Gemini 生成描述的变量)
PROMPT_TEMPLATE_DOUBLE_PLACEHOLDER = (
    "<image_fps><image_map>\n" # BOS 和 两个不同的图像占位符 (顺序可调)
    "The first visual input (<image_fps>) shows the first-person view, described as: \"{fps_text_description}\". " # --- 关联 FPS 描述到第一个占位符 ---
    "The second visual input (<image_map>) is the overhead radar map. " # --- 指明第二个占位符 ---
    "By relating these two views, predict the precise camera coordinates (x, y, z) and view angles (pitch, yaw). " # --- 任务指令 ---
    "Output:"
)

# --- 如何在代码中使用 ---
# fps_text_description = get_caption_for_image(...)
# full_text_prompt = PROMPT_TEMPLATE_DOUBLE_PLACEHOLDER.format(fps_text_description=fps_text_description)


# "<image_fps><image_map>\n"
# "USER: The environment shows: \"{fps_text_description}\". "
# "Given this first-person view and the associated radar map, what are the precise camera coordinates (x, y, z) and view angles (pitch, yaw)?\n"
# "ASSISTANT: The predicted coordinates and angles are: "

DIALOGUE_PROMPT_TEMPLATE = (
    "USER: You are a spatial localization assistant. "
    "Here is the overhead radar map: <image_map>\n"
    "Here is the first-person view: <image_fps>\n"
    "This view is described as: \"{fps_text_description}\".\n"
    "Find the precise coordinates (x, y, z) and view angles (pitch, yaw) of the first-person view on the radar map.\n"

    "ASSISTANT: The predicted coordinates and angles are: "
)
INSTRUCT_FINETUNE_PROMPT_TEMPLATE = (
    "**Analyze the following data to predict camera pose (x, y, z, pitch, yaw).**\n\n"
    "**Overhead Radar Map:**\n<image_map>\n\n"
    "**First-Person View:**\n<image_fps>\n\n"
    "**View Description (for First-Person View):**\n\"{fps_text_description}\"\n\n"
    "**Predicted Pose (x, y, z, pitch, yaw):**\n"
)
STRUCTURE_EXPERT_PROMPT_TEMPLATE = (
    "[CONTEXT]\n"
    "Image 1 (Overhead Radar Map): <image_map>\n"
    "Image 2 (First-Person View): <image_fps>\n"
    "Description (for Image 2): \"{fps_text_description}\"\n\n"

    "[TASK]\n"
    "Analyze the First-Person View (Image 2) and its Description. "
    "Locate this view on the Overhead Radar Map (Image 1). "
    "Output the precise 5D pose (x, y, z, pitch, yaw) of the camera.\n\n"

    "ASSISTANT: The predicted coordinates and angles are: "
)

IGNORE_INDEX = -100

class LocVLATokenTrainDataset(torch.utils.data.Dataset):
    def __init__(self, config, base_tokenizer = None, loc_tokenizer: LocTokenizer = None):
        self.config = config
        self.base_tokenizer = base_tokenizer
        self.loc_tokenizer = loc_tokenizer
        self.prompt_template = STRUCTURE_EXPERT_PROMPT_TEMPLATE
        caption_file = config['data'].get('caption_file', 'qwen_captions.json')
        default_caption = "No description provided."

        self.data_entries = []
        self.map_z_range = {}
        for map_name in config["data"]["train_maps"]:
            # --- a. 加载位置数据 ---
            with open(f"{config['data']['data_dir']}/{map_name}/positions.json", "r", encoding="utf-8") as f:
                positions_data = json.load(f)
            # --- b. 加载该地图的 Z 轴范围 ---
            max_z, min_z = -float('inf'), float('inf')
            for data in positions_data:
                if data['z'] > max_z:
                    max_z = data['z']
                if data['z'] < min_z:
                    min_z = data['z']
            self.map_z_range[map_name] = {
                'max_z':max_z,
                'min_z':min_z
            }
            # --- c. 加载该地图的 Caption 数据 ---
            with open(f"{config['data']['data_dir']}/{map_name}/{caption_file}", "r", encoding="utf-8") as f:
                captions_data = json.load(f) # 假设格式为 {"full/path/to/img.png": "caption..."}

            print(f"len({config['data']['data_dir']}/{map_name}/{caption_file}): {len(captions_data)}")
            # --- d. 合并数据 ---
            for pos_data in positions_data:
                file_frame = pos_data['file_frame']
                map_name = pos_data['map']
                # 重建 caption 字典的 key
                caption_key = f"{config['data']['data_dir']}/{map_name}/imgs/{file_frame}.png"
                # 获取 caption，如果找不到则提供一个默认值
                caption = captions_data.get(caption_key, default_caption)
                # 将所有需要的信息合并到一个字典中
                entry = {
                    'map': map_name,
                    'file_frame': file_frame,
                    'x': pos_data['x'],
                    'y': pos_data['y'],
                    'z': pos_data['z'],
                    'angle_v': pos_data['angle_v'],
                    'angle_h': pos_data['angle_h'],
                    'caption': caption
                }
                self.data_entries.append(entry)

        self.data_entries = [
            entry for entry in self.data_entries
            if entry.get('caption') and entry['caption'] != default_caption and entry['caption'].strip() != ""
        ]
        print(f"🗑️ Filtered out data_entries with missing or default captions.")
        print(f"📊 Final total entries after filtering: {len(self.data_entries)}")

        self.fps_transform, self.map_transform = self.get_transform(config)

        if config['debug']:
            # indices = [3574, 180, 2456, 2540, 2606, 4466, 892, 31, 4779, 1279, 4014, 1939, 2121, 2897, 3275, 806, 1350, 2474, 1724, 3549, 2798, 4542, 1392, 3934, 672, 2886, 1174, 3905, 4848, 556, 1443, 2800]
            indices = [335, 535, 707, 288, 21, 240, 20, 30, 809, 423, 857, 459, 557, 882, 893, 406, 24, 477, 407, 427, 453, 923, 925, 399, 752, 867, 547, 563, 424, 217, 789, 681]
            self.data_entries = [self.data_entries[i] for i in indices]
        else:
            self.data_entries = self.data_entries[:config['data'].get('debug_num_train_data', len(self.data_entries))]

    def __len__(self):
        return len(self.data_entries)

    def __getitem__(self, idx):
        # 1. (修改) 获取完整的数据条目
        data = self.data_entries[idx]
        map_name = data['map']

        # 2. (不变) 加载和转换图像
        map_img_path = f"{self.config['data']['data_dir']}/{map_name}/{map_name}_radar_psd.png"
        map_img = Image.open(map_img_path).convert('RGB')
        map_img = self.map_transform(map_img) # -> radar_img_tensor

        fps_img_path = f"{self.config['data']['data_dir']}/{map_name}/imgs/{data['file_frame']}.png"
        fps_img = Image.open(fps_img_path).convert('RGB')
        fps_img = self.fps_transform(fps_img) # -> fps_img_tensor

        # 3. (新增) 获取 FPS 图像描述
        fps_text_description = data['caption']

        # 4. (修改) 归一化坐标值 (用于填入 target 字符串)
        #    (与原代码逻辑相同)
        x_norm = data['x'] / 1024
        y_norm = data['y'] / 1024
        z_norm = (data['z'] - self.map_z_range[map_name]['min_z']) / (self.map_z_range[map_name]['max_z'] - self.map_z_range[map_name]['min_z'])
        v_norm = data['angle_v'] / (2 * np.pi)
        h_norm = data['angle_h'] / (2 * np.pi)

        # 5. (新增) 格式化 Prompt 和 Target 字符串
        prompt_string = self.prompt_template.format(fps_text_description=fps_text_description)
        loc_array = np.array([x_norm, y_norm, z_norm, v_norm, h_norm])
        answer_length = len(loc_array)

        # 6. (新增) Tokenize 完整序列以进行训练
        #    这是 VLM 训练的标准做法


        # a. Tokenize 提示 (prompt) 部分 (用于计算 labels 的掩码)
        #    我们设置 add_special_tokens=True 来添加 <s> (BOS token)
        loc_token_str = self.loc_tokenizer(loc_array)
        if self.config['data'].get('is_train_with_s_space', False):
            # prompt_string += '<s>' + ' '
            prompt_tokenized = self.base_tokenizer(
                prompt_string,
                add_special_tokens=True, # Add <s>
                return_tensors="pt"
            )
            loc_tokenized = self.base_tokenizer(
                loc_token_str,
                add_special_tokens=True, # Add <s>
                return_tensors="pt"
            )
            if self.config['model']['llm_backbone_id'] == "phi-2-3b":
                full_input_ids = torch.cat([torch.tensor([self.base_tokenizer.bos_token_id], dtype=torch.long), prompt_tokenized.input_ids[0], loc_tokenized.input_ids[0], torch.tensor([self.base_tokenizer.eos_token_id], dtype=torch.long)], dim=0)
            else:
                full_input_ids = torch.cat([prompt_tokenized.input_ids[0], loc_tokenized.input_ids[0], torch.tensor([self.base_tokenizer.eos_token_id], dtype=torch.long)], dim=0)

        else:
            # full_tokenized = self.base_tokenizer(
            #     prompt_string + loc_token_str,
            #     add_special_tokens=True, # Add <s>
            #     return_tensors="pt"
            # )
            #llama_space_id_tensor = torch.tensor([29871], dtype=torch.long)
            # full_input_ids = torch.cat([full_tokenized.input_ids[0], torch.tensor([self.base_tokenizer.eos_token_id], dtype=torch.long)], dim=0)

            prompt_tokenized = self.base_tokenizer(
                prompt_string,
                add_special_tokens=True, # Add <s>
                return_tensors="pt"
            )
            loc_tokenized = self.base_tokenizer(
                loc_token_str,
                add_special_tokens=False, # Add <s>
                return_tensors="pt"
            )

            if self.config['model']['llm_backbone_id'] == "phi-2-3b":
                full_input_ids = torch.cat([torch.tensor([self.base_tokenizer.bos_token_id], dtype=torch.long), prompt_tokenized.input_ids[0], loc_tokenized.input_ids[0], torch.tensor([self.base_tokenizer.eos_token_id], dtype=torch.long)], dim=0)
            else:
                full_input_ids = torch.cat([prompt_tokenized.input_ids[0], loc_tokenized.input_ids[0], torch.tensor([self.base_tokenizer.eos_token_id], dtype=torch.long)], dim=0)

            # 9. Create Labels tensor
            prompt_length = len(full_input_ids) - (answer_length + 1)

            # if full_input_ids[prompt_length-1] != 29871:
            #     full_input_ids = torch.cat(
            #         [
            #             full_input_ids[:prompt_length],
            #             torch.tensor([29871], dtype=torch.long),
            #             full_input_ids[-(answer_length + 1):]
            #         ], dim=0
            #     )

        # Start with IGNORE_INDEX for prompt, then add answer tokens
        full_label_ids = full_input_ids.clone()
        if self.config['data'].get('is_train_with_s_space', False) and self.config['data'].get('is_train_loss_with_s_space'):
            full_label_ids[: -(2 + answer_length + 1)] = IGNORE_INDEX
        else:
            full_label_ids[: -(answer_length + 1)] = IGNORE_INDEX

        # 10. Apply Padding & Truncation to the **full sequence**
        max_len = self.base_tokenizer.model_max_length

        # Truncate if necessary (from the right, affecting mostly the answer part if too long)
        input_ids = full_input_ids[:max_len]
        label_ids = full_label_ids[:max_len]

        # Create attention mask based on actual length *before* padding
        actual_length = len(input_ids)
        attention_mask = torch.ones(actual_length, dtype=torch.long)

        # Pad if necessary (right padding)
        padding_length = max_len - actual_length
        if padding_length > 0:
            input_ids = torch.cat([
                input_ids,
                torch.full((padding_length,), self.base_tokenizer.pad_token_id, dtype=torch.long)
            ], dim=0)
            label_ids = torch.cat([
                label_ids,
                torch.full((padding_length,), IGNORE_INDEX, dtype=torch.long)
            ], dim=0)
            attention_mask = torch.cat([
                attention_mask,
                torch.zeros(padding_length, dtype=torch.long)
            ], dim=0)

        # Final check on shapes
        assert input_ids.shape[0] == max_len
        assert attention_mask.shape[0] == max_len
        assert label_ids.shape[0] == max_len

        # 11. Get map_id integer (same as before)
        map_id_int = map_to_id_dict.get(map_name, -1)

        return {
            "fps_img": fps_img,
            "map_img": map_img,
            "input_ids": input_ids,         # [max_len]
            "attention_mask": attention_mask, # [max_len]
            "label_ids": label_ids,               # [max_len] with IGNORE_INDEX
            "map_id": map_id_int

        }

    def get_transform(self, config):

        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

        if config["data"]["is_fps_aug"]:
            fps_transform = transforms.Compose([
                transforms.Resize((config['data']['fps_size'][0], config['data']['fps_size'][1]), interpolation=InterpolationMode.BICUBIC),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                normalize,
                CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.5),
                GridDropout(grid_size=4, p=0.3),
                RandomErasing(probability=config['data']['erasing_p'], mean=[0.0, 0.0, 0.0])
            ])
        else:
            fps_transform = transforms.Compose([
                transforms.Resize((config['data']['fps_size'][0], config['data']['fps_size'][1]), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])

        map_transform = transforms.Compose([
            transforms.Resize((config['data']['map_size'][0], config['data']['map_size'][1]), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])

        return fps_transform, map_transform


class LocVLATokenEvalDataset(torch.utils.data.Dataset):
    def __init__(self, config, base_tokenizer = None, loc_tokenizer: LocTokenizer = None):
        self.config = config
        self.base_tokenizer = base_tokenizer
        self.loc_tokenizer = loc_tokenizer

        self.prompt_template = STRUCTURE_EXPERT_PROMPT_TEMPLATE
        caption_file = config['data'].get('caption_file', 'qwen_captions.json')
        default_caption = "No description provided."

        self.data_entries = []
        self.map_z_range = {}
        for map_name in config["data"]["val_maps"]:
            # --- a. 加载位置数据 ---
            with open(f"{config['data']['data_dir']}/{map_name}/positions.json", "r", encoding="utf-8") as f:
                positions_data = json.load(f)
            # --- b. 加载该地图的 Z 轴范围 ---
            max_z, min_z = -float('inf'), float('inf')
            for data in positions_data:
                if data['z'] > max_z:
                    max_z = data['z']
                if data['z'] < min_z:
                    min_z = data['z']
            self.map_z_range[map_name] = {
                'max_z':max_z,
                'min_z':min_z
            }
            # --- c. 加载该地图的 Caption 数据 ---
            with open(f"{config['data']['data_dir']}/{map_name}/{caption_file}", "r", encoding="utf-8") as f:
                captions_data = json.load(f) # 假设格式为 {"full/path/to/img.png": "caption..."}
            # --- d. 合并数据 ---
            for pos_data in positions_data:
                file_frame = pos_data['file_frame']
                map_name = pos_data['map']
                # 重建 caption 字典的 key
                caption_key = f"{config['data']['data_dir']}/{map_name}/imgs/{file_frame}.png"
                # 获取 caption，如果找不到则提供一个默认值
                caption = captions_data.get(caption_key, "No description provided.")
                # 将所有需要的信息合并到一个字典中
                entry = {
                    'map': map_name,
                    'file_frame': file_frame,
                    'x': pos_data['x'],
                    'y': pos_data['y'],
                    'z': pos_data['z'],
                    'angle_v': pos_data['angle_v'],
                    'angle_h': pos_data['angle_h'],
                    'caption': caption
                }
                self.data_entries.append(entry)

        self.data_entries = [
            entry for entry in self.data_entries
            if entry.get('caption') and entry['caption'] != default_caption and entry['caption'].strip() != ""
        ]
        print(f"🗑️ Filtered out data_entries with missing or default captions.")
        print(f"📊 Final total entries after filtering: {len(self.data_entries)}")

        self.fps_transform, self.map_transform = self.get_transform(config)

        if config['debug']:
            # indices = [3574, 180, 2456, 2540, 2606, 4466, 892, 31, 4779, 1279, 4014, 1939, 2121, 2897, 3275, 806, 1350, 2474, 1724, 3549, 2798, 4542, 1392, 3934, 672, 2886, 1174, 3905, 4848, 556, 1443, 2800]
            indices = [335, 535, 707, 288, 21, 240, 20, 30, 809, 423, 857, 459, 557, 882, 893, 406, 24, 477, 407, 427, 453, 923, 925, 399, 752, 867, 547, 563, 424, 217, 789, 681]
            self.data_entries = [self.data_entries[i] for i in indices]
        else:
            self.data_entries = self.data_entries[:config['data'].get('debug_num_val_data', len(self.data_entries))]

    def __len__(self):
        return len(self.data_entries)

    def __getitem__(self, idx):
        # 1. (修改) 获取完整的数据条目
        data = self.data_entries[idx]
        map_name = data['map']

        # 2. (不变) 加载和转换图像
        map_img_path = f"{self.config['data']['data_dir']}/{map_name}/{map_name}_radar_psd.png"
        map_img = Image.open(map_img_path).convert('RGB')
        map_img = self.map_transform(map_img) # -> radar_img_tensor

        fps_img_path = f"{self.config['data']['data_dir']}/{map_name}/imgs/{data['file_frame']}.png"
        fps_img = Image.open(fps_img_path).convert('RGB')
        fps_img = self.fps_transform(fps_img) # -> fps_img_tensor

        # 3. (新增) 获取 FPS 图像描述
        fps_text_description = data['caption']

        # 4. (修改) 归一化坐标值 (用于填入 target 字符串)
        #    (与原代码逻辑相同)
        x_norm = data['x'] / 1024
        y_norm = data['y'] / 1024
        z_norm = (data['z'] - self.map_z_range[map_name]['min_z']) / (self.map_z_range[map_name]['max_z'] - self.map_z_range[map_name]['min_z'])
        v_norm = data['angle_v'] / (2 * np.pi)
        h_norm = data['angle_h'] / (2 * np.pi)

        # 5. (新增) 格式化 Prompt 和 Target 字符串
        prompt_string = self.prompt_template.format(fps_text_description=fps_text_description)
        loc_array = np.array([x_norm, y_norm, z_norm, v_norm, h_norm])
        answer_length = len(loc_array)
        gt_coords = torch.tensor(loc_array, dtype=torch.float32) # Shape [5]

        # 6. (新增) Tokenize 完整序列以进行训练
        #    这是 VLM 训练的标准做法

        # a. Tokenize 提示 (prompt) 部分 (用于计算 labels 的掩码)
        #    我们设置 add_special_tokens=True 来添加 <s> (BOS token)
        prompt_tokenized = self.base_tokenizer(
            prompt_string,
            add_special_tokens=True,
            # padding="max_length",
            # truncation=True,
            # max_length=self.base_tokenizer.model_max_length, # 使用完整长度
            return_tensors="pt"
        )
        if self.config['data'].get('is_generate_with_s_space', False):
            input_ids_prompt = torch.cat([prompt_tokenized.input_ids[0], torch.tensor([self.base_tokenizer.bos_token_id], dtype=torch.long), torch.tensor([29871], dtype=torch.long)], dim=0)
            attention_mask_prompt = torch.cat([prompt_tokenized.input_ids[0], torch.tensor([1, 1], dtype=torch.long)], dim=0)
        else:
            input_ids_prompt = prompt_tokenized.input_ids[0]
            attention_mask_prompt = prompt_tokenized.attention_mask[0]

        if self.config['model']['llm_backbone_id'] == "phi-2-3b":
            input_ids_prompt = torch.cat([torch.tensor([self.base_tokenizer.bos_token_id], dtype=torch.long), input_ids_prompt], dim=0)
            attention_mask_prompt = torch.cat([torch.tensor([1], dtype=torch.long), attention_mask_prompt], dim=0)

        max_len = self.base_tokenizer.model_max_length
        input_ids_prompt = input_ids_prompt[:max_len]
        attention_mask_prompt = attention_mask_prompt[:max_len]
        # actual_length = len(input_ids_prompt)
        # padding_length = max_len - actual_length
        # if padding_length > 0:
        #     input_ids_prompt = torch.cat([
        #         input_ids_prompt,
        #         torch.full((padding_length,), self.base_tokenizer.pad_token_id, dtype=torch.long)
        #     ], dim=0)
        #     attention_mask_prompt = torch.cat([
        #         attention_mask_prompt,
        #         torch.zeros(padding_length, dtype=torch.long)
        #     ], dim=0)

        # 完整label，对齐forward和generate的teacher-forcing training过程
        loc_token_str = self.loc_tokenizer(loc_array)
        if self.config['data'].get('is_generate_with_s_space', False):
            # prompt_string += '<s>' + ' '
            prompt_tokenized = self.base_tokenizer(
                prompt_string,
                add_special_tokens=True, # Add <s>
                return_tensors="pt"
            )
            loc_tokenized = self.base_tokenizer(
                loc_token_str,
                add_special_tokens=True, # Add <s>
                return_tensors="pt"
            )
            if self.config['model']['llm_backbone_id'] == "phi-2-3b":
                full_input_ids = torch.cat([torch.tensor([self.base_tokenizer.bos_token_id], dtype=torch.long), prompt_tokenized.input_ids[0], loc_tokenized.input_ids[0], torch.tensor([self.base_tokenizer.eos_token_id], dtype=torch.long)], dim=0)
            else:
                full_input_ids = torch.cat([prompt_tokenized.input_ids[0], loc_tokenized.input_ids[0],  torch.tensor([self.base_tokenizer.eos_token_id], dtype=torch.long)], dim=0)
        else:
            # full_tokenized = self.base_tokenizer(
            #     prompt_string + loc_token_str,
            #     add_special_tokens=True, # Add <s>
            #     return_tensors="pt"
            # )
            # full_input_ids = torch.cat([full_tokenized.input_ids[0], torch.tensor([self.base_tokenizer.eos_token_id], dtype=torch.long)], dim=0)

            prompt_tokenized = self.base_tokenizer(
                prompt_string,
                add_special_tokens=True, # Add <s>
                return_tensors="pt"
            )
            loc_tokenized = self.base_tokenizer(
                loc_token_str,
                add_special_tokens=False, # Add <s>
                return_tensors="pt"
            )

            if self.config['model']['llm_backbone_id'] == "phi-2-3b":
                full_input_ids = torch.cat([torch.tensor([self.base_tokenizer.bos_token_id], dtype=torch.long), prompt_tokenized.input_ids[0], loc_tokenized.input_ids[0], torch.tensor([self.base_tokenizer.eos_token_id], dtype=torch.long)], dim=0)
            else:
                full_input_ids = torch.cat([prompt_tokenized.input_ids[0], loc_tokenized.input_ids[0], torch.tensor([self.base_tokenizer.eos_token_id], dtype=torch.long)], dim=0)

            # prompt_length = len(full_input_ids) - (answer_length + 1)
            # if full_input_ids[prompt_length-1] != 29871:
            #     full_input_ids = torch.cat(
            #         [
            #             full_input_ids[:prompt_length],
            #             torch.tensor([29871], dtype=torch.long),
            #             full_input_ids[-(answer_length + 1):]
            #         ], dim=0
            #     )

        full_label_ids = full_input_ids.clone()
        full_label_ids[: -(answer_length + 1)] = IGNORE_INDEX
        max_len = self.base_tokenizer.model_max_length
        input_ids = full_input_ids[:max_len]
        label_ids = full_label_ids[:max_len]
        actual_length = len(input_ids)
        attention_mask = torch.ones(actual_length, dtype=torch.long)
        # padding_length = max_len - actual_length
        # if padding_length > 0:
        #     input_ids = torch.cat([
        #         input_ids,
        #         torch.full((padding_length,), self.base_tokenizer.pad_token_id, dtype=torch.long)
        #     ], dim=0)
        #     label_ids = torch.cat([
        #         label_ids,
        #         torch.full((padding_length,), IGNORE_INDEX, dtype=torch.long)
        #     ], dim=0)
        #     attention_mask = torch.cat([
        #         attention_mask,
        #         torch.zeros(padding_length, dtype=torch.long)
        #     ], dim=0)




        map_id_int = map_to_id_dict[map_name]

        return {
            "fps_img": fps_img,                   # Tensor [C, H, W]
            "map_img": map_img,                   # Tensor [C, H, W]
            "prompt_input_ids": input_ids_prompt, # LongTensor [SeqLen_max] (Prompt only)
            "prompt_attention_mask": attention_mask_prompt, # LongTensor [SeqLen_max] (Prompt only)
            "gt_coords": gt_coords, # FloatTensor [5] (For L2 score)
            "map_id": map_id_int,                  # int

            "input_ids": input_ids,         # [max_len]
            "attention_mask": attention_mask, # [max_len]
            "label_ids": label_ids,               # [max_len] with IGNORE_INDEX
        }

    def get_transform(self, config):

        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

        fps_transform = transforms.Compose([
                transforms.Resize((config['data']['fps_size'][0], config['data']['fps_size'][1]), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])

        map_transform = transforms.Compose([
            transforms.Resize((config['data']['map_size'][0], config['data']['map_size'][1]), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])

        return fps_transform, map_transform



def create_gaussian_heatmap_label(
    gt_x: float,
    gt_y: float,
    map_size: int = 1024,
    heatmap_size: int = 128,
    sigma: float = 1.5
) -> torch.Tensor:
    """
    为单个 (x, y) 坐标生成一个 2D 高斯热力图标签。

    Args:
        gt_x (float): 1024x1024 空间中的 X 坐标。
        gt_y (float): 1024x1024 空间中的 Y 坐标。
        map_size (int): 原始地图尺寸 (例如 1024)。
        heatmap_size (int): 目标热力图尺寸 (例如 128)。
        sigma (float): 高斯核的标准差 (控制点的大小)。

    Returns:
        torch.Tensor: [heatmap_size, heatmap_size] 的热力图标签。
    """

    # 1. 缩放坐标
    scale_factor = heatmap_size / map_size
    x_scaled = gt_x * scale_factor
    y_scaled = gt_y * scale_factor

    # 2. 创建坐标网格
    x_grid, y_grid = np.meshgrid(np.arange(heatmap_size), np.arange(heatmap_size))

    # 3. 计算高斯分布
    # (i - x_scaled)^2 + (j - y_scaled)^2
    dist_sq = (x_grid - x_scaled)**2 + (y_grid - y_scaled)**2

    # exp( - dist_sq / (2 * sigma^2) )
    heatmap = np.exp(-dist_sq / (2 * sigma**2))

    # 确保峰值为 1.0，并处理浮点精度问题
    heatmap[int(y_scaled), int(x_scaled)] = 1.0

    return torch.tensor(heatmap, dtype=torch.float32)

# --- 示例用法 ---
# 假设真实坐标在 (500, 300)
# gt_heatmap_label = create_gaussian_heatmap_label(500, 300, 1024, 128, sigma=1.5)
# print(gt_heatmap_label.shape) # torch.Size([128, 128])
