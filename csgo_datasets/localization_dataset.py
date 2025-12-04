
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

from csgo_datasets.utils import CoarseDropout, GridDropout
from csgo_datasets.random_erasing import RandomErasing
from loc_tokenizer import LocTokenizer



IGNORE_INDEX = -100



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



INSTRUCTION_TUNING_PROMPT_TEMPLATE = (
    "The following visual data has been fused and inserted into this sequence:\n"
    "1. First-Person View (FPV) Features.\n"
    "2. Overhead Radar Map (RADAR) Features.\n"
    "Analyze the spatial relationship between the FPV and the RADAR Map to determine the precise camera pose. "
    "Predict the 5D pose (x, y, z, pitch, yaw) in the required format.\n"
    "{action_token_sequence}"
)

class CsgoTrainDataset_IT(torch.utils.data.Dataset):
    def __init__(self, config, base_tokenizer = None, loc_tokenizer: LocTokenizer = None):
        self.config = config
        self.prompt_template = INSTRUCTION_TUNING_PROMPT_TEMPLATE

        self.data_entries = []
        self.map_z_range = {}
        for map_name in config["train_maps"]:
            # --- a. 加载位置数据 ---
            with open(f"{config['data_dir']}/{map_name}/positions.json", "r", encoding="utf-8") as f:
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
            # --- d. 合并数据 ---
            for pos_data in positions_data:
                file_frame = pos_data['file_frame']
                map_name = pos_data['map']
                entry = {
                    'map': map_name,
                    'file_frame': file_frame,
                    'x': pos_data['x'],
                    'y': pos_data['y'],
                    'z': pos_data['z'],
                    'angle_v': pos_data['angle_v'],
                    'angle_h': pos_data['angle_h'],
                }
                self.data_entries.append(entry)

        print(f"📊 Final total entries : {len(self.data_entries)}")
        self.data_entries = [data for data in self.data_entries if data['x']!=562 and data['y']!=736]
        print(f"📊 after filter damaged entries: {len(self.data_entries)}")
        self.data_entries = self.data_entries[:-2000]
        print(f"📊 Final train entries : {len(self.data_entries)}")

        self.fps_transform, self.map_transform = self.get_transform(config)

        if config['debug']:
            indices = [335, 535, 707, 288, 21, 240, 20, 30, 809, 423, 857, 459, 557, 882, 893, 406, 24, 477, 407, 427, 453, 923, 925, 399, 752, 867, 547, 563, 424, 217, 789, 681]
            self.data_entries = [self.data_entries[i] for i in indices]
        else:
            self.data_entries = self.data_entries[:config.get('debug_num_train_data', len(self.data_entries))]

    def __len__(self):
        return len(self.data_entries)

    def __getitem__(self, idx):
        # 获取完整的数据条目
        data = self.data_entries[idx]
        map_name = data['map']

        # 加载和转换图像
        map_img_path = f"{self.config['data_dir']}/{map_name}/{map_name}_radar_psd.png"
        map_img = Image.open(map_img_path).convert('RGB')
        map_img = self.map_transform(map_img) # -> radar_img_tensor

        fps_img_path = f"{self.config['data_dir']}/{map_name}/imgs/{data['file_frame']}.png"
        fps_img = Image.open(fps_img_path).convert('RGB')
        fps_img = self.fps_transform(fps_img) # -> fps_img_tensor

        # 归一化坐标值
        x_norm = data['x'] / 1024
        y_norm = data['y'] / 1024
        z_norm = (data['z'] - self.map_z_range[map_name]['min_z']) / (self.map_z_range[map_name]['max_z'] - self.map_z_range[map_name]['min_z'])
        v_norm = data['angle_v'] / (2 * np.pi)
        h_norm = data['angle_h'] / (2 * np.pi)
        loc_array = np.array([x_norm, y_norm, z_norm, v_norm, h_norm])
        gt_coords = torch.tensor(loc_array, dtype=torch.float32)

        # Instrcution Tuning Prompt Template
        prompt_string = self.prompt_template.split("{action_token_sequence}")[0].strip()

        # Get map_id integer
        map_id_int = map_to_id_dict.get(map_name, -1)
        return {
            "image": fps_img,
            "wrist_image": map_img,
            "state": gt_coords,
            "prompt": prompt_string,
            "actions": gt_coords.unsqueeze(0),
            "map_id": map_id_int

        }

    def get_transform(self, config):

        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

        if config["is_fps_aug"]:
            fps_transform = transforms.Compose([
                transforms.Resize((config['fps_size'][0], config['fps_size'][1]), interpolation=InterpolationMode.BICUBIC),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                normalize,
                CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.5),
                GridDropout(grid_size=4, p=0.3),
                RandomErasing(probability=config['erasing_p'], mean=[0.0, 0.0, 0.0])
            ])
        else:
            fps_transform = transforms.Compose([
                transforms.Resize((config['fps_size'][0], config['fps_size'][1]), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])

        map_transform = transforms.Compose([
            transforms.Resize((config['map_size'][0], config['map_size'][1]), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])

        return fps_transform, map_transform



class CsgoEvalDataset_IT(torch.utils.data.Dataset):
    def __init__(self, config, base_tokenizer = None, loc_tokenizer: LocTokenizer = None):
        self.config = config
        self.prompt_template = INSTRUCTION_TUNING_PROMPT_TEMPLATE

        self.data_entries = []
        self.map_z_range = {}
        for map_name in config["val_maps"]:
            # --- a. 加载位置数据 ---
            with open(f"{config['data_dir']}/{map_name}/positions.json", "r", encoding="utf-8") as f:
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
            # --- d. 合并数据 ---
            for pos_data in positions_data:
                file_frame = pos_data['file_frame']
                map_name = pos_data['map']
                entry = {
                    'map': map_name,
                    'file_frame': file_frame,
                    'x': pos_data['x'],
                    'y': pos_data['y'],
                    'z': pos_data['z'],
                    'angle_v': pos_data['angle_v'],
                    'angle_h': pos_data['angle_h'],
                }
                self.data_entries.append(entry)

        print(f"📊 Final total entries: {len(self.data_entries)}")
        self.data_entries = [data for data in self.data_entries if data['x']!=562 and data['y']!=736]
        print(f"📊 after filter damaged entries: {len(self.data_entries)}")
        # print(len([data for data in self.data_entries if data['x']==562 and data['y']==736])) #87000
        self.data_entries = self.data_entries[-2000:]
        print(f"📊 Final eval entries : {len(self.data_entries)}")

        self.fps_transform, self.map_transform = self.get_transform(config)

        if config['debug']:
            indices = [335, 535, 707, 288, 21, 240, 20, 30, 809, 423, 857, 459, 557, 882, 893, 406, 24, 477, 407, 427, 453, 923, 925, 399, 752, 867, 547, 563, 424, 217, 789, 681]
            self.data_entries = [self.data_entries[i] for i in indices]
        else:
            self.data_entries = self.data_entries[:config.get('debug_num_val_data', len(self.data_entries))]

    def __len__(self):
        return len(self.data_entries)

    def __getitem__(self, idx):
        # 获取完整的数据条目
        data = self.data_entries[idx]
        map_name = data['map']

        # 加载和转换图像
        map_img_path = f"{self.config['data_dir']}/{map_name}/{map_name}_radar_psd.png"
        map_img = Image.open(map_img_path).convert('RGB')
        # map_img = self.map_transform(map_img) # -> radar_img_tensor

        fps_img_path = f"{self.config['data_dir']}/{map_name}/imgs/{data['file_frame']}.png"
        fps_img = Image.open(fps_img_path).convert('RGB')
        # fps_img = self.fps_transform(fps_img) # -> fps_img_tensor

        # 归一化坐标值 (用于填入 target 字符串)
        x_norm = data['x'] / 1024
        y_norm = data['y'] / 1024
        z_norm = (data['z'] - self.map_z_range[map_name]['min_z']) / (self.map_z_range[map_name]['max_z'] - self.map_z_range[map_name]['min_z'])
        v_norm = data['angle_v'] / (2 * np.pi)
        h_norm = data['angle_h'] / (2 * np.pi)
        loc_array = np.array([x_norm, y_norm, z_norm, v_norm, h_norm])
        gt_coords = torch.tensor(loc_array, dtype=torch.float32)

        # Instrcution Tuning Prompt Template
        prompt_string = self.prompt_template.split("{action_token_sequence}")[0].strip()

        # Get map_id integer
        map_id_int = map_to_id_dict[map_name]
        return {
            "image": fps_img,
            "wrist_image": map_img,
            "state": gt_coords,
            "prompt": prompt_string,
            "actions": gt_coords.unsqueeze(0),
            "map_id": map_id_int
        }

    def get_transform(self, config):

        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

        fps_transform = transforms.Compose([
                transforms.Resize((config['fps_size'][0], config['fps_size'][1]), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])

        map_transform = transforms.Compose([
            transforms.Resize((config['map_size'][0], config['map_size'][1]), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])

        return fps_transform, map_transform

