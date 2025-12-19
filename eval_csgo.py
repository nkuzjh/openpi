
import dataclasses
import gc
import logging
import os
import platform
import shutil
import time
import random
import datetime
import argparse
from collections import defaultdict

import jax
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.parallel
from torch.utils.data import DataLoader
import tqdm
import wandb

import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data

from openpi.training import config as _config
from openpi.policies import droid_policy
from openpi.policies import policy_config
from openpi.shared import download

from csgo_datasets.localization_dataset import id_to_map_dict, map_to_id_dict, CsgoTrainDataset_IT, CsgoEvalDataset_IT

import PIL
from PIL import Image, ImageDraw



def set_seed(seed=42):
    # 1. Python 内置 random
    random.seed(seed)
    # 2. 操作系统环境 (这对某些哈希操作是必须的，如 set/dict 的顺序)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 3. NumPy
    np.random.seed(seed)
    # 4. PyTorch CPU
    torch.manual_seed(seed)
    # 5. PyTorch GPU (如果可用)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # 如果有多张显卡，为所有显卡设置
    # 6. 设置 CuDNN 后端以确保确定性 (会降低性能)
    # 如果你非常看重结果的逐位一致性，必须开启 deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"随机种子已设置为: {seed}")



@torch.no_grad()
def evaluate(config, policy, dataloader, device, epoch, logger):
    total_loss = 0.0
    total_score = 0.0
    num_samples = 0
    num_batches = len(dataloader)

    predictions = []
    labels = []
    map_ids = set()
    for batch_idx, batch in tqdm.tqdm(enumerate(dataloader), total=num_batches):
        fps_img = batch["image"]#.to(device)
        map_img = batch["wrist_image"]#.to(device)
        state = batch["state"]#.to(device)
        prompt = batch["prompt"]#.to(device)
        actions = batch["actions"]#.to(device)
        map_id = batch["map_id"]
        current_batch_size = actions.size(0)

        # example = libero_policy.make_libero_example()
        # action_chunk = policy.infer(example)["actions"]
        example  = {
            "observation/image": fps_img[0],
            "observation/wrist_image": map_img[0],
            "observation/state": state[0],
            "prompt": prompt[0],
            "actions": actions[0],
            "map_id": map_id[0],
        }
        # Run inference
        action_chunk = policy.infer(example)["actions"]

        pred_coords = action_chunk
        padding_length = 5 - pred_coords.shape[1]
        if padding_length > 0:
            pred_coords = torch.cat([
                pred_coords,
                torch.full((pred_coords.size(0), padding_length,), 0.5, dtype=torch.long).to(pred_coords.device)
            ], dim=1)
        gt_coords = actions.squeeze(1)

        # 存储解码后的坐标用于绘图
        predictions.append(pred_coords)#.detach().cpu().numpy())
        labels.append(gt_coords.detach().cpu().numpy())

        # 计算 L2 距离 (avg_score)
        dist = torch.norm(torch.tensor(pred_coords) - gt_coords, dim=1) # [B]
        total_score += dist.sum().item()

        # 计算评估损失
        if 0:
            pass
            # is_batch_eval = False
            # if not is_batch_eval:
            #     key = jax.random.key(0)
            #     # Create a model from the checkpoint.
            #     model = config.model.load(_model.restore_params(checkpoint_dir / "params"))
            #     # We can create fake observations and actions to test the model.
            #     obs, act = config.model.fake_obs(), config.model.fake_act()
            #     # Sample actions from the model.
            #     loss = model.compute_loss(key, obs, act)
            # else:
            #     # Reduce the batch size to reduce memory usage.
            #     config = dataclasses.replace(config, batch_size=2)
            #     # Load a single batch of data. This is the same data that will be used during training.
            #     # NOTE: In order to make this example self-contained, we are skipping the normalization step
            #     # since it requires the normalization statistics to be generated using `compute_norm_stats`.
            #     loader = _data_loader.create_data_loader(config, num_batches=1, skip_norm_stats=True)
            #     obs, act = next(iter(loader))
            #     # Sample actions from the model.
            #     loss = model.compute_loss(key, obs, act)

        else:
            loss = torch.tensor(0.0) # 模拟损失
        total_loss += loss.item() * current_batch_size

        for _id in map_id:
            map_ids.add(_id.item())
        # map_ids.add(map_id)

        num_samples += current_batch_size

        if batch_idx % 1 == 0:
            logger.info(f"Epoch {epoch}, Eval {batch_idx}/{num_batches}, Loss: {loss.item():.6f}")

        if batch_idx < 2 or batch_idx in [int(len(dataloader)/10),int(1+len(dataloader)/10), int(2*len(dataloader)/10),int(1+2*len(dataloader)/10)]  or len(dataloader)-batch_idx < 3:
            logger.info(f"\n        batch_idx {batch_idx}: ")

            logger.info(f"      action_chunk: {action_chunk}")
            logger.info(f"      pred_coords: {pred_coords}")
            logger.info(f"      actions: {actions}")
            logger.info(f"      gt_coords: {gt_coords}")

    predictions = np.concatenate(predictions, axis=0)
    labels = np.concatenate(labels, axis=0)
    avg_loss = total_loss / num_samples
    avg_score = total_score / num_samples
    logger.info(f"Epoch {epoch} - Eval Loss: {avg_loss:.6f}, Mean L2 Score: {avg_score:.6f}")

    return {'loss_total': avg_loss}, avg_score, predictions, labels, list(map_ids)


def inspect_dataset(dataset, name="dataset"):
    print(f"\n===== Inspecting {name}[0] =====")
    try:
        sample = dataset[0]
        print(f"Keys: {list(sample.keys())}")

        for k, v in sample.items():
            # 检查 PyTorch Tensor
            if isinstance(v, torch.Tensor):
                print(f"  [{k}] Tensor shape: {v.shape}, dtype: {v.dtype}")
            # 检查 NumPy Array
            elif isinstance(v, np.ndarray):
                print(f"  [{k}] Numpy shape: {v.shape}, dtype: {v.dtype}")
            # 其他类型 (如 int, str, dict 等)
            else:
                print(f"  [{k}] Type: {type(v)}, Value: {v}")

    except Exception as e:
        print(f"❌ Error inspecting {name}: {e}")


# from torchvision.transforms.functional import to_pil_image
def tensor_to_pil_img(img_tensor):
    img_array = img_tensor.detach().cpu().numpy()#.transpose(1,2,0)
    if img_array.shape[0]==3:
        img_array = img_array.transpose(1,2,0)
    img_array = (img_array - img_array.min())/(img_array.max() - img_array.min()) * 255
    pil_img = PIL.Image.fromarray(img_array.astype(np.uint8))
    return pil_img

def vertical_concat(images):
    """
    将多张 PIL Image 纵向拼接成一张图

    Args:
        images: List[PIL.Image.Image]，至少包含一张图像

    Returns:
        PIL.Image.Image: 拼接后的图像
    """
    if not images:
        raise ValueError("图像列表不能为空")

    # 获取宽度（假设所有图像宽度一致）
    width = images[0].width
    total_height = sum(img.height for img in images)

    # 创建新图像（模式与第一张图一致，如 'RGB' 或 'L'）
    new_image = Image.new(images[0].mode, (width, total_height))

    # 依次粘贴每张图像
    y_offset = 0
    for img in images:
        # 可选：如果宽度不一致，可在此处 resize
        if img.width != width:
            img = img.resize((width, img.height), Image.LANCZOS)
        new_image.paste(img, (0, y_offset))
        y_offset += img.height

    return new_image

def horizontal_concat_pad(img1, img2, bg_color=(255, 255, 255)):
    """通过 padding 统一高度（居中）"""
    img1 = img1.convert('RGB')
    img2 = img2.convert('RGB')

    max_h = max(img1.height, img2.height)
    total_w = img1.width + img2.width

    new_img = Image.new('RGB', (total_w, max_h), bg_color)

    # 居中粘贴 img1
    y1 = (max_h - img1.height) // 2
    new_img.paste(img1, (0, y1))

    # 居中粘贴 img2
    y2 = (max_h - img2.height) // 2
    new_img.paste(img2, (img1.width, y2))

    return new_img



def main(args, config):
    set_seed()

    cur_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/{config.name}/{config.exp_name}/eval_{cur_time_str}"
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
            format=f'%(asctime)s - %(levelname)s - %(message)s - (%(process)d:%(filename)s:%(lineno)s)',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'eval.log')),
                logging.StreamHandler()
            ],
            force=True
    )
    logger = logging.getLogger(__name__)

    logger.info(f"CsgoTrainConfig: {args.config}")
    logger.info(f"CsgoTrainConfig: {config}")
    checkpoint_dir = args.checkpoint
    logger.info(f"checkpoint: {checkpoint_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#"cpu" #
    logger.info(f"Using device: {device}")

    # Create a trained policy (automatically detects PyTorch format)
    policy = policy_config.create_trained_policy(config, checkpoint_dir)
    logger.info(f"evaluation policy: {policy}")

    # Dataset & DataLoader
    train_dataset = CsgoTrainDataset_IT(config.data.csgo_config)
    test_dataset = CsgoEvalDataset_IT(config.data.csgo_config)
    logger.info(f" test_dataset: {len(test_dataset)}")
    if 1:
        inspect_dataset(train_dataset, "train_dataset")
        inspect_dataset(test_dataset, "test_dataset")
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=16
    )
    logger.info(f"test_loader: {len(test_loader)}")
    if 1:
        from visual_utils import visualize_batch_from_dataloader
        debug_batch = next(iter(test_loader))
        debug_batch_cpu = {k: v.cpu() for k, v in debug_batch.items() if hasattr(v, 'cpu')}#debug_batch['image'].shape=torch.Size([1, 224, 224, 3])
        visualize_batch_from_dataloader(
            batch=debug_batch_cpu,
            id_to_map_dict=id_to_map_dict, # 确保 id_to_map_dict 已导入
            save_dir=log_dir, # 保存到日志目录
            prefix="DEBUG_BATCH_TEST"
        )

    if 1:
        eval_start_time = time.time()
        eval_losses = defaultdict(list)
        eval_scores = []
        eval_loss, eval_score, predictions, labels, map_ids = evaluate(config, policy, test_loader, device, -999, logger)
        eval_end_time = time.time()
        logger.info(f"test set evaluate time: {eval_end_time - eval_start_time} seconds")
        if isinstance(eval_loss, dict):
            for key, value in eval_loss.items():
                eval_losses[key].append(value)
        else:
            eval_losses['loss_total'].append(eval_loss)
        eval_scores.append(eval_score)

        color_list = [
            (255, 0, 0, 100),
            (0, 255, 0, 100),
            (0, 0, 255, 100),
            (255, 255, 0, 100),
            (255, 0, 255, 100)
        ]
        color_list2 = [
            (0, 114, 178),
            (230, 159, 0),
            (0, 158, 115),
            (240, 228, 66),
            (204, 121, 167)
        ]

        if 1:
            for map_id in map_ids:
                map_name = id_to_map_dict[map_id]
                map = Image.open(f"{config.data.csgo_config['data_dir']}/{map_name}/{map_name}_radar_psd.png").convert('RGBA')
                overlay_preds = Image.new("RGBA", map.size, (0, 0, 0, 0))
                overlay_label = Image.new("RGBA", map.size, (0, 0, 0, 0))
                preds_draw = ImageDraw.Draw(overlay_preds)
                label_draw = ImageDraw.Draw(overlay_label)
                for i, pos in enumerate(labels[:5]):
                    x, y = int(pos[0]*map.size[0]), int(pos[1]*map.size[0])
                    label_draw.ellipse((x-5, y-5, x+5, y+5), fill=color_list[i], outline='black', width=2)

                for i, pos in enumerate(predictions[:5]):
                    x, y = int(pos[0]*map.size[0]), int(pos[1]*map.size[0])
                    preds_draw.ellipse((x-9, y-9, x+9, y+9), fill=color_list[i])

                combined_image = Image.alpha_composite(map, overlay_label)
                combined_image = Image.alpha_composite(combined_image, overlay_preds)
                # combined_image.save('debug.png')
                combined_image.save(os.path.join(log_dir, f'visual_test_{map_name}.png'))

            fps_img_list = []
            for i in range(5):
                sample_dict = test_dataset[i]
                fps_img = sample_dict['image']
                map_id = sample_dict['map_id']
                fps_img = tensor_to_pil_img(torch.tensor(fps_img))
                fps_draw = ImageDraw.Draw(fps_img)
                fps_draw.ellipse((1, 1, 10, 10), fill=color_list[i])
                fps_img.save(os.path.join(log_dir, f'visual_test_{id_to_map_dict[map_id]}_fps_img{i}.png'))
                fps_img_list.append(fps_img)
            fps_imgs = vertical_concat(fps_img_list)
            fps_imgs.save(os.path.join(log_dir, f'visual_test_{map_name}_fps_imgs.png'))

            show_examples = horizontal_concat_pad(combined_image, fps_imgs)
            show_examples.save(os.path.join(log_dir, f'visual_test_{map_name}_show_examples.png'))

        logger.info(f"Evaluating completed. Test Loss: {eval_loss}, Test Score: {eval_score}")



if __name__ == "__main__":
    import torch
    print(torch.cuda.get_arch_list())

    parser = argparse.ArgumentParser(description="CSGO Cross-View Localization Pi0.5 Evaluation")
    parser.add_argument("--config", type=str, required=True, help="name of pi05_config")
    parser.add_argument("--exp_name", type=str, required=True, help="exp name of training pi05_config")
    parser.add_argument("--checkpoint", type=str, required=True, help="checkpoint dir of finetuned model")
    args = parser.parse_args()

    config = _config.get_config(args.config)
    config = dataclasses.replace(config, exp_name = args.exp_name)
    config = dataclasses.replace(
        config,
        data=dataclasses.replace(
            config.data,
            csgo_config={**config.data.csgo_config, "debug": False} #需要随机可视化5/32个postions时将debug=True,否则eval 2000个样本时debug=False
        )
    )

    main(args, config)














