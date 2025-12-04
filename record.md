# Configs
- LiberoInputs and LiberoOutputs: Defines the data mapping from the LIBERO environment to the model and vice versa. Will be used for both, training and inference.
- LeRobotLiberoDataConfig: Defines how to process raw LIBERO data from LeRobot dataset for training.
- TrainConfig: Defines fine-tuning hyperparameters, data config, and weight loader.

# 训练命令

## scripts/train.py flax.nnx框架
- 计算数据集统计值:
    `` uv run scripts/compute_norm_stats.py --config-name pi05_libero ``
- 训练:
    `` CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero --exp-name=my_experiment --overwrite ``

## train_pytorch.py pytorch框架
- Single GPU training:
`` uv run scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval> ``

    - Example:
    `` uv run scripts/train_pytorch.py debug --exp_name pytorch_test ``
    `` uv run scripts/train_pytorch.py debug --exp_name pytorch_test --resume `` Resume from latest checkpoint
    `` CUDA_VISIBLE_DEVICES=0 uv run scripts/train_pytorch.py debug_pi05 --exp_name pytorch_test ``
    `` CUDA_VISIBLE_DEVICES=0 uv run scripts/train_pytorch.py pi05_libero --exp_name pytorch_test ``

- Multi-GPU training (single node):
`` uv run torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name> ``

    - Example:
    `` uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test ``
    `` uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test --resume ``
    `` uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi05_libero --exp_name pytorch_ddp_test ``

- Multi-Node Training:
`` uv run torchrun --nnodes=<num_nodes> --nproc_per_node=<gpus_per_node> --node_rank=<rank_of_node> --master_addr=<master_ip> --master_port=<port> scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval> ``

## train_csgo.py
- uv训练
 `` CUDA_VISIBLE_DEVICES=1 uv run train_csgo.py pi05_csgo --exp_name exp0_debug ``
 - 利用uv的.venv/bin/python执行nohup
 `` CUDA_VISIBLE_DEVICES=1 nohup .venv/bin/python -u train_csgo.py pi05_csgo --exp_name exp0_debug > out.out 2>&1 & ``
 - 启用uv的.venv环境执行nohup
 `` source .venv/bin/activate; CUDA_VISIBLE_DEVICES=1 nohup python -u train_csgo.py pi05_csgo --exp_name exp0_debug > out.out 2>&1 &``

