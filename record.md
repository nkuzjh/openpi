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

## scripts/train_pytorch.py pytorch框架
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
 `` CUDA_VISIBLE_DEVICES=1 uv run train_csgo.py pi05_csgo --exp_name exp0_debug ``  `` --resume ``

 - 利用uv的.venv/bin/python执行nohup
 `` CUDA_VISIBLE_DEVICES=1 nohup .venv/bin/python -u train_csgo.py pi05_csgo --exp_name exp0_debug > out.out 2>&1 & ``

 - 启用uv的.venv环境执行nohup
 `` source .venv/bin/activate; CUDA_VISIBLE_DEVICES=1 nohup python -u train_csgo.py pi05_csgo --exp_name exp0_debug > out.out 2>&1 &``

## eval_csgo.py
``    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --config pi05_csgo --exp_name exp0_debug --checkpoint checkpoints/pi05_csgo/exp0_debug/5000    ``
``    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --config pi05_csgo --exp_name exp0_debug --checkpoint checkpoints/pi05_csgo/exp0_debug/10000    ``
``    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --config pi05_csgo --exp_name exp0_debug --checkpoint checkpoints/pi05_csgo/exp0_debug/15000    ``



# exp

## pi05_csgo/exp0_debug
- 自定义数据集中的img_aug(fps_aug)
- 使用libero机器人数据集的norm_stats训练和推理(libero_norm)
- padding_resize + pi05_aug
- peak_lr=5e-5; warmup_steps=10_000; num_train_steps=30_000 (默认超参)
 ``     CUDA_VISIBLE_DEVICES=0 uv run train_csgo.py pi05_csgo --exp_name exp0_debug --resume    ``

**eval_csgo.py**
``    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --config pi05_csgo --exp_name exp0_debug --checkpoint checkpoints/pi05_csgo/exp0_debug/15000    ``
``    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --config pi05_csgo --exp_name exp0_debug --checkpoint checkpoints/pi05_csgo/exp0_debug/30000    ``


## pi05_csgo_exp1
- 去除自定义数据集中的img_aug
- 使用libero机器人数据集的norm_stats训练和推理(libero_norm)
- padding_resize + pi05_aug
- peak_lr=2e-5; warmup_steps=1000; num_train_steps=10_000
 ``     CUDA_VISIBLE_DEVICES=1 uv run train_csgo.py pi05_csgo_exp1 --exp_name pi05_csgo_exp1    ``
 ``     CUDA_VISIBLE_DEVICES=1 python train_csgo.py pi05_csgo_exp1 --exp_name pi05_csgo_exp1    ``
 ``     CUDA_VISIBLE_DEVICES=1 nohup python train_csgo.py pi05_csgo_exp1 --exp_name pi05_csgo_exp1 > outs/pi05_csgo_exp1.out 2>&1 &     ``

**eval_csgo.py**
``    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --config pi05_csgo_exp1 --exp_name pi05_csgo_exp1 --checkpoint checkpoints/pi05_csgo_exp1/pi05_csgo_exp/9999    ``


## pi05_csgo_exp2
- 去除自定义数据集中的img_aug
- 去除pi05原有的action预先计算norm_stats和训练正则化过程(wo/norm)
- padding_resize + pi05_aug
 ``     CUDA_VISIBLE_DEVICES=1 nohup python train_csgo.py pi05_csgo_exp2 --exp_name pi05_csgo_exp2 > outs/pi05_csgo_exp2.out 2>&1 &``

 ``     CUDA_VISIBLE_DEVICES=1 python train_csgo.py pi05_csgo_exp2 --exp_name pi05_csgo_exp2    ``

**eval_csgo.py**
``    CUDA_VISIBLE_DEVICES=1 python eval_csgo.py --config pi05_csgo_exp2 --exp_name pi05_csgo_exp2 --checkpoint checkpoints/pi05_csgo_exp2/pi05_csgo_exp2/10000    ``


## pi05_csgo_exp3
- 去除自定义数据集中的img_aug
- 去除pi05原有的action预先计算norm_stats和训练正则化过程(wo/norm)
- fps_dropout + padding_resize + pi05_aug
 ``     CUDA_VISIBLE_DEVICES=1 nohup python train_csgo.py pi05_csgo_exp3 --exp_name pi05_csgo_exp3 > outs/pi05_csgo_exp3.out 2>&1 &     ``
 ``     CUDA_VISIBLE_DEVICES=0 python train_csgo.py pi05_csgo_exp3 --exp_name pi05_csgo_exp3    ``

**eval_csgo.py**
``    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --config pi05_csgo_exp3 --exp_name pi05_csgo_exp3 --checkpoint checkpoints/pi05_csgo_exp3/pi05_csgo_exp3/5000    ``


## pi05_csgo_exp4
- 去除自定义数据集中的img_aug
- 去除pi05原有的action预先计算norm_stats和训练正则化过程(wo/norm)
- fps_resize_dropout + padding_resize + pi05_aug
 ``     CUDA_VISIBLE_DEVICES=1 nohup python train_csgo.py pi05_csgo_exp4 --exp_name pi05_csgo_exp4 > outs/pi05_csgo_exp4.out 2>&1 &     ``
 ``     CUDA_VISIBLE_DEVICES=1 python train_csgo.py pi05_csgo_exp4 --exp_name pi05_csgo_exp4    ``


## pi05_csgo_exp5
- 去除自定义数据集中的img_aug
- 去除pi05原有的action预先计算norm_stats和训练正则化过程(wo/norm)
- fps_dropout + padding_resize + pi05_aug
- LAPE from LLaVA-ST


## pi05_csgo_exp6
- Pi0_fast
- 去除自定义数据集中的img_aug
- 去除pi05原有的action预先计算norm_stats和训练正则化过程(wo/norm)
- fps_dropout + padding_resize + pi05_aug


## pi05_csgo_exp7
- Pi0
- 去除自定义数据集中的img_aug
- 去除pi05原有的action预先计算norm_stats和训练正则化过程(wo/norm)
- fps_dropout + padding_resize + pi05_aug



