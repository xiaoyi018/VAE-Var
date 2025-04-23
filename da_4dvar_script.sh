#!/bin/bash

gpus=1
node_num=1
single_gpus=`expr $gpus / $node_num`

cpus=16

# export NCCL_IB_DISABLE=1
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL

srun -p ai4earth --quotatype=auto --ntasks-per-node=$single_gpus --cpus-per-task=$cpus -N $node_num -o job/%j.out --gres=gpu:$single_gpus --async -u python da_4dvar.py --prefix="test" --da_mode=vae4dvar --scale_factor=2.0 --da_win=1 --obs_std=0.005  --obs_type="column_random_0001" --modify_tp=2 --start_time="2022-01-01 00:00:00" --end_time="2023-01-01 12:00:00"  --save_interval=1 --Nit=4 --q_type=1 --obs_coeff=1.0 --filter_coeff=0.1 --coeff_dir="dataset/bq_info_lr/" --param_str="parameters0_old" --vae_ckpt="nf_model/ckpts/vae_parameters0_old_sigma2.00_epoch4_19790101_20151231_finetuned.pt"

sleep 2
rm -f batchscript-*
