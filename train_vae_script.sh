#!/bin/bash

gpus=1
node_num=1
single_gpus=`expr $gpus / $node_num`

cpus=4

# export NCCL_IB_DISABLE=1
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# -w SH-IDC1-10-140-24-47 


PORT=$((((RANDOM<<15)|RANDOM)%49152 + 10000))

echo $PORT

srun -p ai4earth --quotatype=auto --ntasks-per-node=$single_gpus --cpus-per-task=$cpus -N $node_num -o job/%j.out --gres=gpu:$single_gpus --async -u python train_vae.py \
--init_method 'tcp://127.0.0.1:'$PORT   \
--world_size $gpus        \
--per_cpus $cpus    \
--length  5      \
--predict_len 4    \
--batch_size  1    \
--sigma 2.0 \
--cfgdir '/mnt/petrelfs/xiaoyi/projects/fengwu-hr/output/model/model_0.25degree' \
--param_str 'parameters0_old' \
--start_year '2014-01-01 00:00:00' \

sleep 2
rm -f batchscript-*
