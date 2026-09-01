
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29501
export NNODES=1
export NODE_RANK=0
export NPROC_PER_NODE=6

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m scripts_train.train configs/holocap-qwen2.5-0.5b-lora.yaml

