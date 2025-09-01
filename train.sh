

##Eval for single gpu interactive session
export LOCAL_RANK=0
export RANK=0
export WORLD_SIZE=2
export MASTER_ADDR=localhost
export MASTER_PORT=13720 # Or any available port
export CUDA_VISIBLE_DEVICES=0,1
# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_USE_CUDA_DSA=1
# export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# export MASTER_PORT=$(( 60000 + (${SLURM_JOBID} % 1000) ))
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export NCCL_P2P_DISABLE=1
# export OMP_NUM_THREADS=1
# torchrun --nproc_per_node=2 main.py --llama-path '/home/rmfrieske/LLaMA_Models/LLaMA3_1/Meta-Llama-3.1-8B-Instruct' \
# --data-path '/home/rmfrieske/ERIT' \
# --result-path '/home/rmfrieske/results' \

# EXP_NAME="MVSA_ERIT_petrain_1e-3_50_epoch_31_layer1"
DATE=$(date +%Y-%m-%d)

torchrun  --master_port=$MASTER_PORT  --nproc_per_node=$WORLD_SIZE --nnodes=1 --node_rank=0 turntaking/voice_adapter/train_adapter.py \
    --adapter_type linear_seq \
    --streaming \
    --chunk_size 1600 \
    --batch_size 128 \
    --epochs 1 \
    --num_workers 8 \
    --max_samples 1000 \
    --enable_ddp \
    --lr 0.01 \
    --target_seq_len 63

# 
# python turntaking/voice_adapter/train_adapter.py \
#     --adapter_type cnn \
#     --streaming \
#     --chunk_size 1600 \
#     --batch_size 16 \
#     --epochs 5 \
#     --num_workers 0 \
#     --max_samples 1000 \

    # 'cnn': CNNAdapter,
    #         'linear': LinearAdapter,
    #         'linear_attn': LinearAttentionAdapter,
    #         'linear_seq': LinearSequenceAdapter,           # New
    #         'linear_attn_seq': LinearAt