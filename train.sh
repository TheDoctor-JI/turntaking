

##Eval for single gpu interactive session
export LOCAL_RANK=0
export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=localhost
export MASTER_PORT=13768 # Or any available port
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
python turntaking/train.py \
# --run_validation \
# --save_model \
# --max_eval_step 10 \
# --adapter_path '/scratch/vemotionsys/rmfrieske/LLaMA_Models/LLaMA3_1_Adapter/MVSA_ERIT_finetune/2025-05-16/COCO_Flikr_pretrain_0.001llama_gate_full_batch64_8gpu218897/' \

# python  pretrain.py \
# --llama_path /scratch/vemotionsys/rmfrieske/LLaMA_Models/Meta-Llama-3.1-8B/ \
# --lr=0.0001 \
# --eval_data_path '/home/rmfrieske/datasets/csv/affectnet_val.csv'  \
# --data_path  '/home/rmfrieske/datasets/csv/affectnet_train.csv'   '/home/rmfrieske/datasets/csv/emotion-detection-fer_train.csv' \
# --epochs 10 \
# --batch_size 16 \
# --experiment_name ${EXP_NAME} \
# --pure_bf16  \
# --gradient_clipping \
# --gradient_clipping_threshold 0.5 \
# --patience 10 \
# --contrastive_loss \
# --max_train_step 400 \

# # ##2GPU VERSUION
# torchrun  --master_port=$MASTER_PORT   --nnodes=1  --
# export MASTER_ADDR=localhost
# export WORLD_SIZE=2
# export MASTER_PORT=13742 # Or any available port
# export CUDA_VISIBLE_DEVICES=0,1
# ##2GPU VERSUION
# torchrun  --master_port=$MASTER_PORT   --nnodes=1  --nproc_per_node=$WORLD_SIZE pretrain.py  \
# --llama_path '/scratch/vemotionsys/rmfrieske/LLaMA_Models/LLaMA3_1/Meta-Llama-3.1-8B-Instruct' \
# --lr=0.001 \
# --eval_data_path '/home/rmfrieske/datasets/csv/affectnet_val.csv'  \
# --data_path  '/home/rmfrieske/datasets/csv/affectnet_train.csv'   '/home/rmfrieske/datasets/csv/emotion-detection-fer_train.csv' \
# --epochs 100 \
# --batch_size 16 \
# --experiment_name ${EXP_NAME} \
# --pure_bf16  \
# --gradient_clipping \
# --gradient_clipping_threshold 0.5 \
# --patience 7 \
# --num_workers 8 \
# --tiny_dataset \
# --enable_ddp \
# --run_validation \
# --save_model \


# --contrastive_loss \
# --adapter_path '/scratch/vemotionsys/rmfrieske/LLaMA_Models/LLaMA3_1_Adapter/MVSA_ERIT_finetune/2025-05-06/FER_AFFECTNET_EMOTIC_pretrain_1e-1_clip_none_llama_gate_full_batch8_1gpu_weight_decay0.1_dropout_0.1_213423/' \
# 
# --max_train_step 200 \
# --max_eval_step 10 \

# --experiment_name ${EXP_NAME} \
# --pure_bf16  \
# --gradient_clipping \
# --gradient_clipping_threshold 0.5 \
# --patience 7 \
# --run_validation \
# --save_model \
# --enable_ddp \
# --num_workers 8 \
# --contrastive_loss \
# --max_train_step 200 \
# --max_eval_step 10 \




