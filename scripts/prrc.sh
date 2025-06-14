# 1.Train PRRC model

torchrun --nproc_per_node=4  src/train_singletask.py \
    --data_folder DATA_FOLDER_CONTAINING_TRAIN_VAL_TEST \
    --pretrained_model_name_or_path ModernBERT-base \
    --output_dir ModernBERT-base-pro-2k-v1 \
    --seed 42 \
    --max_length 2048 \
    --batch_size 16 \
    --dimension professionalism \
    --num_workers 16

# 2.Test PRRC model
torchrun --nproc_per_node=4  src/test_singletask.py \
    --model_path ModernBERT-base-pro-2k-v1/best_ckpt \
    --data_folder DATA_FOLDER_CONTAINING_TRAIN_VAL_TEST \
    --dimension professionalism \
    --max_length 4096 \
    --batch_size 64 \
    --num_workers 16
