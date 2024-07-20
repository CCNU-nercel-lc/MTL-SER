#!/bin/bash
# 在.sh文件中初始化conda并指定conda虚拟环境 # python run_emotion4_2.py --report_to none \
# 初始化conda
. /root/miniconda3/etc/profile.d/conda.sh
cd /2f592ba9440443f8863ede3d2d2b4927/liuchuan/coding/interspeech21_emotion-main/
conda activate baiduMTL
TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true

export MODEL=wav2vec2-base
export TOKENIZER=wav2vec2-base
export ALPHA=0.1
export BETA=0
export LR=5e-5
export ACC=4 # batch size * acc = 8
export WORKER_NUM=4

python run_emotion3.py \
--output_dir=output3/final2/alpha0.1beta0 \
--cache_dir=cache/ \
--num_train_epochs=100 \
--per_device_train_batch_size="2" \
--per_device_eval_batch_size="2" \
--gradient_accumulation_steps=$ACC \
--alpha $ALPHA \
--beta $BETA \
--dataset_name emotion \
--split_id 01F \
--evaluation_strategy="epoch" \
--save_total_limit="1" \
--save_steps="500" \
--eval_steps="500" \
--logging_steps="50" \
--logging_dir="final_log2/beta0/alpha0.1beta0" \
--load_best_model_at_end=True \
--metric_for_best_model="eval_acc" \
--do_train \
--do_eval \
--learning_rate=$LR \
--model_name_or_path=facebook/$MODEL \
--tokenizer facebook/$TOKENIZER \
--preprocessing_num_workers=$WORKER_NUM \
--dataloader_num_workers $WORKER_NUM
# --freeze_feature_extractor \
# --gradient_checkpointing true \
# --fp16 \
